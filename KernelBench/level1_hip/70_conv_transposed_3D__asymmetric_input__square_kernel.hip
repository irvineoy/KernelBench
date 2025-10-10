// ConvTranspose3d (transposed 3D convolution) HIP kernel for PyTorch
// Assumptions based on provided model defaults:
//   stride = 1, padding = 0, dilation = 1, output_padding = 0, groups = 1, bias = False
// Weight layout for ConvTranspose3d in PyTorch: [in_channels, out_channels, kD, kH, kW]
// Input  layout: [N, in_channels, D, H, W]
// Output layout: [N, out_channels, D + kD - 1, H + kH - 1, W + kW - 1]
//
// The kernel computes output via gather (no atomics):
//   y[n, oc, od, oh, ow] = sum_ic sum_kd,kh,kw x[n, ic, od-kd, oh-kh, ow-kw] * w[ic, oc, kd, kh, kw]
//
// Optimizations:
// - Wave-friendly block size (256 threads): blockDim=(128,2,1)
// - Coalesced access across W dimension (threadIdx.x -> ow)
// - Shared memory cache of the entire weight slice for current oc (IC * KD * KH * KW)
// - Tight bounds for kernel loops to reduce branches
// - 64-bit indexing safety, restricted pointers

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef TILE_W
#define TILE_W 128   // multiple of 64 (wavefront), maps to output width
#endif

#ifndef TILE_H
#define TILE_H 2     // small tile in height to keep 256 threads/block
#endif

// Launch bounds must match the launch configuration below (256 threads)
__global__ __launch_bounds__(TILE_W * TILE_H, 1)
void conv_transpose3d_kernel(
    const float* __restrict__ input,    // [N, IC, D, H, W]
    const float* __restrict__ weight,   // [IC, OC, KD, KH, KW]
    float* __restrict__ output,         // [N, OC, OD, OH, OW]
    const int32_t* __restrict__ meta    // packed ints with sizes
) {
    // Unpack metadata
    const int N  = meta[0];
    const int IC = meta[1];
    const int D  = meta[2];
    const int H  = meta[3];
    const int W  = meta[4];
    const int OC = meta[5];
    const int KD = meta[6];
    const int KH = meta[7];
    const int KW = meta[8];
    const int OD = meta[9];
    const int OH = meta[10];
    const int OW = meta[11];
    // meta[12..15] reserved

    // Thread mapping
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;

    int z = blockIdx.z;         // combine (n, oc, od)
    const int od = z % OD; z /= OD;
    const int oc = z % OC; z /= OC;
    const int n  = z;           // 0..N-1

    // Load weight slice for current oc into shared memory:
    // s_w layout: [IC, KD, KH, KW] flattened
    extern __shared__ float s_w[];
    const int threads = blockDim.x * blockDim.y;
    const int t = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_w_elems = IC * KD * KH * KW;

    // global weight layout strides
    const int64_t w_stride_kw = 1;
    const int64_t w_stride_kh = KW;
    const int64_t w_stride_kd = (int64_t)KH * KW;
    const int64_t w_stride_oc = (int64_t)KD * KH * KW;
    const int64_t w_stride_ic = (int64_t)OC * w_stride_oc;

    for (int idx = t; idx < total_w_elems; idx += threads) {
        int tmp = idx;
        const int kw = tmp % KW; tmp /= KW;
        const int kh = tmp % KH; tmp /= KH;
        const int kd = tmp % KD; tmp /= KD;
        const int ic = tmp;      // 0..IC-1

        const int64_t w_index =
            ((int64_t)ic) * w_stride_ic +
            ((int64_t)oc) * w_stride_oc +
            ((int64_t)kd) * w_stride_kd +
            ((int64_t)kh) * w_stride_kh +
            ((int64_t)kw) * w_stride_kw;

        s_w[idx] = weight[w_index];
    }
    __syncthreads();

    if (ow >= OW || oh >= OH) return;  // out-of-bounds threads skip work

    // Precompute valid kernel ranges to avoid inner-branching
    // For id = od - kd in [0, D-1] => kd in [max(0, od-(D-1)) .. min(KD-1, od)]
    const int kd_start = max(0, od - (D - 1));
    const int kd_end   = min(KD - 1, od);

    const int kh_start = max(0, oh - (H - 1));
    const int kh_end   = min(KH - 1, oh);

    const int kw_start = max(0, ow - (W - 1));
    const int kw_end   = min(KW - 1, ow);

    float acc = 0.0f;

    // Input strides
    const int64_t in_stride_w = 1;
    const int64_t in_stride_h = W;
    const int64_t in_stride_d = (int64_t)H * W;
    const int64_t in_stride_c = (int64_t)D * H * W;
    const int64_t in_stride_n = (int64_t)IC * D * H * W;

    // Output index base not needed until final store

    // Accumulate
    for (int ic = 0; ic < IC; ++ic) {
        // Flattened base offset for weight slice in shared memory for this ic
        const int w_ic_base = ic * (KD * KH * KW);

        // Precompute base offset for input for this (n, ic)
        const int64_t in_nic_base = ((int64_t)n) * in_stride_n + ((int64_t)ic) * in_stride_c;

        for (int kd = kd_start; kd <= kd_end; ++kd) {
            const int id = od - kd;  // guaranteed 0..D-1
            const int64_t in_d_base = in_nic_base + ((int64_t)id) * in_stride_d;
            const int w_ic_kd_base = w_ic_base + kd * (KH * KW);

            for (int kh = kh_start; kh <= kh_end; ++kh) {
                const int ih = oh - kh;  // guaranteed 0..H-1
                const int64_t in_h_base = in_d_base + ((int64_t)ih) * in_stride_h;
                const int w_ic_kdkh_base = w_ic_kd_base + kh * KW;

                // Iterate over kw with coalesced input access along width
                #pragma unroll 3
                for (int kw = kw_start; kw <= kw_end; ++kw) {
                    const int iw = ow - kw;  // guaranteed 0..W-1
                    const float xval = input[in_h_base + iw];
                    const float wval = s_w[w_ic_kdkh_base + kw];
                    acc += xval * wval;
                }
            }
        }
    }

    // Store output
    const int64_t out_stride_w = 1;
    const int64_t out_stride_h = OW;
    const int64_t out_stride_d = (int64_t)OH * OW;
    const int64_t out_stride_c = (int64_t)OD * OH * OW;
    const int64_t out_stride_n = (int64_t)OC * OD * OH * OW;

    const int64_t out_index =
        ((int64_t)n) * out_stride_n +
        ((int64_t)oc) * out_stride_c +
        ((int64_t)od) * out_stride_d +
        ((int64_t)oh) * out_stride_h +
        ((int64_t)ow) * out_stride_w;

    output[out_index] = acc;
}

at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "only float32 supported for input");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "only float32 supported for weight");
    TORCH_CHECK(input.dim() == 5, "input must be 5D [N, IC, D, H, W]");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D [IC, OC, KD, KH, KW] for ConvTranspose3d");
    TORCH_CHECK(input.device() == weight.device(), "input and weight must be on the same device");

    // Extract shapes
    const int64_t N  = input.size(0);
    const int64_t IC = input.size(1);
    const int64_t D  = input.size(2);
    const int64_t H  = input.size(3);
    const int64_t W  = input.size(4);

    const int64_t W_IC = weight.size(0);
    const int64_t W_OC = weight.size(1);
    const int64_t KD   = weight.size(2);
    const int64_t KH   = weight.size(3);
    const int64_t KW   = weight.size(4);

    TORCH_CHECK(W_IC == IC, "weight.size(0) must equal input.channels (in_channels)");
    // groups assumed = 1 => out_channels == W_OC
    const int64_t OC = W_OC;

    // Defaults: stride=1, padding=0, dilation=1, output_padding=0
    const int64_t OD = D + KD - 1;
    const int64_t OH = H + KH - 1;
    const int64_t OW = W + KW - 1;

    // Ensure contiguous
    auto in  = input.contiguous();
    auto w   = weight.contiguous();
    auto out = at::empty({N, OC, OD, OH, OW}, in.options());

    // Build metadata tensor on CPU, then move to GPU
    auto meta_cpu = at::empty({16}, at::TensorOptions().dtype(at::kInt).device(at::kCPU));
    int32_t* m = meta_cpu.data_ptr<int32_t>();
    m[0]  = static_cast<int32_t>(N);
    m[1]  = static_cast<int32_t>(IC);
    m[2]  = static_cast<int32_t>(D);
    m[3]  = static_cast<int32_t>(H);
    m[4]  = static_cast<int32_t>(W);
    m[5]  = static_cast<int32_t>(OC);
    m[6]  = static_cast<int32_t>(KD);
    m[7]  = static_cast<int32_t>(KH);
    m[8]  = static_cast<int32_t>(KW);
    m[9]  = static_cast<int32_t>(OD);
    m[10] = static_cast<int32_t>(OH);
    m[11] = static_cast<int32_t>(OW);
    m[12] = 0; m[13] = 0; m[14] = 0; m[15] = 0; // reserved
    auto meta = meta_cpu.to(in.device(), /*non_blocking=*/false, /*copy=*/true);

    // Launch configuration
    dim3 block(TILE_W, TILE_H, 1); // 128x2 = 256 threads (multiple of 64)
    dim3 grid(
        static_cast<unsigned int>((OW + block.x - 1) / block.x),
        static_cast<unsigned int>((OH + block.y - 1) / block.y),
        static_cast<unsigned int>(N * OC * OD)
    );

    // Shared memory size: IC * KD * KH * KW floats
    size_t shmem_bytes = static_cast<size_t>(IC * KD * KH * KW) * sizeof(float);

    hipLaunchKernelGGL(
        conv_transpose3d_kernel,
        grid, block, shmem_bytes, 0,
        in.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        meta.data_ptr<int32_t>());

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Transposed 3D Convolution (ConvTranspose3d) - HIP optimized (stride=1, pad=0, dil=1, groups=1)");
}