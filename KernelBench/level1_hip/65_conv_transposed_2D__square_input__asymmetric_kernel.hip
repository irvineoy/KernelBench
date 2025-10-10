// ConvTranspose2d (deconvolution) NCHW -> NCHW for stride=1, padding=0, output_padding=0, dilation=1, groups=1
// Weight layout follows PyTorch: [in_channels, out_channels, kH, kW] (groups=1 assumed).
// The kernel caches per-(oc) weight tiles in shared memory by chunks of input channels to reduce DRAM traffic.

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#ifndef TILE_W
#define TILE_W 16
#endif
#ifndef TILE_H
#define TILE_H 16
#endif
#ifndef IC_TILE
#define IC_TILE 16  // input-channel chunk size for shared-memory weight staging
#endif

// Wavefront size on AMD is 64; use 256 threads per block (4 waves)
__global__ __launch_bounds__(TILE_W * TILE_H, 2)
void conv_transpose2d_kernel(
    const float* __restrict__ x,   // [N, C_in, H, W]
    const float* __restrict__ w,   // [C_in, C_out, kH, kW]  (groups=1)
    float* __restrict__ y,         // [N, C_out, H_out, W_out]
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int H_out, int W_out)
{
    extern __shared__ float s_w[]; // size = IC_TILE * kH * kW floats

    // Thread coordinates within the output tile
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Output coordinates
    const int ow = blockIdx.x * TILE_W + tx;
    const int oh = blockIdx.y * TILE_H + ty;

    // Batch and output-channel derived from grid.z
    const int oc = blockIdx.z % C_out;
    const int n  = blockIdx.z / C_out;

    if (n >= N || oc >= C_out || oh >= H_out || ow >= W_out) return;

    // Accumulator
    float sum = 0.0f;

    const int kwH = kH * kW;
    const int threads_per_block = blockDim.x * blockDim.y;
    const int tid = ty * blockDim.x + tx;

    // Loop over input channels in chunks
    for (int ic0 = 0; ic0 < C_in; ic0 += IC_TILE) {
        const int ic_chunk = min(IC_TILE, C_in - ic0);
        const int chunk_elems = ic_chunk * kwH;

        // Stage weights for this (oc) and current input-channel chunk into shared memory
        // s_w layout: [ic_chunk][kH*kW]
        for (int idx = tid; idx < chunk_elems; idx += threads_per_block) {
            int lic   = idx / kwH;          // local ic in chunk
            int rem   = idx - lic * kwH;
            int kh    = rem / kW;
            int kw    = rem - kh * kW;
            int ic    = ic0 + lic;

            // Weight index (contiguous): (((ic) * C_out + oc) * kH + kh) * kW + kw
            int w_idx = (((ic * C_out) + oc) * kH + kh) * kW + kw;
            s_w[idx] = w[w_idx];
        }
        __syncthreads();

        // Accumulate this chunk's contribution
        // Formula for stride=1, pad=0, dilation=1 (conv_transpose):
        // y[n, oc, oh, ow] += sum_{ic,kh,kw} x[n, ic, ih=oh - kh, iw=ow - kw] * w[ic, oc, kh, kw]
        // Only valid if ih, iw in range.
        #pragma unroll 1
        for (int lic = 0; lic < ic_chunk; ++lic) {
            const int ic = ic0 + lic;
            const float* __restrict__ w_ic = &s_w[lic * kwH];

            // Unroll kernel loops moderately
            #pragma unroll 4
            for (int kh = 0; kh < kH; ++kh) {
                const int ih = oh - kh;
                if ((unsigned)ih >= (unsigned)H) continue;

                #pragma unroll 4
                for (int kw = 0; kw < kW; ++kw) {
                    const int iw = ow - kw;
                    if ((unsigned)iw >= (unsigned)W) continue;

                    // x index: (((n*C_in + ic) * H + ih) * W + iw)
                    const int x_idx = (((n * C_in + ic) * H + ih) * W + iw);
                    const float xv = x[x_idx];
                    const float wv = w_ic[kh * kW + kw];

                    sum += xv * wv;
                }
            }
        }
        __syncthreads();
    }

    // Store result
    // y index: (((n*C_out + oc) * H_out + oh) * W_out + ow)
    const int y_idx = (((n * C_out + oc) * H_out + oh) * W_out + ow);
    y[y_idx] = sum;
}

at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 tensors are supported");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Only float32 weights are supported");
    TORCH_CHECK(input.dim() == 4, "input must be NCHW");
    TORCH_CHECK(weight.dim() == 4, "weight must be [C_in, C_out, kH, kW] for ConvTranspose2d (groups=1)");

    // Enforce contiguous layout
    auto x = input.contiguous();
    auto w = weight.contiguous();

    // Extract dimensions
    const int N    = static_cast<int>(x.size(0));
    const int C_in = static_cast<int>(x.size(1));
    const int H    = static_cast<int>(x.size(2));
    const int W    = static_cast<int>(x.size(3));

    const int C_in_w = static_cast<int>(w.size(0));
    const int C_out  = static_cast<int>(w.size(1));
    const int kH     = static_cast<int>(w.size(2));
    const int kW     = static_cast<int>(w.size(3));

    // Assume groups=1, stride=1, padding=0, output_padding=0, dilation=1 (matches provided model defaults)
    TORCH_CHECK(C_in == C_in_w, "Mismatch: input C_in and weight's first dimension must match (groups=1).");

    const int stride_h = 1, stride_w = 1;
    const int pad_h = 0, pad_w = 0;
    const int dil_h = 1, dil_w = 1;
    const int out_pad_h = 0, out_pad_w = 0;

    // Output size formula (PyTorch): (H-1)*stride - 2*padding + dilation*(k-1) + output_padding + 1
    const int H_out = (H - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + out_pad_h + 1;
    const int W_out = (W - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + out_pad_w + 1;

    auto y = at::empty({N, C_out, H_out, W_out}, x.options());

    // Launch configuration
    dim3 block(TILE_W, TILE_H);
    dim3 grid(
        (W_out + TILE_W - 1) / TILE_W,
        (H_out + TILE_H - 1) / TILE_H,
        N * C_out
    );

    // Dynamic shared memory size for staged weights of one IC_TILE chunk
    size_t shared_bytes = static_cast<size_t>(IC_TILE) * static_cast<size_t>(kH) * static_cast<size_t>(kW) * sizeof(float);

    // Launch
    hipLaunchKernelGGL(
        conv_transpose2d_kernel,
        grid, block, shared_bytes, 0,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C_in, H, W,
        C_out, kH, kW,
        H_out, W_out
    );

    // Error checks
    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "ConvTranspose2d (deconvolution) with stride=1, padding=0, groups=1 (HIP)");
}