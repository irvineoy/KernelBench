// Conv3D (N, C, D, H, W) with groups, stride=1, padding=0, dilation=1
// Optimized for AMD MI300X (gfx942). Coalesced along W, small shared cache for weights.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Tile sizes (multiple of 64 threads per block recommended)
#define TILE_W 16
#define TILE_H 16

// Kernel: Each block computes a TILE_H x TILE_W tile for a fixed (n, oc, od).
// Grid mapping: grid.z = N * OC * OD
// Memory layout (contiguous): input [N, C, D, H, W], weight [OC, CinPerG, KD, KH, KW], output [N, OC, OD, OH, OW]
__global__ void conv3d_ncdhw_kernel(
    const float* __restrict__ x,        // [N, C, D, H, W]
    const float* __restrict__ w,        // [OC, CinPerG, KD, KH, KW]
    const float* __restrict__ b,        // [OC] or nullptr
    float* __restrict__ y,              // [N, OC, OD, OH, OW]
    // Scalar metadata (host-only derived; allowed here)
    int N, int C, int D, int H, int W,
    int OC, int OD, int OH, int OW,
    int KD, int KH, int KW,
    int CinPerG, int G,
    int use_shm, int has_bias)
{
    extern __shared__ float s_w[]; // Size S = CinPerG * KD * KH * KW (when use_shm != 0)

    // Decode (n, oc, od) from grid.z
    int z = blockIdx.z;
    int n  = z / (OC * OD);
    int r  = z % (OC * OD);
    int oc = r / OD;
    int od = r % OD;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ow = blockIdx.x * blockDim.x + tx;
    int oh = blockIdx.y * blockDim.y + ty;

    if (n >= N || oc >= OC || od >= OD || oh >= OH || ow >= OW) return;

    // Group mapping
    int oc_per_group = OC / G;
    int g = oc / oc_per_group;
    int c_start = g * CinPerG;

    // Shared-cache weights per (oc)
    const int S = CinPerG * KD * KH * KW;
    int lid = ty * blockDim.x + tx;

    if (use_shm) {
        // Load weights for this oc into shared memory
        // Global base offset for oc: oc * (CinPerG*KD*KH*KW)
        int w_base = oc * S;
        for (int i = lid; i < S; i += blockDim.x * blockDim.y) {
            s_w[i] = w[w_base + i];
        }
        __syncthreads();
    }

    // Accumulator (init with optional bias)
    float acc = 0.0f;
    if (has_bias) {
        acc = b[oc];
    }

    // Precompute output and input strides
    // Input strides
    long stride_c = (long)D * H * W;
    long stride_d = (long)H * W;
    long stride_h = (long)W;
    // Weight strides (within one oc slice)
    int w_stride_ic = KD * KH * KW;
    int w_stride_kd = KH * KW;
    int w_stride_kh = KW;

    // Compute
    // Since stride=1, pad=0, dil=1: in_d = od + kd, in_h = oh + kh, in_w = ow + kw
    // Bounds guaranteed by output size computation.
    #pragma unroll 1
    for (int ic = 0; ic < CinPerG; ++ic) {
        int c = c_start + ic;

        long in_base = ((long)n * C + c) * stride_c + (long)(od) * stride_d + (long)(oh) * stride_h + (long)ow;
        int w_ic_base = ic * w_stride_ic;

        #pragma unroll 4
        for (int kd = 0; kd < KD; ++kd) {
            long in_d_base = in_base + (long)kd * stride_d;
            int w_kd_base = w_ic_base + kd * w_stride_kd;

            #pragma unroll 4
            for (int kh = 0; kh < KH; ++kh) {
                long in_h_base = in_d_base + (long)kh * stride_h;
                int w_kh_base = w_kd_base + kh * w_stride_kh;

                // Unroll KW loop moderately; KW could be small (e.g., 7)
                #pragma unroll 8
                for (int kw = 0; kw < KW; ++kw) {
                    float xv = __ldg(&x[in_h_base + kw]); // coalesced along W
                    float ww;
                    if (use_shm) {
                        ww = s_w[w_kh_base + kw];
                    } else {
                        // Global offset within oc-slice:
                        ww = w[oc * S + w_kh_base + kw];
                    }
                    acc = fmaf(xv, ww, acc);
                }
            }
        }
    }

    // Store
    long out_idx = ((((long)n * OC + oc) * OD + od) * OH + oh) * OW + ow;
    y[out_idx] = acc;
}

// Host wrapper: tensors only. We derive all metadata from tensor sizes.
// Stride=1, padding=0, dilation=1, groups inferred from weight shape.
static at::Tensor run_impl(at::Tensor input, at::Tensor weight, c10::optional<at::Tensor> bias_opt) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA/HIP device");
    TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA/HIP device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt->is_cuda(), "bias must be on CUDA/HIP device");
        TORCH_CHECK(bias_opt->scalar_type() == at::kFloat, "bias must be float32");
    }

    // Enforce contiguous
    auto x = input.contiguous();
    auto w = weight.contiguous();
    at::Tensor b;
    bool has_bias = bias_opt.has_value();
    if (has_bias) b = bias_opt->contiguous();

    // Input sizes: x[N, C, D, H, W]
    TORCH_CHECK(x.dim() == 5, "input must be 5D (N, C, D, H, W)");
    int N = static_cast<int>(x.size(0));
    int C = static_cast<int>(x.size(1));
    int D = static_cast<int>(x.size(2));
    int H = static_cast<int>(x.size(3));
    int W = static_cast<int>(x.size(4));

    // Weight sizes: w[OC, CinPerG, KD, KH, KW]
    TORCH_CHECK(w.dim() == 5, "weight must be 5D (OC, CinPerG, KD, KH, KW)");
    int OC = static_cast<int>(w.size(0));
    int CinPerG = static_cast<int>(w.size(1));
    int KD = static_cast<int>(w.size(2));
    int KH = static_cast<int>(w.size(3));
    int KW = static_cast<int>(w.size(4));

    // Infer groups
    TORCH_CHECK(C % CinPerG == 0, "Input channels not divisible by weight CinPerG");
    int G = C / CinPerG;
    TORCH_CHECK(OC % G == 0, "Out channels not divisible by groups");

    // Stride=1, padding=0, dilation=1 (as per model default)
    int OD = D - KD + 1;
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    TORCH_CHECK(OD > 0 && OH > 0 && OW > 0, "Invalid output size; check kernel vs input dims");

    // Allocate output
    auto y = at::empty({N, OC, OD, OH, OW}, x.options());

    // Launch config
    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid((OW + TILE_W - 1) / TILE_W,
              (OH + TILE_H - 1) / TILE_H,
              N * OC * OD);

    // Decide shared weight cache usage (per-block)
    size_t S = static_cast<size_t>(CinPerG) * KD * KH * KW;  // elements per oc
    size_t bytes_shm = 0;
    int use_shm = 0;
    // LDS per CU is 64KB; be conservative
    if (S * sizeof(float) <= 64 * 1024) {
        bytes_shm = S * sizeof(float);
        use_shm = 1;
    }

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    const float* b_ptr = has_bias ? b.data_ptr<float>() : nullptr;
    float* y_ptr = y.data_ptr<float>();

    hipLaunchKernelGGL(
        conv3d_ncdhw_kernel,
        grid, block, bytes_shm, 0,
        x_ptr, w_ptr, b_ptr, y_ptr,
        N, C, D, H, W,
        OC, OD, OH, OW,
        KD, KH, KW,
        CinPerG, G,
        use_shm, has_bias ? 1 : 0
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

// Overloads that accept only tensors (no scalar config)
at::Tensor run(at::Tensor input, at::Tensor weight) {
    return run_impl(std::move(input), std::move(weight), c10::nullopt);
}

at::Tensor run_bias(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    return run_impl(std::move(input), std::move(weight), std::move(bias));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Conv3D NCDHW, stride=1, pad=0, dil=1 (HIP)");
    m.def("run", &run_bias, "Conv3D NCDHW with bias, stride=1, pad=0, dil=1 (HIP)");
}