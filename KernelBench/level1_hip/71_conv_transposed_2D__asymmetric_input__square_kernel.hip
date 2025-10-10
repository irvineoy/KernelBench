// Transposed Conv2D (ConvTranspose2d) optimized HIP kernel for AMD MI300X (gfx942)
// Assumes stride=1, padding=0, dilation=1, output_padding=0
// Groups default to 1 (cannot be inferred from tensors without extra metadata).
// Weight layout matches PyTorch ConvTranspose2d: [Cin, Cout/groups, Kh, Kw]

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef TILE_W
#define TILE_W 16
#endif

#ifndef TILE_H
#define TILE_H 16
#endif

// Utility for integer ceil-div
__host__ __device__ inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

// Naive gather kernel (no shared memory). Computes one output element per thread.
// Output layout: [N, Cout, H_out, W_out]
// Input layout:  [N, Cin,  H_in,  W_in]
__global__ void deconv2d_naive_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,   // [Cin, CoutPG, Kh, Kw]
    const float* __restrict__ bias,     // [Cout] or nullptr
    float* __restrict__ output,
    int N, int Cin, int H_in, int W_in,
    int Cout, int Kh, int Kw,
    int H_out, int W_out,
    int groups) {

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int ow = blockIdx.x * blockDim.x + tx;
    const int oh = blockIdx.y * blockDim.y + ty;

    const int oc = blockIdx.z % Cout;
    const int n  = blockIdx.z / Cout;

    if (n >= N || oc >= Cout || oh >= H_out || ow >= W_out) return;

    // Grouped setup (defaults groups=1)
    const int cout_pg = Cout / groups;
    const int cin_pg  = Cin / groups;
    const int g       = oc / cout_pg;
    const int oc_in_g = oc - g * cout_pg;
    const int ic_base = g * cin_pg;

    // Determine kernel range that maps to valid input
    const int kh_start = max(0, oh - (H_in - 1));
    const int kh_end   = min(Kh - 1, oh);
    const int kw_start = max(0, ow - (W_in - 1));
    const int kw_end   = min(Kw - 1, ow);

    float sum = 0.0f;

    // Iterate over input channels in this group
    for (int ic_local = 0; ic_local < cin_pg; ++ic_local) {
        const int ic = ic_base + ic_local;

        // For given (oh, ow), valid ih = oh - kh, iw = ow - kw
        // Unroll small Kh/Kw partially for common 3x3/5x5 cases
        #pragma unroll 4
        for (int kh = kh_start; kh <= kh_end; ++kh) {
            const int ih = oh - kh;
            const int in_row_base = (((n * Cin + ic) * H_in + ih) * W_in);

            // Weight base for this (ic_local, oc_in_g, kh)
            const int w_base = (((ic) * cout_pg + oc_in_g) * Kh + kh) * Kw;

            #pragma unroll 4
            for (int kw = kw_start; kw <= kw_end; ++kw) {
                const int iw = ow - kw;

                const float x = input[in_row_base + iw];
                const float w = weight[w_base + kw];

                sum += x * w;
            }
        }
    }

    if (bias) sum += bias[oc];

    output[(((n * Cout + oc) * H_out + oh) * W_out + ow)] = sum;
}

// Shared-memory optimized kernel: cache per-(block, oc) weights in LDS.
// Each block handles one (n, oc) tile over spatial output.
extern "C"
__global__ void deconv2d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,   // [Cin, CoutPG, Kh, Kw]
    const float* __restrict__ bias,     // [Cout] or nullptr
    float* __restrict__ output,
    int N, int Cin, int H_in, int W_in,
    int Cout, int Kh, int Kw,
    int H_out, int W_out,
    int groups) {

    extern __shared__ float s_weight[]; // size = (Cin/groups)*Kh*Kw

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int t  = ty * blockDim.x + tx;

    const int ow = blockIdx.x * blockDim.x + tx;
    const int oh = blockIdx.y * blockDim.y + ty;

    const int oc = blockIdx.z % Cout;
    const int n  = blockIdx.z / Cout;

    if (n >= N || oc >= Cout) return;

    // Grouped setup (defaults groups=1)
    const int cout_pg = Cout / groups;
    const int cin_pg  = Cin / groups;
    const int g       = oc / cout_pg;
    const int oc_in_g = oc - g * cout_pg;
    const int ic_base = g * cin_pg;

    // Load weights into shared memory for this (oc) across all ic_local, kh, kw
    const int weight_elems = cin_pg * Kh * Kw;
    for (int idx = t; idx < weight_elems; idx += blockDim.x * blockDim.y) {
        const int ic_local = idx / (Kh * Kw);
        const int rem      = idx % (Kh * Kw);
        const int kh       = rem / Kw;
        const int kw       = rem % Kw;

        const int ic = ic_base + ic_local;

        // Global weight index: [ic, oc_in_g, kh, kw]
        const int w_idx = (((ic) * cout_pg + oc_in_g) * Kh + kh) * Kw + kw;
        s_weight[idx] = weight[w_idx];
    }
    __syncthreads();

    // Guard for threads out of bounds
    if (oh >= H_out || ow >= W_out) return;

    // Determine kernel range that maps to valid input
    const int kh_start = max(0, oh - (H_in - 1));
    const int kh_end   = min(Kh - 1, oh);
    const int kw_start = max(0, ow - (W_in - 1));
    const int kw_end   = min(Kw - 1, ow);

    float sum = 0.0f;

    // Iterate over input channels in this group
    for (int ic_local = 0; ic_local < cin_pg; ++ic_local) {
        const int ic = ic_base + ic_local;

        #pragma unroll 4
        for (int kh = kh_start; kh <= kh_end; ++kh) {
            const int ih = oh - kh;
            const int in_row_base = (((n * Cin + ic) * H_in + ih) * W_in);

            // s_weight layout: [ic_local, kh, kw] tightly packed
            const int sw_base = (ic_local * Kh + kh) * Kw;

            #pragma unroll 4
            for (int kw = kw_start; kw <= kw_end; ++kw) {
                const int iw = ow - kw;

                const float x = input[in_row_base + iw];
                const float w = s_weight[sw_base + kw];

                sum += x * w;
            }
        }
    }

    if (bias) sum += bias[oc];

    output[(((n * Cout + oc) * H_out + oh) * W_out + ow)] = sum;
}

// Host wrapper: accepts input and weight (and optional bias) tensors only.
// Infers configuration from tensor shapes and uses stride=1, padding=0, dilation=1, output_padding=0, groups=1 (default).
static at::Tensor run_impl(at::Tensor input, at::Tensor weight, c10::optional<at::Tensor> bias_opt) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Only float32 weight supported");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "Bias must be a CUDA/HIP tensor");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "Only float32 bias supported");
    }

    auto x = input.contiguous();
    auto w = weight.contiguous();
    at::Tensor b;
    const float* d_bias = nullptr;
    if (bias_opt.has_value()) {
        b = bias_opt.value().contiguous();
        d_bias = b.data_ptr<float>();
    }

    // Extract dimensions
    const int N    = x.size(0);
    const int Cin  = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);

    // Weight: [Cin, Cout/groups, Kh, Kw]
    const int Kh = w.size(2);
    const int Kw = w.size(3);

    // We cannot infer groups from tensors reliably; use default groups=1 per instructions.
    const int groups = 1;
    const int Cout_per_group = w.size(1);
    const int Cout = Cout_per_group * groups;

    // Validate bias if provided
    if (bias_opt.has_value()) {
        TORCH_CHECK(b.numel() == Cout, "Bias size mismatch: expected ", Cout, " got ", b.numel());
    }

    // Output dimensions for stride=1, padding=0, dilation=1, output_padding=0
    const int H_out = H_in + Kh - 1;
    const int W_out = W_in + Kw - 1;

    // Allocate output
    auto y = at::zeros({N, Cout, H_out, W_out}, x.options());

    // Launch configuration
    dim3 block(TILE_W, TILE_H);
    dim3 grid(iDivUp(W_out, TILE_W), iDivUp(H_out, TILE_H), N * Cout);

    const float* d_x = x.data_ptr<float>();
    const float* d_w = w.data_ptr<float>();
    float* d_y = y.data_ptr<float>();

    // Decide whether to use shared-memory kernel
    // Shared memory per block: (Cin/groups) * Kh * Kw * sizeof(float)
    size_t s_elems = static_cast<size_t>(Cin / groups) * static_cast<size_t>(Kh) * static_cast<size_t>(Kw);
    size_t s_bytes = s_elems * sizeof(float);

    // Use shared kernel if it fits within a safe LDS budget (<= 60 KB)
    const size_t LDS_BUDGET = 60 * 1024;

    if (s_bytes > 0 && s_bytes <= LDS_BUDGET) {
        hipLaunchKernelGGL(
            deconv2d_shared_kernel,
            grid, block, s_bytes, 0,
            d_x, d_w, d_bias, d_y,
            N, Cin, H_in, W_in,
            Cout, Kh, Kw,
            H_out, W_out,
            groups
        );
    } else {
        hipLaunchKernelGGL(
            deconv2d_naive_kernel,
            grid, block, 0, 0,
            d_x, d_w, d_bias, d_y,
            N, Cin, H_in, W_in,
            Cout, Kh, Kw,
            H_out, W_out,
            groups
        );
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

// Overloads exposed to Python
at::Tensor run(at::Tensor input, at::Tensor weight) {
    return run_impl(input, weight, c10::nullopt);
}
at::Tensor run_bias(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    return run_impl(input, weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "ConvTranspose2d (stride=1, pad=0, groups=1) - HIP");
    m.def("run_bias", &run_bias, "ConvTranspose2d with bias (stride=1, pad=0, groups=1) - HIP");
}