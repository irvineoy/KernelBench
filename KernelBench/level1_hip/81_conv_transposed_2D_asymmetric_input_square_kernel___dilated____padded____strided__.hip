#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef TILE_X
#define TILE_X 16
#endif
#ifndef TILE_Y
#define TILE_Y 16
#endif

// Hard-coded model configuration inferred from the provided target model instantiation
// ConvTranspose2d with stride=5, padding=1, dilation=2, square kernel
// Note: kernel size is inferred from weight tensor dimensions
#define STRIDE    5
#define PADDING   1
#define DILATION  2

// Utility to check HIP errors
static inline void hipCheck(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        TORCH_CHECK(false, msg, ": ", hipGetErrorString(err));
    }
}

// Gather-style transposed convolution kernel with per-(batch, out_channel) weight cache in shared memory.
// Each block computes a TILE_Y x TILE_X tile of output for a fixed (batch b, out_channel oc).
// This avoids atomics and ensures deterministic accumulation.
__global__ void __launch_bounds__(TILE_X * TILE_Y, 4)
deconv2d_transpose_gather_shared_kernel(
    const float* __restrict__ x,      // [N, Cin, Hin, Win]
    const float* __restrict__ w,      // [Cin, Cout, Kh, Kw] (PyTorch ConvTranspose2d weight layout)
    const float* __restrict__ bias,   // [Cout] or nullptr
    float* __restrict__ y,            // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Kh, int Kw,
    int Hout, int Wout)
{
    extern __shared__ float s_weight[]; // size = Cin * Kh * Kw

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int out_w = blockIdx.x * blockDim.x + tx;
    const int out_h = blockIdx.y * blockDim.y + ty;
    const int bc = blockIdx.z; // combined batch and out_channel
    const int oc = bc % Cout;
    const int b  = bc / Cout;

    // Load weights for this (oc) into shared memory once per block
    const int weight_elems = Cin * Kh * Kw;
    for (int idx = tid; idx < weight_elems; idx += blockDim.x * blockDim.y) {
        int cin = idx / (Kh * Kw);
        int rem = idx % (Kh * Kw);
        int kh  = rem / Kw;
        int kw  = rem % Kw;
        // Weight layout: [Cin, Cout, Kh, Kw]
        int w_idx = (((cin * Cout + oc) * Kh + kh) * Kw + kw);
        s_weight[idx] = w[w_idx];
    }
    __syncthreads();

    if (b >= N || oc >= Cout || out_h >= Hout || out_w >= Wout) {
        return;
    }

    float acc = 0.0f;

    // For each input channel and kernel position, compute corresponding input indices
    // Condition for valid contribution:
    // ih such that out_h = ih*STRIDE - PADDING + kh*DILATION  => ih = (out_h + PADDING - kh*DILATION) / STRIDE, must be integer and in [0, Hin)
    // iw such that out_w = iw*STRIDE - PADDING + kw*DILATION  => iw = (out_w + PADDING - kw*DILATION) / STRIDE, must be integer and in [0, Win)
    for (int cin = 0; cin < Cin; ++cin) {
        const int base = cin * (Kh * Kw);
        #pragma unroll
        for (int kh = 0; kh < Kh; ++kh) {
            int tH = out_h + PADDING - kh * DILATION;
            // Check divisibility by STRIDE
            if ((tH % STRIDE) != 0) continue;
            int ih = tH / STRIDE;
            if (ih < 0 || ih >= Hin) continue;

            #pragma unroll
            for (int kw = 0; kw < Kw; ++kw) {
                int tW = out_w + PADDING - kw * DILATION;
                if ((tW % STRIDE) != 0) continue;
                int iw = tW / STRIDE;
                if (iw < 0 || iw >= Win) continue;

                int x_idx = (((b * Cin + cin) * Hin + ih) * Win + iw);
                float xv = x[x_idx];
                float wv = s_weight[base + kh * Kw + kw];
                acc += xv * wv;
            }
        }
    }

    if (bias != nullptr) {
        acc += bias[oc];
    }

    int y_idx = (((b * Cout + oc) * Hout + out_h) * Wout + out_w);
    y[y_idx] = acc;
}

// Fallback kernel without shared memory (weights read from global memory)
// Used when Cin*Kh*Kw exceeds shared memory budget; correctness-first.
__global__ void __launch_bounds__(TILE_X * TILE_Y, 4)
deconv2d_transpose_gather_global_kernel(
    const float* __restrict__ x,      // [N, Cin, Hin, Win]
    const float* __restrict__ w,      // [Cin, Cout, Kh, Kw]
    const float* __restrict__ bias,   // [Cout] or nullptr
    float* __restrict__ y,            // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Kh, int Kw,
    int Hout, int Wout)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int out_w = blockIdx.x * blockDim.x + tx;
    const int out_h = blockIdx.y * blockDim.y + ty;
    const int bc = blockIdx.z;
    const int oc = bc % Cout;
    const int b  = bc / Cout;

    if (b >= N || oc >= Cout || out_h >= Hout || out_w >= Wout) {
        return;
    }

    float acc = 0.0f;

    for (int cin = 0; cin < Cin; ++cin) {
        #pragma unroll
        for (int kh = 0; kh < Kh; ++kh) {
            int tH = out_h + PADDING - kh * DILATION;
            if ((tH % STRIDE) != 0) continue;
            int ih = tH / STRIDE;
            if (ih < 0 || ih >= Hin) continue;

            #pragma unroll
            for (int kw = 0; kw < Kw; ++kw) {
                int tW = out_w + PADDING - kw * DILATION;
                if ((tW % STRIDE) != 0) continue;
                int iw = tW / STRIDE;
                if (iw < 0 || iw >= Win) continue;

                int x_idx = (((b * Cin + cin) * Hin + ih) * Win + iw);
                int w_idx = (((cin * Cout + oc) * Kh + kh) * Kw + kw);
                acc += x[x_idx] * w[w_idx];
            }
        }
    }

    if (bias != nullptr) {
        acc += bias[oc];
    }

    int y_idx = (((b * Cout + oc) * Hout + out_h) * Wout + out_w);
    y[y_idx] = acc;
}

// Host wrapper: input and weight (and optional bias) are the ONLY runtime arguments.
// All configuration (kernel size, channels) are derived from tensor shapes.
// Stride, padding, dilation are fixed to STRIDE/PADDING/DILATION as above, matching the target model.
at::Tensor run_impl(at::Tensor input, at::Tensor weight, c10::optional<at::Tensor> bias_opt) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Only float32 weights supported");
    input = input.contiguous();
    weight = weight.contiguous();

    const int64_t N   = input.size(0);
    const int64_t Cin = input.size(1);
    const int64_t Hin = input.size(2);
    const int64_t Win = input.size(3);

    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D [Cin, Cout, Kh, Kw] for ConvTranspose2d");
    TORCH_CHECK(weight.size(0) == Cin, "Weight Cin mismatch: got ", weight.size(0), " expected ", Cin);

    const int64_t Cin_w = weight.size(0);
    const int64_t Cout  = weight.size(1);
    const int64_t Kh    = weight.size(2);
    const int64_t Kw    = weight.size(3);
    TORCH_CHECK(Cin_w == Cin, "Input channels and weight[0] mismatch");

    at::Tensor bias;
    const float* d_bias = nullptr;
    if (bias_opt.has_value()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA/HIP tensor");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == Cout, "Bias must be 1D with size Cout");
        TORCH_CHECK(bias.scalar_type() == at::kFloat, "Bias must be float32");
        bias = bias.contiguous();
        d_bias = bias.data_ptr<float>();
    }

    // Compute output dims for ConvTranspose2d:
    // Hout = (Hin - 1) * stride - 2*padding + dilation * (Kh - 1) + 1
    // Wout = (Win - 1) * stride - 2*padding + dilation * (Kw - 1) + 1
    const int64_t Hout = (Hin - 1) * STRIDE - 2 * PADDING + DILATION * (Kh - 1) + 1;
    const int64_t Wout = (Win - 1) * STRIDE - 2 * PADDING + DILATION * (Kw - 1) + 1;

    auto output = at::zeros({N, Cout, Hout, Wout}, input.options());

    const float* d_x = input.data_ptr<float>();
    const float* d_w = weight.data_ptr<float>();
    float* d_y = output.data_ptr<float>();

    dim3 block(TILE_X, TILE_Y, 1);
    dim3 grid(
        static_cast<unsigned int>((Wout + TILE_X - 1) / TILE_X),
        static_cast<unsigned int>((Hout + TILE_Y - 1) / TILE_Y),
        static_cast<unsigned int>(N * Cout)
    );

    // Shared memory size per block for weight cache
    size_t smem_bytes = static_cast<size_t>(Cin) * static_cast<size_t>(Kh) * static_cast<size_t>(Kw) * sizeof(float);

    // Use shared-memory kernel if within LDS budget; else fallback to global-read kernel
    // MI300X has 64KB LDS per CU; keep some headroom for occupancy.
    const size_t LDS_BUDGET = 48 * 1024; // 48KB to allow multiple blocks per CU
    if (smem_bytes <= LDS_BUDGET) {
        hipLaunchKernelGGL(
            deconv2d_transpose_gather_shared_kernel,
            grid, block, smem_bytes, 0,
            d_x, d_w, d_bias, d_y,
            static_cast<int>(N), static_cast<int>(Cin), static_cast<int>(Hin), static_cast<int>(Win),
            static_cast<int>(Cout), static_cast<int>(Kh), static_cast<int>(Kw),
            static_cast<int>(Hout), static_cast<int>(Wout)
        );
    } else {
        hipLaunchKernelGGL(
            deconv2d_transpose_gather_global_kernel,
            grid, block, 0, 0,
            d_x, d_w, d_bias, d_y,
            static_cast<int>(N), static_cast<int>(Cin), static_cast<int>(Hin), static_cast<int>(Win),
            static_cast<int>(Cout), static_cast<int>(Kh), static_cast<int>(Kw),
            static_cast<int>(Hout), static_cast<int>(Wout)
        );
    }

    hipError_t err = hipDeviceSynchronize();
    hipCheck(err, "HIP device synchronize failed after kernel");

    err = hipGetLastError();
    hipCheck(err, "HIP kernel launch failed");

    return output;
}

// Overloads to match harness calling conventions (with or without bias)
at::Tensor run(at::Tensor input, at::Tensor weight) {
    return run_impl(input, weight, c10::nullopt);
}
at::Tensor run_bias(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    return run_impl(input, weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "ConvTranspose2d (deconvolution) forward - HIP (no bias)");
    m.def("run", &run_bias, "ConvTranspose2d (deconvolution) forward - HIP (with bias)");
}