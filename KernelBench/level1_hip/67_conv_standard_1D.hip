// Compile with: hipcc -O3 --offload-arch=gfx9-4-generic -std=c++17 conv1d_hip.cpp -o conv1d_hip
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef TILE_L
#define TILE_L 256  // number of output-length positions per block (must equal blockDim.x)
#endif

// Generic grouped=1 Conv1D kernel (stride=1, padding=0, dilation=1)
// Each block processes one (batch n, out_channel co) pair over a tile of output positions along length.
// We cache the full filter for (co) in LDS (shared memory) for reuse across all threads in the block.
__global__ __launch_bounds__(256)
void conv1d_g1_generic_kernel(
    const float* __restrict__ x,         // [N, Cin, Lin]
    const float* __restrict__ w,         // [Cout, Cin, K]
    const float* __restrict__ bias,      // [Cout] or nullptr
    float* __restrict__ y,               // [N, Cout, Lout]
    int N, int Cin, int Cout,
    int Lin, int K, int Lout,
    int has_bias)
{
    extern __shared__ float s_w[]; // size = Cin*K floats
    const int n  = blockIdx.z;     // batch index
    const int co = blockIdx.y;     // output channel
    const int l0 = blockIdx.x * blockDim.x + threadIdx.x; // output position

    // Load weights for this output channel into LDS
    const int w_base_g = co * Cin * K;
    for (int idx = threadIdx.x; idx < Cin * K; idx += blockDim.x) {
        s_w[idx] = w[w_base_g + idx];
    }
    __syncthreads();

    if (n >= N || co >= Cout || l0 >= Lout) return;

    float acc = 0.0f;
    if (has_bias) {
        acc = bias[co];
    }

    // Compute convolution: stride=1, pad=0, dilation=1
    // y[n, co, l] = sum_{ci,k} x[n, ci, l + k] * w[co, ci, k]
    // With valid conv, 0 <= l <= Lin-K, so l+k in [0, Lin-1]
    const int x_n_off = n * Cin * Lin;
    const int y_off   = (n * Cout + co) * Lout;

    // Loop over input channels
    for (int ci = 0; ci < Cin; ++ci) {
        const int wi = ci * K;                         // in shared mem
        const int xi = (x_n_off + ci * Lin) + l0;      // start at l0
        // Loop over kernel taps
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            acc += x[xi + k] * s_w[wi + k];
        }
    }

    y[y_off + l0] = acc;
}

// Specialized K=3 grouped=1 Conv1D kernel for small kernels (common case), unrolled inner loop
__global__ __launch_bounds__(256)
void conv1d_g1_k3_kernel(
    const float* __restrict__ x,         // [N, Cin, Lin]
    const float* __restrict__ w,         // [Cout, Cin, 3]
    const float* __restrict__ bias,      // [Cout] or nullptr
    float* __restrict__ y,               // [N, Cout, Lout]
    int N, int Cin, int Cout,
    int Lin, int Lout,
    int has_bias)
{
    extern __shared__ float s_w[]; // size = Cin*3 floats
    const int n  = blockIdx.z;
    const int co = blockIdx.y;
    const int l0 = blockIdx.x * blockDim.x + threadIdx.x;

    // Load weights for this output channel into LDS
    const int w_base_g = co * Cin * 3;
    for (int idx = threadIdx.x; idx < Cin * 3; idx += blockDim.x) {
        s_w[idx] = w[w_base_g + idx];
    }
    __syncthreads();

    if (n >= N || co >= Cout || l0 >= Lout) return;

    float acc = 0.0f;
    if (has_bias) {
        acc = bias[co];
    }

    const int x_n_off = n * Cin * Lin;
    const int y_off   = (n * Cout + co) * Lout;

    // Unrolled K=3
    for (int ci = 0; ci < Cin; ++ci) {
        const int wi = ci * 3;
        const float w0 = s_w[wi + 0];
        const float w1 = s_w[wi + 1];
        const float w2 = s_w[wi + 2];

        const int xi = (x_n_off + ci * Lin) + l0;
        const float x0 = x[xi + 0];
        const float x1 = x[xi + 1];
        const float x2 = x[xi + 2];

        acc += x0 * w0 + x1 * w1 + x2 * w2;
    }

    y[y_off + l0] = acc;
}

// Specialized K=1 grouped=1 Conv1D kernel (pointwise 1x1 convolution along channels)
__global__ __launch_bounds__(256)
void conv1d_g1_k1_kernel(
    const float* __restrict__ x,         // [N, Cin, Lin]
    const float* __restrict__ w,         // [Cout, Cin, 1]
    const float* __restrict__ bias,      // [Cout] or nullptr
    float* __restrict__ y,               // [N, Cout, Lout==Lin]
    int N, int Cin, int Cout,
    int Lin, int Lout,
    int has_bias)
{
    extern __shared__ float s_w[]; // size = Cin floats
    const int n  = blockIdx.z;
    const int co = blockIdx.y;
    const int l0 = blockIdx.x * blockDim.x + threadIdx.x;

    // Load weights for this output channel into LDS
    const int w_base_g = co * Cin; // since K=1
    for (int idx = threadIdx.x; idx < Cin; idx += blockDim.x) {
        s_w[idx] = w[w_base_g + idx];
    }
    __syncthreads();

    if (n >= N || co >= Cout || l0 >= Lout) return;

    float acc = 0.0f;
    if (has_bias) acc = bias[co];

    const int x_n_off = n * Cin * Lin;
    const int y_off   = (n * Cout + co) * Lout;

    // y[n,co,l] = sum_ci x[n,ci,l] * w[co,ci,0]
    for (int ci = 0; ci < Cin; ++ci) {
        const float wi = s_w[ci];
        const float xi = x[x_n_off + ci * Lin + l0];
        acc += xi * wi;
    }
    y[y_off + l0] = acc;
}

// PyTorch entry point
// Signature must only contain tensors: input, weight, [bias(optional)]
at::Tensor run(at::Tensor input, at::Tensor weight, at::Tensor bias = at::Tensor()) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must have shape [N, Cin, Lin]");
    TORCH_CHECK(weight.dim() == 3, "weight must have shape [Cout, Cin, K] for groups=1");

    // Enforce contiguous memory layout
    auto x = input.contiguous();
    auto w = weight.contiguous();

    const int64_t N   = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t Lin = x.size(2);

    const int64_t Cout = w.size(0);
    const int64_t Win  = w.size(1); // should be Cin for groups=1
    const int64_t K    = w.size(2);

    TORCH_CHECK(Win == Cin, "Only groups=1 is supported: weight.size(1) must equal input.size(1)");
    TORCH_CHECK(K >= 1, "Kernel size must be >= 1");

    // Optional bias
    const bool has_bias = bias.defined() && bias.numel() == Cout;
    at::Tensor b;
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be on CUDA/HIP");
        TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == Cout, "bias must have shape [Cout]");
        b = bias.contiguous();
    }

    // Fixed config: stride=1, padding=0, dilation=1 (valid conv)
    const int64_t Lout = Lin - K + 1;
    TORCH_CHECK(Lout >= 0, "Invalid configuration results in negative output length; ensure Lin >= K");

    auto y = at::empty({N, Cout, Lout}, x.options());

    if (N == 0 || Cout == 0 || Lout == 0) {
        return y; // nothing to do
    }

    const int block_size = 256; // multiple of wavefront size (64)
    const int grid_x = static_cast<int>((Lout + TILE_L - 1) / TILE_L);
    dim3 block(block_size, 1, 1);
    dim3 grid(grid_x, static_cast<unsigned int>(Cout), static_cast<unsigned int>(N));

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    const float* b_ptr = has_bias ? b.data_ptr<float>() : nullptr;
    float* y_ptr = y.data_ptr<float>();

    // Shared memory size needed = Cin*K*sizeof(float) (or smaller for specialized kernels)
    size_t shmem_bytes = static_cast<size_t>(Cin) * static_cast<size_t>(K) * sizeof(float);

    // Choose specialized kernels for small K
    hipError_t err;
    if (K == 1) {
        shmem_bytes = static_cast<size_t>(Cin) * sizeof(float);
        hipLaunchKernelGGL(
            conv1d_g1_k1_kernel,
            grid, block, shmem_bytes, 0,
            x_ptr, w_ptr, b_ptr, y_ptr,
            static_cast<int>(N), static_cast<int>(Cin), static_cast<int>(Cout),
            static_cast<int>(Lin), static_cast<int>(Lout),
            has_bias ? 1 : 0
        );
    } else if (K == 3) {
        shmem_bytes = static_cast<size_t>(Cin) * 3 * sizeof(float);
        hipLaunchKernelGGL(
            conv1d_g1_k3_kernel,
            grid, block, shmem_bytes, 0,
            x_ptr, w_ptr, b_ptr, y_ptr,
            static_cast<int>(N), static_cast<int>(Cin), static_cast<int>(Cout),
            static_cast<int>(Lin), static_cast<int>(Lout),
            has_bias ? 1 : 0
        );
    } else {
        hipLaunchKernelGGL(
            conv1d_g1_generic_kernel,
            grid, block, shmem_bytes, 0,
            x_ptr, w_ptr, b_ptr, y_ptr,
            static_cast<int>(N), static_cast<int>(Cin), static_cast<int>(Cout),
            static_cast<int>(Lin), static_cast<int>(K), static_cast<int>(Lout),
            has_bias ? 1 : 0
        );
    }

    err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Optimized Conv1D (groups=1, stride=1, padding=0, dilation=1) on HIP",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = at::Tensor());
}