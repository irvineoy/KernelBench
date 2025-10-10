// tanh_hip_kernel.hip
// High-performance HIP kernel for elementwise Tanh activation on AMD MI300X (gfx942)
// Implements the PyTorch model: return torch.tanh(x)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <iostream>
#include <cmath>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256  // multiple of 64 for wavefront alignment
#endif

// Vectorized elementwise Tanh for float4
__device__ __forceinline__ float4 tanh_float4(const float4 v) {
    float4 r;
    r.x = tanhf(v.x);
    r.y = tanhf(v.y);
    r.z = tanhf(v.z);
    r.w = tanhf(v.w);
    return r;
}

// Vectorized kernel: processes 4 floats per thread using float4 loads/stores
// n_vec = number of float4 elements (i.e., N / 4)
__global__ void tanh_kernel_vec4(const float4* __restrict__ in,
                                 float4* __restrict__ out,
                                 int64_t n_vec) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n_vec; i += stride) {
        float4 v = in[i];
        out[i] = tanh_float4(v);
    }
}

// Scalar kernel: processes remaining elements
__global__ void tanh_kernel_scalar(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        out[i] = tanhf(in[i]);
    }
}

// C++ wrapper exposed to PyTorch. Only tensor parameters allowed.
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on HIP device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 tensors are supported");

    // Ensure contiguous for coalesced/vectorized access
    auto x = input.contiguous();
    auto N = x.numel();

    auto out = at::empty_like(x);

    if (N == 0) {
        return out;
    }

    // Raw pointers
    float* out_ptr = out.data_ptr<float>();
    const float* in_ptr = x.data_ptr<float>();

    // Try vectorized path if 16-byte aligned and N divisible by 4
    bool aligned16 = ((reinterpret_cast<uintptr_t>(in_ptr) % 16u) == 0) &&
                     ((reinterpret_cast<uintptr_t>(out_ptr) % 16u) == 0);
    int64_t n_vec = 0;
    int64_t n_tail = N;

    dim3 block(BLOCK_SIZE);
    // Choose a grid size that gives enough work per SM without going too large
    // Cap grid size to avoid extremely large launches; use grid-stride in kernels
    const int max_grid = 4 * 1024 * 1024;  // reasonable upper bound
    if (aligned16 && (N % 4 == 0)) {
        n_vec = N / 4;
        n_tail = 0;
        int64_t grid_x = (n_vec + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid_x > max_grid) grid_x = max_grid;

        hipLaunchKernelGGL(
            tanh_kernel_vec4,
            dim3((unsigned int)grid_x), block, 0, 0,
            reinterpret_cast<const float4*>(in_ptr),
            reinterpret_cast<float4*>(out_ptr),
            n_vec
        );

        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            TORCH_CHECK(false, "HIP kernel (vec4) failed: ", hipGetErrorString(err));
        }

        err = hipGetLastError();
        if (err != hipSuccess) {
            TORCH_CHECK(false, "HIP kernel (vec4) launch failed: ", hipGetErrorString(err));
        }
    } else {
        // Scalar fallback for any case (unaligned or N not divisible by 4)
        int64_t grid_x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid_x > max_grid) grid_x = max_grid;

        hipLaunchKernelGGL(
            tanh_kernel_scalar,
            dim3((unsigned int)grid_x), block, 0, 0,
            in_ptr, out_ptr, (int64_t)N
        );

        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            TORCH_CHECK(false, "HIP kernel (scalar) failed: ", hipGetErrorString(err));
        }

        err = hipGetLastError();
        if (err != hipSuccess) {
            TORCH_CHECK(false, "HIP kernel (scalar) launch failed: ", hipGetErrorString(err));
        }
    }

    // No tail expected in vec4 path because we only take it if N % 4 == 0
    // If desired, we could add a hybrid path (vec4 + tail), but for simplicity and
    // to respect alignment guarantees, we choose either vec4 or scalar exclusively.

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Elementwise Tanh activation (HIP, optimized)");
}