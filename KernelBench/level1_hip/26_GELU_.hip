// gelu_hip_kernel.hip
// High-performance HIP kernel for GELU activation (PyTorch extension)
// Targets AMD MI300X (gfx942)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <iostream>

#ifndef GELU_EXACT
// Default to fast tanh-based approximation for performance.
// Define GELU_EXACT to use the exact erf-based formula.
#endif

// Constants
#define WAVE_SIZE 64
#define BLOCK_SIZE 256  // Multiple of 64 (wavefront size)

// Device GELU implementations
__device__ __forceinline__ float gelu_tanh_approx(float x) {
    // 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    float t = tanhf(k0 * (x + k1 * x3));
    return 0.5f * x * (1.0f + t);
}

__device__ __forceinline__ float gelu_exact(float x) {
    // 0.5 * x * (1 + erf(x / sqrt(2)))
    const float inv_sqrt2 = 0.7071067811865475f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

__device__ __forceinline__ float gelu_func(float x) {
#ifdef GELU_EXACT
    return gelu_exact(x);
#else
    return gelu_tanh_approx(x);
#endif
}

// Scalar grid-stride kernel
__global__ __launch_bounds__(BLOCK_SIZE)
void gelu_kernel_scalar(const float* __restrict__ in,
                        float* __restrict__ out,
                        int64_t N) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < N; i += stride) {
        float x = in[i];
        out[i] = gelu_func(x);
    }
}

// Vectorized (float4) grid-stride kernel
__global__ __launch_bounds__(BLOCK_SIZE)
void gelu_kernel_vec4(const float4* __restrict__ in4,
                      float4* __restrict__ out4,
                      int64_t N4) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < N4; i += stride) {
        float4 v = in4[i];
        v.x = gelu_func(v.x);
        v.y = gelu_func(v.y);
        v.z = gelu_func(v.z);
        v.w = gelu_func(v.w);
        out4[i] = v;
    }
}

// Run wrapper (PyTorch extension entry point)
// Only tensor parameters are allowed (matches get_inputs())
at::Tensor run(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA/HIP device");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 tensors are supported");
    auto input = x.contiguous();
    auto output = at::empty_like(input);

    int64_t N = input.numel();
    if (N == 0) return output;

    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    // Decide vectorization strategy on host
    uintptr_t in_addr = reinterpret_cast<uintptr_t>(in_ptr);
    uintptr_t out_addr = reinterpret_cast<uintptr_t>(out_ptr);
    bool aligned16 = ((in_addr | out_addr) & 0xF) == 0;  // 16-byte alignment
    int64_t N4 = aligned16 ? (N >> 2) : 0;
    int64_t rem = aligned16 ? (N & 3) : N;

    // Launch configuration (balance occupancy and scheduling)
    int block = BLOCK_SIZE;

    // Heuristic: cap grid to a large value to avoid excessively large grids
    auto ceil_div = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };
    int64_t grid_vec = (N4 > 0) ? ceil_div(N4, block) : 0;
    int64_t grid_scalar = (rem > 0) ? ceil_div(rem, block) : 0;

    // Limit grid to a large positive int to satisfy HIP API
    int max_grid = 2147483647; // INT_MAX
    if (grid_vec > max_grid) grid_vec = max_grid;
    if (grid_scalar > max_grid) grid_scalar = max_grid;

    hipError_t err;

    // Vectorized pass for the bulk (if aligned and size >= 4)
    if (N4 > 0) {
        const float4* in4 = reinterpret_cast<const float4*>(in_ptr);
        float4* out4 = reinterpret_cast<float4*>(out_ptr);

        dim3 grid((unsigned int)grid_vec);
        dim3 blockDim(block);

        hipLaunchKernelGGL(gelu_kernel_vec4, grid, blockDim, 0, 0, in4, out4, N4);
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP launch failed (vec4): ", hipGetErrorString(err));
        err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP kernel failed (vec4): ", hipGetErrorString(err));
    }

    // Tail or full scalar pass
    if (rem > 0) {
        // For tail, offset pointers if we already processed N4 * 4 elements
        const float* in_tail = in_ptr + (N - rem);
        float* out_tail = out_ptr + (N - rem);

        dim3 grid((unsigned int)grid_scalar);
        dim3 blockDim(block);

        hipLaunchKernelGGL(gelu_kernel_scalar, grid, blockDim, 0, 0, in_tail, out_tail, rem);
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP launch failed (scalar): ", hipGetErrorString(err));
        err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP kernel failed (scalar): ", hipGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "GELU activation (HIP, vectorized)");
}