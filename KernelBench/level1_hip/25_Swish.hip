// Swish activation (x * sigmoid(x)) optimized HIP kernel for AMD MI300X (gfx942)
// Single-file Torch extension with vectorized and scalar paths

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Scalar kernel: grid-stride loop, coalesced access
__global__ void swish_scalar_kernel(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;

    for (size_t i = tid; i < n; i += stride) {
        float v = x[i];
        float s = 1.0f / (1.0f + expf(-v));
        y[i] = v * s;
    }
}

// Vectorized kernel (float4): 4 elements per thread, grid-stride loop
__global__ void swish_vec4_kernel(const float4* __restrict__ x4,
                                  float4* __restrict__ y4,
                                  size_t n4) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;

    for (size_t i = tid; i < n4; i += stride) {
        float4 v = x4[i];

        float s0 = 1.0f / (1.0f + expf(-v.x));
        float s1 = 1.0f / (1.0f + expf(-v.y));
        float s2 = 1.0f / (1.0f + expf(-v.z));
        float s3 = 1.0f / (1.0f + expf(-v.w));

        float4 out;
        out.x = v.x * s0;
        out.y = v.y * s1;
        out.z = v.z * s2;
        out.w = v.w * s3;

        y4[i] = out;
    }
}

// Wrapper exposed to PyTorch. Only tensor arguments are accepted.
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor (on GPU).");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 tensors are supported.");
    // Ensure contiguous for coalesced/vectorized access
    auto x = input.contiguous();

    // Allocate output
    auto y = at::empty_like(x);

    const size_t n = static_cast<size_t>(x.numel());
    if (n == 0) {
        return y;
    }

    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    // Kernel launch configuration
    int device = -1;
    hipGetDevice(&device);
    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, device);
    // Choose a wavefront-aligned block size
    int block_size = 256; // good balance on MI300X (multiple of 64)
    // Use oversubscription to hide latency; clamp grid size to a reasonable value
    // Use grid-stride loop, so even a modest grid saturates the GPU
    int mp = std::max(1, prop.multiProcessorCount);
    int grid_size = mp * 20; // 20x oversubscription

    // Try vectorized path when memory is 16-byte aligned and n % 4 == 0
    bool aligned = ((reinterpret_cast<uintptr_t>(x_ptr) % 16u) == 0u) &&
                   ((reinterpret_cast<uintptr_t>(y_ptr) % 16u) == 0u);
    if (aligned && (n % 4 == 0)) {
        size_t n4 = n / 4;
        const float4* x4 = reinterpret_cast<const float4*>(x_ptr);
        float4* y4 = reinterpret_cast<float4*>(y_ptr);
        hipLaunchKernelGGL(
            swish_vec4_kernel,
            dim3(grid_size), dim3(block_size), 0, 0,
            x4, y4, n4
        );
    } else {
        // Fallback scalar path
        hipLaunchKernelGGL(
            swish_scalar_kernel,
            dim3(grid_size), dim3(block_size), 0, 0,
            x_ptr, y_ptr, n
        );
    }

    // Error checking and synchronization
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        TORCH_CHECK(false, "HIP device sync failed: ", hipGetErrorString(err));
    }
    err = hipGetLastError();
    if (err != hipSuccess) {
        TORCH_CHECK(false, "HIP kernel launch failed: ", hipGetErrorString(err));
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Swish activation (x * sigmoid(x)) - Optimized HIP");
}