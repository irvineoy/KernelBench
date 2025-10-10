#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>

#define THREADS_PER_BLOCK 256
#define WAVE_SIZE 64

// SELU constants (PyTorch defaults)
__device__ __constant__ float SELU_ALPHA = 1.6732632423543772848170429916717f;
__device__ __constant__ float SELU_SCALE = 1.0507009873554804934193349852946f;

__device__ __forceinline__ float selu_scalar(float x) {
    // Branch keeps us from calling expf() for positive inputs (dominant case for rand())
    if (x > 0.0f) {
        return SELU_SCALE * x;
    } else {
        return SELU_SCALE * SELU_ALPHA * (expf(x) - 1.0f);
    }
}

__global__ void selu_kernel_scalar(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t i = tid; i < n; i += stride) {
        float v = x[i];
        y[i] = selu_scalar(v);
    }
}

__global__ void selu_kernel_vec4(const float4* __restrict__ x4,
                                 float4* __restrict__ y4,
                                 size_t n_vec4) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t i = tid; i < n_vec4; i += stride) {
        float4 v = x4[i];
        float4 o;
        o.x = selu_scalar(v.x);
        o.y = selu_scalar(v.y);
        o.z = selu_scalar(v.z);
        o.w = selu_scalar(v.w);
        y4[i] = o;
    }
}

at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a HIP/CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be float32");
    auto x = input.contiguous();
    auto y = at::empty_like(x);

    size_t n = static_cast<size_t>(x.numel());
    if (n == 0) return y;

    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    // Decide on vectorized path if 16-byte aligned
    uintptr_t x_addr = reinterpret_cast<uintptr_t>(x_ptr);
    uintptr_t y_addr = reinterpret_cast<uintptr_t>(y_ptr);
    bool aligned16 = ((x_addr | y_addr) & 0xF) == 0;

    int block = THREADS_PER_BLOCK;

    hipError_t err;

    if (aligned16) {
        size_t n_vec4 = n / 4;
        size_t n_tail = n - n_vec4 * 4;

        if (n_vec4 > 0) {
            int grid = static_cast<int>((n_vec4 + block - 1) / block);
            // Launch vectorized kernel
            hipLaunchKernelGGL(
                selu_kernel_vec4,
                dim3(grid), dim3(block), 0, 0,
                reinterpret_cast<const float4*>(x_ptr),
                reinterpret_cast<float4*>(y_ptr),
                n_vec4
            );
            err = hipDeviceSynchronize();
            TORCH_CHECK(err == hipSuccess, "HIP error (vector kernel): ", hipGetErrorString(err));
            err = hipGetLastError();
            TORCH_CHECK(err == hipSuccess, "HIP launch error (vector kernel): ", hipGetErrorString(err));
        }

        if (n_tail > 0) {
            const float* x_tail = x_ptr + (n - n_tail);
            float* y_tail = y_ptr + (n - n_tail);
            int grid_tail = static_cast<int>((n_tail + block - 1) / block);
            hipLaunchKernelGGL(
                selu_kernel_scalar,
                dim3(grid_tail), dim3(block), 0, 0,
                x_tail, y_tail, n_tail
            );
            err = hipDeviceSynchronize();
            TORCH_CHECK(err == hipSuccess, "HIP error (tail kernel): ", hipGetErrorString(err));
            err = hipGetLastError();
            TORCH_CHECK(err == hipSuccess, "HIP launch error (tail kernel): ", hipGetErrorString(err));
        }
    } else {
        // Fallback scalar path
        int grid = static_cast<int>((n + block - 1) / block);
        hipLaunchKernelGGL(
            selu_kernel_scalar,
            dim3(grid), dim3(block), 0, 0,
            x_ptr, y_ptr, n
        );
        err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP error (scalar kernel): ", hipGetErrorString(err));
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP launch error (scalar kernel): ", hipGetErrorString(err));
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "SELU activation (HIP, optimized for AMD MI300X)");
}