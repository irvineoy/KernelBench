// ELU activation HIP kernel optimized for AMD MI300X (gfx942)
// Applies ELU elementwise: y = x if x > 0 else alpha * (exp(x) - 1)
// Alpha is a construction-time configuration; per task rules we hard-code alpha = 1.0f.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>
#include <cstdint>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Use __expf if available for speed
__device__ __forceinline__ float fast_exp(float x) {
#if defined(__HIPCC_RTC__) || defined(__HIP_DEVICE_COMPILE__)
    // On device, __expf is available for fast single-precision exp
    return __expf(x);
#else
    // Host path (shouldn't be used in kernel)
    return expf(x);
#endif
}

__device__ __forceinline__ float elu_op(float x, float alpha) {
    // Branch-based to avoid expensive exp for positive inputs
    return (x > 0.0f) ? x : alpha * (fast_exp(x) - 1.0f);
}

// Vectorized float4 ELU kernel with grid-stride loop
__global__ __launch_bounds__(256)
void elu_vec4_kernel(const float4* __restrict__ in4,
                     float4* __restrict__ out4,
                     uint64_t n_vec4,
                     float alpha) {
    uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

    for (uint64_t i = idx; i < n_vec4; i += stride) {
        float4 v = in4[i];
        float4 r;
        r.x = elu_op(v.x, alpha);
        r.y = elu_op(v.y, alpha);
        r.z = elu_op(v.z, alpha);
        r.w = elu_op(v.w, alpha);
        out4[i] = r;
    }
}

// Scalar ELU kernel for tail elements or generic path
__global__ __launch_bounds__(256)
void elu_scalar_kernel(const float* __restrict__ in,
                       float* __restrict__ out,
                       uint64_t n,
                       float alpha) {
    uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

    for (uint64_t i = idx; i < n; i += stride) {
        float x = in[i];
        out[i] = (x > 0.0f) ? x : alpha * (fast_exp(x) - 1.0f);
    }
}

// Host wrapper: takes only tensors (no scalar configs), returns output tensor.
// Signature: run(input)
// Alpha is a model construction-time constant; per task, we fix alpha = 1.0f.
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "Only float32 tensors are supported");
    auto x = input.contiguous();  // ensure dense, aligned storage

    // Create output with same shape and dtype
    auto out = at::empty_like(x);

    const uint64_t N = static_cast<uint64_t>(x.numel());
    if (N == 0) return out;

    const float alpha = 1.0f;  // From model init; not passed at runtime per task rules

    const float* d_in = x.data_ptr<float>();
    float* d_out = out.data_ptr<float>();

    // Vectorization decision: require 16-byte alignment and N >= 4
    const uintptr_t in_addr = reinterpret_cast<uintptr_t>(d_in);
    const uintptr_t out_addr = reinterpret_cast<uintptr_t>(d_out);
    const bool aligned16 = ((in_addr | out_addr) & 0xF) == 0;
    const uint64_t n_vec4 = aligned16 ? (N >> 2) : 0;  // number of float4 elements
    const uint64_t n_tail = aligned16 ? (N & 3ULL) : N; // if not aligned, process all in scalar

    // Launch parameters
    const int block = 256;
    // Cap grid to a large, reasonable number to keep scheduler happy while enabling grid-stride loops
    auto ceil_div = [](uint64_t a, uint64_t b) -> uint64_t { return (a + b - 1) / b; };
    int grid_vec = 0;
    int grid_sca = 0;

    // Launch vectorized kernel if possible
    if (n_vec4 > 0) {
        grid_vec = static_cast<int>(std::min<uint64_t>(ceil_div(n_vec4, block), 1048576ULL)); // up to 1M blocks
        hipLaunchKernelGGL(
            elu_vec4_kernel,
            dim3(grid_vec), dim3(block), 0, 0,
            reinterpret_cast<const float4*>(d_in),
            reinterpret_cast<float4*>(d_out),
            n_vec4, alpha
        );
        hipError_t err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP launch failed (vec4): ", hipGetErrorString(err));
    }

    // Launch scalar kernel for tail or full array if not aligned
    if (n_tail > 0) {
        const float* tail_in = aligned16 ? (d_in + (n_vec4 << 2)) : d_in;
        float* tail_out = aligned16 ? (d_out + (n_vec4 << 2)) : d_out;

        grid_sca = static_cast<int>(std::min<uint64_t>(ceil_div(n_tail, block), 1048576ULL));
        hipLaunchKernelGGL(
            elu_scalar_kernel,
            dim3(grid_sca), dim3(block), 0, 0,
            tail_in, tail_out, n_tail, alpha
        );
        hipError_t err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP launch failed (scalar): ", hipGetErrorString(err));
    }

    hipError_t err_sync = hipDeviceSynchronize();
    TORCH_CHECK(err_sync == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err_sync));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "ELU activation (alpha=1.0) - HIP optimized");
}