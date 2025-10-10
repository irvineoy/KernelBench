// Softplus HIP kernel optimized for AMD MI300X (gfx942)
// Single-file extension: kernels + PyTorch bindings

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Choose a block size that's a multiple of wave size; 256 is a good balance.
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Softplus defaults from PyTorch: beta=1.0f, threshold=20.0f
__device__ __forceinline__ float softplus_elem(float x) {
    // Threshold path matches PyTorch: when x > 20, return x directly
    if (x > 20.0f) return x;

    // Stable formulation: log1p(exp(x)) = max(0, x) + log1p(exp(-abs(x)))
    float ax = fabsf(x);
    return fmaxf(0.0f, x) + log1pf(expf(-ax));
}

__device__ __forceinline__ void softplus_vec4(float4 v, float4& r) {
    r.x = softplus_elem(v.x);
    r.y = softplus_elem(v.y);
    r.z = softplus_elem(v.z);
    r.w = softplus_elem(v.w);
}

// Vectorized kernel: process 4 floats per thread using float4
__global__ __launch_bounds__(BLOCK_SIZE)
void softplus_vec4_kernel(const float* __restrict__ in,
                          float* __restrict__ out,
                          int64_t n_vec4) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_vec4) return;

    const float4* __restrict__ in4 = reinterpret_cast<const float4*>(in);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(out);

    float4 v = in4[idx];
    float4 r;
    softplus_vec4(v, r);
    out4[idx] = r;
}

// Scalar tail kernel for remaining elements (n_tail in {0,1,2,3})
__global__ __launch_bounds__(BLOCK_SIZE)
void softplus_scalar_kernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            int64_t start,
                            int64_t n_tail) {
    int tid = threadIdx.x;
    if (tid >= n_tail) return;
    int64_t i = start + tid;
    out[i] = softplus_elem(in[i]);
}

// PyTorch entry point: only tensor arguments allowed
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA/ROCm device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input tensor must be float32");

    auto x = input.contiguous();
    auto y = at::empty_like(x);

    int64_t N = x.numel();
    if (N == 0) return y;

    const float* in_ptr = x.data_ptr<float>();
    float* out_ptr = y.data_ptr<float>();

    // Vectorization by 4 (float4). PyTorch contiguous tensors are typically 16B-aligned.
    const int64_t n_vec4 = N / 4;
    const int64_t n_tail = N - n_vec4 * 4;

    // Launch configuration
    dim3 block(BLOCK_SIZE);
    if (n_vec4 > 0) {
        int64_t grid64 = (n_vec4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // dim3 takes unsigned ints; n_vec4 here is well within limits for practical sizes.
        dim3 grid(static_cast<unsigned int>(grid64));
        hipLaunchKernelGGL(
            softplus_vec4_kernel,
            grid, block, 0, 0,
            in_ptr, out_ptr, n_vec4
        );

        hipError_t err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP kernel (vec4) failed: ", hipGetErrorString(err));
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP kernel (vec4) launch failed: ", hipGetErrorString(err));
    }

    if (n_tail > 0) {
        // Handle the last 0..3 elements with a tiny block
        dim3 grid(1);
        dim3 block_tail(BLOCK_SIZE);
        hipLaunchKernelGGL(
            softplus_scalar_kernel,
            grid, block_tail, 0, 0,
            in_ptr, out_ptr, n_vec4 * 4, n_tail
        );

        hipError_t err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP kernel (tail) failed: ", hipGetErrorString(err));
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP kernel (tail) launch failed: ", hipGetErrorString(err));
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Softplus activation (HIP, optimized, beta=1, threshold=20)");
}