// HIP kernel for elementwise Sigmoid activation optimized for AMD MI300X (gfx942)
// Single file contains kernels and PyTorch bindings.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Numerically stable sigmoid using a single exp
__device__ __forceinline__ float sigmoid_stable(float x) {
    float ax = fabsf(x);
    float z = expf(-ax);                 // z = exp(-|x|)
    // if x >= 0: 1/(1+exp(-x)) = 1/(1+z)
    // else     : exp(x)/(1+exp(x)) = z/(1+z)
    float pos = 1.0f / (1.0f + z);
    float neg = z / (1.0f + z);
    // Avoid branch divergence via predication
    return (x >= 0.0f) ? pos : neg;
}

// Scalar kernel with grid-stride loop
__global__ void sigmoid_kernel_scalar(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        float x = in[i];
        out[i] = sigmoid_stable(x);
    }
}

// Vectorized kernel (float4) with grid-stride loop
__global__ void sigmoid_kernel_vec4(const float4* __restrict__ in4,
                                    float4* __restrict__ out4,
                                    int64_t n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int64_t i4 = idx; i4 < n4; i4 += stride) {
        float4 v = in4[i4];
        v.x = sigmoid_stable(v.x);
        v.y = sigmoid_stable(v.y);
        v.z = sigmoid_stable(v.z);
        v.w = sigmoid_stable(v.w);
        out4[i4] = v;
    }
}

static inline int compute_grid_from_size(int64_t n, int block) {
    // Cap grid size to avoid excessive tiny blocks; use large-stride loops inside kernels
    const int max_grid = 1048576; // 1M blocks cap
    int64_t grid = (n + block - 1) / block;
    if (grid > max_grid) grid = max_grid;
    return static_cast<int>(grid);
}

at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 tensors are supported");

    auto in = input.contiguous();
    auto out = at::empty_like(in);

    const int64_t n = in.numel();
    if (n == 0) return out;

    const int block = 256; // multiple of 64 (wavefront)
    // Try vectorized path on the bulk
    const int64_t n4 = n / 4;
    const int64_t rem = n - n4 * 4;

    const float* in_ptr = in.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    if (n4 > 0) {
        int grid = compute_grid_from_size(n4, block);
        // Launch vectorized kernel over n4 float4 elements
        hipLaunchKernelGGL(sigmoid_kernel_vec4,
                           dim3(grid), dim3(block), 0, 0,
                           reinterpret_cast<const float4*>(in_ptr),
                           reinterpret_cast<float4*>(out_ptr),
                           n4);
        hipError_t err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP launch failed (vec4): ", hipGetErrorString(err));
    }

    if (rem > 0) {
        // Handle tail elements with scalar kernel
        const float* in_tail = in_ptr + n4 * 4;
        float* out_tail = out_ptr + n4 * 4;
        int grid_tail = compute_grid_from_size(rem, block);
        hipLaunchKernelGGL(sigmoid_kernel_scalar,
                           dim3(grid_tail), dim3(block), 0, 0,
                           in_tail, out_tail, rem);
        hipError_t err2 = hipGetLastError();
        TORCH_CHECK(err2 == hipSuccess, "HIP launch failed (scalar tail): ", hipGetErrorString(err2));
    }

    hipError_t sync_err = hipDeviceSynchronize();
    TORCH_CHECK(sync_err == hipSuccess, "HIP kernel execution failed: ", hipGetErrorString(sync_err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Elementwise Sigmoid activation (HIP, optimized for MI300X)");
}