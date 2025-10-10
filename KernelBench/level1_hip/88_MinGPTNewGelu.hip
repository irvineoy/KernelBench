#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <iostream>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Approximate GELU as used in BERT/GPT:
// 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
__device__ __forceinline__ float gelu_approx(float x) {
    const float SQRT_2_OVER_PI = 0.7978845608028654f; // sqrt(2/pi)
    const float C = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    float s = fmaf(C, x3, x);                // x + 0.044715*x^3
    float t = tanhf(SQRT_2_OVER_PI * s);     // tanh(...)
    return 0.5f * x * (1.0f + t);
}

__global__ void gelu_scalar_kernel(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;

    for (size_t i = idx; i < N; i += stride) {
        out[i] = gelu_approx(in[i]);
    }
}

__global__ void gelu_vec4_kernel(const float4* __restrict__ in4,
                                 float4* __restrict__ out4,
                                 size_t N4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;

    for (size_t i = idx; i < N4; i += stride) {
        float4 v = in4[i];
        float4 r;
        r.x = gelu_approx(v.x);
        r.y = gelu_approx(v.y);
        r.z = gelu_approx(v.z);
        r.w = gelu_approx(v.w);
        out4[i] = r;
    }
}

static inline int compute_grid_from_size(size_t work_items, int block_size, int max_blocks_hint) {
    int grid = static_cast<int>((work_items + block_size - 1) / block_size);
    if (max_blocks_hint > 0 && grid > max_blocks_hint) grid = max_blocks_hint;
    if (grid < 1) grid = 1;
    return grid;
}

// PyTorch entry point: only tensor parameters (matches get_inputs())
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA/HIP device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input tensor must be float32");
    auto x = input.contiguous();
    auto y = at::empty_like(x);

    const size_t N = static_cast<size_t>(x.numel());
    if (N == 0) return y;

    const float* in_ptr  = x.data_ptr<float>();
    float* out_ptr       = y.data_ptr<float>();

    // Configure launch parameters
    const int block_size = 256; // multiple of 64 (wavefront)
    int max_blocks_hint = 0;    // will be set to #CUs * 8 if available

    // Query device to size the launch grid reasonably
    int device_id = -1;
    hipError_t herr = hipGetDevice(&device_id);
    TORCH_CHECK(herr == hipSuccess, "hipGetDevice failed: ", hipGetErrorString(herr));

    hipDeviceProp_t prop{};
    herr = hipGetDeviceProperties(&prop, device_id);
    TORCH_CHECK(herr == hipSuccess, "hipGetDeviceProperties failed: ", hipGetErrorString(herr));
    if (prop.multiProcessorCount > 0) {
        max_blocks_hint = prop.multiProcessorCount * 8; // 8 blocks/CU hint
    }

    // Vectorized path if 16-byte aligned and size divisible by 4
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(in_ptr)  & 0xF) == 0) &&
        ((reinterpret_cast<uintptr_t>(out_ptr) & 0xF) == 0);

    size_t N4 = 0;
    if (aligned) {
        N4 = N / 4;
    }

    if (N4 > 0) {
        const float4* in4  = reinterpret_cast<const float4*>(in_ptr);
        float4* out4       = reinterpret_cast<float4*>(out_ptr);
        int grid_size_vec4 = compute_grid_from_size(N4, block_size, max_blocks_hint);

        hipLaunchKernelGGL(gelu_vec4_kernel,
                           dim3(grid_size_vec4), dim3(block_size), 0, 0,
                           in4, out4, N4);
        herr = hipGetLastError();
        TORCH_CHECK(herr == hipSuccess, "gelu_vec4_kernel launch failed: ", hipGetErrorString(herr));
    }

    // Handle tail elements (if any)
    const size_t processed = N4 * 4;
    const size_t tail = N - processed;
    if (tail > 0) {
        const float* in_tail = in_ptr + processed;
        float* out_tail = out_ptr + processed;
        int grid_size_tail = compute_grid_from_size(tail, block_size, max_blocks_hint);

        hipLaunchKernelGGL(gelu_scalar_kernel,
                           dim3(grid_size_tail), dim3(block_size), 0, 0,
                           in_tail, out_tail, tail);
        herr = hipGetLastError();
        TORCH_CHECK(herr == hipSuccess, "gelu_scalar_kernel (tail) launch failed: ", hipGetErrorString(herr));
    }

    herr = hipDeviceSynchronize();
    TORCH_CHECK(herr == hipSuccess, "HIP device synchronize failed: ", hipGetErrorString(herr));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Approximate GELU activation (HIP, vectorized)");
}