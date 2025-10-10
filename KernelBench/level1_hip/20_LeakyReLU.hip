// LeakyReLU (float32) optimized HIP kernel for AMD GPUs (MI300X/gfx942)
// Single file: kernels + PyTorch bindings

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <algorithm>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Branchless LeakyReLU: y = max(x,0) + slope * min(x,0)
__device__ __forceinline__ float leaky_relu_elem(float x, float slope) {
    return fmaxf(x, 0.0f) + slope * fminf(x, 0.0f);
}

// Vectorized elementwise LeakyReLU kernel with grid-stride loops.
// Processes main body as float4 for coalesced 16-byte transactions, then handles tail scalars.
__global__ void leaky_relu_kernel_vec4(
    const float* __restrict__ in,
    float* __restrict__ out,
    int64_t N,
    float negative_slope)
{
    // Each thread handles multiple elements via grid-stride
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(gridDim.x) * blockDim.x;

    // Vectorized path (4 elements per iteration)
    const int64_t total_vec = N >> 2; // N / 4
    const float4* __restrict__ in4  = reinterpret_cast<const float4*>(in);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(out);

    for (int64_t i = tid; i < total_vec; i += stride) {
        float4 v = in4[i];
        float4 r;
        r.x = leaky_relu_elem(v.x, negative_slope);
        r.y = leaky_relu_elem(v.y, negative_slope);
        r.z = leaky_relu_elem(v.z, negative_slope);
        r.w = leaky_relu_elem(v.w, negative_slope);
        out4[i] = r;
    }

    // Tail scalars
    const int64_t tail_start = total_vec << 2; // total_vec * 4
    for (int64_t i = tail_start + tid; i < N; i += stride) {
        float x = in[i];
        out[i] = leaky_relu_elem(x, negative_slope);
    }
}

// PyTorch-visible entry point: only tensor parameters.
// Matches the model.forward signature: run(input)
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA/HIP device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "This kernel supports only float32 tensors");

    // Ensure contiguous memory for coalesced access and vectorized loads
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    auto N = input.numel();
    auto output = at::empty_like(input);

    if (N == 0) {
        return output;
    }

    // Hard-coded negative_slope = 0.01 as per provided model (get_init_inputs() = [])
    const float negative_slope = 0.01f;

    // Launch configuration
    const int block = 256; // multiple of 64 (wavefront), good balance
    // Grid size to cover all elements; grid-stride loop handles any size
    int64_t grid64 = (N + block - 1) / block;
    // Clamp to int range if needed (HIP supports large grids; this is conservative)
    const int grid = static_cast<int>(std::min<int64_t>(grid64, 1'073'741'823LL)); // ~1e9 blocks upper bound

    // Launch kernel
    hipLaunchKernelGGL(
        leaky_relu_kernel_vec4,
        dim3(grid), dim3(block), 0, 0,
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int64_t>(N),
        negative_slope
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "LeakyReLU activation (HIP, float32)");
}