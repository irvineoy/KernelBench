// ReLU HIP kernel optimized for AMD MI300X (gfx942)
// Single file: kernels + PyTorch extension bindings

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <algorithm>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Choose a good default block size (multiple of wave size)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Scalar ReLU kernel with grid-stride loop
__global__ void relu_kernel_scalar(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = int64_t(blockDim.x) * gridDim.x;

    // Use <= 0 to preserve NaNs (NaN <= 0 is false → we keep NaN)
    for (int64_t i = idx; i < N; i += stride) {
        float v = in[i];
        out[i] = (v <= 0.0f) ? 0.0f : v;
    }
}

// Vectorized float4 ReLU kernel for aligned, multiple-of-4 tensors
__global__ void relu_kernel_vec4(const float4* __restrict__ in4,
                                 float4* __restrict__ out4,
                                 int64_t N4) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = int64_t(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < N4; i += stride) {
        float4 v = in4[i];
        // Use <= 0.0f to ensure NaN propagates
        v.x = (v.x <= 0.0f) ? 0.0f : v.x;
        v.y = (v.y <= 0.0f) ? 0.0f : v.y;
        v.z = (v.z <= 0.0f) ? 0.0f : v.z;
        v.w = (v.w <= 0.0f) ? 0.0f : v.w;
        out4[i] = v;
    }
}

// PyTorch-facing wrapper: ONLY tensor parameters (matches model.forward signature)
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor.");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 tensors are supported.");

    // Ensure contiguous memory for coalesced access
    at::Tensor in = input.contiguous();
    at::Tensor out = at::empty_like(in);

    const int64_t N = in.numel();
    if (N == 0) {
        return out;
    }

    const float* in_ptr = in.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    // Launch configuration
    const int block = BLOCK_SIZE;
    // Use grid-stride loop with capped grid size to avoid excessive grids
    auto compute_grid = [&](int64_t work_items) -> dim3 {
        int64_t grid_x = (work_items + block - 1) / block;
        grid_x = std::min<int64_t>(grid_x, 65535); // safe cap
        return dim3(static_cast<unsigned int>(grid_x), 1, 1);
    };

    // Fast path: vectorized float4 if aligned and N % 4 == 0
    bool ptr_aligned = (reinterpret_cast<uintptr_t>(in_ptr) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(out_ptr) % 16 == 0);
    bool can_vec4 = ptr_aligned && (N % 4 == 0);

    hipError_t err;

    if (can_vec4) {
        const int64_t N4 = N / 4;
        dim3 grid = compute_grid(N4);
        dim3 block_dim(block);

        hipLaunchKernelGGL(
            relu_kernel_vec4,
            grid, block_dim, 0, 0,
            reinterpret_cast<const float4*>(in_ptr),
            reinterpret_cast<float4*>(out_ptr),
            N4
        );

        err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP kernel (vec4) failed: ", hipGetErrorString(err));
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP kernel (vec4) launch failed: ", hipGetErrorString(err));
    } else {
        dim3 grid = compute_grid(N);
        dim3 block_dim(block);

        hipLaunchKernelGGL(
            relu_kernel_scalar,
            grid, block_dim, 0, 0,
            in_ptr, out_ptr, N
        );

        err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP kernel (scalar) failed: ", hipGetErrorString(err));
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP kernel (scalar) launch failed: ", hipGetErrorString(err));
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "ReLU activation (HIP, optimized for MI300X)");
}