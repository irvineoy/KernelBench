// HardTanh (clamp to [-1, 1]) optimized HIP kernel for AMD MI300X (gfx942)
// Single file containing kernels and PyTorch extension bindings.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Use 256 threads per block (4 waves) - good balance on MI300X
constexpr int THREADS_PER_BLOCK = 256;

// Device-side clamp
__device__ __forceinline__ float clamp_to_unit(float x) {
    // HardTanh in [-1, 1]
    x = fmaxf(x, -1.0f);
    x = fminf(x,  1.0f);
    return x;
}

// Vectorized kernel: each thread processes 4 contiguous elements (float4)
// Tail elements (when N not divisible by 4) are handled safely.
__global__ __launch_bounds__(THREADS_PER_BLOCK, 8)
void hardtanh_vec4_kernel(const float* __restrict__ in,
                          float* __restrict__ out,
                          int64_t N)
{
    // Global thread id in units of float4
    int64_t tid4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t base = tid4 * 4;  // starting element index

    if (base + 3 < N) {
        // Fast path: fully in-bounds vector transaction
        // Assumes 16-byte alignment; wrapper ensures alignment before choosing this kernel.
        const float4* __restrict__ in4  = reinterpret_cast<const float4*>(in);
        float4* __restrict__ out4 = reinterpret_cast<float4*>(out);
        float4 v = in4[tid4];

        v.x = clamp_to_unit(v.x);
        v.y = clamp_to_unit(v.y);
        v.z = clamp_to_unit(v.z);
        v.w = clamp_to_unit(v.w);

        out4[tid4] = v;
    } else {
        // Tail path: process remaining 0..3 scalars safely
        for (int i = 0; i < 4; ++i) {
            int64_t idx = base + i;
            if (idx < N) {
                float v = in[idx];
                out[idx] = clamp_to_unit(v);
            }
        }
    }
}

// Scalar kernel: one element per thread (used when pointers are not 16B aligned)
__global__ __launch_bounds__(THREADS_PER_BLOCK, 8)
void hardtanh_scalar_kernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            int64_t N)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = in[idx];
    out[idx] = clamp_to_unit(v);
}

// PyTorch extension entry point.
// Signature must only contain tensors from get_inputs() and model parameters (none here).
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 tensors are supported");
    // Ensure contiguous for coalesced memory access
    at::Tensor x = input.contiguous();

    const int64_t N = x.numel();
    auto output = at::empty_like(x);

    if (N == 0) return output;

    const float* in_ptr  = x.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    // Determine if we can safely use vectorized path:
    // - pointers 16B aligned
    // - at least 4 elements
    uintptr_t in_addr  = reinterpret_cast<uintptr_t>(in_ptr);
    uintptr_t out_addr = reinterpret_cast<uintptr_t>(out_ptr);
    bool aligned16 = ((in_addr | out_addr) & 0xF) == 0;
    bool use_vec4 = aligned16 && (N >= 4);

    if (use_vec4) {
        // Each thread handles 4 elements
        int64_t work_items = (N + 3) / 4;  // number of float4 items
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(static_cast<unsigned int>((work_items + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK));
        hipLaunchKernelGGL(hardtanh_vec4_kernel, grid, block, 0, 0, in_ptr, out_ptr, N);
    } else {
        // Fallback scalar kernel
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(static_cast<unsigned int>((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK));
        hipLaunchKernelGGL(hardtanh_scalar_kernel, grid, block, 0, 0, in_ptr, out_ptr, N);
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return output.view_as(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "HardTanh activation (clamp to [-1,1]) - HIP optimized");
}