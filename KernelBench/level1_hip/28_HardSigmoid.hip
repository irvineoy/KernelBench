// HardSigmoid (hardsigmoid) optimized HIP kernel for AMD MI300X (gfx942)
// Builds as a single PyTorch extension file.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <iostream>

#ifndef CHECK_HIP
#define CHECK_HIP(expr)                                                       \
  do {                                                                        \
    hipError_t err = (expr);                                                  \
    if (err != hipSuccess) {                                                  \
      TORCH_CHECK(false, "HIP error: ", hipGetErrorString(err),               \
                  " at ", __FILE__, ":", __LINE__);                           \
    }                                                                         \
  } while (0)
#endif

// Device-side hard-sigmoid for float
__device__ __forceinline__ float hsig(float x) {
    // y = clamp(x/6 + 0.5, 0, 1), using FMA for precision
    const float inv6 = 0.1666666667f; // 1/6
    float y = fmaf(x, inv6, 0.5f);
    y = fminf(fmaxf(y, 0.0f), 1.0f);
    return y;
}

// Vectorized kernel: processes 4 elements per thread via float4
template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
hardsigmoid_vec4_kernel(const float4* __restrict__ in,
                        float4* __restrict__ out,
                        long long N4) {
    long long tid   = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = tid; i < N4; i += stride) {
        float4 v = in[i];
        v.x = hsig(v.x);
        v.y = hsig(v.y);
        v.z = hsig(v.z);
        v.w = hsig(v.w);
        out[i] = v;
    }
}

// Scalar kernel: processes 1 element per thread with grid-stride loop
template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
hardsigmoid_scalar_kernel(const float* __restrict__ in,
                          float* __restrict__ out,
                          long long N) {
    long long tid   = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = tid; i < N; i += stride) {
        float x = in[i];
        out[i] = hsig(x);
    }
}

static inline int compute_grid_dim(long long work_items, int block_size) {
    // Choose a grid size that balances launch overhead and occupancy, using grid-stride loops.
    int device = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&prop, device) != hipSuccess) {
        // Fallback if device query fails
        int fallback = 8192;
        long long need = (work_items + block_size - 1) / block_size;
        if (need <= 0) need = 1;
        return (int)std::min<long long>(need, fallback);
    }
    int sm = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
    int target = sm * 32; // 32 wavesets per CU
    long long need = (work_items + block_size - 1) / block_size;
    if (need <= 0) need = 1;
    return (int)std::min<long long>(need, target);
}

// Entry point called by the benchmarking harness: only tensor arguments allowed.
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA/HIP device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 is supported");

    // Make contiguous to guarantee coalesced accesses
    at::Tensor x = input.contiguous();
    at::Tensor y = at::empty_like(x);

    const long long N = x.numel();
    if (N == 0) return y;

    const int BLOCK = 256; // multiple of 64 (wavefront)

    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    // Vectorization eligibility: 16-byte aligned pointers
    bool aligned16 = ((reinterpret_cast<uintptr_t>(x_ptr) & 0xF) == 0) &&
                     ((reinterpret_cast<uintptr_t>(y_ptr) & 0xF) == 0);

    long long N4 = 0;
    if (aligned16 && N >= 4) {
        N4 = N / 4;
        const float4* x4 = reinterpret_cast<const float4*>(x_ptr);
        float4* y4 = reinterpret_cast<float4*>(y_ptr);
        int grid_v = compute_grid_dim(N4, BLOCK);
        hipLaunchKernelGGL((hardsigmoid_vec4_kernel<BLOCK>), dim3(grid_v), dim3(BLOCK), 0, 0,
                           x4, y4, N4);
        CHECK_HIP(hipGetLastError());
    }

    long long rem = N - 4 * N4;
    if (rem > 0) {
        const float* xr = x_ptr + 4 * N4;
        float* yr = y_ptr + 4 * N4;
        int grid_s = compute_grid_dim(rem, BLOCK);
        hipLaunchKernelGGL((hardsigmoid_scalar_kernel<BLOCK>), dim3(grid_s), dim3(BLOCK), 0, 0,
                           xr, yr, rem);
        CHECK_HIP(hipGetLastError());
    }

    CHECK_HIP(hipDeviceSynchronize());
    return y.view(input.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "HardSigmoid activation (HIP, optimized)");
}