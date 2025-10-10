// Softsign activation (y = x / (1 + |x|)) optimized HIP kernel for AMD MI300X (gfx942)
// Single-file PyTorch extension with vectorized and scalar paths.
//
// Build example:
//   python setup.py install
// or JIT via torch.utils.cpp_extension.load with hipcc -O3

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Choose a block size that is a multiple of wavefront size (64)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Clamp grid size to avoid launching an excessive number of blocks.
// We rely on a grid-stride loop to cover all elements.
static inline int compute_grid_size(int64_t work_items, int block_size) {
    int64_t blocks = (work_items + block_size - 1) / block_size;
    const int64_t max_blocks = 65535;  // conservative upper bound
    if (blocks > max_blocks) blocks = max_blocks;
    return static_cast<int>(blocks);
}

// Scalar kernel: processes one element per thread in a grid-stride loop.
__global__ __launch_bounds__(BLOCK_SIZE)
void softsign_kernel_scalar(const float* __restrict__ in,
                            float* __restrict__ out,
                            uint64_t N) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

    for (uint64_t i = tid; i < N; i += stride) {
        float x = in[i];
        float y = x / (1.0f + fabsf(x));
        out[i] = y;
    }
}

// Vectorized kernel: processes float4 at a time (4 elements per thread access).
__global__ __launch_bounds__(BLOCK_SIZE)
void softsign_kernel_vec4(const float4* __restrict__ in4,
                          float4* __restrict__ out4,
                          uint64_t N4) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

    for (uint64_t i = tid; i < N4; i += stride) {
        float4 v = in4[i];
        float4 r;
        r.x = v.x / (1.0f + fabsf(v.x));
        r.y = v.y / (1.0f + fabsf(v.y));
        r.z = v.z / (1.0f + fabsf(v.z));
        r.w = v.w / (1.0f + fabsf(v.w));
        out4[i] = r;
    }
}

// PyTorch entry point: takes only tensors (no scalar config arguments)
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor on device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 tensors are supported");
    auto x = input.contiguous();  // ensure dense, contiguous memory
    auto out = at::empty_like(x);

    const int64_t N = x.numel();
    if (N == 0) return out;

    float* d_out = out.data_ptr<float>();
    const float* d_in = x.data_ptr<float>();

    // Check 16-byte alignment for safe/efficient float4 vectorization
    uintptr_t in_addr = reinterpret_cast<uintptr_t>(d_in);
    uintptr_t out_addr = reinterpret_cast<uintptr_t>(d_out);
    bool aligned16 = ((in_addr & 0xF) == 0) && ((out_addr & 0xF) == 0);

    hipError_t err;
    if (aligned16 && (N >= 4)) {
        // Vectorized main path
        uint64_t N4 = static_cast<uint64_t>(N) >> 2;  // N / 4
        int grid_vec = compute_grid_size(static_cast<int64_t>(N4), BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        dim3 grid(grid_vec);

        hipLaunchKernelGGL(softsign_kernel_vec4, grid, block, 0, 0,
                           reinterpret_cast<const float4*>(d_in),
                           reinterpret_cast<float4*>(d_out),
                           N4);
        err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP device sync error after vec4 kernel: ", hipGetErrorString(err));
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP launch error for vec4 kernel: ", hipGetErrorString(err));

        // Handle remaining tail elements (N % 4)
        uint64_t tail = static_cast<uint64_t>(N) & 3ULL;
        if (tail) {
            const float* tail_in = d_in + (N - tail);
            float* tail_out = d_out + (N - tail);
            int grid_tail = 1;
            dim3 grid2(grid_tail);
            dim3 block2(64);  // small block for tiny tail
            hipLaunchKernelGGL(softsign_kernel_scalar, grid2, block2, 0, 0,
                               tail_in, tail_out, tail);
            err = hipDeviceSynchronize();
            TORCH_CHECK(err == hipSuccess, "HIP device sync error after tail kernel: ", hipGetErrorString(err));
            err = hipGetLastError();
            TORCH_CHECK(err == hipSuccess, "HIP launch error for tail kernel: ", hipGetErrorString(err));
        }
    } else {
        // Fallback scalar path
        int grid_scalar = compute_grid_size(N, BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        dim3 grid(grid_scalar);

        hipLaunchKernelGGL(softsign_kernel_scalar, grid, block, 0, 0,
                           d_in, d_out, static_cast<uint64_t>(N));
        err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP device sync error after scalar kernel: ", hipGetErrorString(err));
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP launch error for scalar kernel: ", hipGetErrorString(err));
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Softsign activation (HIP, optimized for MI300X)");
}