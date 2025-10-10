// Mean reduction over dim=1 for 3D tensor (B, D1, D2)
// Optimized for AMD MI300X (gfx942) with coalesced memory access across D2.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 256  // multiple of 64 for MI300X wavefront
#endif

// Kernel: each thread computes mean over dim1 for one output element (b, k)
// Grid mapping:
//   grid.y = B (batch dimension)
//   grid.x covers D2 with BLOCK_SIZE_X threads per block using a grid-stride loop
// Memory pattern:
//   For fixed (b, i), threads in a wavefront read consecutive k values -> coalesced loads.
__global__ __launch_bounds__(BLOCK_SIZE_X, 8)
void mean_dim1_kernel(const float* __restrict__ x,
                      float* __restrict__ out,
                      int64_t B,
                      int64_t D1,
                      int64_t D2) {
    int b = blockIdx.y;
    if (b >= B) return;

    int tx = threadIdx.x;
    int64_t k0 = static_cast<int64_t>(blockIdx.x) * blockDim.x + tx;
    int64_t k_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    const int64_t batch_offset = static_cast<int64_t>(b) * D1 * D2;

    for (int64_t k = k0; k < D2; k += k_stride) {
        const float* __restrict__ row_ptr = x + batch_offset + k; // points to (b, 0, k)
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;

        int64_t offset = 0;
        int64_t i = 0;

        // Unroll by 8 to improve ILP
        for (; i + 7 < D1; i += 8) {
            sum0 += row_ptr[offset + 0 * D2];
            sum1 += row_ptr[offset + 1 * D2];
            sum2 += row_ptr[offset + 2 * D2];
            sum3 += row_ptr[offset + 3 * D2];
            sum4 += row_ptr[offset + 4 * D2];
            sum5 += row_ptr[offset + 5 * D2];
            sum6 += row_ptr[offset + 6 * D2];
            sum7 += row_ptr[offset + 7 * D2];
            offset += 8 * D2;
        }

        float acc = (((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7)));

        // Tail
        for (; i < D1; ++i) {
            acc += row_ptr[static_cast<int64_t>(i) * D2];
        }

        out[static_cast<int64_t>(b) * D2 + k] = acc / static_cast<float>(D1);
    }
}

// PyTorch entry point: accepts only tensors (no scalar config).
// Implements torch.mean(x, dim=1) for x of shape (B, D1, D2).
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA/ROCm device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input tensor must be float32");
    TORCH_CHECK(input.dim() == 3, "Input tensor must have 3 dimensions (B, D1, D2)");

    // Ensure contiguous (PyTorch rand returns contiguous by default, but be safe)
    auto x = input.contiguous();

    const int64_t B = x.size(0);
    const int64_t D1 = x.size(1);
    const int64_t D2 = x.size(2);

    // Output shape: (B, D2)
    auto out = at::empty({B, D2}, x.options());

    // Launch configuration
    dim3 block(BLOCK_SIZE_X, 1, 1);
    int grid_x = static_cast<int>((D2 + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);
    if (grid_x < 1) grid_x = 1;
    dim3 grid(grid_x, static_cast<unsigned int>(B), 1);

    hipLaunchKernelGGL(
        mean_dim1_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, D1, D2
    );

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        TORCH_CHECK(false, "HIP device sync failed: ", hipGetErrorString(err));
    }
    err = hipGetLastError();
    if (err != hipSuccess) {
        TORCH_CHECK(false, "HIP kernel launch failed: ", hipGetErrorString(err));
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Mean reduction over dim=1 (HIP, optimized for MI300X)");
}