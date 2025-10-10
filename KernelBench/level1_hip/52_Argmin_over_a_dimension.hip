// Argmin along dimension 1 for 3D tensors: input shape [B, M, N] -> output shape [B, N]
// Returns int64 indices (PyTorch default) for the minimum value along dim=1.
//
// Build: hipcc -O3 --offload-arch=gfx9-4-generic argmin_dim1.hip -shared -fPIC -o argmin_dim1.so
// Usage via PyTorch extension: module.run(x)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Kernel: For each (b, n), scan m in [0, M) and find argmin along dim=1.
// Thread mapping:
//   - grid.y over batch B
//   - grid.x * blockDim.x over N (last dimension)
// Coalesced access: at each iteration over m, threads within a wavefront access
// consecutive n indices from the same (b, m) row, yielding contiguous global loads.
__global__ void argmin_dim1_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ out_idx,
    int B,
    int M,
    int N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;  // column index (0..N-1)
    int b = blockIdx.y;                              // batch index (0..B-1)

    if (b >= B || n >= N) return;

    // Base offset for batch b
    size_t batch_base = static_cast<size_t>(b) * static_cast<size_t>(M) * static_cast<size_t>(N);

    // Initialize from first row (m=0) to preserve "first occurrence" tie-breaking like PyTorch
    size_t idx0 = batch_base + static_cast<size_t>(n);
    float min_val = x[idx0];
    int64_t min_arg = 0;

    // Scan remaining rows
    for (int m = 1; m < M; ++m) {
        size_t idx = batch_base + static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n);
        float v = x[idx];
        if (v < min_val) {
            min_val = v;
            min_arg = static_cast<int64_t>(m);
        }
    }

    // Write result: output shape [B, N]
    out_idx[static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n)] = min_arg;
}

at::Tensor run(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA/ROCm device");
    TORCH_CHECK(x.dim() == 3, "Expected a 3D tensor of shape [B, M, N]");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Expected float32 input tensor");

    // Ensure contiguous memory layout for predictable strides and coalesced access
    auto x_c = x.contiguous();

    int64_t B64 = x_c.size(0);
    int64_t M64 = x_c.size(1);
    int64_t N64 = x_c.size(2);

    TORCH_CHECK(B64 > 0 && M64 > 0 && N64 > 0, "Dimensions must be positive");

    // Allocate output tensor of indices (int64), shape [B, N]
    auto out = at::empty({B64, N64}, x_c.options().dtype(at::kLong));

    const int B = static_cast<int>(B64);
    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);

    // Launch configuration: 256 threads/block (multiple of wavefront=64)
    const int block_size = 256;
    dim3 block(block_size, 1, 1);
    dim3 grid((N + block_size - 1) / block_size, B, 1);

    hipLaunchKernelGGL(
        argmin_dim1_kernel,
        grid, block, 0, 0,
        x_c.data_ptr<float>(),
        out.data_ptr<int64_t>(),
        B, M, N
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Computes argmin along dim=1 for a 3D tensor [B, M, N] -> [B, N]
    m.def("run", &run, "Argmin along dim=1 (HIP, optimized for MI300X)");
}