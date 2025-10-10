// Argmax over dimension 1 for 3D tensor [B, D1, D2]
// Optimized for AMD MI300X (gfx942) using coalesced row-wise loads.
//
// Build (example):
//   hipcc -O3 --offload-arch=gfx9-4-generic argmax_dim1.hip -shared -fPIC -I/path/to/pytorch/include -I/path/to/pytorch/include/torch/csrc/api/include -L/path/to/pytorch/lib -ltorch -ltorch_cpu -lc10

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cfloat>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256  // multiple of wavefront size (64)
#endif

// Each block processes a tile of columns (j) for one batch (b).
// Threads within a block correspond to consecutive j indices so row loads are coalesced.
// For each column j, a thread scans all D1 rows to find argmax index.
__global__ __launch_bounds__(BLOCK_SIZE)
void argmax_dim1_kernel(const float* __restrict__ x,
                        int64_t* __restrict__ out,
                        int B, int D1, int D2)
{
    int b = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || j >= D2) return;

    // Base pointer for batch b
    size_t batch_base = static_cast<size_t>(b) * static_cast<size_t>(D1) * static_cast<size_t>(D2);

    float best_val = -FLT_MAX;
    int64_t best_idx = 0;

    // Iterate over dim1 with coalesced access across threads for each row
    // Don't unroll - D1 can be very large (e.g., 4096)
    for (int i = 0; i < D1; ++i) {
        const float* __restrict__ row_ptr = x + batch_base + static_cast<size_t>(i) * static_cast<size_t>(D2);
        float v = row_ptr[j];
        if (v > best_val) {
            best_val = v;
            best_idx = static_cast<int64_t>(i);
        }
    }

    // Write argmax index for (b, j) to output of shape [B, D2]
    out[static_cast<size_t>(b) * static_cast<size_t>(D2) + static_cast<size_t>(j)] = best_idx;
}

at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA (ROCm) tensor");
    TORCH_CHECK(input.dim() == 3, "Input must be a 3D tensor [B, D1, D2]");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input dtype must be float32");
    auto x = input.contiguous();

    const int B  = static_cast<int>(x.size(0));
    const int D1 = static_cast<int>(x.size(1));
    const int D2 = static_cast<int>(x.size(2));

    // Output: argmax over dim=1 removes that dimension -> shape [B, D2], dtype int64
    auto out = at::empty({B, D2}, x.options().dtype(at::kLong));

    dim3 block(BLOCK_SIZE);
    dim3 grid((D2 + BLOCK_SIZE - 1) / BLOCK_SIZE, B);

    hipLaunchKernelGGL(argmax_dim1_kernel, grid, block, 0, 0,
                       x.data_ptr<float>(),
                       out.data_ptr<int64_t>(),
                       B, D1, D2);

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Argmax over dim=1 for 3D tensor (HIP, optimized for MI300X)");
}