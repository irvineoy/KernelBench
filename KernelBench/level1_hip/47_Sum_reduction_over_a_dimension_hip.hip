// Sum reduction over dim=1 (keepdim=True) for a 3D tensor [B, N, M].
// Optimized for AMD MI300X (gfx942) with coalesced memory access across M.
//
// Build example:
//   hipcc -O3 --offload-arch=gfx9-4-generic reduction_dim1.hip -o reduction
//
// PyTorch extension entry point: run(input: Tensor) -> Tensor
// Returns tensor of shape [B, 1, M]

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <iostream>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Use a multiple of 64 (AMD wavefront) for best performance
#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 256
#endif

// Kernel: each thread computes the sum over N for one (b, m) position.
// Access pattern is coalesced along M for each N step.
__global__ void __launch_bounds__(BLOCK_SIZE_X)
reduce_dim1_kernel(const float* __restrict__ x,
                   float* __restrict__ y,
                   long B, long N, long M) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;

    if (b >= B || m >= M) return;

    // Row base for the current batch index
    // x is laid out as [B, N, M] contiguous: index = ((b * N) + n) * M + m
    long row_base = static_cast<long>(b) * N * M;

    float sum = 0.0f;

    // Unroll in chunks of 4 for ILP
    long n = 0;
    long rb = row_base;
    long M1 = M;

    // Process 4 rows per iteration when possible
    for (; n + 3 < N; n += 4) {
        // For fixed n, threads in a wavefront access contiguous x[rb + m],
        // hence accesses are coalesced.
        sum += x[rb + m];
        sum += x[rb + M1 + m];
        sum += x[rb + 2 * M1 + m];
        sum += x[rb + 3 * M1 + m];
        rb += 4 * M1;
    }

    // Tail
    for (; n < N; ++n) {
        sum += x[rb + m];
        rb += M1;
    }

    // Output layout [B, 1, M], contiguous
    // Index = (b * 1 + 0) * M + m = b * M + m
    y[static_cast<long>(b) * M + m] = sum;
}

at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must be a 3D tensor [B, N, M]");

    // Ensure contiguous for predictable strides
    auto x = input.contiguous();

    const long B = x.size(0);
    const long N = x.size(1);
    const long M = x.size(2);

    // Output shape [B, 1, M], keepdim=True along dim=1
    auto y = at::empty({B, 1, M}, x.options());

    if (B == 0 || M == 0) {
        // Nothing to do, return empty output
        return y;
    }

    // Launch configuration
    dim3 block(BLOCK_SIZE_X, 1, 1);
    dim3 grid(static_cast<unsigned int>((M + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X),
              static_cast<unsigned int>(B),
              1);

    // Launch kernel
    hipStream_t stream = c10::hip::getCurrentHIPStream();
    hipLaunchKernelGGL(
        reduce_dim1_kernel,
        grid, block, 0, stream,
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        B, N, M
    );

    // Error checking
    hipError_t err_sync = hipGetLastError();
    TORCH_CHECK(err_sync == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err_sync));
    err_sync = hipStreamSynchronize(stream);
    TORCH_CHECK(err_sync == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err_sync));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Sum reduction over dim=1 (keepdim=True) [HIP]");
}