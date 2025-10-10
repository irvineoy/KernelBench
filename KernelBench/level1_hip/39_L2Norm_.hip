// L2 normalization along dim=1 for a 2D tensor (batch_size, dim)
// Optimized for AMD MI300X (gfx942)

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256   // Multiple of 64 (wavefront size on AMD); good balance for MI300X
#endif

// Kernel: one thread block processes one row (batch element).
// Phase 1: parallel reduction to compute L2 norm of the row.
// Phase 2: scale the row by the reciprocal of the norm.
__global__ void l2norm_rows_kernel(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   long rows,
                                   long cols) {
    const int tid = threadIdx.x;
    const long row = static_cast<long>(blockIdx.x);
    if (row >= rows) return;

    // Shared memory for reduction and broadcasting the inverse norm
    __shared__ float ssum[BLOCK_SIZE];
    __shared__ float s_inv_norm;

    const long base = row * cols;

    // Accumulate partial sum of squares for this thread
    float local = 0.0f;

    // Coalesced access: consecutive threads read consecutive columns
    for (long c = tid; c < cols; c += blockDim.x) {
        float v = x[base + c];
        local = fmaf(v, v, local);  // v*v + local using FMA
    }

    // Block-level reduction
    ssum[tid] = local;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }

    // Compute inverse norm once per block and broadcast via shared memory
    if (tid == 0) {
        float norm = sqrtf(ssum[0]);
        // No epsilon - match PyTorch's exact behavior: division by zero yields inf/NaN as appropriate
        s_inv_norm = 1.0f / norm;  // If norm==0 -> inf, which matches x / 0 behavior on IEEE754
    }
    __syncthreads();

    const float inv = s_inv_norm;

    // Scale the row by inv norm (multiply is faster and consistent with division semantics, incl. inf/NaN)
    for (long c = tid; c < cols; c += blockDim.x) {
        y[base + c] = x[base + c] * inv;
    }
}

// PyTorch entry point: accepts only tensors (no scalar configuration).
// Input: x of shape [batch_size, dim], dtype float32, contiguous.
// Output: y with same shape and dtype, L2-normalized along dim=1.
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor on device");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch_size, dim)");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 is supported");
    auto x = input.contiguous();

    const long rows = x.size(0);
    const long cols = x.size(1);

    auto y = at::empty_like(x);

    dim3 block(BLOCK_SIZE);
    dim3 grid(static_cast<unsigned int>(rows));  // rows fits in grid.x for typical sizes

    hipLaunchKernelGGL(
        l2norm_rows_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        rows, cols
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel execution failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "L2 normalization along dim=1 (HIP, MI300X-optimized)");
}