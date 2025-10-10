// L1 normalization along dim=1 for a 2D tensor [N, D]
// y[i, j] = x[i, j] / mean_j(|x[i, j]|)
// Fused, block-per-row kernel optimized for AMD MI300X (gfx942)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <cmath>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Threads per block must be a multiple of 64 (wavefront size)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

static_assert(BLOCK_SIZE % WARP_SIZE == 0, "BLOCK_SIZE must be a multiple of wavefront size (64).");
static_assert(BLOCK_SIZE <= 1024, "BLOCK_SIZE must be <= 1024.");

__global__ void l1norm_rows_fused_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long rows,
    long long cols)
{
    const long long row = static_cast<long long>(blockIdx.x);
    if (row >= rows) return;

    const float* __restrict__ xrow = x + (size_t)row * (size_t)cols;
    float* __restrict__ yrow = y + (size_t)row * (size_t)cols;

    // Partial sum of |x| for this row
    float local_sum = 0.0f;

    // Coalesced strided access across columns
    for (long long j = threadIdx.x; j < cols; j += BLOCK_SIZE) {
        float v = xrow[j];
        local_sum += fabsf(v);
    }

    // Block reduction in LDS
    __shared__ float sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Tree reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Compute inverse mean and broadcast via LDS
    if (threadIdx.x == 0) {
        // mean = sum / cols; inv_mean = cols / sum
        float sum = sdata[0];
        float inv_mean = static_cast<float>(cols) / sum;  // matches PyTorch behavior (0/0 -> NaN)
        sdata[0] = inv_mean;
    }
    __syncthreads();

    float inv_mean = sdata[0];

    // Scale and write output (coalesced)
    for (long long j = threadIdx.x; j < cols; j += BLOCK_SIZE) {
        yrow[j] = xrow[j] * inv_mean;
    }
}

// PyTorch wrapper: accepts a single tensor input and returns normalized tensor
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU (HIP).");
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor of shape [N, D].");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 tensors are supported.");

    auto x = input.contiguous();  // ensure contiguous for coalesced access
    const long long rows = x.size(0);
    const long long cols = x.size(1);

    auto y = at::empty_like(x);

    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(static_cast<unsigned int>(rows), 1, 1);

    hipLaunchKernelGGL(
        l1norm_rows_fused_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        rows, cols
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "L1 normalization along dim=1 (HIP, fused reduction+scaling)");
}