// Frobenius norm normalization (x / ||x||_F) optimized for AMD MI300X (gfx942)
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <cmath>
#include <iostream>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Choose 256 threads per block (multiple of 64 for AMD wavefront)
constexpr int BLOCK_SIZE = 256;

// First-pass reduction: compute per-block partial sums of squares in double precision
__global__ __launch_bounds__(BLOCK_SIZE)
void reduce_sum_squares_kernel(const float* __restrict__ x,
                               double* __restrict__ partial,
                               int64_t N) {
    __shared__ double sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t global_tid = blockIdx.x * blockDim.x + tid;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    double sum = 0.0;

    // Grid-stride loop to cover all elements
    for (int64_t i = global_tid; i < N; i += stride) {
        float v = x[i];
        sum += (double)v * (double)v;
    }

    // Store to LDS and reduce within the block
    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write per-block result
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

// Final reduction kernel: reduce an array of doubles to a single double
__global__ __launch_bounds__(BLOCK_SIZE)
void finalize_reduce_kernel(const double* __restrict__ partial,
                            int num_partials,
                            double* __restrict__ out_sum) {
    __shared__ double sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    double sum = 0.0;

    // Each thread accumulates multiple partials
    for (int i = tid; i < num_partials; i += blockDim.x) {
        sum += partial[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_sum = sdata[0];
    }
}

// Normalize kernel (scalar path): y[i] = x[i] / sqrt(sum_of_squares)
__global__ __launch_bounds__(BLOCK_SIZE)
void normalize_kernel_scalar(const float* __restrict__ x,
                             float* __restrict__ y,
                             int64_t N,
                             const double* __restrict__ sum_ptr) {
    __shared__ float s_inv_norm;
    if (threadIdx.x == 0) {
        double s = *sum_ptr;
        // Compute 1 / sqrt(s). If s == 0, result will be INF; behavior matches x / 0 -> NaN for non-zero x.
        double inv = 1.0 / sqrt(s);
        s_inv_norm = static_cast<float>(inv);
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < N; i += stride) {
        y[i] = x[i] * s_inv_norm;
    }
}

// Normalize kernel (vectorized path): process float4 when aligned and N % 4 == 0
__global__ __launch_bounds__(BLOCK_SIZE)
void normalize_kernel_vectorized(const float4* __restrict__ x4,
                                 float4* __restrict__ y4,
                                 int64_t N4,
                                 const double* __restrict__ sum_ptr) {
    __shared__ float s_inv_norm;
    if (threadIdx.x == 0) {
        double s = *sum_ptr;
        double inv = 1.0 / sqrt(s);
        s_inv_norm = static_cast<float>(inv);
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < N4; i += stride) {
        float4 vx = x4[i];
        float4 vy;
        vy.x = vx.x * s_inv_norm;
        vy.y = vx.y * s_inv_norm;
        vy.z = vx.z * s_inv_norm;
        vy.w = vx.w * s_inv_norm;
        y4[i] = vy;
    }
}

// Wrapper: accepts only runtime tensors (matches model.forward signature)
at::Tensor run(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(x.dtype() == at::kFloat, "Only float32 input is supported");

    at::Tensor x_contig = x.contiguous();
    auto options_f32 = x_contig.options();
    auto options_f64 = x_contig.options().dtype(at::kDouble);

    int64_t N = x_contig.numel();
    at::Tensor y = at::empty_like(x_contig);

    if (N == 0) {
        return y;  // nothing to do
    }

    // Launch config
    const int threads = BLOCK_SIZE;
    // Limit number of blocks to keep partial buffer small and ensure good occupancy
    int64_t max_blocks = 32768;  // cap to avoid oversized temp buffer
    int64_t blocks_needed = (N + threads - 1) / threads;
    int grid_reduce = static_cast<int>(blocks_needed > max_blocks ? max_blocks : blocks_needed);
    if (grid_reduce < 1) grid_reduce = 1;

    // Temporary buffers for reduction
    at::Tensor partial = at::empty({grid_reduce}, options_f64);
    at::Tensor sum = at::empty({1}, options_f64);

    // 1) First-pass reduction
    hipLaunchKernelGGL(
        reduce_sum_squares_kernel,
        dim3(grid_reduce), dim3(threads), 0, 0,
        x_contig.data_ptr<float>(),
        partial.data_ptr<double>(),
        N
    );
    {
        hipError_t err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP kernel (reduce_sum_squares_kernel) failed: ", hipGetErrorString(err));
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP launch (reduce_sum_squares_kernel) failed: ", hipGetErrorString(err));
    }

    // 2) Final reduction to one scalar (sum of squares)
    hipLaunchKernelGGL(
        finalize_reduce_kernel,
        dim3(1), dim3(threads), 0, 0,
        partial.data_ptr<double>(),
        grid_reduce,
        sum.data_ptr<double>()
    );
    {
        hipError_t err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP kernel (finalize_reduce_kernel) failed: ", hipGetErrorString(err));
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP launch (finalize_reduce_kernel) failed: ", hipGetErrorString(err));
    }

    // 3) Normalize: y = x / sqrt(sum)
    // Try vectorized path if pointers are 16-byte aligned and N multiple of 4
    const void* x_ptr = x_contig.data_ptr<float>();
    void* y_ptr = y.data_ptr<float>();
    bool aligned16 = ((reinterpret_cast<uintptr_t>(x_ptr) & 0xF) == 0) && ((reinterpret_cast<uintptr_t>(y_ptr) & 0xF) == 0);
    bool vectorizable = aligned16 && (N % 4 == 0);

    int64_t work_items = vectorizable ? (N / 4) : N;
    int64_t blocks_norm_needed = (work_items + threads - 1) / threads;
    // Use a large grid for the normalization pass (memory bound); cap to a reasonable number
    int grid_norm = static_cast<int>(blocks_norm_needed > 131072 ? 131072 : blocks_norm_needed);
    if (grid_norm < 1) grid_norm = 1;

    if (vectorizable) {
        const float4* x4 = reinterpret_cast<const float4*>(x_contig.data_ptr<float>());
        float4* y4 = reinterpret_cast<float4*>(y.data_ptr<float>());
        hipLaunchKernelGGL(
            normalize_kernel_vectorized,
            dim3(grid_norm), dim3(threads), 0, 0,
            x4, y4, work_items, sum.data_ptr<double>()
        );
    } else {
        hipLaunchKernelGGL(
            normalize_kernel_scalar,
            dim3(grid_norm), dim3(threads), 0, 0,
            x_contig.data_ptr<float>(),
            y.data_ptr<float>(),
            work_items,
            sum.data_ptr<double>()
        );
    }
    {
        hipError_t err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP kernel (normalize) failed: ", hipGetErrorString(err));
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP launch (normalize) failed: ", hipGetErrorString(err));
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Frobenius norm normalization (x / ||x||_F) - HIP optimized");
}