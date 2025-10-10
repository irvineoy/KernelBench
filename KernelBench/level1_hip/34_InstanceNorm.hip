// InstanceNorm2d (affine=False, track_running_stats=False) optimized HIP kernel for MI300X
// Single file: kernels + PyTorch extension bindings
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <iostream>

#ifndef CHECK_HIP
#define CHECK_HIP(call)                                                                   \
    do {                                                                                  \
        hipError_t err__ = (call);                                                        \
        if (err__ != hipSuccess) {                                                        \
            TORCH_CHECK(false, "HIP error at ", __FILE__, ":", __LINE__, ": ",            \
                        hipGetErrorString(err__));                                        \
        }                                                                                 \
    } while (0)
#endif

// Choose a block size that is a multiple of AMD wavefront (64)
constexpr int BLOCK_SIZE = 256;

// Kernel 1: compute per-(n,c) mean and inverse std over H*W
// Input:  x [N, C, H, W]
// Outputs: mean [N*C], inv_std [N*C]
__global__ void __launch_bounds__(BLOCK_SIZE)
compute_stats_kernel(const float* __restrict__ x,
                     float* __restrict__ mean,
                     float* __restrict__ inv_std,
                     int N, int C, int H, int W, float eps) {
    int nc = blockIdx.x; // one block per (n,c)
    if (nc >= N * C) return;

    int M = H * W;
    int tid = threadIdx.x;

    // Decode n, c
    int n = nc / C;
    int c = nc % C;

    // Base pointer for this (n,c) slice
    size_t base = (static_cast<size_t>(n) * C + c) * static_cast<size_t>(H) * static_cast<size_t>(W);

    // Accumulate sum and sum of squares across spatial elements
    float thread_sum = 0.0f;
    float thread_sumsq = 0.0f;

    // Stride over the spatial elements assigned to this block
    // Coalesced: consecutive threads read consecutive elements
    for (int i = tid; i < M; i += blockDim.x) {
        float v = x[base + i];
        thread_sum   += v;
        thread_sumsq += v * v;
    }

    __shared__ float s_sum[BLOCK_SIZE];
    __shared__ float s_sumsq[BLOCK_SIZE];
    s_sum[tid]   = thread_sum;
    s_sumsq[tid] = thread_sumsq;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid]   += s_sum[tid + s];
            s_sumsq[tid] += s_sumsq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float m = s_sum[0] / static_cast<float>(M);
        float ex2 = s_sumsq[0] / static_cast<float>(M);
        float var = ex2 - m * m; // biased variance (population), matches InstanceNorm
        float inv = rsqrtf(var + eps);
        mean[nc]    = m;
        inv_std[nc] = inv;
    }
}

// Kernel 2: normalize using computed mean and inv_std
// y = (x - mean) * inv_std
__global__ void __launch_bounds__(BLOCK_SIZE)
normalize_kernel(const float* __restrict__ x,
                 const float* __restrict__ mean,
                 const float* __restrict__ inv_std,
                 float* __restrict__ y,
                 int N, int C, int H, int W) {
    int nc = blockIdx.z; // select (n,c)
    if (nc >= N * C) return;

    int tid = threadIdx.x;
    int M = H * W;

    int n = nc / C;
    int c = nc % C;

    float m   = mean[nc];
    float inv = inv_std[nc];

    // Number of tiles across spatial dimension for this (n,c)
    int tile_idx = blockIdx.x; // along spatial tiles
    int idx = tile_idx * blockDim.x + tid;
    if (idx >= M) return;

    size_t base = (static_cast<size_t>(n) * C + c) * static_cast<size_t>(H) * static_cast<size_t>(W);
    float v = x[base + idx];
    y[base + idx] = (v - m) * inv;
}

// PyTorch entry point: only tensors as arguments (input)
// InstanceNorm2d with affine=False, track_running_stats=False
at::Tensor run(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(x.dim() == 4, "Expected input of shape [N, C, H, W]");

    auto x_contig = x.contiguous();
    int64_t N64 = x_contig.size(0);
    int64_t C64 = x_contig.size(1);
    int64_t H64 = x_contig.size(2);
    int64_t W64 = x_contig.size(3);

    TORCH_CHECK(N64 > 0 && C64 > 0 && H64 > 0 && W64 > 0, "Invalid input dimensions");

    int N = static_cast<int>(N64);
    int C = static_cast<int>(C64);
    int H = static_cast<int>(H64);
    int W = static_cast<int>(W64);

    // Allocate output and workspace
    auto options = x_contig.options();
    at::Tensor y = at::empty_like(x_contig);
    at::Tensor mean = at::empty({N * C}, options);
    at::Tensor inv_std = at::empty({N * C}, options);

    float eps = 1.0e-5f; // PyTorch default

    // Launch compute_stats_kernel: grid.x = N*C, block.x = 256
    dim3 block_stats(BLOCK_SIZE);
    dim3 grid_stats(N * C);
    hipLaunchKernelGGL(
        compute_stats_kernel,
        grid_stats, block_stats, 0, 0,
        x_contig.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        N, C, H, W, eps
    );
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipGetLastError());

    // Launch normalize_kernel:
    // grid.z over N*C, grid.x over tiles of H*W
    int M = H * W;
    int tiles = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_norm(BLOCK_SIZE);
    dim3 grid_norm(tiles, 1, N * C);

    hipLaunchKernelGGL(
        normalize_kernel,
        grid_norm, block_norm, 0, 0,
        x_contig.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W
    );
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipGetLastError());

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "InstanceNorm2d (affine=False, track_running_stats=False) - HIP");
}