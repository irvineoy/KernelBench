// Min-reduction over dimension 1 for a 3D tensor [N, M, K] -> [N, K]
// Optimized for AMD MI300X (gfx942)
// Build as a PyTorch extension.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <limits>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Kernel: each thread computes one output element out[n, k] = min_m x[n, m, k]
// Mapping:
//   - blockDim.x is a multiple of 64 (wavefront size)
//   - grid.x tiles the K dimension
//   - grid.y maps 1-to-1 to N (batch) dimension
// Memory access pattern:
//   For each fixed (n, m), threads in a wavefront iterate over contiguous k values
//   -> coalesced global loads
__global__ __launch_bounds__(256)
void reduce_min_dim1_kernel_f32(const float* __restrict__ x,
                                float* __restrict__ out,
                                int64_t N, int64_t M, int64_t K) {
    int n = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N || k >= K) return;

    // Base offset for this n
    const int64_t n_base = (int64_t)n * M * K;

    // Initialize with +inf
    float minv = INFINITY;

    // Unroll by 4 over M when possible
    int64_t m = 0;
    for (; m + 3 < M; m += 4) {
        int64_t base0 = n_base + (m + 0) * (int64_t)K;
        int64_t base1 = n_base + (m + 1) * (int64_t)K;
        int64_t base2 = n_base + (m + 2) * (int64_t)K;
        int64_t base3 = n_base + (m + 3) * (int64_t)K;

        float v0 = x[base0 + k];
        float v1 = x[base1 + k];
        float v2 = x[base2 + k];
        float v3 = x[base3 + k];

        // Reduce within registers
        float t0 = v0 < v1 ? v0 : v1;
        float t1 = v2 < v3 ? v2 : v3;
        float t  = t0 < t1 ? t0 : t1;
        minv = minv < t ? minv : t;
    }

    // Tail
    for (; m < M; ++m) {
        float v = x[n_base + m * (int64_t)K + k];
        minv = minv < v ? minv : v;
    }

    out[n * (int64_t)K + k] = minv;
}

at::Tensor run(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA/HIP device");
    TORCH_CHECK(x.dim() == 3, "Expected 3D tensor [N, M, K]");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 tensors are supported");

    auto x_contig = x.contiguous();

    const int64_t N = x_contig.size(0);
    const int64_t M = x_contig.size(1);  // reduce over this dimension (dim=1)
    const int64_t K = x_contig.size(2);

    // Output shape [N, K]
    auto out = at::empty({N, K}, x_contig.options());

    // Launch configuration
    const int threads = 256;  // multiple of 64 (wavefront)
    dim3 block(threads, 1, 1);
    dim3 grid((K + threads - 1) / threads, N, 1);

    hipLaunchKernelGGL(
        reduce_min_dim1_kernel_f32,
        grid, block, 0, 0,
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        N, M, K
    );

    // Error checking
    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Min reduction over dim=1 (HIP, float32)");
}