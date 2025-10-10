// Max reduction over dim=1 for 3D tensors: (B, D1, D2) -> (B, D2)
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256  // Multiple of 64 (wavefront size on AMD)
#endif

// Kernel maps one thread to one output column j for a fixed batch b.
// Each thread scans across D1 (the reduction dimension) with stride D2,
// which yields fully coalesced loads across threads for each i iteration.
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void max_reduce_dim1_kernel(const float* __restrict__ x,
                            float* __restrict__ out,
                            int B, int D1, int D2) {
    int b = blockIdx.y;
    if (b >= B) return;

    // Grid-stride loop over output columns (dimension 2)
    for (int j = blockIdx.x * blockDim.x + threadIdx.x;
         j < D2;
         j += gridDim.x * blockDim.x) {

        // Base pointer for (b, 0, j)
        const float* p = x + (static_cast<long long>(b) * D1 * D2) + j;

        // Initialize to -inf
        float maxv = -INFINITY;

        // Unroll by 4 over D1 for throughput
        int i = 0;
        int D2x1 = D2;
        int D2x2 = D2 << 1;        // 2 * D2
        int D2x3 = D2 + D2x2;      // 3 * D2
        for (; i + 3 < D1; i += 4) {
            float v0 = p[0];
            float v1 = p[D2x1];
            float v2 = p[D2x2];
            float v3 = p[D2x3];
            maxv = fmaxf(maxv, v0);
            maxv = fmaxf(maxv, v1);
            maxv = fmaxf(maxv, v2);
            maxv = fmaxf(maxv, v3);
            p += (D2x3 + D2x1); // advance by 4*D2
        }
        // Remainder
        for (; i < D1; ++i) {
            float v = *p;
            maxv = fmaxf(maxv, v);
            p += D2;
        }

        out[static_cast<long long>(b) * D2 + j] = maxv;
    }
}

// Host wrapper: performs max reduction over dim=1
// Input: x of shape (B, D1, D2), dtype float32, contiguous
at::Tensor run(at::Tensor x) {
    TORCH_CHECK(x.dim() == 3, "Expected a 3D tensor (B, D1, D2). Got dim = ", x.dim());
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Expected float32 tensor.");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA/HIP device (ROCm).");

    auto x_contig = x.contiguous();
    const int64_t B64  = x_contig.size(0);
    const int64_t D164 = x_contig.size(1);  // reduction dimension
    const int64_t D264 = x_contig.size(2);

    TORCH_CHECK(B64 >= 1 && D164 >= 1 && D264 >= 1, "All dimensions must be >= 1");
    TORCH_CHECK(B64 <= INT_MAX && D164 <= INT_MAX && D264 <= INT_MAX, "Dimensions exceed int32 range.");

    const int B  = static_cast<int>(B64);
    const int D1 = static_cast<int>(D164);
    const int D2 = static_cast<int>(D264);

    auto out = at::empty({B64, D264}, x_contig.options());

    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid((D2 + BLOCK_SIZE - 1) / BLOCK_SIZE, B, 1);

    hipLaunchKernelGGL(
        max_reduce_dim1_kernel,
        grid, block, 0, 0,
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        B, D1, D2
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));
    err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel execution failed: ", hipGetErrorString(err));

    return out;
}

// Optional: expose alternative entry points if you need other dims.
// Not used by the harness; provided for convenience.
/*
static at::Tensor reduce_dim0(at::Tensor x) { // (B, D1, D2) -> (D1, D2)
    // Implement similarly by remapping indices; omitted for brevity.
    return x; 
}
static at::Tensor reduce_dim2(at::Tensor x) { // (B, D1, D2) -> (B, D1)
    // Implement similarly by remapping indices; omitted for brevity.
    return x;
}
*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Max reduction over dim=1 for 3D tensors (HIP, optimized for MI300X)");
    // m.def("run_dim0", &reduce_dim0, "Max over dim=0");
    // m.def("run_dim2", &reduce_dim2, "Max over dim=2");
}