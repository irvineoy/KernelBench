// RMSNorm (NCHW) optimized HIP kernel for AMD MI300X (gfx942)
// Single-file PyTorch extension: build with hipcc -O3 --offload-arch=gfx9-4-generic

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Use a block size that is a multiple of the wavefront size (64)
#ifndef BLOCK_X
#define BLOCK_X 256
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 1
#endif

// Two-pass RMSNorm over channel dimension (C) for NCHW layout:
// For each (n, h, w):
//   1) compute mean of squares across C
//   2) compute inv_rms = 1/sqrt(mean + eps)
//   3) write y[n, c, h, w] = x[n, c, h, w] * inv_rms for all c
//
// Memory access pattern:
// - Threads in a wavefront iterate the same channel 'c' simultaneously,
//   thus accessing contiguous memory along width (w) -> coalesced loads/stores.
__global__ __launch_bounds__(BLOCK_X * BLOCK_Y)
void rmsnorm_nchw_kernel(const float* __restrict__ x,
                         float* __restrict__ y,
                         int N, int C, int H, int W,
                         float eps)
{
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int nh = blockIdx.y * blockDim.y + threadIdx.y;

    const int NH = N * H;
    if (w >= W || nh >= NH) return;

    const int n = nh / H;
    const int h = nh - n * H;

    const size_t HW = static_cast<size_t>(H) * static_cast<size_t>(W);
    const size_t cstride = HW;

    // Base index for (n, c=0, h, w)
    const size_t base = (static_cast<size_t>(n) * C * HW) + (static_cast<size_t>(h) * W) + w;

    // Pass 1: accumulate sum of squares across channels
    float sumsq = 0.0f;

    // Unroll by 4 across channel dimension
    int c = 0;
    const int C4 = (C / 4) * 4;
    size_t idx = base;

    for (; c < C4; c += 4) {
        float v0 = x[idx];
        float v1 = x[idx + cstride];
        float v2 = x[idx + 2 * cstride];
        float v3 = x[idx + 3 * cstride];
        sumsq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        idx += 4 * cstride;
    }
    for (; c < C; ++c) {
        float v = x[base + static_cast<size_t>(c) * cstride];
        sumsq += v * v;
    }

    float mean = sumsq / static_cast<float>(C);
    // Use fast reciprocal sqrt; numerical tolerance requirement is 1e-2
    float inv_rms = rsqrtf(mean + eps);

    // Pass 2: scale and write
    c = 0;
    idx = base;
    for (; c < C4; c += 4) {
        float v0 = x[idx];
        float v1 = x[idx + cstride];
        float v2 = x[idx + 2 * cstride];
        float v3 = x[idx + 3 * cstride];
        y[idx]                 = v0 * inv_rms;
        y[idx + cstride]       = v1 * inv_rms;
        y[idx + 2 * cstride]   = v2 * inv_rms;
        y[idx + 3 * cstride]   = v3 * inv_rms;
        idx += 4 * cstride;
    }
    for (; c < C; ++c) {
        size_t ii = base + static_cast<size_t>(c) * cstride;
        y[ii] = x[ii] * inv_rms;
    }
}

// PyTorch-facing entry point: accepts only tensors (no scalar config arguments)
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 tensors are supported");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D NCHW");

    // Ensure contiguous NCHW layout
    auto x = input.contiguous();

    const int N = static_cast<int>(x.size(0));
    const int C = static_cast<int>(x.size(1));
    const int H = static_cast<int>(x.size(2));
    const int W = static_cast<int>(x.size(3));

    auto y = at::empty_like(x);

    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((W + block.x - 1) / block.x,
              ((N * H) + block.y - 1) / block.y,
              1);

    // Default eps from the model (not passed at runtime)
    const float eps = 1.0e-5f;

    hipLaunchKernelGGL(
        rmsnorm_nchw_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        eps
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "RMSNorm (NCHW) optimized HIP kernel");
}