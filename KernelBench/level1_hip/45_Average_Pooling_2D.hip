// AvgPool2d (kernel=11, stride=11, padding=0) optimized HIP kernel for MI300X
// Single .hip file with PyTorch bindings

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef TILE_X
#define TILE_X 16  // threads along output width
#endif

#ifndef TILE_Y
#define TILE_Y 16  // threads along output height
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Hard-coded configuration for the provided model:
// nn.AvgPool2d(kernel_size=11, stride=None -> stride=11, padding=0, ceil_mode=False)
#ifndef KERNEL_K
#define KERNEL_K 11
#endif
#ifndef KERNEL_STRIDE
#define KERNEL_STRIDE 11
#endif
#ifndef KERNEL_PAD
#define KERNEL_PAD 0
#endif

// Device kernel: Each thread computes one output element (N, C, oh, ow)
// Uses 64-bit indexing to safely address up to and including 2^32 elements.
__global__ __launch_bounds__(TILE_X * TILE_Y)
void avgpool2d_k11s11p0_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int ow = blockIdx.x * blockDim.x + tx;
    const int oh = blockIdx.y * blockDim.y + ty;

    const int nc = blockIdx.z; // fuse N and C in grid.z
    const int n = nc / C;
    const int c = nc % C;

    if (n >= N || c >= C || oh >= outH || ow >= outW) {
        return;
    }

    // For AvgPool2d: with stride = kernel = 11 and padding = 0, all windows are full-size
    constexpr int K = KERNEL_K;
    constexpr int S = KERNEL_STRIDE;
    constexpr int P = KERNEL_PAD;

    const int ih0 = oh * S - P; // P = 0
    const int iw0 = ow * S - P; // P = 0

    // Base offset for (n, c)
    const int64_t HW = static_cast<int64_t>(H) * static_cast<int64_t>(W);
    const int64_t base_nc = (static_cast<int64_t>(n) * C + c) * HW;

    float sum = 0.0f;

    // Accumulate KxK window
    #pragma unroll
    for (int ky = 0; ky < K; ++ky) {
        const int ih = ih0 + ky; // guaranteed in-bounds for valid pooling window
        const int64_t row_start = base_nc + static_cast<int64_t>(ih) * static_cast<int64_t>(W) + iw0;

        // Unroll inner loop partially for better ILP
        #pragma unroll
        for (int kx = 0; kx < K; ++kx) {
            sum += input[row_start + kx];
        }
    }

    const float inv_area = 1.0f / static_cast<float>(K * K);

    // Write output
    const int64_t outHW = static_cast<int64_t>(outH) * static_cast<int64_t>(outW);
    const int64_t out_idx = (static_cast<int64_t>(n) * C + c) * outHW
                          + static_cast<int64_t>(oh) * static_cast<int64_t>(outW)
                          + ow;
    output[out_idx] = sum * inv_area;
}

// PyTorch wrapper: takes only runtime tensors (input). No scalar args allowed.
// Assumes AvgPool2d(kernel_size=11, stride=11, padding=0, ceil_mode=False).
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 inputs are supported");
    TORCH_CHECK(input.dim() == 4, "Input must be NCHW (4D)");

    // Ensure contiguous for coalesced access
    auto x = input.contiguous();

    const int N = static_cast<int>(x.size(0));
    const int C = static_cast<int>(x.size(1));
    const int H = static_cast<int>(x.size(2));
    const int W = static_cast<int>(x.size(3));

    // Hard-coded pooling parameters per target model
    constexpr int K = KERNEL_K;
    constexpr int S = KERNEL_STRIDE;
    constexpr int P = KERNEL_PAD;

    // ceil_mode=False (default), padding=0:
    // out = floor((H + 2P - K) / S) + 1
    TORCH_CHECK(P == 0, "This kernel assumes padding=0");
    TORCH_CHECK(S == K, "This kernel assumes stride == kernel_size");
    TORCH_CHECK(K > 0, "Kernel size must be > 0");
    TORCH_CHECK(H >= K && W >= K, "Input must be at least as large as kernel");

    const int outH = (H - K) / S + 1;
    const int outW = (W - K) / S + 1;

    auto y = at::empty({N, C, outH, outW}, x.options());

    // Launch configuration
    dim3 block(TILE_X, TILE_Y, 1); // 256 threads (multiple of wavefront 64)
    dim3 grid(
        (outW + TILE_X - 1) / TILE_X,
        (outH + TILE_Y - 1) / TILE_Y,
        N * C
    );

    hipLaunchKernelGGL(
        avgpool2d_k11s11p0_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W, outH, outW
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "AvgPool2d (kernel=11, stride=11, padding=0) - HIP");
}