// depthwise_kh1.hip
// Optimized HIP implementation of depthwise 2D convolution with asymmetric kernel (kernel_h, 1)
// Matches PyTorch nn.Conv2d(groups=in_channels, kernel_size=(k,1)) with default stride=1, padding=0, dilation=1.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef TILE_X
#define TILE_X 32   // More threads along width to improve coalescing
#endif
#ifndef TILE_Y
#define TILE_Y 8
#endif

#define WAVE_SIZE 64
#define THREADS_PER_BLOCK (TILE_X * TILE_Y)

// Upper bound for kernel height cached in LDS (sufficient for common sizes)
#ifndef MAX_KH
#define MAX_KH 128
#endif

// Naive depthwise conv kernel (kernel width = 1), good for very small kernel_h
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
depthwise_kh1_naive_kernel(
    const float* __restrict__ input,   // NCHW
    const float* __restrict__ weight,  // [C, 1, KH, 1]
    float* __restrict__ output,        // NCHW_out
    int N, int C, int H, int W,
    int KH, int OH, int OW,
    int stride, int padding, int dilation)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int out_w = blockIdx.x * TILE_X + tx;
    int out_h = blockIdx.y * TILE_Y + ty;

    int cz = blockIdx.z;          // linear over N*C
    int c = cz % C;
    int n = cz / C;

    if (n >= N || c >= C || out_h >= OH || out_w >= OW) return;

    float acc = 0.0f;

    // Only iterate over kernel height; kernel width is 1
    #pragma unroll
    for (int kh = 0; kh < KH; ++kh) {
        int in_h = out_h * stride - padding + kh * dilation;
        int in_w = out_w * stride - padding; // kw = 0 (since kernel_w == 1)

        if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
            // Indexing: ((n*C + c)*H + in_h)*W + in_w
            int in_idx = ((n * C + c) * H + in_h) * W + in_w;
            int w_idx  = c * KH + kh;  // weight[c, 0, kh, 0]
            acc += input[in_idx] * weight[w_idx];
        }
    }

    int out_idx = ((n * C + c) * OH + out_h) * OW + out_w;
    output[out_idx] = acc;
}

// Shared-memory-optimized kernel: caches weights in LDS for reuse across TILE_X*TILE_Y threads
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
depthwise_kh1_shared_kernel(
    const float* __restrict__ input,   // NCHW
    const float* __restrict__ weight,  // [C, 1, KH, 1]
    float* __restrict__ output,        // NCHW_out
    int N, int C, int H, int W,
    int KH, int OH, int OW,
    int stride, int padding, int dilation)
{
    __shared__ float s_weight[MAX_KH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int out_w = blockIdx.x * TILE_X + tx;
    int out_h = blockIdx.y * TILE_Y + ty;

    int cz = blockIdx.z;          // linear over N*C
    int c = cz % C;
    int n = cz / C;

    // Load weights to LDS (one block works on a fixed (n,c) pair)
    for (int i = tid; i < KH; i += blockDim.x * blockDim.y) {
        s_weight[i] = weight[c * KH + i];
    }
    __syncthreads();

    if (n >= N || c >= C || out_h >= OH || out_w >= OW) return;

    float acc = 0.0f;

    // Unroll a bit for small/medium KH
    #pragma unroll 4
    for (int kh = 0; kh < KH; ++kh) {
        int in_h = out_h * stride - padding + kh * dilation;
        int in_w = out_w * stride - padding; // kw = 0 (since kernel_w == 1)

        if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
            int in_idx = ((n * C + c) * H + in_h) * W + in_w;
            acc += input[in_idx] * s_weight[kh];
        }
    }

    int out_idx = ((n * C + c) * OH + out_h) * OW + out_w;
    output[out_idx] = acc;
}

at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Only float32 weights supported");

    // Force contiguous for predictable indexing
    auto x = input.contiguous();
    auto w = weight.contiguous();

    TORCH_CHECK(x.dim() == 4, "Input must be NCHW");
    TORCH_CHECK(w.dim() == 4, "Weight must be [C, 1, KH, KW] for depthwise");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Weight layout for depthwise: [C, 1, KH, 1]
    int Cout = w.size(0);
    int Cin_group = w.size(1);
    int KH = w.size(2);
    int KW = w.size(3);

    TORCH_CHECK(Cout == C, "Depthwise: weight.size(0) must equal in/out channels");
    TORCH_CHECK(Cin_group == 1, "Depthwise: weight.size(1) must be 1");
    TORCH_CHECK(KW == 1, "Asymmetric kernel expected with kernel_w == 1");
    TORCH_CHECK(KH > 0 && KH <= MAX_KH, "Kernel height must be in (0, ", MAX_KH, "]");

    // Defaults per instructions (no scalars passed at runtime)
    const int stride = 1;
    const int padding = 0;
    const int dilation = 1;

    // Output sizes (general conv formula)
    int OH = (H + 2 * padding - dilation * (KH - 1) - 1) / stride + 1;
    int OW = (W + 2 * padding - dilation * (KW - 1) - 1) / stride + 1;

    auto y = at::empty({N, C, OH, OW}, x.options());

    dim3 block(TILE_X, TILE_Y);
    dim3 grid((OW + TILE_X - 1) / TILE_X,
              (OH + TILE_Y - 1) / TILE_Y,
              N * C);

    // Choose kernel: cache weights in LDS when KH is moderate
    const int KH_THRESHOLD_USE_SHARED = 5;

    if (KH <= KH_THRESHOLD_USE_SHARED) {
        hipLaunchKernelGGL(
            depthwise_kh1_naive_kernel,
            grid, block, 0, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            y.data_ptr<float>(),
            N, C, H, W,
            KH, OH, OW,
            stride, padding, dilation
        );
    } else {
        hipLaunchKernelGGL(
            depthwise_kh1_shared_kernel,
            grid, block, 0, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            y.data_ptr<float>(),
            N, C, H, W,
            KH, OH, OW,
            stride, padding, dilation
        );
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device synchronize failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Depthwise 2D convolution (kernel_h,1), groups=in_channels (HIP)");
}