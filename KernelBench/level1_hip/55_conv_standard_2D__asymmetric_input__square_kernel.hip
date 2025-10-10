// hip_conv2d_nchw.hip
// Optimized 2D convolution (NCHW) for AMD MI300X (gfx942) with stride=1, padding=0, dilation=1, groups=1.
// Inputs: input[N,Cin,H,W], weight[Cout,Cin,K,K], optional bias[Cout]
// Output: output[N,Cout,H-K+1,W-K+1]
// This file contains HIP kernels and PyTorch extension bindings.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef TILE_W
#define TILE_W 32   // width tile (multiple of 64 across two rows)
#endif

#ifndef TILE_H
#define TILE_H 8    // height tile, TILE_W * TILE_H = 256 threads
#endif

// Kernel: Each block computes a TILE_H x TILE_W tile of output for a single (batch b, out_channel oc).
// We cache all weights for oc in shared memory when affordable.
// Indexing is NCHW, contiguous.
__global__ void conv2d_nchw_sharedw_kernel(
    const float* __restrict__ input,   // [N, Cin, H, W]
    const float* __restrict__ weight,  // [Cout, Cin, K, K]
    const float* __restrict__ bias,    // [Cout] or nullptr
    float* __restrict__ output,        // [N, Cout, Ho, Wo]
    int N, int Cin, int H, int W,
    int Cout, int K, int Ho, int Wo)
{
    // Determine (b, oc) this block works on
    int bz = blockIdx.z;
    int b  = bz / Cout;
    int oc = bz % Cout;

    // Tile coordinates
    int ow0 = blockIdx.x * TILE_W + threadIdx.x;
    int oh0 = blockIdx.y * TILE_H + threadIdx.y;

    // Shared memory for this oc's weights: size = Cin*K*K
    extern __shared__ float s_w[];

    // Flattened number of weights per oc
    int WperOC = Cin * K * K;

    // Cooperative load of weights for this oc into LDS
    // Thread linear id in block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int idx = tid; idx < WperOC; idx += blockDim.x * blockDim.y) {
        s_w[idx] = weight[oc * WperOC + idx];
    }
    __syncthreads();

    // Bounds check on batch/out_channel index
    if (b >= N || oc >= Cout) return;

    // Output bounds check for this thread
    if (oh0 >= Ho || ow0 >= Wo) return;

    // Initialize accumulator with bias if provided
    float acc = 0.0f;
    if (bias != nullptr) {
        acc = bias[oc];
    }

    // Compute convolution for this output element
    // input index base for batch b and channel ic will be computed in-loop
    // Access pattern: threads differ by ow0 (coalesced on W)
    // No padding, no dilation, stride=1, so in_h = oh0 + kh; in_w = ow0 + kw
    #pragma unroll 1
    for (int ic = 0; ic < Cin; ++ic) {
        // Pointers to input base for this (b, ic)
        int in_base = ((b * Cin + ic) * H) * W;
        // Weight base in shared memory for this (oc, ic)
        int sw_base = ic * (K * K);

        // Unroll small K loops to help ILP (K is typically small like 3)
        #pragma unroll 4
        for (int kh = 0; kh < K; ++kh) {
            int in_h = oh0 + kh;
            int in_row = (in_base + in_h * W);
            int sw_row = sw_base + kh * K;

            #pragma unroll 4
            for (int kw = 0; kw < K; ++kw) {
                int in_w = ow0 + kw;
                float v = input[in_row + in_w];          // Coalesced across threadIdx.x
                float wv = s_w[sw_row + kw];             // Broadcast from LDS
                acc = fmaf(v, wv, acc);
            }
        }
    }

    // Store result
    int out_idx = ((b * Cout + oc) * Ho + oh0) * Wo + ow0;
    output[out_idx] = acc;
}

// Fallback kernel without shared-memory weight cache (for very large Cin*K*K)
__global__ void conv2d_nchw_noshared_kernel(
    const float* __restrict__ input,   // [N, Cin, H, W]
    const float* __restrict__ weight,  // [Cout, Cin, K, K]
    const float* __restrict__ bias,    // [Cout] or nullptr
    float* __restrict__ output,        // [N, Cout, Ho, Wo]
    int N, int Cin, int H, int W,
    int Cout, int K, int Ho, int Wo)
{
    int bz = blockIdx.z;
    int b  = bz / Cout;
    int oc = bz % Cout;

    int ow0 = blockIdx.x * TILE_W + threadIdx.x;
    int oh0 = blockIdx.y * TILE_H + threadIdx.y;

    if (b >= N || oc >= Cout) return;
    if (oh0 >= Ho || ow0 >= Wo) return;

    float acc = 0.0f;
    if (bias != nullptr) {
        acc = bias[oc];
    }

    int WperOC = Cin * K * K;
    int w_oc_base = oc * WperOC;

    #pragma unroll 1
    for (int ic = 0; ic < Cin; ++ic) {
        int in_base = ((b * Cin + ic) * H) * W;
        int w_ic_base = w_oc_base + ic * (K * K);

        #pragma unroll 4
        for (int kh = 0; kh < K; ++kh) {
            int in_h = oh0 + kh;
            int in_row = in_base + in_h * W;
            int w_kh_base = w_ic_base + kh * K;

            #pragma unroll 4
            for (int kw = 0; kw < K; ++kw) {
                int in_w = ow0 + kw;
                float v = input[in_row + in_w];
                float wv = weight[w_kh_base + kw];
                acc = fmaf(v, wv, acc);
            }
        }
    }

    int out_idx = ((b * Cout + oc) * Ho + oh0) * Wo + ow0;
    output[out_idx] = acc;
}

// Host wrapper: no bias
at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(weight.dtype() == at::kFloat, "Only float32 weights supported");
    TORCH_CHECK(input.dim() == 4, "input must be NCHW");
    TORCH_CHECK(weight.dim() == 4, "weight must be [Cout,Cin,K,K]");

    // Enforce contiguous for coalesced access
    auto x = input.contiguous();
    auto w = weight.contiguous();

    int64_t N = x.size(0);
    int64_t Cin = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    int64_t Cout = w.size(0);
    int64_t Kcin = w.size(1);
    int64_t K = w.size(2);
    int64_t Kw = w.size(3);
    TORCH_CHECK(K == Kw, "Kernel must be square KxK");
    TORCH_CHECK(Kcin == Cin, "weight.size(1) must equal input channels (groups=1 only)");

    // Stride=1, padding=0, dilation=1
    int64_t Ho = H - K + 1;
    int64_t Wo = W - K + 1;
    TORCH_CHECK(Ho >= 0 && Wo >= 0, "Invalid output size; input smaller than kernel");

    auto out = at::empty({N, Cout, Ho, Wo}, x.options());

    if (Ho == 0 || Wo == 0 || N == 0 || Cout == 0) {
        return out;
    }

    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid((Wo + TILE_W - 1) / TILE_W,
              (Ho + TILE_H - 1) / TILE_H,
              N * Cout);

    // Choose kernel based on shared memory requirement
    size_t shmem_bytes = static_cast<size_t>(Cin) * static_cast<size_t>(K) * static_cast<size_t>(K) * sizeof(float);
    const float* bias_ptr = nullptr;

    if (shmem_bytes <= 48 * 1024) {
        hipLaunchKernelGGL(
            conv2d_nchw_sharedw_kernel,
            grid, block, shmem_bytes, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            bias_ptr,
            out.data_ptr<float>(),
            (int)N, (int)Cin, (int)H, (int)W,
            (int)Cout, (int)K, (int)Ho, (int)Wo
        );
    } else {
        hipLaunchKernelGGL(
            conv2d_nchw_noshared_kernel,
            grid, block, 0, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            bias_ptr,
            out.data_ptr<float>(),
            (int)N, (int)Cin, (int)H, (int)W,
            (int)Cout, (int)K, (int)Ho, (int)Wo
        );
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

// Host wrapper: with bias
at::Tensor run(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA/HIP tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(weight.dtype() == at::kFloat, "Only float32 weights supported");
    TORCH_CHECK(bias.dtype() == at::kFloat, "Only float32 bias supported");
    TORCH_CHECK(input.dim() == 4, "input must be NCHW");
    TORCH_CHECK(weight.dim() == 4, "weight must be [Cout,Cin,K,K]");
    TORCH_CHECK(bias.dim() == 1, "bias must be [Cout]");

    auto x = input.contiguous();
    auto w = weight.contiguous();
    auto b = bias.contiguous();

    int64_t N = x.size(0);
    int64_t Cin = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    int64_t Cout = w.size(0);
    int64_t Kcin = w.size(1);
    int64_t K = w.size(2);
    int64_t Kw = w.size(3);
    TORCH_CHECK(K == Kw, "Kernel must be square KxK");
    TORCH_CHECK(Kcin == Cin, "weight.size(1) must equal input channels (groups=1 only)");
    TORCH_CHECK(b.size(0) == Cout, "bias.size(0) must match out channels");

    int64_t Ho = H - K + 1;
    int64_t Wo = W - K + 1;
    TORCH_CHECK(Ho >= 0 && Wo >= 0, "Invalid output size; input smaller than kernel");

    auto out = at::empty({N, Cout, Ho, Wo}, x.options());

    if (Ho == 0 || Wo == 0 || N == 0 || Cout == 0) {
        return out;
    }

    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid((Wo + TILE_W - 1) / TILE_W,
              (Ho + TILE_H - 1) / TILE_H,
              N * Cout);

    size_t shmem_bytes = static_cast<size_t>(Cin) * static_cast<size_t>(K) * static_cast<size_t>(K) * sizeof(float);

    if (shmem_bytes <= 48 * 1024) {
        hipLaunchKernelGGL(
            conv2d_nchw_sharedw_kernel,
            grid, block, shmem_bytes, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            b.data_ptr<float>(),
            out.data_ptr<float>(),
            (int)N, (int)Cin, (int)H, (int)W,
            (int)Cout, (int)K, (int)Ho, (int)Wo
        );
    } else {
        hipLaunchKernelGGL(
            conv2d_nchw_noshared_kernel,
            grid, block, 0, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            b.data_ptr<float>(),
            out.data_ptr<float>(),
            (int)N, (int)Cin, (int)H, (int)W,
            (int)Cout, (int)K, (int)Ho, (int)Wo
        );
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Overloaded "run" to support optional bias
    m.def("run", py::overload_cast<at::Tensor, at::Tensor>(&run),
          "Conv2D NCHW (stride=1, pad=0, dil=1, groups=1) without bias");
    m.def("run", py::overload_cast<at::Tensor, at::Tensor, at::Tensor>(&run),
          "Conv2D NCHW (stride=1, pad=0, dil=1, groups=1) with bias");
}