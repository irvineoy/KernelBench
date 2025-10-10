// Depthwise-separable Conv2D (depthwise + pointwise 1x1) optimized for AMD MI300X (gfx942)
// This file provides HIP kernels and a PyTorch extension entry point.
// It assumes stride=1, dilation=1, and uses "same" padding for odd kernels (pad = floor((k-1)/2)).
// Inputs: x, depthwise_weight[, depthwise_bias], pointwise_weight[, pointwise_bias]

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Tunables
#define TILE_DW 16      // Tile for depthwise output (height x width)
#define TM 16           // Tile size along M dimension (pixels) for pointwise GEMM
#define TN 16           // Tile size along N dimension (out_channels) for pointwise GEMM

// Depthwise convolution kernel (groups = in_channels)
// Weight layout: [C, 1, kH, kW], Input: [B, C, H, W], Output: [B, C, outH, outW]
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,   // nullable
    float* __restrict__ output,
    int B, int C, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int outH, int outW
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int ow = blockIdx.x * TILE_DW + tx;
    int oh = blockIdx.y * TILE_DW + ty;
    int bc = blockIdx.z;  // index over B*C
    int c = bc % C;
    int b = bc / C;

    if (b >= B || c >= C || oh >= outH || ow >= outW) return;

    float acc = 0.0f;

    // Convolution over kernel window
    #pragma unroll
    for (int kh = 0; kh < kH; ++kh) {
        int ih = oh - padH + kh;
        if (ih < 0 || ih >= H) continue;

        #pragma unroll
        for (int kw = 0; kw < kW; ++kw) {
            int iw = ow - padW + kw;
            if (iw < 0 || iw >= W) continue;

            int in_idx = ((b * C + c) * H + ih) * W + iw;
            int w_idx = ((c * kH + kh) * kW + kw); // [C, 1, kH, kW]
            acc += input[in_idx] * weight[w_idx];
        }
    }

    if (bias != nullptr) {
        acc += bias[c];
    }

    int out_idx = ((b * C + c) * outH + oh) * outW + ow;
    output[out_idx] = acc;
}

// Pointwise 1x1 convolution kernel implemented as GEMM-like outer-product microkernel.
// Y(MxN) = A(MxK) * B(KxN) + bias(N)
// A is depthwise output flattened over pixels: M = B*H*W, K = C_in
// B is pointwise weight transposed to shape [K, N] (i.e., B[k, n] = W[n, k])
// We minimize global loads by sharing A row (TM elements) and B column (TN elements) via LDS per K-step.
__global__ void pointwise_1x1_kernel(
    const float* __restrict__ A,        // [B, C_in, H, W] flattened as MxK via indexing
    const float* __restrict__ BkN,      // [K, N] contiguous along N (transposed weights)
    const float* __restrict__ bias,     // nullable, length N
    float* __restrict__ Y,              // [B, N, H, W]
    int bs, int H, int W,               // for indexing M <-> (b,h,w)
    int K,                              // in_channels
    int N                               // out_channels
) {
    __shared__ float sA[TM];   // TM A-elements for current K-step
    __shared__ float sB[TN];   // TN B-elements for current K-step

    // Block covers [TM x TN] tile in [M x N]
    int m_tile = blockIdx.x * TM;
    int n_tile = blockIdx.y * TN;

    int tx = threadIdx.x; // [0..TM-1]
    int ty = threadIdx.y; // [0..TN-1]

    int M = bs * H * W;

    int m = m_tile + tx;
    int n = n_tile + ty;

    // Initialize accumulator with bias if available
    float acc = 0.0f;
    if (n < N && bias != nullptr) {
        acc = bias[n];
    }

    // Loop over K dimension
    for (int k = 0; k < K; ++k) {
        // Load A tile: sA[tx] = A[m, k], loaded by threads with ty==0
        if (ty == 0) {
            if (m < M) {
                // Map m -> (b, h, w)
                int tmp = m;
                int b = tmp / (H * W);
                tmp -= b * (H * W);
                int h = tmp / W;
                int w = tmp - h * W;

                int a_idx = ((b * K + k) * H + h) * W + w; // A[b, k, h, w]
                sA[tx] = A[a_idx];
            } else {
                sA[tx] = 0.0f;
            }
        }

        // Load B tile: sB[ty] = B[k, n], loaded by threads with tx==0
        if (tx == 0) {
            if (n < N) {
                int b_idx = k * N + n; // B[k, n] contiguous along n
                sB[ty] = BkN[b_idx];
            } else {
                sB[ty] = 0.0f;
            }
        }

        __syncthreads();

        if (m < M && n < N) {
            acc = fmaf(sA[tx], sB[ty], acc);
        }

        __syncthreads();
    }

    if (m < M && n < N) {
        // Map m back to (b,h,w)
        int tmp = m;
        int b = tmp / (H * W);
        tmp -= b * (H * W);
        int h = tmp / W;
        int w = tmp - h * W;

        int y_idx = ((b * N + n) * H + h) * W + w; // Y[b, n, h, w]
        Y[y_idx] = acc;
    }
}

// Transpose pointwise weights from [N, K, 1, 1] -> [K, N] contiguous along N
// Small helper kernel; size is small (<= a few tens of KB)
__global__ void transpose_pw_weight_NK_to_KN(
    const float* __restrict__ W_NK11,
    float* __restrict__ W_KN,
    int N, int K
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x; // out_channels
    int k = blockIdx.y * blockDim.y + threadIdx.y; // in_channels
    if (n >= N || k >= K) return;

    int src_idx = n * K + k;    // [N, K]
    int dst_idx = k * N + n;    // [K, N]
    W_KN[dst_idx] = W_NK11[src_idx];
}

// Compute "same" padding for odd kernels, else 0 (stride=1, dilation=1 assumed)
static inline std::pair<int,int> compute_same_pad(int kH, int kW) {
    int padH = (kH % 2 == 1) ? (kH / 2) : 0;
    int padW = (kW % 2 == 1) ? (kW / 2) : 0;
    return {padH, padW};
}

// Core runner implementing depthwise + pointwise; biases optional via nullable pointers
static at::Tensor run_impl(
    at::Tensor input,
    at::Tensor dw_weight,
    const at::Tensor* dw_bias_opt,    // nullable
    at::Tensor pw_weight,
    const at::Tensor* pw_bias_opt     // nullable
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA/HIP tensor");
    TORCH_CHECK(dw_weight.is_cuda() && pw_weight.is_cuda(), "weights must be CUDA/HIP tensors");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(dw_weight.scalar_type() == at::kFloat && pw_weight.scalar_type() == at::kFloat, "weights must be float32");

    auto x = input.contiguous();
    auto Wdw = dw_weight.contiguous();
    auto Wpw = pw_weight.contiguous();

    const float* dw_bias_ptr = nullptr;
    const float* pw_bias_ptr = nullptr;
    at::Tensor dw_bias_c;
    at::Tensor pw_bias_c;
    if (dw_bias_opt != nullptr) {
        TORCH_CHECK(dw_bias_opt->defined(), "dw_bias is not defined");
        TORCH_CHECK(dw_bias_opt->is_cuda(), "dw_bias must be CUDA/HIP tensor");
        TORCH_CHECK(dw_bias_opt->scalar_type() == at::kFloat, "dw_bias must be float32");
        dw_bias_c = dw_bias_opt->contiguous();
        dw_bias_ptr = dw_bias_c.data_ptr<float>();
    }
    if (pw_bias_opt != nullptr) {
        TORCH_CHECK(pw_bias_opt->defined(), "pw_bias is not defined");
        TORCH_CHECK(pw_bias_opt->is_cuda(), "pw_bias must be CUDA/HIP tensor");
        TORCH_CHECK(pw_bias_opt->scalar_type() == at::kFloat, "pw_bias must be float32");
        pw_bias_c = pw_bias_opt->contiguous();
        pw_bias_ptr = pw_bias_c.data_ptr<float>();
    }

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Depthwise weight: [C, 1, kH, kW]
    TORCH_CHECK(Wdw.size(0) == C && Wdw.size(1) == 1, "depthwise weight must be [C,1,kH,kW]");
    int64_t kH = Wdw.size(2);
    int64_t kW = Wdw.size(3);

    // Pointwise weight: [N, C, 1, 1]
    TORCH_CHECK(Wpw.size(1) == C && Wpw.size(2) == 1 && Wpw.size(3) == 1, "pointwise weight must be [N,C,1,1]");
    int64_t N = Wpw.size(0);

    // Use stride=1, dilation=1, and SAME padding for odd kernels by default
    auto pad = compute_same_pad((int)kH, (int)kW);
    int padH = pad.first;
    int padW = pad.second;

    int outH = (int)(H + 2 * padH - kH + 1);
    int outW = (int)(W + 2 * padW - kW + 1);

    // Allocate intermediate depthwise output and final output
    auto dw_out = at::empty({B, C, outH, outW}, x.options());
    auto y = at::empty({B, N, outH, outW}, x.options());

    // Launch depthwise kernel
    dim3 blockDW(TILE_DW, TILE_DW);
    dim3 gridDW(
        (outW + TILE_DW - 1) / TILE_DW,
        (outH + TILE_DW - 1) / TILE_DW,
        B * C
    );

    hipLaunchKernelGGL(
        depthwise_conv2d_kernel,
        gridDW, blockDW, 0, 0,
        x.data_ptr<float>(),
        Wdw.data_ptr<float>(),
        dw_bias_ptr,
        dw_out.data_ptr<float>(),
        (int)B, (int)C, (int)H, (int)W,
        (int)kH, (int)kW,
        (int)padH, (int)padW,
        (int)outH, (int)outW
    );
    {
        hipError_t err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP depthwise kernel failed: ", hipGetErrorString(err));
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP depthwise kernel launch failed: ", hipGetErrorString(err));
    }

    // Transpose pointwise weight to [K, N] contiguous along N for coalesced loads
    auto Wpw_t = at::empty({C, N}, Wpw.options());
    {
        dim3 blockT(16, 16); // 256 threads/block (multiple of 64)
        dim3 gridT(
            (N + blockT.x - 1) / blockT.x,
            (C + blockT.y - 1) / blockT.y
        );
        hipLaunchKernelGGL(
            transpose_pw_weight_NK_to_KN,
            gridT, blockT, 0, 0,
            Wpw.data_ptr<float>(),
            Wpw_t.data_ptr<float>(),
            (int)N, (int)C
        );
        hipError_t err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP transpose kernel failed: ", hipGetErrorString(err));
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP transpose kernel launch failed: ", hipGetErrorString(err));
    }

    // Launch pointwise 1x1 kernel (GEMM-like)
    int M = (int)(B * outH * outW);
    dim3 blockPW(TM, TN);  // 256 threads (multiple of 64)
    dim3 gridPW(
        (M + TM - 1) / TM,
        (N + TN - 1) / TN
    );

    hipLaunchKernelGGL(
        pointwise_1x1_kernel,
        gridPW, blockPW, 0, 0,
        dw_out.data_ptr<float>(),
        Wpw_t.data_ptr<float>(),
        pw_bias_ptr,
        y.data_ptr<float>(),
        (int)B, (int)outH, (int)outW,
        (int)C, (int)N
    );
    {
        hipError_t err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP pointwise kernel failed: ", hipGetErrorString(err));
        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP pointwise kernel launch failed: ", hipGetErrorString(err));
    }

    return y;
}

// PyTorch entry points (no scalar configuration arguments allowed):
// 1) No bias case: (input, depthwise_weight, pointwise_weight)
at::Tensor run(at::Tensor input, at::Tensor depthwise_weight, at::Tensor pointwise_weight) {
    return run_impl(input, depthwise_weight, nullptr, pointwise_weight, nullptr);
}

// 2) Bias case: (input, depthwise_weight, depthwise_bias, pointwise_weight, pointwise_bias)
at::Tensor run_biased(at::Tensor input,
                      at::Tensor depthwise_weight,
                      at::Tensor depthwise_bias,
                      at::Tensor pointwise_weight,
                      at::Tensor pointwise_bias) {
    return run_impl(input, depthwise_weight, &depthwise_bias, pointwise_weight, &pointwise_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Overloads to support presence/absence of bias
    m.def("run", &run, "Depthwise-separable Conv2D (no bias, HIP)");
    m.def("run", &run_biased, "Depthwise-separable Conv2D (with bias, HIP)");
}