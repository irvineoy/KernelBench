// Depthwise Conv2D (groups = in_channels) optimized HIP kernel for AMD MI300X (gfx942)
// Supports channel multiplier (out_channels = in_channels * multiplier)
// Assumes stride = 1, padding = 0, dilation = 1 (as per the provided PyTorch model test)
//
// Single-file Torch extension: kernels + wrappers + PyBind

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <iostream>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Tile configuration
#ifndef TILE_H
#define TILE_H 16
#endif

#ifndef TILE_W
#define TILE_W 16
#endif

// Depthwise conv2d with shared memory tiling
// - input:  NCHW, float32
// - weight: [C_out, 1, K, K]
// - bias:   [C_out] or nullptr
// - output: [N, C_out, H_out, W_out]
// Notes:
// - groups = in_channels (depthwise) with channel_multiplier = C_out / C_in
// - stride = 1, padding = 0, dilation = 1
__global__
void __launch_bounds__(TILE_H * TILE_W, 4)
dwconv2d_shared_kernel(const float* __restrict__ input,
                       const float* __restrict__ weight,
                       const float* __restrict__ bias, // may be nullptr
                       float* __restrict__ output,
                       int N, int C_in, int H, int W,
                       int C_out, int K,
                       int H_out, int W_out,
                       int channel_multiplier)
{
    extern __shared__ float smem[];

    // Grid decomposition
    // blockIdx.x -> output tile along width
    // blockIdx.y -> output tile along height
    // blockIdx.z -> fused (batch, out_channel)
    const int tx = threadIdx.x; // 0..TILE_W-1
    const int ty = threadIdx.y; // 0..TILE_H-1

    const int out_w0 = blockIdx.x * TILE_W;
    const int out_h0 = blockIdx.y * TILE_H;

    const int fused = blockIdx.z;
    const int n = fused / C_out;
    const int oc = fused % C_out;

    if (n >= N || oc >= C_out) {
        return;
    }

    // Depthwise mapping: each oc maps to exactly one ic
    // oc = ic * channel_multiplier + m
    const int ic = oc / channel_multiplier;

    // Output tile extents (handle edges)
    const int tile_out_h = min(TILE_H, H_out - out_h0);
    const int tile_out_w = min(TILE_W, W_out - out_w0);

    if (tile_out_h <= 0 || tile_out_w <= 0) {
        return;
    }

    // Input tile extents for stride=1, pad=0: tile_out + (K - 1)
    const int tile_in_h = tile_out_h + K - 1;
    const int tile_in_w = tile_out_w + K - 1;

    // LDS layout: add +1 padding on width to avoid bank conflicts
    const int lds_ld = tile_in_w + 1;

    float* s_input = smem;
    float* s_weight = s_input + (tile_in_h * lds_ld);

    // Load weights into shared memory
    const int tid_linear = ty * TILE_W + tx;
    const int weight_elems = K * K;
    for (int i = tid_linear; i < weight_elems; i += (TILE_W * TILE_H)) {
        // weight shape: [C_out, 1, K, K] — contiguous
        const int kh = i / K;
        const int kw = i % K;
        const int widx = oc * (K * K) + kh * K + kw;
        s_weight[i] = weight[widx];
    }

    // Base offsets for input/output
    const int input_base = ((n * C_in + ic) * H) * W;
    const int output_base = ((n * C_out + oc) * H_out) * W_out;

    // For stride=1, pad=0 input tile starts at (out_h0, out_w0)
    // These are guaranteed in-bounds when computing valid convolution.
    // Still, compute bounds and use them for safety (no divergence here).
    const int in_h0 = out_h0;
    const int in_w0 = out_w0;

    // Load input tile into shared memory (coalesced)
    for (int ih = ty; ih < tile_in_h; ih += TILE_H) {
        const int h_global = in_h0 + ih;
        const int row_in_global = input_base + h_global * W;

        int s_row = ih * lds_ld;
        for (int iw = tx; iw < tile_in_w; iw += TILE_W) {
            const int w_global = in_w0 + iw;
            s_input[s_row + iw] = input[row_in_global + w_global];
        }
    }
    __syncthreads();

    // Compute outputs for this tile
    for (int oh = ty; oh < tile_out_h; oh += TILE_H) {
        const int out_h = out_h0 + oh;
        const int s_row_base = oh * lds_ld; // starting row in s_input
        for (int ow = tx; ow < tile_out_w; ow += TILE_W) {
            const int out_w = out_w0 + ow;

            float acc = 0.0f;

            // Convolution sum over KxK
            // s_input indexing: (oh + kh, ow + kw)
            // Flattened: (oh + kh) * lds_ld + (ow + kw)
            // s_weight: kh*K + kw
            #pragma unroll 4
            for (int kh = 0; kh < K; ++kh) {
                const int s_row = s_row_base + kh * lds_ld;
                const int w_row = kh * K;
                #pragma unroll 4
                for (int kw = 0; kw < K; ++kw) {
                    acc += s_input[s_row + (ow + kw)] * s_weight[w_row + kw];
                }
            }

            // Bias
            if (bias != nullptr) {
                acc += bias[oc];
            }

            // Write out
            output[output_base + out_h * W_out + out_w] = acc;
        }
    }
}

// Naive fallback kernel (no shared memory), used for very large K if needed
__global__
void dwconv2d_naive_kernel(const float* __restrict__ input,
                           const float* __restrict__ weight,
                           const float* __restrict__ bias,
                           float* __restrict__ output,
                           int N, int C_in, int H, int W,
                           int C_out, int K,
                           int H_out, int W_out,
                           int channel_multiplier)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int out_w = blockIdx.x * blockDim.x + tx;
    const int out_h = blockIdx.y * blockDim.y + ty;

    const int fused = blockIdx.z;
    const int n = fused / C_out;
    const int oc = fused % C_out;

    if (n >= N || oc >= C_out || out_h >= H_out || out_w >= W_out) return;

    // Depthwise mapping
    const int ic = oc / channel_multiplier;

    // Base pointers
    const int input_base = ((n * C_in + ic) * H) * W;
    const int output_base = ((n * C_out + oc) * H_out) * W_out;

    float acc = 0.0f;

    // Stride=1, pad=0 => input origin = (out_h, out_w)
    const int in_h0 = out_h;
    const int in_w0 = out_w;

    #pragma unroll 2
    for (int kh = 0; kh < K; ++kh) {
        const int ih = in_h0 + kh;
        const int row_in = input_base + ih * W;
        const int w_row = oc * (K * K) + kh * K;
        #pragma unroll 2
        for (int kw = 0; kw < K; ++kw) {
            const int iw = in_w0 + kw;
            acc += input[row_in + iw] * weight[w_row + kw];
        }
    }

    if (bias != nullptr) {
        acc += bias[oc];
    }

    output[output_base + out_h * W_out + out_w] = acc;
}

// Host wrapper (no scalar args exposed in the API)
// Overload 1: without bias
at::Tensor run(at::Tensor input, at::Tensor weight) {
    // Validate types and layout
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Weight must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be NCHW 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be [C_out, 1, K, K] 4D");

    auto x = input.contiguous();
    auto w = weight.contiguous();

    const int64_t N = x.size(0);
    const int64_t C_in = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    const int64_t C_out = w.size(0);
    const int64_t Cin_per_group = w.size(1); // should be 1 for depthwise
    const int64_t K = w.size(2);
    TORCH_CHECK(w.size(3) == K, "Kernel must be square");

    TORCH_CHECK(Cin_per_group == 1, "Depthwise Conv expects weight.size(1) == 1");
    TORCH_CHECK(C_out % C_in == 0, "out_channels must be a multiple of in_channels for depthwise with multiplier");
    const int channel_multiplier = static_cast<int>(C_out / C_in);

    // Assume stride=1, padding=0, dilation=1
    const int H_out = static_cast<int>(H - K + 1);
    const int W_out = static_cast<int>(W - K + 1);
    TORCH_CHECK(H_out > 0 && W_out > 0, "Invalid output size; ensure H >= K and W >= K for stride=1, padding=0");

    auto y = at::empty({N, C_out, H_out, W_out}, x.options());

    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid((W_out + TILE_W - 1) / TILE_W,
              (H_out + TILE_H - 1) / TILE_H,
              N * C_out);

    // Dynamic shared memory size: input tile + weights
    size_t smem_input_elems = static_cast<size_t>((TILE_H + K - 1) * (TILE_W + K - 1 + 1));
    size_t smem_weight_elems = static_cast<size_t>(K * K);
    size_t smem_bytes = (smem_input_elems + smem_weight_elems) * sizeof(float);

    // Heuristic: use shared kernel for reasonable K, otherwise naive
    bool use_shared = (K <= 31); // safe and performant range
    if (use_shared) {
        hipLaunchKernelGGL(
            dwconv2d_shared_kernel, grid, block, smem_bytes, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            nullptr,
            y.data_ptr<float>(),
            static_cast<int>(N),
            static_cast<int>(C_in),
            static_cast<int>(H),
            static_cast<int>(W),
            static_cast<int>(C_out),
            static_cast<int>(K),
            static_cast<int>(H_out),
            static_cast<int>(W_out),
            static_cast<int>(channel_multiplier)
        );
    } else {
        hipLaunchKernelGGL(
            dwconv2d_naive_kernel, grid, block, 0, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            nullptr,
            y.data_ptr<float>(),
            static_cast<int>(N),
            static_cast<int>(C_in),
            static_cast<int>(H),
            static_cast<int>(W),
            static_cast<int>(C_out),
            static_cast<int>(K),
            static_cast<int>(H_out),
            static_cast<int>(W_out),
            static_cast<int>(channel_multiplier)
        );
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

// Overload 2: with bias
at::Tensor run(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(), "All tensors must be CUDA/HIP tensors");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "Bias must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be NCHW 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be [C_out, 1, K, K] 4D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D of size C_out");

    auto x = input.contiguous();
    auto w = weight.contiguous();
    auto b = bias.contiguous();

    const int64_t N = x.size(0);
    const int64_t C_in = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    const int64_t C_out = w.size(0);
    const int64_t Cin_per_group = w.size(1); // should be 1
    const int64_t K = w.size(2);
    TORCH_CHECK(w.size(3) == K, "Kernel must be square");
    TORCH_CHECK(Cin_per_group == 1, "Depthwise Conv expects weight.size(1) == 1");
    TORCH_CHECK(b.size(0) == C_out, "Bias size must match out_channels");

    TORCH_CHECK(C_out % C_in == 0, "out_channels must be a multiple of in_channels for depthwise with multiplier");
    const int channel_multiplier = static_cast<int>(C_out / C_in);

    // stride=1, pad=0, dilation=1
    const int H_out = static_cast<int>(H - K + 1);
    const int W_out = static_cast<int>(W - K + 1);
    TORCH_CHECK(H_out > 0 && W_out > 0, "Invalid output size; ensure H >= K and W >= K for stride=1, padding=0");

    auto y = at::empty({N, C_out, H_out, W_out}, x.options());

    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid((W_out + TILE_W - 1) / TILE_W,
              (H_out + TILE_H - 1) / TILE_H,
              N * C_out);

    // Dynamic shared memory
    size_t smem_input_elems = static_cast<size_t>((TILE_H + K - 1) * (TILE_W + K - 1 + 1));
    size_t smem_weight_elems = static_cast<size_t>(K * K);
    size_t smem_bytes = (smem_input_elems + smem_weight_elems) * sizeof(float);

    bool use_shared = (K <= 31);
    if (use_shared) {
        hipLaunchKernelGGL(
            dwconv2d_shared_kernel, grid, block, smem_bytes, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            b.data_ptr<float>(),
            y.data_ptr<float>(),
            static_cast<int>(N),
            static_cast<int>(C_in),
            static_cast<int>(H),
            static_cast<int>(W),
            static_cast<int>(C_out),
            static_cast<int>(K),
            static_cast<int>(H_out),
            static_cast<int>(W_out),
            static_cast<int>(channel_multiplier)
        );
    } else {
        hipLaunchKernelGGL(
            dwconv2d_naive_kernel, grid, block, 0, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            b.data_ptr<float>(),
            y.data_ptr<float>(),
            static_cast<int>(N),
            static_cast<int>(C_in),
            static_cast<int>(H),
            static_cast<int>(W),
            static_cast<int>(C_out),
            static_cast<int>(K),
            static_cast<int>(H_out),
            static_cast<int>(W_out),
            static_cast<int>(channel_multiplier)
        );
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", static_cast<at::Tensor(*)(at::Tensor, at::Tensor)>(&run),
          "Depthwise 2D Convolution (HIP) - no bias");
    m.def("run", static_cast<at::Tensor(*)(at::Tensor, at::Tensor, at::Tensor)>(&run),
          "Depthwise 2D Convolution (HIP) - with bias");
}