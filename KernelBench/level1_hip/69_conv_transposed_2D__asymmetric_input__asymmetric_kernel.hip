// Transposed 2D Convolution (ConvTranspose2d) for AMD GPUs (HIP, gfx942-optimized)
// Assumes defaults: stride=(1,1), padding=(0,0), dilation=(1,1), output_padding=(0,0), groups=1
// Weight layout (PyTorch ConvTranspose2d): [in_channels, out_channels, kernel_h, kernel_w]
// Input:  [N, in_channels, H_in, W_in]
// Output: [N, out_channels, H_out, W_out], where H_out = H_in + kernel_h - 1; W_out = W_in + kernel_w - 1

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

#define TILE_X 16
#define TILE_Y 16

// Gather-based implementation: each thread computes one output element (b, oc, oh, ow)
// Grid z-dimension spans batch*out_channels so a block processes a single (b, oc) pair,
// allowing us to share weights for that (ic, oc, :, :) across the whole block via LDS.
//
// We only support float tensors here for performance/simplicity.
__global__ void __launch_bounds__(TILE_X * TILE_Y)
conv_transpose2d_kernel_g1_s1_p0_d1(
    const float* __restrict__ input,     // [N, C_in, H_in, W_in]
    const float* __restrict__ weight,    // [C_in, C_out, kH, kW]
    const float* __restrict__ bias,      // [C_out] or nullptr
    float* __restrict__ output,          // [N, C_out, H_out, W_out]
    int N, int C_in,
    int H_in, int W_in,
    int C_out,
    int kH, int kW,
    int H_out, int W_out)
{
    // Tile coordinates
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // Resolve (b, oc) from z-dimension
    const int bz = blockIdx.z;
    const int oc = bz % C_out;
    const int b  = bz / C_out;

    // Output spatial indices for this thread
    const int ow = blockIdx.x * blockDim.x + tx;
    const int oh = blockIdx.y * blockDim.y + ty;

    if (b >= N || oc >= C_out || oh >= H_out || ow >= W_out) {
        return;
    }

    // Dynamic shared memory for weights of size kH*kW for current (ic, oc)
    extern __shared__ float sW[];

    // Compute valid kernel ranges for this (oh, ow) to avoid per-iteration branching:
    // ih = oh - kh in [0, H_in-1] => kh in [oh - (H_in - 1), oh]
    // kw similarly
    int kh_start = oh - (H_in - 1);
    if (kh_start < 0) kh_start = 0;
    if (kh_start > kH) kh_start = kH; // empty

    int kh_end = oh + 1;
    if (kh_end > kH) kh_end = kH;
    if (kh_end < 0) kh_end = 0; // empty

    int kw_start = ow - (W_in - 1);
    if (kw_start < 0) kw_start = 0;
    if (kw_start > kW) kw_start = kW;

    int kw_end = ow + 1;
    if (kw_end > kW) kw_end = kW;
    if (kw_end < 0) kw_end = 0;

    float acc = 0.0f;

    // Loop over input channels
    for (int ic = 0; ic < C_in; ++ic) {
        // Load weights for current (ic, oc, :, :) into shared memory
        // Weight index base for (ic, oc): ((ic * C_out + oc) * kH) * kW
        int wbase = ((ic * C_out) + oc) * kH * kW;

        // Coalesce loading of sW by first K threads
        int K = kH * kW;
        for (int i = tid; i < K; i += blockDim.x * blockDim.y) {
            sW[i] = weight[wbase + i];
        }
        __syncthreads();

        // Now compute contributions from this ic
        // Iterate only over valid kh, kw ranges computed above
        // These loops are small (e.g., 3x5), so this is compute-light
        for (int kh = kh_start; kh < kh_end; ++kh) {
            int ih = oh - kh; // valid by construction
            // Base index for input row
            int in_row_base = (((b * C_in + ic) * H_in) + ih) * W_in;

            // Weight row base in shared memory
            int w_row_base = kh * kW;

            // Unroll kw loop lightly for common small kernels
            #pragma unroll 4
            for (int kw = kw_start; kw < kw_end; ++kw) {
                int iw = ow - kw; // valid by construction
                float x = input[in_row_base + iw];
                float w = sW[w_row_base + kw];
                acc = fmaf(x, w, acc);
            }
        }
        __syncthreads(); // Ensure all threads done before reusing sW for next ic
    }

    if (bias != nullptr) {
        acc += bias[oc];
    }

    // Write result
    int out_idx = (((b * C_out) + oc) * H_out + oh) * W_out + ow;
    output[out_idx] = acc;
}

// Wrapper: input and weight only (no bias)
at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "Only float32 tensors are supported");
    TORCH_CHECK(weight.dtype() == at::kFloat, "Only float32 tensors are supported");

    auto x = input.contiguous();
    auto w = weight.contiguous();

    TORCH_CHECK(x.dim() == 4, "Input must be 4D [N, C_in, H_in, W_in]");
    TORCH_CHECK(w.dim() == 4, "Weight must be 4D [C_in, C_out, kH, kW] (ConvTranspose2d)");

    int64_t N    = x.size(0);
    int64_t C_in = x.size(1);
    int64_t H_in = x.size(2);
    int64_t W_in = x.size(3);

    int64_t W_Cin = w.size(0);
    int64_t C_out = w.size(1);
    int64_t kH    = w.size(2);
    int64_t kW    = w.size(3);

    TORCH_CHECK(W_Cin == C_in, "Weight in_channels (dim 0) must match input C_in. Got ", W_Cin, " vs ", C_in);

    // Defaults: stride=1, padding=0, dilation=1, output_padding=0, groups=1
    int64_t H_out = H_in + kH - 1;
    int64_t W_out = W_in + kW - 1;

    auto y = at::empty({N, C_out, H_out, W_out}, x.options());

    dim3 block(TILE_X, TILE_Y, 1);
    dim3 grid((W_out + TILE_X - 1) / TILE_X,
              (H_out + TILE_Y - 1) / TILE_Y,
              N * C_out);

    size_t shmem_bytes = static_cast<size_t>(kH * kW) * sizeof(float);

    hipLaunchKernelGGL(
        conv_transpose2d_kernel_g1_s1_p0_d1,
        grid, block,
        shmem_bytes, 0,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        nullptr,
        y.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C_in),
        static_cast<int>(H_in), static_cast<int>(W_in),
        static_cast<int>(C_out),
        static_cast<int>(kH), static_cast<int>(kW),
        static_cast<int>(H_out), static_cast<int>(W_out)
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

// Wrapper: input, weight, bias
at::Tensor run(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA/HIP tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA/HIP tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "Only float32 tensors are supported");
    TORCH_CHECK(weight.dtype() == at::kFloat, "Only float32 tensors are supported");
    TORCH_CHECK(bias.dtype() == at::kFloat, "Only float32 tensors are supported");

    auto x = input.contiguous();
    auto w = weight.contiguous();
    auto b = bias.contiguous();

    TORCH_CHECK(x.dim() == 4, "Input must be 4D [N, C_in, H_in, W_in]");
    TORCH_CHECK(w.dim() == 4, "Weight must be 4D [C_in, C_out, kH, kW] (ConvTranspose2d)");
    TORCH_CHECK(b.dim() == 1, "Bias must be 1D [C_out]");

    int64_t N    = x.size(0);
    int64_t C_in = x.size(1);
    int64_t H_in = x.size(2);
    int64_t W_in = x.size(3);

    int64_t W_Cin = w.size(0);
    int64_t C_out = w.size(1);
    int64_t kH    = w.size(2);
    int64_t kW    = w.size(3);

    TORCH_CHECK(W_Cin == C_in, "Weight in_channels (dim 0) must match input C_in. Got ", W_Cin, " vs ", C_in);
    TORCH_CHECK(b.size(0) == C_out, "Bias size must equal C_out. Got ", b.size(0), " vs ", C_out);

    // Defaults: stride=1, padding=0, dilation=1, output_padding=0, groups=1
    int64_t H_out = H_in + kH - 1;
    int64_t W_out = W_in + kW - 1;

    auto y = at::empty({N, C_out, H_out, W_out}, x.options());

    dim3 block(TILE_X, TILE_Y, 1);
    dim3 grid((W_out + TILE_X - 1) / TILE_X,
              (H_out + TILE_Y - 1) / TILE_Y,
              N * C_out);

    size_t shmem_bytes = static_cast<size_t>(kH * kW) * sizeof(float);

    hipLaunchKernelGGL(
        conv_transpose2d_kernel_g1_s1_p0_d1,
        grid, block,
        shmem_bytes, 0,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C_in),
        static_cast<int>(H_in), static_cast<int>(W_in),
        static_cast<int>(C_out),
        static_cast<int>(kH), static_cast<int>(kW),
        static_cast<int>(H_out), static_cast<int>(W_out)
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Overloads: with and without bias
    m.def("run", (at::Tensor(*)(at::Tensor, at::Tensor)) &run,
          "ConvTranspose2d (groups=1, stride=1, padding=0, dilation=1) without bias (HIP)");
    m.def("run", (at::Tensor(*)(at::Tensor, at::Tensor, at::Tensor)) &run,
          "ConvTranspose2d (groups=1, stride=1, padding=0, dilation=1) with bias (HIP)");
}