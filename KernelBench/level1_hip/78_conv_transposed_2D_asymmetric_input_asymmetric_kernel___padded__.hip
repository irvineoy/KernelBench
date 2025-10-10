// ConvTranspose2d (stride=1) with asymmetric kernel, padding inferred as floor(kernel/2)
// Single-file HIP kernel + PyTorch extension for AMD MI300X (gfx942)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        TORCH_CHECK(false, "HIP error: ", hipGetErrorString(e), " at ", __FILE__, ":", __LINE__); \
    } \
} while(0)
#endif

// Tiling parameters
#define TILE_H 16
#define TILE_W 16
#define IC_CHUNK 8  // number of input channels processed per iteration

// Kernel: 2D Transposed Convolution (ConvTranspose2d) with stride=1, dilation=1, groups=1
// Padding is inferred as pad_h = kernel_h/2, pad_w = kernel_w/2 (floor), which matches common usage and the target model.
// Weight layout (PyTorch ConvTranspose2d): [in_channels, out_channels, kH, kW]
// Input layout (NCHW): [B, C_in, H, W]
// Output layout (NCHW): [B, C_out, H_out, W_out], where
// H_out = H - 2*pad_h + kH, W_out = W - 2*pad_w + kW  (since stride=1, dilation=1, output_padding=0)
__global__ void conv_transpose2d_tiled_kernel(
    const float* __restrict__ x,       // [B, C_in, H, W]
    const float* __restrict__ w,       // [C_in, C_out, kH, kW]
    float* __restrict__ y,             // [B, C_out, H_out, W_out]
    int B, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int H_out, int W_out,
    int pad_h, int pad_w)
{
    // Block identifies a tile in output space for one (b, oc)
    int tx = threadIdx.x;  // [0, TILE_W)
    int ty = threadIdx.y;  // [0, TILE_H)
    int tid = ty * blockDim.x + tx;

    int tile_ow0 = blockIdx.x * TILE_W;
    int tile_oh0 = blockIdx.y * TILE_H;

    int oc = blockIdx.z % C_out;
    int b  = blockIdx.z / C_out;

    // Global output coordinates for this thread
    int ow = tile_ow0 + tx;
    int oh = tile_oh0 + ty;

    // Precompute input tile origin corresponding to this output tile
    // For conv transpose with stride=1:
    // in_y used = oh + pad_h - ky, ky in [0..kH-1]
    // The minimal in_y across ky (for this tile) is (tile_oh0 + pad_h - (kH-1))
    // Similarly for in_x.
    const int in_y0 = tile_oh0 + pad_h - (kH - 1);
    const int in_x0 = tile_ow0 + pad_w - (kW - 1);

    // Dimensions for shared memory tiles
    const int sH = TILE_H + kH - 1;         // input tile height per IC chunk
    const int sW_no_pad = TILE_W + kW - 1;  // input tile width per IC chunk (real)
    const int sW = sW_no_pad + 1;           // +1 pad to avoid bank conflicts

    // Dynamic shared memory layout:
    // [ sIn (IC_CHUNK * sH * sW) | sWght (IC_CHUNK * kH * kW) ]
    extern __shared__ float smem[];
    float* sIn   = smem;
    float* sWght = sIn + (IC_CHUNK * sH * sW);

    float acc = 0.0f; // accumulator for this thread's output pixel

    // Iterate over input channels in chunks
    for (int ic0 = 0; ic0 < C_in; ic0 += IC_CHUNK) {
        int chunk_len = C_in - ic0;
        if (chunk_len > IC_CHUNK) chunk_len = IC_CHUNK;

        // Load weight chunk into shared memory: [chunk_len, kH, kW]
        int w_elems = chunk_len * kH * kW;
        for (int idx = tid; idx < w_elems; idx += blockDim.x * blockDim.y) {
            int ic = idx / (kH * kW);
            int rem = idx % (kH * kW);
            int ky = rem / kW;
            int kx = rem % kW;

            int ic_g = ic0 + ic; // global input channel
            // w layout: [C_in, C_out, kH, kW]
            int w_idx = (((ic_g * C_out + oc) * kH + ky) * kW + kx);
            sWght[(ic * kH + ky) * kW + kx] = w[w_idx];
        }

        // Load input tile chunk into shared memory: [chunk_len, sH, sW]
        int in_elems = chunk_len * sH * sW;
        for (int idx = tid; idx < in_elems; idx += blockDim.x * blockDim.y) {
            int ic = idx / (sH * sW);
            int rem = idx % (sH * sW);
            int sy = rem / sW;
            int sx = rem % sW;

            float val = 0.0f;
            // Only load real data for sx < sW_no_pad; last column is padding to avoid bank conflicts
            if (sx < sW_no_pad) {
                int iy = in_y0 + sy;
                int ix = in_x0 + sx;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    int ic_g = ic0 + ic;
                    int x_idx = (((b * C_in + ic_g) * H + iy) * W + ix);
                    val = x[x_idx];
                }
            }
            sIn[(ic * sH + sy) * sW + sx] = val;
        }

        __syncthreads();

        // Compute accumulation for this thread's output pixel from the shared tiles
        if (oh < H_out && ow < W_out) {
            // For this thread's output at (oh, ow), the corresponding input indices within the tile are:
            // sy = ty + (kH - 1 - ky), sx = tx + (kW - 1 - kx)
            // These indices are guaranteed to be within [0, sH) and [0, sW_no_pad) for ky, kx in range.
            #pragma unroll 4
            for (int ic = 0; ic < chunk_len; ++ic) {
                #pragma unroll 4
                for (int ky = 0; ky < kH; ++ky) {
                    int sy = ty + (kH - 1 - ky);
                    int sIn_base = (ic * sH + sy) * sW;
                    int sW_base  = (ic * kH + ky) * kW;
                    #pragma unroll 4
                    for (int kx = 0; kx < kW; ++kx) {
                        int sx = tx + (kW - 1 - kx);
                        float in_val = sIn[sIn_base + sx];     // sx < sW_no_pad by construction
                        float w_val  = sWght[sW_base + kx];
                        acc = fmaf(in_val, w_val, acc);
                    }
                }
            }
        }

        __syncthreads();
    }

    // Write result
    if (oh < H_out && ow < W_out) {
        int y_idx = (((b * C_out + oc) * H_out + oh) * W_out + ow);
        y[y_idx] = acc;
    }
}

// PyTorch entry point: Only tensors (no scalar hyper-parameters).
// Signature: run(input, weight)
// input:  [B, C_in, H, W]
// weight: [C_in, C_out, kH, kW]
// Returns: output [B, C_out, H - 2*floor(kH/2) + kH, W - 2*floor(kW/2) + kW]
// Assumptions: stride=1, dilation=1, groups=1, bias absent (matches provided model)
at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.dim() == 4, "input must be NCHW 4D");
    TORCH_CHECK(weight.dim() == 4, "weight must be [C_in, C_out, kH, kW]");

    // Ensure float32 for this kernel
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input dtype must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight dtype must be float32");

    auto x = input.contiguous();
    auto w = weight.contiguous();

    int64_t B = x.size(0);
    int64_t C_in = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    int64_t W_cin = w.size(0);
    int64_t C_out = w.size(1);
    int64_t kH = w.size(2);
    int64_t kW = w.size(3);

    TORCH_CHECK(W_cin == C_in, "weight.size(0) must equal input channels");

    // Infer padding as floor(kernel/2) (common convention; matches target model: (3,7)->(1,3))
    int pad_h = static_cast<int>(kH / 2);
    int pad_w = static_cast<int>(kW / 2);

    // stride=1, dilation=1, output_padding=0
    int H_out = static_cast<int>(H - 2 * pad_h + kH);
    int W_out = static_cast<int>(W - 2 * pad_w + kW);

    auto y = at::empty({B, C_out, H_out, W_out}, x.options());

    dim3 block(TILE_W, TILE_H, 1); // 16x16 = 256 threads (multiple of 64)
    dim3 grid((W_out + TILE_W - 1) / TILE_W,
              (H_out + TILE_H - 1) / TILE_H,
              B * C_out);

    // Compute dynamic shared memory size
    int sH = TILE_H + static_cast<int>(kH) - 1;
    int sW_no_pad = TILE_W + static_cast<int>(kW) - 1;
    int sWpad = sW_no_pad + 1;

    int sInMaxElems = IC_CHUNK * sH * sWpad;
    int sWMaxElems  = IC_CHUNK * static_cast<int>(kH) * static_cast<int>(kW);
    size_t sharedBytes = static_cast<size_t>(sInMaxElems + sWMaxElems) * sizeof(float);

    hipStream_t stream = at::hip::getCurrentHIPStream();

    hipLaunchKernelGGL(
        conv_transpose2d_tiled_kernel,
        grid, block, sharedBytes, stream,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(B), static_cast<int>(C_in), static_cast<int>(H), static_cast<int>(W),
        static_cast<int>(C_out), static_cast<int>(kH), static_cast<int>(kW),
        static_cast<int>(H_out), static_cast<int>(W_out),
        pad_h, pad_w
    );

    CHECK_HIP( hipGetLastError() );
    CHECK_HIP( hipStreamSynchronize(stream) );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "ConvTranspose2d (stride=1, padding=floor(k/2)) - HIP optimized");
}