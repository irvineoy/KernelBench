// Depthwise Conv2D (asymmetric kernel) optimized for AMD MI300X (gfx942)
// Single-file HIP + PyTorch extension
// Build: hipcc -O3 --offload-arch=gfx9-4-generic depthwise_dw_conv2d.hip -shared -fPIC $(python3 -c "import torch, sys; print(' '.join(torch.utils.cpp_extension.include_paths()))") -o depthwise_dw_conv2d.so

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Tunable tile sizes (keep threads/block a multiple of 64)
#ifndef TILE_H
#define TILE_H 16
#endif

#ifndef TILE_W
#define TILE_W 32
#endif

// Kernel uses only tensor parameters (config packed into a small int32 tensor)
__global__ void depthwise_conv2d_tiled_kernel(
    const float* __restrict__ input,   // [N, C, H, W]
    const float* __restrict__ weight,  // [C, 1, kH, kW] (depthwise)
    float* __restrict__ output,        // [N, C, outH, outW]
    const int32_t* __restrict__ meta   // packed config
) {
    // Unpack metadata (kept as tensor to satisfy "tensor-only" signature rule)
    // meta layout:
    // 0:N, 1:C, 2:H, 3:W, 4:kH, 5:kW, 6:outH, 7:outW,
    // 8:stride_h, 9:stride_w, 10:pad_h, 11:pad_w, 12:dil_h, 13:dil_w
    int N        = meta[0];
    int C        = meta[1];
    int H        = meta[2];
    int W        = meta[3];
    int kH       = meta[4];
    int kW       = meta[5];
    int outH     = meta[6];
    int outW     = meta[7];
    int stride_h = meta[8];
    int stride_w = meta[9];
    int pad_h    = meta[10];
    int pad_w    = meta[11];
    int dil_h    = meta[12];
    int dil_w    = meta[13];

    // Block coordinates cover an output tile for a single (b,c)
    int out_w0 = blockIdx.x * TILE_W;
    int out_h0 = blockIdx.y * TILE_H;

    int bc = blockIdx.z;        // fused batch-channel index
    int b = bc / C;
    int c = bc % C;

    if (b >= N || c >= C || out_h0 >= outH || out_w0 >= outW) return;

    // Threads
    int tx = threadIdx.x; // along W
    int ty = threadIdx.y; // along H
    int block_threads = blockDim.x * blockDim.y;
    int tid = ty * blockDim.x + tx;

    // Output tile extents (clamped at boundaries)
    int tile_out_h = min(TILE_H, outH - out_h0);
    int tile_out_w = min(TILE_W, outW - out_w0);

    // For generality, compute needed input tile dimensions for this output tile
    // Effective receptive field size along H/W: (k - 1) * dil + 1
    int eff_kH = (kH - 1) * dil_h + 1;
    int eff_kW = (kW - 1) * dil_w + 1;

    // For stride s, the input coverage for a tile of output (tile_out_{h,w}) is:
    // (tile_out_{h,w} - 1) * s + eff_k{H,W}
    int tile_in_h = (tile_out_h - 1) * stride_h + eff_kH;
    int tile_in_w = (tile_out_w - 1) * stride_w + eff_kW;

    // Compute starting input coordinates for the tile (top-left of the receptive field of (out_h0, out_w0))
    int in_h0 = out_h0 * stride_h - pad_h;
    int in_w0 = out_w0 * stride_w - pad_w;

    // Dynamic shared memory layout:
    // [0 ... tile_in_h*tile_in_w - 1]               : input tile
    // [tile_in_h*tile_in_w ... tile_in_h*tile_in_w + kH*kW - 1] : weight for channel c
    extern __shared__ float smem[];
    float* s_input = smem;
    float* s_weight = smem + (tile_in_h * tile_in_w);

    // Load channel weights into shared memory
    // weight layout: [C, 1, kH, kW] contiguous
    int weights_per_channel = kH * kW;
    int w_base = c * weights_per_channel;
    for (int idx = tid; idx < weights_per_channel; idx += block_threads) {
        s_weight[idx] = weight[w_base + idx];
    }

    // Load input tile to shared memory with zero-padding at borders
    // Each thread loads strided elements
    for (int idx = tid; idx < tile_in_h * tile_in_w; idx += block_threads) {
        int ih = idx / tile_in_w;
        int iw = idx % tile_in_w;

        int gh = in_h0 + ih;  // global input h
        int gw = in_w0 + iw;  // global input w

        float val = 0.0f;
        if (gh >= 0 && gh < H && gw >= 0 && gw < W) {
            size_t input_idx = (((size_t)b * C + (size_t)c) * H + (size_t)gh) * W + (size_t)gw;
            val = input[input_idx];
        }
        s_input[ih * tile_in_w + iw] = val;
    }

    __syncthreads();

    // Compute output values for this tile
    if (tx < tile_out_w && ty < tile_out_h) {
        float acc = 0.0f;

        // For each kernel element (kH x kW) with dilation
        // Map from output tile position (ty, tx) to input tile position:
        // in_tile_h = ty * stride_h + kh * dil_h
        // in_tile_w = tx * stride_w + kw * dil_w
        #pragma unroll 1
        for (int kh = 0; kh < kH; ++kh) {
            int in_th = ty * stride_h + kh * dil_h;
            #pragma unroll 1
            for (int kw = 0; kw < kW; ++kw) {
                int in_tw = tx * stride_w + kw * dil_w;
                float a = s_input[in_th * tile_in_w + in_tw];
                float wv = s_weight[kh * kW + kw];
                acc += a * wv;
            }
        }

        // Write output
        int oh = out_h0 + ty;
        int ow = out_w0 + tx;
        size_t out_idx = (((size_t)b * C + (size_t)c) * outH + (size_t)oh) * outW + (size_t)ow;
        output[out_idx] = acc;
    }
}

// Host wrapper visible to PyTorch: only tensor parameters
// Signature: run(input, weight) where
//   input:  [N, C, H, W] float32
//   weight: [C, 1, kH, kW] float32 (depthwise conv, groups=C), bias is not used in provided model
at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D NCHW");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D [C, 1, kH, kW]");

    auto x = input.contiguous();
    auto w = weight.contiguous();

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(w.size(0) == C, "weight.size(0) must equal input channels (depthwise)");
    TORCH_CHECK(w.size(1) == 1, "weight.size(1) must be 1 for depthwise conv");

    int64_t kH = w.size(2);
    int64_t kW = w.size(3);

    // Defaults per problem statement (not passed at runtime)
    int stride_h = 1, stride_w = 1;
    int pad_h = 0, pad_w = 0;
    int dil_h = 1, dil_w = 1;

    // Output dimensions (standard conv2d formula)
    int64_t outH = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    int64_t outW = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;

    TORCH_CHECK(outH > 0 && outW > 0, "Invalid output size. Check kernel/padding/dilation.");

    auto y = at::empty({N, C, outH, outW}, x.options());

    // Pack metadata into a small int32 tensor (device) to keep kernel tensor-only
    int32_t hmeta[14];
    hmeta[0]  = static_cast<int32_t>(N);
    hmeta[1]  = static_cast<int32_t>(C);
    hmeta[2]  = static_cast<int32_t>(H);
    hmeta[3]  = static_cast<int32_t>(W);
    hmeta[4]  = static_cast<int32_t>(kH);
    hmeta[5]  = static_cast<int32_t>(kW);
    hmeta[6]  = static_cast<int32_t>(outH);
    hmeta[7]  = static_cast<int32_t>(outW);
    hmeta[8]  = static_cast<int32_t>(stride_h);
    hmeta[9]  = static_cast<int32_t>(stride_w);
    hmeta[10] = static_cast<int32_t>(pad_h);
    hmeta[11] = static_cast<int32_t>(pad_w);
    hmeta[12] = static_cast<int32_t>(dil_h);
    hmeta[13] = static_cast<int32_t>(dil_w);

    auto meta = at::empty({14}, x.options().dtype(at::kInt));
    hipError_t err = hipMemcpy(meta.data_ptr<int32_t>(), hmeta, sizeof(hmeta), hipMemcpyHostToDevice);
    TORCH_CHECK(err == hipSuccess, "hipMemcpy meta failed: ", hipGetErrorString(err));

    // Launch configuration
    dim3 block(TILE_W, TILE_H);  // 32 x 16 = 512 threads (multiple of 64)
    dim3 grid((outW + TILE_W - 1) / TILE_W,
              (outH + TILE_H - 1) / TILE_H,
              N * C);

    // Dynamic shared memory size
    // Worst-case per block: ( (TILE_H-1)*stride_h + eff_kH ) * ( (TILE_W-1)*stride_w + eff_kW )
    // But we use max tile dims (TILE_H, TILE_W) to size shared memory conservatively.
    int eff_kH = (static_cast<int>(kH) - 1) * dil_h + 1;
    int eff_kW = (static_cast<int>(kW) - 1) * dil_w + 1;
    int tile_in_h_max = (TILE_H - 1) * stride_h + eff_kH;
    int tile_in_w_max = (TILE_W - 1) * stride_w + eff_kW;
    size_t smem_floats = static_cast<size_t>(tile_in_h_max) * static_cast<size_t>(tile_in_w_max) + static_cast<size_t>(kH) * static_cast<size_t>(kW);
    size_t smem_bytes = smem_floats * sizeof(float);

    hipLaunchKernelGGL(
        depthwise_conv2d_tiled_kernel,
        grid, block, smem_bytes, 0,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        y.data_ptr<float>(),
        meta.data_ptr<int32_t>());

    err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));

    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Depthwise 2D convolution (asymmetric kernel) - HIP optimized");
}