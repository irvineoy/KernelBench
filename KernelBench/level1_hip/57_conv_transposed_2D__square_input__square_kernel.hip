// ConvTranspose2d (deconvolution) optimized HIP kernel for MI300X (gfx942)
// Assumptions for this model from get_init_inputs(): stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=False
// Weight layout (PyTorch): [C_in, C_out, kH, kW]
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef TILE_W
#define TILE_W 16
#endif
#ifndef TILE_H
#define TILE_H 16
#endif

// Tiled output-oriented ConvTranspose2d (no groups, stride=1, padding=0, dilation=1)
// Each block computes a TILE_H x TILE_W tile for a single (batch, out_channel) pair.
// Shared memory:
//  - s_w: cache of all weights for this (out_channel) across all input channels (C_in * kH * kW floats)
//  - s_in: input tile with halo per input channel ( (TILE_H + kH - 1) x (TILE_W + kW - 1 + 1) floats; +1 padding to avoid bank conflicts)
__global__ __launch_bounds__(TILE_W * TILE_H)
void conv_transpose2d_tiled_kernel(
    const float* __restrict__ input,   // [B, C_in, H_in, W_in]
    const float* __restrict__ weight,  // [C_in, C_out, kH, kW]
    float* __restrict__ output,        // [B, C_out, H_out, W_out]
    int B, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int H_out, int W_out)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int out_w = blockIdx.x * TILE_W + tx;
    const int out_h = blockIdx.y * TILE_H + ty;

    const int co = blockIdx.z % C_out;  // output channel
    const int b  = blockIdx.z / C_out;  // batch index
    if (b >= B || co >= C_out) return;

    // Shared memory layout
    extern __shared__ float smem[];
    const int w_elems = C_in * kH * kW;

    float* s_w = smem;                      // size: C_in * kH * kW
    float* s_in = s_w + w_elems;            // size: (TILE_H + kH - 1) * (TILE_W + kW - 1 + 1)

    const int TH = TILE_H + kH - 1;
    const int TW = TILE_W + kW - 1;
    const int TW_PADDED = TW + 1;           // +1 to mitigate LDS bank conflicts

    // Load per-(co) weights into shared memory:
    // weight layout: [ci, co, kh, kw]
    for (int idx = tid; idx < w_elems; idx += blockDim.x * blockDim.y) {
        int tmp = idx;
        int kw = tmp % kW; tmp /= kW;
        int kh = tmp % kH; tmp /= kH;
        int ci = tmp; // 0..C_in-1

        const int w_index =
            (((ci * C_out + co) * kH + kh) * kW + kw);
        s_w[idx] = weight[w_index];
    }
    __syncthreads();

    // Compute base coords for input tile with halo
    const int base_ho = blockIdx.y * TILE_H;
    const int base_wo = blockIdx.x * TILE_W;
    const int base_hi = base_ho - (kH - 1);
    const int base_wi = base_wo - (kW - 1);

    float acc = 0.0f;

    // Loop over input channels; for each, load its input tile and accumulate
    for (int ci = 0; ci < C_in; ++ci) {
        // Load input tile for this channel into shared memory
        // s_in[iy, ix] corresponds to input[h = base_hi + iy, w = base_wi + ix]
        for (int t = tid; t < TH * TW_PADDED; t += blockDim.x * blockDim.y) {
            int iy = t / TW_PADDED;
            int ix = t - iy * TW_PADDED;
            float val = 0.0f;
            if (ix < TW) {
                int gh = base_hi + iy;
                int gw = base_wi + ix;
                if (gh >= 0 && gh < H_in && gw >= 0 && gw < W_in) {
                    const int in_index =
                        (((b * C_in + ci) * H_in + gh) * W_in + gw);
                    val = input[in_index];
                }
            }
            s_in[iy * TW_PADDED + ix] = val;
        }
        __syncthreads();

        // Accumulate contribution for this input channel
        if (out_h < H_out && out_w < W_out) {
            // sum over filter window
            // s_in indices: iy = ty + (kH - 1) - kh; ix = tx + (kW - 1) - kw
            // s_w index for this ci: ((ci * kH + kh) * kW + kw)
            float sum_ci = 0.0f;
            const int base_in_iy = ty + (kH - 1);
            const int base_in_ix = tx + (kW - 1);

            const int w_base_ci = ci * (kH * kW);
            #pragma unroll 4
            for (int kh = 0; kh < kH; ++kh) {
                const int iy = base_in_iy - kh;
                const int w_row = kh * kW;
                #pragma unroll 4
                for (int kw = 0; kw < kW; ++kw) {
                    const int ix = base_in_ix - kw;
                    const float v_in = s_in[iy * TW_PADDED + ix];
                    const float v_w  = s_w[w_base_ci + w_row + kw];
                    sum_ci += v_in * v_w;
                }
            }
            acc += sum_ci;
        }
        __syncthreads();
    }

    // Write output
    if (out_h < H_out && out_w < W_out) {
        const int out_index =
            (((b * C_out + co) * H_out + out_h) * W_out + out_w);
        output[out_index] = acc;
    }
}

at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(input.dim() == 4, "input must be NCHW");
    TORCH_CHECK(weight.dim() == 4, "weight must be [C_in, C_out, kH, kW] for ConvTranspose2d");

    auto x = input.contiguous();
    auto w = weight.contiguous();

    const int B    = x.size(0);
    const int C_in = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);

    TORCH_CHECK(w.size(0) == C_in, "weight.size(0) must equal input C_in for groups=1 ConvTranspose2d");

    const int C_out = w.size(1);
    const int kH    = w.size(2);
    const int kW    = w.size(3);

    // Assumed defaults per task: stride=1, padding=0, dilation=1, output_padding=0, groups=1
    const int H_out = H_in + kH - 1;
    const int W_out = W_in + kW - 1;

    auto out = at::empty({B, C_out, H_out, W_out}, x.options());

    // Launch config
    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid((W_out + TILE_W - 1) / TILE_W,
              (H_out + TILE_H - 1) / TILE_H,
              B * C_out);

    // Dynamic shared memory size (floats):
    // s_w: C_in * kH * kW
    // s_in: (TILE_H + kH - 1) x (TILE_W + kW - 1 + 1)
    const int TH = TILE_H + kH - 1;
    const int TW = TILE_W + kW - 1;
    const int TW_PADDED = TW + 1;
    size_t shared_floats = (size_t)(C_in * kH * kW) + (size_t)(TH * TW_PADDED);
    size_t shared_bytes = shared_floats * sizeof(float);

    hipLaunchKernelGGL(
        conv_transpose2d_tiled_kernel,
        grid, block, shared_bytes, 0,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C_in, H_in, W_in, C_out, kH, kW, H_out, W_out
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP error (sync): ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "ConvTranspose2d (stride=1, pad=0, dil=1, groups=1) - HIP optimized");
}