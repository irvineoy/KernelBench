// Depthwise Conv2d (groups = in_channels) optimized for AMD MI300X (gfx942)
// Single-file HIP kernel + PyTorch extension bindings
// Assumptions: float32 tensors, NCHW layout, stride=1, padding=0, dilation=1

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef TILE
#define TILE 16  // 16x16 tile -> 256 threads/block (multiple of 64 wavefront)
#endif

// Depthwise conv2d with LDS tiling (shared memory)
// Kernel computes valid convolution (stride=1, padding=0, dilation=1)
__global__ void dwconv2d_lds_kernel(
    const float* __restrict__ input,   // [B, C, H, W]
    const float* __restrict__ weight,  // [C, 1, kH, kW]
    const float* __restrict__ bias,    // [C] or nullptr
    float* __restrict__ output,        // [B, C, outH, outW]
    int B, int C, int H, int W,
    int kH, int kW, int outH, int outW)
{
    // Block tile origin in output space
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bcz = blockIdx.z;

    const int c = bcz % C;
    const int b = bcz / C;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int oh0 = by * TILE;
    const int ow0 = bx * TILE;

    // Per-block pointers
    const int in_bc_offset  = ((b * C + c) * H) * W;
    const int out_bc_offset = ((b * C + c) * outH) * outW;
    const int w_c_offset    = (c * kH) * kW;

    // Dynamic shared memory layout:
    // [ weights kH*kW | input tile (TILE + kH - 1) x (TILE + kW - 1 + 1 padding) ]
    extern __shared__ float smem[];
    float* s_weight = smem;
    // LDS pitch with +1 padding to avoid bank conflicts
    const int s_pitch = (TILE + kW - 1) + 1;
    float* s_input = s_weight + (kH * kW);

    // 1) Load weights for this channel into shared memory
    for (int idx = tid; idx < kH * kW; idx += blockDim.x * blockDim.y) {
        s_weight[idx] = weight[w_c_offset + idx];
    }
    __syncthreads();

    // 2) Load input tile (with halo) into shared memory.
    // Global region to load: gh in [oh0, oh0 + TILE + kH - 2], gw in [ow0, ow0 + TILE + kW - 2]
    const int sH = TILE + kH - 1;
    const int sW = TILE + kW - 1;

    const int smem_elems = sH * s_pitch; // we will fill sH rows, s_pitch cols (last column is padding)
    for (int idx = tid; idx < smem_elems; idx += blockDim.x * blockDim.y) {
        int sh = idx / s_pitch;
        int sw = idx % s_pitch;

        float val = 0.0f;
        if (sh < sH && sw < sW) {
            int gh = oh0 + sh; // stride=1, pad=0
            int gw = ow0 + sw;

            if (gh >= 0 && gh < H && gw >= 0 && gw < W) {
                val = input[in_bc_offset + gh * W + gw];
            } else {
                val = 0.0f;
            }
        }
        s_input[sh * s_pitch + sw] = val;
    }
    __syncthreads();

    // 3) Compute output for this thread if within bounds
    const int oh = oh0 + ty;
    const int ow = ow0 + tx;

    if (oh < outH && ow < outW) {
        float acc = 0.0f;

        // Convolution: for ky, kx
        // s_input is arranged so that the top-left pixel of the receptive field for (oh, ow)
        // is at s_input[ty * s_pitch + tx]
        int s_base = ty * s_pitch + tx;

        // Iterate over kernel
        for (int ky = 0; ky < kH; ++ky) {
            int s_row = s_base + ky * s_pitch;
            int w_row = ky * kW;
            // Unroll kW loop modestly for small kernels (most common: 3)
            #pragma unroll 8
            for (int kx = 0; kx < kW; ++kx) {
                acc += s_input[s_row + kx] * s_weight[w_row + kx];
            }
        }

        if (bias != nullptr) {
            acc += bias[c];
        }

        output[out_bc_offset + oh * outW + ow] = acc;
    }
}

// Naive fallback kernel (no shared memory), used for very large kernels if needed.
__global__ void dwconv2d_naive_kernel(
    const float* __restrict__ input,   // [B, C, H, W]
    const float* __restrict__ weight,  // [C, 1, kH, kW]
    const float* __restrict__ bias,    // [C] or nullptr
    float* __restrict__ output,        // [B, C, outH, outW]
    int B, int C, int H, int W,
    int kH, int kW, int outH, int outW)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bcz = blockIdx.z;

    const int c = bcz % C;
    const int b = bcz / C;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int oh = by * blockDim.y + ty;
    const int ow = bx * blockDim.x + tx;

    if (oh >= outH || ow >= outW) return;

    const int in_bc_offset  = ((b * C + c) * H) * W;
    const int out_bc_offset = ((b * C + c) * outH) * outW;
    const int w_c_offset    = (c * kH) * kW;

    float acc = 0.0f;
    // Receptive field top-left in input for (oh, ow)
    const int ih0 = oh;
    const int iw0 = ow;

    for (int ky = 0; ky < kH; ++ky) {
        for (int kx = 0; kx < kW; ++kx) {
            int ih = ih0 + ky;
            int iw = iw0 + kx;
            float v = input[in_bc_offset + ih * W + iw];
            float w = weight[w_c_offset + ky * kW + kx];
            acc += v * w;
        }
    }
    if (bias != nullptr) {
        acc += bias[c];
    }
    output[out_bc_offset + oh * outW + ow] = acc;
}

// Host launcher (no bias)
at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Only float32 weight supported");
    TORCH_CHECK(input.dim() == 4, "Input must be NCHW (4D)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be [C, 1, kH, kW]");

    auto x = input.contiguous();
    auto w = weight.contiguous();

    const int B  = x.size(0);
    const int C  = x.size(1);
    const int H  = x.size(2);
    const int W  = x.size(3);

    const int OC = w.size(0);
    const int ICg= w.size(1); // should be 1 for depthwise
    const int kH = w.size(2);
    const int kW = w.size(3);

    TORCH_CHECK(OC == C, "Depthwise conv expects weight.size(0) == in_channels");
    TORCH_CHECK(ICg == 1, "Depthwise conv expects weight.size(1) == 1");
    TORCH_CHECK(kH >= 1 && kW >= 1, "Kernel size must be >= 1");
    TORCH_CHECK(H >= kH && W >= kW, "Input smaller than kernel with padding=0");

    // stride=1, padding=0, dilation=1 (per task requirements)
    const int outH = H - kH + 1;
    const int outW = W - kW + 1;

    auto y = at::empty({B, C, outH, outW}, x.options());

    dim3 block(TILE, TILE, 1);
    dim3 grid((outW + TILE - 1) / TILE,
              (outH + TILE - 1) / TILE,
              B * C);

    // Compute shared memory size for LDS kernel
    const int s_pitch = (TILE + kW - 1) + 1;
    const int sH = TILE + kH - 1;
    const size_t s_weights_bytes = static_cast<size_t>(kH) * static_cast<size_t>(kW) * sizeof(float);
    const size_t s_input_bytes   = static_cast<size_t>(s_pitch) * static_cast<size_t>(sH) * sizeof(float);
    const size_t shared_bytes = s_weights_bytes + s_input_bytes;

    // Use LDS kernel when shared memory requirement is within 64KB budget; else fallback to naive
    const size_t LDS_BUDGET = 64 * 1024; // 64KB per CU
    if (shared_bytes <= LDS_BUDGET) {
        hipLaunchKernelGGL(dwconv2d_lds_kernel,
                           grid, block, shared_bytes, 0,
                           x.data_ptr<float>(),
                           w.data_ptr<float>(),
                           nullptr,
                           y.data_ptr<float>(),
                           B, C, H, W, kH, kW, outH, outW);
    } else {
        hipLaunchKernelGGL(dwconv2d_naive_kernel,
                           grid, block, 0, 0,
                           x.data_ptr<float>(),
                           w.data_ptr<float>(),
                           nullptr,
                           y.data_ptr<float>(),
                           B, C, H, W, kH, kW, outH, outW);
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

// Host launcher (with bias)
at::Tensor run(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA/HIP tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Only float32 weight supported");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "Only float32 bias supported");
    TORCH_CHECK(input.dim() == 4, "Input must be NCHW (4D)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be [C, 1, kH, kW]");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D [C]");

    auto x = input.contiguous();
    auto w = weight.contiguous();
    auto b = bias.contiguous();

    const int B  = x.size(0);
    const int C  = x.size(1);
    const int H  = x.size(2);
    const int W  = x.size(3);

    const int OC = w.size(0);
    const int ICg= w.size(1); // 1
    const int kH = w.size(2);
    const int kW = w.size(3);

    TORCH_CHECK(OC == C, "Depthwise conv expects weight.size(0) == in_channels");
    TORCH_CHECK(ICg == 1, "Depthwise conv expects weight.size(1) == 1");
    TORCH_CHECK(b.size(0) == C, "Bias size must equal channels");
    TORCH_CHECK(H >= kH && W >= kW, "Input smaller than kernel with padding=0");

    // stride=1, padding=0, dilation=1
    const int outH = H - kH + 1;
    const int outW = W - kW + 1;

    auto y = at::empty({B, C, outH, outW}, x.options());

    dim3 block(TILE, TILE, 1);
    dim3 grid((outW + TILE - 1) / TILE,
              (outH + TILE - 1) / TILE,
              B * C);

    // Shared memory size
    const int s_pitch = (TILE + kW - 1) + 1;
    const int sH = TILE + kH - 1;
    const size_t s_weights_bytes = static_cast<size_t>(kH) * static_cast<size_t>(kW) * sizeof(float);
    const size_t s_input_bytes   = static_cast<size_t>(s_pitch) * static_cast<size_t>(sH) * sizeof(float);
    const size_t shared_bytes = s_weights_bytes + s_input_bytes;

    const size_t LDS_BUDGET = 64 * 1024;
    if (shared_bytes <= LDS_BUDGET) {
        hipLaunchKernelGGL(dwconv2d_lds_kernel,
                           grid, block, shared_bytes, 0,
                           x.data_ptr<float>(),
                           w.data_ptr<float>(),
                           b.data_ptr<float>(),
                           y.data_ptr<float>(),
                           B, C, H, W, kH, kW, outH, outW);
    } else {
        hipLaunchKernelGGL(dwconv2d_naive_kernel,
                           grid, block, 0, 0,
                           x.data_ptr<float>(),
                           w.data_ptr<float>(),
                           b.data_ptr<float>(),
                           y.data_ptr<float>(),
                           B, C, H, W, kH, kW, outH, outW);
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", static_cast<at::Tensor(*)(at::Tensor, at::Tensor)>(&run),
          "Depthwise Conv2d (stride=1, pad=0) without bias (HIP)");
    m.def("run", static_cast<at::Tensor(*)(at::Tensor, at::Tensor, at::Tensor)>(&run),
          "Depthwise Conv2d (stride=1, pad=0) with bias (HIP)");
}