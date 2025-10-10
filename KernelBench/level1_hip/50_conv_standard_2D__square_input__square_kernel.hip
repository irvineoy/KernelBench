// Optimized 2D Convolution (Conv2d) for Model: Conv2d(3 -> 96, kernel=11x11, stride=4, padding=2, groups=1)
// Single .hip file with HIP kernel and PyTorch extension bindings

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

// Model-specific constants inferred from the PyTorch definition
// Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, dilation=1, groups=1)
#define STRIDE_H 4
#define STRIDE_W 4
#define PAD_H 2
#define PAD_W 2
#define DILATION_H 1
#define DILATION_W 1
#define TILE_W 16   // tile width in output space
#define TILE_H 8    // tile height in output space

// Use launch bounds only if you always launch with 128 threads/block (16x8)
__global__ __launch_bounds__(TILE_W * TILE_H, 4)
void conv2d_forward_tiled_kernel(
    const float* __restrict__ input,   // [B, C_in, H, W]
    const float* __restrict__ weight,  // [C_out, C_in, KH, KW]
    const float* __restrict__ bias,    // [C_out] or nullptr if no bias
    float* __restrict__ output,        // [B, C_out, OH, OW]
    // Shape metadata
    int B, int C_in, int H, int W,
    int C_out, int KH, int KW, int OH, int OW)
{
    extern __shared__ float s_mem[];

    // Block and thread indices
    const int tx = threadIdx.x; // 0..TILE_W-1
    const int ty = threadIdx.y; // 0..TILE_H-1
    const int tid = ty * blockDim.x + tx;

    const int bx = blockIdx.x;  // tile index in width
    const int by = blockIdx.y;  // tile index in height
    const int bz = blockIdx.z;  // flatten (batch, out_channel)

    const int oc = bz % C_out;
    const int b  = bz / C_out;

    // Output tile base coordinates
    const int oh_base = by * TILE_H;
    const int ow_base = bx * TILE_W;

    // Shared memory partitioning
    const int tile_in_h = TILE_H * STRIDE_H + (KH - 1) * DILATION_H;
    const int tile_in_w = TILE_W * STRIDE_W + (KW - 1) * DILATION_W;
    const int tile_in_w_padded = tile_in_w + 1; // +1 to reduce LDS bank conflicts

    float* s_input = s_mem; // size = C_in * tile_in_h * tile_in_w_padded
    float* s_weight = s_input + (size_t)C_in * tile_in_h * tile_in_w_padded; // size = C_in * KH * KW

    // Load weights for this out-channel into shared memory
    {
        const int weight_elems = C_in * KH * KW;
        for (int idx = tid; idx < weight_elems; idx += blockDim.x * blockDim.y) {
            s_weight[idx] = weight[(size_t)oc * weight_elems + idx];
        }
    }

    // Load input tile into shared memory (with zero-padding at borders)
    const int in_h_start = oh_base * STRIDE_H - PAD_H;
    const int in_w_start = ow_base * STRIDE_W - PAD_W;

    // Ensure all threads cooperate on loading input tile in a coalesced fashion
    for (int ic = 0; ic < C_in; ++ic) {
        for (int ih = ty; ih < tile_in_h; ih += blockDim.y) {
            const int gh = in_h_start + ih * DILATION_H;
            const bool h_in = (gh >= 0 && gh < H);
            // Load across the row with stride blockDim.x to preserve coalescing
            for (int iw = tx; iw < tile_in_w; iw += blockDim.x) {
                const int gw = in_w_start + iw * DILATION_W;
                float val = 0.0f;
                if (h_in && gw >= 0 && gw < W) {
                    const size_t g_idx = (((size_t)b * C_in + ic) * H + gh) * W + gw;
                    val = input[g_idx];
                }
                const size_t s_idx = ((size_t)ic * tile_in_h + ih) * tile_in_w_padded + iw;
                s_input[s_idx] = val;
            }
        }
    }

    __syncthreads();

    // Compute output for this thread's output coordinate (oh, ow) if in range
    const int oh = oh_base + ty;
    const int ow = ow_base + tx;
    if (oh < OH && ow < OW) {
        float acc = (bias != nullptr) ? bias[oc] : 0.0f;

        // Accumulate over input channels and kernel window
        for (int ic = 0; ic < C_in; ++ic) {
            for (int kh = 0; kh < KH; ++kh) {
                const int ih = (oh - oh_base) * STRIDE_H + kh * DILATION_H;
                const size_t s_in_row = ((size_t)ic * tile_in_h + ih) * tile_in_w_padded;
                const size_t w_row = ((size_t)ic * KH + kh) * KW;

                #pragma unroll 4
                for (int kw = 0; kw < KW; ++kw) {
                    const int iw = (ow - ow_base) * STRIDE_W + kw * DILATION_W;
                    acc = fmaf(s_input[s_in_row + iw], s_weight[w_row + kw], acc);
                }
            }
        }

        const size_t out_idx = (((size_t)b * C_out + oc) * OH + oh) * OW + ow;
        output[out_idx] = acc;
    }
}

// Host wrapper: ONLY tensor parameters, no scalar config in the public API
// Signature: run(input, weight, bias)
at::Tensor run(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(input.dim() == 4, "Input must be [B, C_in, H, W]");
    TORCH_CHECK(weight.dim() == 4, "Weight must be [C_out, C_in/groups, KH, KW]");
    TORCH_CHECK(!bias.defined() || bias.scalar_type() == at::kFloat, "Bias must be float32 if provided");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "Bias must be a CUDA/HIP tensor if provided");

    // Make tensors contiguous
    auto x = input.contiguous();
    auto w = weight.contiguous();
    const float* bias_ptr = nullptr;
    if (bias.defined() && bias.numel() > 0) {
        auto bcontig = bias.contiguous();
        bias_ptr = bcontig.data_ptr<float>();
        // keep owning reference to avoid D2H frees before launch
        bias = bcontig;
    }

    const int64_t B64   = x.size(0);
    const int64_t C_in64= x.size(1);
    const int64_t H64   = x.size(2);
    const int64_t W64   = x.size(3);
    const int64_t C_out64 = w.size(0);
    const int64_t KH64  = w.size(2);
    const int64_t KW64  = w.size(3);

    TORCH_CHECK(w.size(1) == C_in64, "groups != 1 not supported in this kernel");

    // Model-specific stride/padding/dilation (hard-coded from the model definition)
    const int stride_h = STRIDE_H;
    const int stride_w = STRIDE_W;
    const int pad_h = PAD_H;
    const int pad_w = PAD_W;
    const int dil_h = DILATION_H;
    const int dil_w = DILATION_W;

    // Compute output dimensions
    const int64_t OH64 = (H64 + 2 * pad_h - dil_h * (KH64 - 1) - 1) / stride_h + 1;
    const int64_t OW64 = (W64 + 2 * pad_w - dil_w * (KW64 - 1) - 1) / stride_w + 1;

    TORCH_CHECK(OH64 > 0 && OW64 > 0, "Invalid output size");

    // Allocate output
    auto y = at::empty({B64, C_out64, OH64, OW64}, x.options());

    // Grid/block configuration
    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid(
        (unsigned int)((OW64 + TILE_W - 1) / TILE_W),
        (unsigned int)((OH64 + TILE_H - 1) / TILE_H),
        (unsigned int)(B64 * C_out64)
    );

    // Dynamic shared memory size
    const int B = static_cast<int>(B64);
    const int C_in = static_cast<int>(C_in64);
    const int H = static_cast<int>(H64);
    const int W = static_cast<int>(W64);
    const int C_out = static_cast<int>(C_out64);
    const int KH = static_cast<int>(KH64);
    const int KW = static_cast<int>(KW64);
    const int OH = static_cast<int>(OH64);
    const int OW = static_cast<int>(OW64);

    const int tile_in_h = TILE_H * STRIDE_H + (KH - 1) * DILATION_H;
    const int tile_in_w = TILE_W * STRIDE_W + (KW - 1) * DILATION_W;
    const int tile_in_w_padded = tile_in_w + 1;

    const size_t s_in_floats = (size_t)C_in * tile_in_h * tile_in_w_padded;
    const size_t s_w_floats  = (size_t)C_in * KH * KW;
    const size_t shmem_bytes = (s_in_floats + s_w_floats) * sizeof(float);

    // Launch
    hipLaunchKernelGGL(
        conv2d_forward_tiled_kernel,
        grid, block, shmem_bytes, 0,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        B, C_in, H, W, C_out, KH, KW, OH, OW
    );

    hipError_t errSync = hipDeviceSynchronize();
    TORCH_CHECK(errSync == hipSuccess, "HIP sync error: ", hipGetErrorString(errSync));
    hipError_t errAsync = hipGetLastError();
    TORCH_CHECK(errAsync == hipSuccess, "HIP launch error: ", hipGetErrorString(errAsync));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Optimized Conv2d 3->96, k=11, s=4, p=2 (HIP, tiled LDS)");
}