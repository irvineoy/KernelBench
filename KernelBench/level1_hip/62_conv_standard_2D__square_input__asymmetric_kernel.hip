// hip_conv2d_asymm.hip
// Optimized NCHW Conv2d (asymmetric kernel) for AMD MI300X (gfx942)
// Implements PyTorch nn.Conv2d forward with defaults: stride=1, padding=0, dilation=1
// Supports groups inferred from tensor shapes, and optional bias.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef TILE_W
#define TILE_W 16
#endif

#ifndef TILE_H
#define TILE_H 16
#endif

// ceil_div utility
static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

// Kernel: NCHW input, weight [C_out, C_in_per_group, KH, KW]
// Grid:
//  - x: tiles over outW
//  - y: tiles over outH
//  - z: N * C_out (each block handles one (n, oc))
// Each block computes a TILE_H x TILE_W tile of (oh, ow).
// For better weight reuse, one block cooperatively caches this (oc)'s weights into LDS.
__global__ __launch_bounds__(TILE_W * TILE_H, 2)
void conv2d_nchw_kernel(
    const float* __restrict__ input,   // [N, C_in, H, W]
    const float* __restrict__ weight,  // [C_out, C_in_per_group, KH, KW]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ output,        // [N, C_out, outH, outW]
    // Scalars
    int N, int C_in, int H, int W,
    int C_out, int G,
    int KH, int KW,
    int outH, int outW,
    int use_bias,
    int use_smem_weights // 1 to cache weights into LDS per oc
) {
    // Compute (n, oc) from z-dimension
    int z = blockIdx.z;
    int oc = z % C_out;
    int n  = z / C_out;

    // All threads must participate in __syncthreads even for invalid tiles.
    bool valid_noc = (n < N) && (oc < C_out);

    // Tile coordinates
    int ow = blockIdx.x * TILE_W + threadIdx.x;
    int oh = blockIdx.y * TILE_H + threadIdx.y;

    // Per-group dimensions
    int cin_per_group = (G > 0) ? (C_in / G) : 0;
    int out_per_group = (G > 0) ? (C_out / G) : 0;

    // Guard against degenerate cases
    if (G <= 0 || cin_per_group * G != C_in || out_per_group * G != C_out) {
        return;
    }

    int group_id = oc / out_per_group;
    int c_start = group_id * cin_per_group;

    // Prepare dynamic shared memory for weights of this oc if requested
    extern __shared__ float s_w[];
    int K = cin_per_group * KH * KW;

    // Flat thread id in block for cooperative loads
    int tpb = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    // Cooperative load of weights for this oc into LDS (if enabled)
    if (use_smem_weights && valid_noc) {
        for (int idx = tid; idx < K; idx += tpb) {
            int ci_rel = idx / (KH * KW);
            int rem    = idx % (KH * KW);
            int kh     = rem / KW;
            int kw     = rem % KW;
            // weight layout: [C_out, cin_per_group, KH, KW]
            int wIdx = (((oc * cin_per_group + ci_rel) * KH) + kh) * KW + kw;
            s_w[idx] = weight[wIdx];
        }
    }
    __syncthreads();

    // Bounds check on output indices
    if (!valid_noc || oh >= outH || ow >= outW) return;

    // Initialize accumulator with bias if present
    float sum = 0.0f;
    if (use_bias) {
        sum = bias[oc];
    }

    // Base pointers/strides
    int in_batch_offset  = n * C_in * H * W;
    int out_batch_offset = n * C_out * outH * outW;

    // Valid convolution (stride=1, pad=0, dilation=1)
    // so the input coordinate (ih, iw) = (oh + kh, ow + kw)
    // Loop ordering chosen to maximize coalescing across width (kw innermost)
    for (int ci_rel = 0; ci_rel < cin_per_group; ++ci_rel) {
        int ci = c_start + ci_rel;
        int in_ch_offset = (in_batch_offset + ci * H * W);
        int w_ci_base = (ci_rel * KH * KW);

        // Unroll KH moderately (runtime known small)
        #pragma unroll 4
        for (int kh = 0; kh < KH; ++kh) {
            int ih = oh + kh; // pad=0, stride=1
            int in_row_offset = in_ch_offset + ih * W;

            // Unroll KW moderately (runtime known small)
            #pragma unroll 4
            for (int kw = 0; kw < KW; ++kw) {
                int iw = ow + kw; // pad=0, stride=1

                float wv;
                if (use_smem_weights) {
                    wv = s_w[w_ci_base + kh * KW + kw];
                } else {
                    int wIdx = (((oc * cin_per_group + ci_rel) * KH) + kh) * KW + kw;
                    wv = weight[wIdx];
                }

                float xv = __builtin_nontemporal_load(&input[in_row_offset + iw]);
                sum = fmaf(xv, wv, sum);
            }
        }
    }

    // Store result
    int out_idx = ((out_batch_offset + oc * outH * outW) + oh * outW + ow);
    output[out_idx] = sum;
}

// Host launcher (internal)
static at::Tensor run_internal(at::Tensor input, at::Tensor weight, c10::optional<at::Tensor> bias_opt) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be NCHW (4D)");
    TORCH_CHECK(weight.dim() == 4, "weight must be [C_out, C_in_per_group, KH, KW]");

    // Ensure contiguous
    auto x = input.contiguous();
    auto w = weight.contiguous();

    // Shapes
    int64_t N  = x.size(0);
    int64_t C  = x.size(1);
    int64_t H  = x.size(2);
    int64_t W  = x.size(3);

    int64_t Co = w.size(0);
    int64_t Cpg = w.size(1);
    int64_t KH  = w.size(2);
    int64_t KW  = w.size(3);

    TORCH_CHECK(Cpg > 0 && KH > 0 && KW > 0, "Invalid weight shape");
    TORCH_CHECK(C % Cpg == 0, "Input channels not divisible by weight cin_per_group");
    int64_t G = C / Cpg;

    // Defaults: stride=1, padding=0, dilation=1
    int64_t outH = H - (KH - 1);
    int64_t outW = W - (KW - 1);
    TORCH_CHECK(outH > 0 && outW > 0, "Output spatial size must be positive (use padding if needed)");

    // Optional bias
    const float* bias_ptr = nullptr;
    bool use_bias = false;
    at::Tensor b;
    if (bias_opt.has_value()) {
        b = bias_opt.value().contiguous();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA/HIP tensor");
        TORCH_CHECK(b.scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(b.dim() == 1 && b.size(0) == Co, "bias must have shape [C_out]");
        bias_ptr = b.data_ptr<float>();
        use_bias = true;
    }

    // Allocate output
    auto y = at::empty({N, Co, outH, outW}, x.options());

    // Launch configuration
    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid(ceil_div_int((int)outW, TILE_W),
              ceil_div_int((int)outH, TILE_H),
              (unsigned int)(N * Co));

    int cin_per_group = (int)Cpg;
    int KH_i = (int)KH;
    int KW_i = (int)KW;

    // Shared memory for one oc's weights: cin_per_group * KH * KW floats
    size_t shmem_bytes = (size_t)cin_per_group * (size_t)KH_i * (size_t)KW_i * sizeof(float);
    // Use LDS only if it comfortably fits (64KB per CU budget); leave headroom for other state.
    int use_smem_weights = (shmem_bytes <= 48 * 1024) ? 1 : 0;
    if (!use_smem_weights) {
        shmem_bytes = 0;
    }

    hipLaunchKernelGGL(
        conv2d_nchw_kernel,
        grid, block, shmem_bytes, 0,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        (int)N, (int)C, (int)H, (int)W,
        (int)Co, (int)G,
        (int)KH, (int)KW,
        (int)outH, (int)outW,
        (int)use_bias,
        (int)use_smem_weights
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

// PyTorch bindings: provide overloaded "run" for optional bias
at::Tensor run_wo_bias(at::Tensor input, at::Tensor weight) {
    return run_internal(input, weight, c10::nullopt);
}

at::Tensor run_w_bias(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    return run_internal(input, weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run_wo_bias, "Conv2d NCHW (HIP) without bias");
    m.def("run", &run_w_bias, "Conv2d NCHW (HIP) with bias");
}