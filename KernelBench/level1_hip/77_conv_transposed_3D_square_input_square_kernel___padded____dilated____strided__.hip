// ConvTranspose3d (deconvolution) HIP kernel for PyTorch
// - Implements nn.ConvTranspose3d with cubic kernel
// - Weight layout matches PyTorch: [in_channels, out_channels, kD, kH, kW]
// - Input: [N, in_channels, D, H, W]
// - Output: [N, out_channels, outD, outH, outW]
// Notes:
//   - Per-output-element kernel (no atomics), caches per-(oc) filter in shared memory
//   - Hard-codes stride=2, padding=1, dilation=2 to match the provided model
//   - Only tensor arguments exposed in the PyTorch extension entry point (run)
//   - Kernel itself takes only tensor pointers; scalar metadata is passed via a small int tensor

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef CHECK_HIP
#define CHECK_HIP(x) do { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        TORCH_CHECK(false, "HIP error: ", hipGetErrorString(err)); \
    } \
} while (0)
#endif

// Kernel block config
#define BLOCK_W 16
#define BLOCK_H 16

// Gather all scalar metadata via a small device tensor of ints
// meta[0] = N
// meta[1] = IC
// meta[2] = OC
// meta[3] = D
// meta[4] = H
// meta[5] = W
// meta[6] = KD
// meta[7] = KH
// meta[8] = KW
// meta[9] = outD
// meta[10] = outH
// meta[11] = outW
// meta[12] = stride   (hard-coded to 2 in wrapper)
// meta[13] = padding  (hard-coded to 1 in wrapper)
// meta[14] = dilation (hard-coded to 2 in wrapper)
// meta[15] = has_bias (0/1)
__global__ void convtranspose3d_per_output_kernel(
    const float* __restrict__ x,            // [N, IC, D, H, W]
    const float* __restrict__ w,            // [IC, OC, KD, KH, KW]
    const float* __restrict__ b,            // [OC] or nullptr
    const int*   __restrict__ meta,         // int metadata array
    float*       __restrict__ y             // [N, OC, outD, outH, outW]
) {
    // Load meta
    const int N   = meta[0];
    const int IC  = meta[1];
    const int OC  = meta[2];
    const int D   = meta[3];
    const int H   = meta[4];
    const int W   = meta[5];
    const int KD  = meta[6];
    const int KH  = meta[7];
    const int KW  = meta[8];
    const int outD = meta[9];
    const int outH = meta[10];
    const int outW = meta[11];
    const int stride = meta[12];
    const int padding = meta[13];
    const int dilation = meta[14];
    const int has_bias = meta[15];

    // Thread coordinates for output spatial tile
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;

    // Pack (n, od, oc) in grid.z
    int tmp = blockIdx.z;
    const int oc = tmp % OC;
    tmp /= OC;
    const int od = tmp % outD;
    const int n  = tmp / outD;

    if (n >= N || oc >= OC || od >= outD || oh >= outH || ow >= outW) return;

    // Shared memory for weights for this (oc)
    extern __shared__ float s_w[]; // size = IC * KD * KH * KW floats
    const int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    const int tid_in_block = threadIdx.z * (blockDim.x * blockDim.y)
                           + threadIdx.y * blockDim.x
                           + threadIdx.x;

    const int64_t w_ic_stride  = (int64_t)OC * KD * KH * KW;
    const int64_t w_oc_stride  = (int64_t)KD * KH * KW;
    const int64_t w_kd_stride  = (int64_t)KH * KW;
    const int64_t w_kh_stride  = (int64_t)KW;

    const int w_elems = IC * KD * KH * KW;

    // Load filter [IC, KD, KH, KW] for this 'oc' into shared memory
    for (int idx = tid_in_block; idx < w_elems; idx += threads_per_block) {
        int t = idx;
        const int kw = t % KW; t /= KW;
        const int kh = t % KH; t /= KH;
        const int kd = t % KD; t /= KD;
        const int ic = t;

        const int64_t w_idx = ((int64_t)ic * OC + oc) * w_oc_stride
                            + (int64_t)kd * w_kd_stride
                            + (int64_t)kh * w_kh_stride
                            + (int64_t)kw;
        s_w[idx] = w[w_idx];
    }
    __syncthreads();

    // Initialize accumulator with bias if provided
    float acc = 0.0f;
    if (has_bias) {
        acc = b[oc];
    }

    // Fast early-out when no contributions can map to this output location.
    // For the provided configuration (stride=2, dilation=2), the congruence
    // (out + pad - k*dil) % stride == 0 reduces to (out + pad) % stride == 0,
    // independent of k. If not satisfied in any dimension, there are no input
    // contributions and output equals bias only.
    const bool fast_mod_indep = (dilation % stride) == 0;
    if (fast_mod_indep) {
        if ( ((od + padding) % stride) != 0 ||
             ((oh + padding) % stride) != 0 ||
             ((ow + padding) % stride) != 0 ) {
            // Write bias-only result
            const int64_t y_od_stride = (int64_t)outH * outW;
            const int64_t y_oc_stride = (int64_t)outD * y_od_stride;
            const int64_t y_n_stride  = (int64_t)OC * y_oc_stride;
            const int64_t y_idx = (int64_t)n * y_n_stride
                                + (int64_t)oc * y_oc_stride
                                + (int64_t)od * y_od_stride
                                + (int64_t)oh * outW
                                + (int64_t)ow;
            y[y_idx] = acc;
            return;
        }
    }

    // Strides for input and output
    const int64_t x_c_stride  = (int64_t)D * H * W;
    const int64_t x_d_stride  = (int64_t)H * W;
    const int64_t x_h_stride  = (int64_t)W;
    const int64_t x_n_stride  = (int64_t)IC * x_c_stride;

    const int64_t y_oc_stride = (int64_t)outD * outH * outW;
    const int64_t y_n_stride  = (int64_t)OC * y_oc_stride;
    const int64_t y_od_stride = (int64_t)outH * outW;
    const int64_t y_oh_stride = (int64_t)outW;

    const int64_t x_base_n = (int64_t)n * x_n_stride;

    // Accumulate from valid input positions
    // Loop order favors locality in kw -> kh -> kd
    #pragma unroll 1
    for (int kd = 0; kd < KD; ++kd) {
        const int in_d_num = od + padding - kd * dilation;
        if (in_d_num < 0) continue;
        if (in_d_num % stride != 0) continue;
        const int id = in_d_num / stride;
        if (id < 0 || id >= D) continue;

        #pragma unroll 1
        for (int kh = 0; kh < KH; ++kh) {
            const int in_h_num = oh + padding - kh * dilation;
            if (in_h_num < 0) continue;
            if (in_h_num % stride != 0) continue;
            const int ih = in_h_num / stride;
            if (ih < 0 || ih >= H) continue;

            #pragma unroll 1
            for (int kw = 0; kw < KW; ++kw) {
                const int in_w_num = ow + padding - kw * dilation;
                if (in_w_num < 0) continue;
                if (in_w_num % stride != 0) continue;
                const int iw = in_w_num / stride;
                if (iw < 0 || iw >= W) continue;

                // Compute input base offset for (n, id, ih, iw)
                const int64_t x_spatial = (int64_t)id * x_d_stride + (int64_t)ih * x_h_stride + (int64_t)iw;
                const int64_t x_base_spatial = x_base_n + x_spatial;

                // s_w index base for (kd,kh,kw)
                const int w_k_base = ((kd * KH) + kh) * KW + kw;

                // Accumulate over input channels
                // Access pattern: input advances by x_c_stride per ic; s_w advances by KD*KH*KW per ic
                #pragma unroll 1
                for (int ic = 0; ic < IC; ++ic) {
                    const float xv = x[x_base_spatial + (int64_t)ic * x_c_stride];
                    const float wv = s_w[ic * (KD * KH * KW) + w_k_base];
                    acc += xv * wv;
                }
            }
        }
    }

    // Write output
    const int64_t y_idx = (int64_t)n * y_n_stride
                        + (int64_t)oc * y_oc_stride
                        + (int64_t)od * y_od_stride
                        + (int64_t)oh * y_oh_stride
                        + (int64_t)ow;
    y[y_idx] = acc;
}

// PyTorch entry points (only tensor parameters)
at::Tensor run_impl(at::Tensor input, at::Tensor weight, c10::optional<at::Tensor> bias_opt) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA/HIP device");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA/HIP device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Only float32 weight supported");
    TORCH_CHECK(input.dim() == 5, "Input must be [N, C_in, D, H, W]");
    TORCH_CHECK(weight.dim() == 5, "Weight must be [C_in, C_out, KD, KH, KW]");

    auto x = input.contiguous();
    auto w = weight.contiguous();

    const int64_t N  = x.size(0);
    const int64_t IC = x.size(1);
    const int64_t D  = x.size(2);
    const int64_t H  = x.size(3);
    const int64_t W  = x.size(4);

    const int64_t IC_w = w.size(0);
    const int64_t OC   = w.size(1);
    const int64_t KD   = w.size(2);
    const int64_t KH   = w.size(3);
    const int64_t KW   = w.size(4);

    TORCH_CHECK(IC == IC_w, "Input channels mismatch: input C=", IC, " vs weight C_in=", IC_w);

    // Hard-code stride/padding/dilation to match the model provided in the prompt
    const int stride   = 2;
    const int padding  = 1;
    const int dilation = 2;
    const int output_padding = 0;

    // Output sizes per PyTorch ConvTranspose3d formula
    const int64_t outD = (D - 1) * stride - 2 * padding + dilation * (KD - 1) + output_padding + 1;
    const int64_t outH = (H - 1) * stride - 2 * padding + dilation * (KH - 1) + output_padding + 1;
    const int64_t outW = (W - 1) * stride - 2 * padding + dilation * (KW - 1) + output_padding + 1;

    at::Tensor bias;
    int has_bias = 0;
    if (bias_opt.has_value()) {
        bias = bias_opt.value().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias must be on CUDA/HIP device");
        TORCH_CHECK(bias.scalar_type() == at::kFloat, "Bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == OC, "Bias shape must be [C_out]");
        has_bias = 1;
    }

    auto y = at::empty({N, OC, outD, outH, outW}, x.options());

    // Prepare meta tensor (int32) on device
    at::Tensor meta = at::empty({16}, x.options().dtype(at::kInt));

    // Fill on host and copy to device (use a small CPU tensor then copy)
    int hmeta[16];
    hmeta[0] = static_cast<int>(N);
    hmeta[1] = static_cast<int>(IC);
    hmeta[2] = static_cast<int>(OC);
    hmeta[3] = static_cast<int>(D);
    hmeta[4] = static_cast<int>(H);
    hmeta[5] = static_cast<int>(W);
    hmeta[6] = static_cast<int>(KD);
    hmeta[7] = static_cast<int>(KH);
    hmeta[8] = static_cast<int>(KW);
    hmeta[9] = static_cast<int>(outD);
    hmeta[10] = static_cast<int>(outH);
    hmeta[11] = static_cast<int>(outW);
    hmeta[12] = stride;
    hmeta[13] = padding;
    hmeta[14] = dilation;
    hmeta[15] = has_bias;

    // Copy meta to device
    at::Tensor meta_cpu = at::from_blob(hmeta, {16}, at::TensorOptions().dtype(at::kInt).device(at::kCPU));
    meta.copy_(meta_cpu, /*non_blocking=*/false);

    // Launch configuration
    dim3 block(BLOCK_W, BLOCK_H, 1);
    dim3 grid( (outW + BLOCK_W - 1) / BLOCK_W,
               (outH + BLOCK_H - 1) / BLOCK_H,
               (unsigned int)(N * OC * outD) );

    // Shared memory for weights per (oc): IC * KD * KH * KW floats
    size_t smem_bytes = static_cast<size_t>(IC * KD * KH * KW) * sizeof(float);

    // Launch
    hipStream_t stream = at::cuda::getCurrentHIPStream();
    hipLaunchKernelGGL(
        convtranspose3d_per_output_kernel,
        grid, block, smem_bytes, stream,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        meta.data_ptr<int>(),
        y.data_ptr<float>()
    );

    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipStreamSynchronize(stream));

    return y;
}

// Overloads for PyTorch
at::Tensor run(at::Tensor input, at::Tensor weight) {
    return run_impl(input, weight, c10::nullopt);
}

at::Tensor run_with_bias(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    return run_impl(input, weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "ConvTranspose3d (HIP, no bias)");
    m.def("run", &run_with_bias, "ConvTranspose3d (HIP, with bias)");
}