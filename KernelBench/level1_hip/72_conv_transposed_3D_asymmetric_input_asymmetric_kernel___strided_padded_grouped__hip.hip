// ConvTranspose3d (deconvolution) HIP kernel for AMD MI300X (gfx942)
// Gather formulation (no atomics). Supports groups, asymmetric kernel/stride/padding/output_padding.
// Kernel signature uses only tensors (scalars are packed into a meta tensor).
// PyTorch extension entry point: run(input, weight, [bias], stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
//                                   output_padding_d, output_padding_h, output_padding_w, groups, [dilation_d, dilation_h, dilation_w])

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// meta layout (int64):
// 0: N, 1: Cin, 2: Din, 3: Hin, 4: Win
// 5: Cout, 6: Dout, 7: Hout, 8: Wout
// 9: kD, 10: kH, 11: kW
// 12: stride_d, 13: stride_h, 14: stride_w
// 15: pad_d, 16: pad_h, 17: pad_w
// 18: outpad_d, 19: outpad_h, 20: outpad_w
// 21: groups
// 22: dil_d, 23: dil_h, 24: dil_w
// 25: in_per_group
// 26: out_per_group
// 27: stride_out_W
// 28: stride_out_H
// 29: stride_out_D
// 30: stride_out_C
// 31: stride_out_N
// 32: stride_in_W
// 33: stride_in_H
// 34: stride_in_D
// 35: stride_in_C
// 36: stride_in_N
// 37: stride_w_Kw
// 38: stride_w_Kh
// 39: stride_w_Kd
// 40: stride_w_CoutPerG
// 41: stride_w_Cin

__device__ __forceinline__ bool divisible_ll(long long x, long long s) {
    // true if x is divisible by s; handles negative x correctly
    long long r = x % s;
    if (r < 0) r += s;
    return r == 0;
}

__global__ void convtrans3d_gather_kernel(
    const float* __restrict__ input,    // [N, Cin, Din, Hin, Win]
    const float* __restrict__ weight,   // [Cin, CoutPerG, kD, kH, kW]
    const float* __restrict__ bias,     // [Cout] or nullptr
    const long long* __restrict__ meta, // see layout above
    float* __restrict__ output          // [N, Cout, Dout, Hout, Wout]
) {
    // Read meta
    const long long N    = meta[0];
    const long long Cin  = meta[1];
    const long long Din  = meta[2];
    const long long Hin  = meta[3];
    const long long Win  = meta[4];
    const long long Cout = meta[5];
    const long long Dout = meta[6];
    const long long Hout = meta[7];
    const long long Wout = meta[8];

    const long long kD = meta[9];
    const long long kH = meta[10];
    const long long kW = meta[11];

    const long long sd = meta[12];
    const long long sh = meta[13];
    const long long sw = meta[14];

    const long long pd = meta[15];
    const long long ph = meta[16];
    const long long pw = meta[17];

    // outpad_* are used only for sizing; the math below uses Dout/Hout/Wout directly

    const long long groups = meta[21];

    const long long dd = meta[22];
    const long long dh = meta[23];
    const long long dw = meta[24];

    const long long in_per_g  = meta[25];
    const long long out_per_g = meta[26];

    // Output strides for contiguous NCDHW layout
    const long long sW_out = meta[27];
    const long long sH_out = meta[28];
    const long long sD_out = meta[29];
    const long long sC_out = meta[30];
    const long long sN_out = meta[31];

    // Input strides
    const long long sW_in = meta[32];
    const long long sH_in = meta[33];
    const long long sD_in = meta[34];
    const long long sC_in = meta[35];
    const long long sN_in = meta[36];

    // Weight strides (contiguous [Cin, CoutPerG, kD, kH, kW])
    const long long sKw_w     = meta[37];
    const long long sKh_w     = meta[38];
    const long long sKd_w     = meta[39];
    const long long sCout_w   = meta[40];
    const long long sCin_w    = meta[41];

    const long long total = N * Cout * Dout * Hout * Wout;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decode linear index into N, C, D, H, W (row-major with W fastest)
    long long t = idx;
    const long long ow = t % Wout; t /= Wout;
    const long long oh = t % Hout; t /= Hout;
    const long long od = t % Dout; t /= Dout;
    const long long oc = t % Cout; t /= Cout;
    const long long nb = t;

    // Determine group and offsets
    const long long g  = oc / out_per_g;       // which group
    const long long oc_in_g = oc % out_per_g;  // index inside the group's output channels
    const long long cin_beg = g * in_per_g;

    // Initialize accumulator with bias if provided
    float acc = 0.0f;
    if (bias != nullptr) {
        acc = bias[oc];
    }

    // Walk kernel positions (kd, kh, kw)
    // We gather only those positions that map exactly to integer input coordinates.
    // Mapping (for each dim): in_x = (out_x + pad_x - kx * dil_x)
    // Requires in_x % stride_x == 0 -> ix = in_x / stride_x within [0, InX)
    for (long long kd = 0; kd < kD; ++kd) {
        const long long t_d = od + pd - kd * dd;
        if (!divisible_ll(t_d, sd)) continue;
        const long long id = t_d / sd;
        if (id < 0 || id >= Din) continue;

        for (long long kh = 0; kh < kH; ++kh) {
            const long long t_h = oh + ph - kh * dh;
            if (!divisible_ll(t_h, sh)) continue;
            const long long ih = t_h / sh;
            if (ih < 0 || ih >= Hin) continue;

            for (long long kw = 0; kw < kW; ++kw) {
                const long long t_w = ow + pw - kw * dw;
                if (!divisible_ll(t_w, sw)) continue;
                const long long iw = t_w / sw;
                if (iw < 0 || iw >= Win) continue;

                // Offsets independent of ci in this inner-most loop
                const long long in_base  = nb * sN_in + id * sD_in + ih * sH_in + iw * sW_in;
                const long long w_base   = kd * sKd_w + kh * sKh_w + kw * sKw_w + oc_in_g * sCout_w;

                // Sum over input channels of this group
                #pragma unroll 2
                for (long long ci = 0; ci < in_per_g; ++ci) {
                    const long long cidx = cin_beg + ci;
                    const float val_in = input[in_base + cidx * sC_in];
                    const float w = weight[w_base + cidx * sCin_w];
                    acc += val_in * w;
                }
            }
        }
    }

    // Write the computed output
    const long long out_off = nb * sN_out + oc * sC_out + od * sD_out + oh * sH_out + ow * sW_out;
    output[out_off] = acc;
}

at::Tensor run(at::Tensor input,
               at::Tensor weight,
               c10::optional<at::Tensor> bias_opt,
               int64_t stride_d,
               int64_t stride_h,
               int64_t stride_w,
               int64_t pad_d,
               int64_t pad_h,
               int64_t pad_w,
               int64_t output_padding_d,
               int64_t output_padding_h,
               int64_t output_padding_w,
               int64_t groups,
               int64_t dilation_d = 1,
               int64_t dilation_h = 1,
               int64_t dilation_w = 1) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 supported for now");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Only float32 supported for now");
    TORCH_CHECK(input.dim() == 5, "input must be [N, Cin, Din, Hin, Win]");
    TORCH_CHECK(weight.dim() == 5, "weight must be [Cin, CoutPerG, kD, kH, kW] for ConvTranspose3d");

    const int64_t N   = input.size(0);
    const int64_t Cin = input.size(1);
    const int64_t Din = input.size(2);
    const int64_t Hin = input.size(3);
    const int64_t Win = input.size(4);

    const int64_t wCin = weight.size(0);
    const int64_t CoutPerG = weight.size(1);
    const int64_t kD = weight.size(2);
    const int64_t kH = weight.size(3);
    const int64_t kW = weight.size(4);

    TORCH_CHECK(Cin == wCin, "weight.size(0) must equal input channels for ConvTranspose3d");
    TORCH_CHECK(groups > 0, "groups must be > 0");
    TORCH_CHECK(Cin % groups == 0, "Cin must be divisible by groups");
    const int64_t in_per_group  = Cin / groups;
    const int64_t out_per_group = CoutPerG;
    const int64_t Cout = out_per_group * groups;

    at::Tensor bias;
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA/HIP tensor");
        TORCH_CHECK(bias.numel() == Cout, "bias numel must equal Cout");
        TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias.data_ptr<float>();
    }

    // Compute output sizes (PyTorch formula)
    // out = (in - 1) * stride - 2*pad + dilation*(kernel - 1) + output_padding + 1
    const int64_t Dout = (Din - 1) * stride_d - 2 * pad_d + dilation_d * (kD - 1) + output_padding_d + 1;
    const int64_t Hout = (Hin - 1) * stride_h - 2 * pad_h + dilation_h * (kH - 1) + output_padding_h + 1;
    const int64_t Wout = (Win - 1) * stride_w - 2 * pad_w + dilation_w * (kW - 1) + output_padding_w + 1;

    // Allocate output (contiguous NCDHW)
    auto output = at::empty({N, Cout, Dout, Hout, Wout}, input.options().dtype(at::kFloat));

    // Ensure tensors are contiguous for predictable strides
    auto input_c  = input.contiguous();
    auto weight_c = weight.contiguous();
    auto output_c = output.contiguous();
    const float* in_ptr  = input_c.data_ptr<float>();
    const float* w_ptr   = weight_c.data_ptr<float>();
    float* out_ptr       = output_c.data_ptr<float>();

    // Precompute and pack meta into device tensor (int64)
    at::Tensor meta = at::empty({42}, input.options().dtype(at::kLong));
    auto meta_acc = meta.data_ptr<long long>();

    // Fill meta
    meta_acc[0] = N;   meta_acc[1] = Cin; meta_acc[2] = Din; meta_acc[3] = Hin; meta_acc[4] = Win;
    meta_acc[5] = Cout; meta_acc[6] = Dout; meta_acc[7] = Hout; meta_acc[8] = Wout;
    meta_acc[9] = kD; meta_acc[10] = kH; meta_acc[11] = kW;

    meta_acc[12] = stride_d; meta_acc[13] = stride_h; meta_acc[14] = stride_w;
    meta_acc[15] = pad_d; meta_acc[16] = pad_h; meta_acc[17] = pad_w;
    meta_acc[18] = output_padding_d; meta_acc[19] = output_padding_h; meta_acc[20] = output_padding_w;

    meta_acc[21] = groups;
    meta_acc[22] = dilation_d; meta_acc[23] = dilation_h; meta_acc[24] = dilation_w;

    meta_acc[25] = in_per_group;
    meta_acc[26] = out_per_group;

    // Compute strides for output [N, C, D, H, W]
    auto os = output_c.strides();
    meta_acc[27] = os[4]; // sW_out
    meta_acc[28] = os[3]; // sH_out
    meta_acc[29] = os[2]; // sD_out
    meta_acc[30] = os[1]; // sC_out
    meta_acc[31] = os[0]; // sN_out

    auto is = input_c.strides();
    meta_acc[32] = is[4]; // sW_in
    meta_acc[33] = is[3]; // sH_in
    meta_acc[34] = is[2]; // sD_in
    meta_acc[35] = is[1]; // sC_in
    meta_acc[36] = is[0]; // sN_in

    auto ws = weight_c.strides();
    // weight [Cin, CoutPerG, kD, kH, kW]
    meta_acc[37] = ws[4]; // sKw_w
    meta_acc[38] = ws[3]; // sKh_w
    meta_acc[39] = ws[2]; // sKd_w
    meta_acc[40] = ws[1]; // sCout_w
    meta_acc[41] = ws[0]; // sCin_w

    // Initialize output with bias if present; else zero in-kernel already uses bias, but to reduce extra work we can memset if no bias
    if (!bias_ptr) {
        output_c.zero_();
    }

    // Launch config
    const int threads = 256; // multiple of 64 (wavefront)
    const long long total = N * Cout * Dout * Hout * Wout;
    const int blocks = (int)((total + threads - 1) / threads);

    hipStream_t stream = c10::hip::getCurrentHIPStream();

    hipLaunchKernelGGL(
        convtrans3d_gather_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        in_ptr, w_ptr, bias_ptr, meta_acc, out_ptr
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    err = hipStreamSynchronize(stream);
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));

    return output_c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run",
        &run,
        "ConvTranspose3d (NCDHW) with groups, asymmetric kernel/stride/padding/output_padding (HIP)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias") = c10::nullopt,
        pybind11::arg("stride_d") = 1,
        pybind11::arg("stride_h") = 1,
        pybind11::arg("stride_w") = 1,
        pybind11::arg("pad_d") = 0,
        pybind11::arg("pad_h") = 0,
        pybind11::arg("pad_w") = 0,
        pybind11::arg("output_padding_d") = 0,
        pybind11::arg("output_padding_h") = 0,
        pybind11::arg("output_padding_w") = 0,
        pybind11::arg("groups") = 1,
        pybind11::arg("dilation_d") = 1,
        pybind11::arg("dilation_h") = 1,
        pybind11::arg("dilation_w") = 1
    );
}