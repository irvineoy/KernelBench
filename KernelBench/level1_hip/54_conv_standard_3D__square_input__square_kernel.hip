// Conv3D (NCDHW) forward HIP kernel optimized for MI300X (gfx942)
// Implements nn.Conv3d with defaults: stride=1, padding=0, dilation=1
// Groups are inferred from weight layout [Cout, Cin/groups, kD, kH, kW]
// Supports optional bias
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>

#ifndef TILE_W
#define TILE_W 64   // threads along W (coalesced across last dimension)
#endif

#ifndef TILE_H
#define TILE_H 2    // threads along H
#endif

// Ensure blockDim.x * blockDim.y == 128 when using __launch_bounds__
#define THREADS_PER_BLOCK (TILE_W * TILE_H)

// Generic 3D convolution kernel (arbitrary kD, kH, kW, groups)
// Access pattern: coalesced across W (innermost dimension)
template <bool HasBias>
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void conv3d_fwd_generic_kernel(
    const float* __restrict__ x,    // [N, Cin, D, H, W]
    const float* __restrict__ w,    // [Cout, Cin/groups, kD, kH, kW]
    const float* __restrict__ b,    // [Cout] or nullptr
    float* __restrict__ y,          // [N, Cout, outD, outH, outW]
    int N, int Cin, int Din, int Hin, int Win,
    int Cout,
    int kD, int kH, int kW,
    int outD, int outH, int outW,
    int groups)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int ow = blockIdx.x * blockDim.x + tx;
    const int oh = blockIdx.y * blockDim.y + ty;

    int linear = blockIdx.z;                // packed as n * Cout * outD + oc * outD + od
    const int od = linear % outD; linear /= outD;
    const int oc = linear % Cout; linear /= Cout;
    const int n  = linear;

    if (n >= N || oc >= Cout || od >= outD || oh >= outH || ow >= outW) return;

    const int ocpg = Cout / groups;
    const int icpg = Cin / groups;
    const int gid  = oc / ocpg;

    float acc = HasBias ? b[oc] : 0.0f;

    // base input positions (stride=1, pad=0, dil=1)
    const int in_d0 = od;
    const int in_h0 = oh;
    const int in_w0 = ow;

    // Accumulate over group input channels and kernel window
    for (int icg = 0; icg < icpg; ++icg) {
        const int ic = gid * icpg + icg;

        // Weight base for this oc, icg
        const int64_t w_base = (((int64_t)oc * icpg) + icg) * (int64_t)(kD * kH * kW);

        for (int kd = 0; kd < kD; ++kd) {
            const int id = in_d0 + kd;
            const int64_t in_plane_base = (((((int64_t)n * Cin) + ic) * Din) + id) * (int64_t)Hin;

            for (int kh = 0; kh < kH; ++kh) {
                const int ih = in_h0 + kh;
                const int64_t in_row_base = (in_plane_base + ih) * (int64_t)Win + in_w0;
                const int64_t w_row_base  = w_base + ((int64_t)kd * kH + kh) * kW;

                #pragma unroll 4
                for (int kw = 0; kw < kW; ++kw) {
                    float xv = x[in_row_base + kw];
                    float wv = w[w_row_base + kw];
                    acc = fmaf(xv, wv, acc);
                }
            }
        }
    }

    const int64_t out_idx = (((((int64_t)n * Cout) + oc) * outD + od) * outH + oh) * outW + ow;
    y[out_idx] = acc;
}

// Specialized fast path for kD=kH=kW=3, stride=1, pad=0, dil=1
template <bool HasBias>
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void conv3d_fwd_k3_kernel(
    const float* __restrict__ x,    // [N, Cin, D, H, W]
    const float* __restrict__ w,    // [Cout, Cin/groups, 3, 3, 3]
    const float* __restrict__ b,    // [Cout] or nullptr
    float* __restrict__ y,          // [N, Cout, outD, outH, outW]
    int N, int Cin, int Din, int Hin, int Win,
    int Cout,
    int outD, int outH, int outW,
    int groups)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int ow = blockIdx.x * blockDim.x + tx;
    const int oh = blockIdx.y * blockDim.y + ty;

    int linear = blockIdx.z;                // packed as n * Cout * outD + oc * outD + od
    const int od = linear % outD; linear /= outD;
    const int oc = linear % Cout; linear /= Cout;
    const int n  = linear;

    if (n >= N || oc >= Cout || od >= outD || oh >= outH || ow >= outW) return;

    const int ocpg = Cout / groups;
    const int icpg = Cin / groups;
    const int gid  = oc / ocpg;

    float acc = HasBias ? b[oc] : 0.0f;

    const int in_d0 = od;
    const int in_h0 = oh;
    const int in_w0 = ow;

    // Fully unrolled 3x3x3
    #pragma unroll
    for (int icg = 0; icg < icpg; ++icg) {
        const int ic = gid * icpg + icg;
        const int64_t w_base = (((int64_t)oc * icpg) + icg) * 27; // 3*3*3 = 27

        #pragma unroll
        for (int kd = 0; kd < 3; ++kd) {
            const int id = in_d0 + kd;
            const int64_t in_plane_base = (((((int64_t)n * Cin) + ic) * Din) + id) * (int64_t)Hin;

            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                const int ih = in_h0 + kh;
                const int64_t in_row_base = (in_plane_base + ih) * (int64_t)Win + in_w0;
                const int64_t w_row_base  = w_base + ((int64_t)kd * 3 + kh) * 3;

                // Manually unrolled kw = 0..2
                float x0 = x[in_row_base + 0];
                float x1 = x[in_row_base + 1];
                float x2 = x[in_row_base + 2];

                float w0 = w[w_row_base + 0];
                float w1 = w[w_row_base + 1];
                float w2 = w[w_row_base + 2];

                acc = fmaf(x0, w0, acc);
                acc = fmaf(x1, w1, acc);
                acc = fmaf(x2, w2, acc);
            }
        }
    }

    const int64_t out_idx = (((((int64_t)n * Cout) + oc) * outD + od) * outH + oh) * outW + ow;
    y[out_idx] = acc;
}

// Host wrapper utilities
static inline void check_inputs(const at::Tensor& x, const at::Tensor& w, const at::Tensor* b_opt) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA/HIP tensor");
    TORCH_CHECK(w.is_cuda(), "weight must be CUDA/HIP tensor");
    if (b_opt) TORCH_CHECK(b_opt->is_cuda(), "bias must be CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "only float32 supported");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "only float32 supported");
    if (b_opt) TORCH_CHECK(b_opt->scalar_type() == at::kFloat, "only float32 bias supported");
    TORCH_CHECK(x.dim() == 5, "input must be [N, C, D, H, W]");
    TORCH_CHECK(w.dim() == 5, "weight must be [Cout, Cin/groups, kD, kH, kW]");
    if (b_opt) TORCH_CHECK(b_opt->dim() == 1 && b_opt->size(0) == w.size(0), "bias must be [Cout]");
}

template <bool HasBias>
static at::Tensor conv3d_run_impl(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    // Contiguous tensors
    auto x = input.contiguous();
    auto w = weight.contiguous();
    at::Tensor b;
    if constexpr (HasBias) b = bias.contiguous();

    const int N   = static_cast<int>(x.size(0));
    const int Cin = static_cast<int>(x.size(1));
    const int Din = static_cast<int>(x.size(2));
    const int Hin = static_cast<int>(x.size(3));
    const int Win = static_cast<int>(x.size(4));

    const int Cout = static_cast<int>(w.size(0));
    const int Cin_per_g = static_cast<int>(w.size(1));
    const int kD  = static_cast<int>(w.size(2));
    const int kH  = static_cast<int>(w.size(3));
    const int kW  = static_cast<int>(w.size(4));

    // Infer groups from weight layout
    TORCH_CHECK(Cin % Cin_per_g == 0, "Cin not divisible by Cin_per_group");
    const int groups = Cin / Cin_per_g;
    TORCH_CHECK(Cout % groups == 0, "Cout not divisible by groups");

    // Defaults: stride=1, pad=0, dil=1
    const int outD = Din - kD + 1;
    const int outH = Hin - kH + 1;
    const int outW = Win - kW + 1;
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "Invalid output dims; ensure stride=1,pad=0,dil=1 and input >= kernel");

    auto y = at::empty({N, Cout, outD, outH, outW}, x.options());

    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid((outW + block.x - 1) / block.x,
              (outH + block.y - 1) / block.y,
              static_cast<unsigned int>(N) * static_cast<unsigned int>(Cout) * static_cast<unsigned int>(outD));

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    const float* b_ptr = HasBias ? b.data_ptr<float>() : nullptr;
    float* y_ptr = y.data_ptr<float>();

    // Select specialized 3x3x3 or generic
    if (kD == 3 && kH == 3 && kW == 3) {
        hipLaunchKernelGGL(
            (conv3d_fwd_k3_kernel<HasBias>),
            grid, block, 0, 0,
            x_ptr, w_ptr, b_ptr, y_ptr,
            N, Cin, Din, Hin, Win,
            Cout,
            outD, outH, outW,
            groups
        );
    } else {
        hipLaunchKernelGGL(
            (conv3d_fwd_generic_kernel<HasBias>),
            grid, block, 0, 0,
            x_ptr, w_ptr, b_ptr, y_ptr,
            N, Cin, Din, Hin, Win,
            Cout,
            kD, kH, kW,
            outD, outH, outW,
            groups
        );
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP launch failed: ", hipGetErrorString(err));

    return y;
}

// PyTorch entry points
at::Tensor run(at::Tensor input, at::Tensor weight) {
    check_inputs(input, weight, nullptr);
    return conv3d_run_impl<false>(input, weight, at::Tensor());
}

at::Tensor run(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    check_inputs(input, weight, &bias);
    return conv3d_run_impl<true>(input, weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Overloads to support with/without bias
    m.def("run", (at::Tensor(*)(at::Tensor, at::Tensor)) &run, "Conv3D forward (HIP) without bias");
    m.def("run", (at::Tensor(*)(at::Tensor, at::Tensor, at::Tensor)) &run, "Conv3D forward (HIP) with bias");
}