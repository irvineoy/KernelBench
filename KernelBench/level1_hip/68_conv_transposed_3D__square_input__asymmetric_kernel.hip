// Transposed 3D Convolution (ConvTranspose3d) optimized HIP kernel for AMD MI300X
// Assumes default stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), output_padding=(0,0,0), groups=1
// Weight layout (PyTorch): [in_channels, out_channels, kD, kH, kW]
// Input layout:  [N, in_channels, D, H, W]
// Output layout: [N, out_channels, D + kD - 1, H + kH - 1, W + kW - 1]

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#ifndef TILE_X
#define TILE_X 16
#endif
#ifndef TILE_Y
#define TILE_Y 16
#endif

// Gather-style ConvTranspose3d: each thread computes one output spatial location (for a fixed n, oc)
// Uses shared memory to cache all weights for the current (oc) across all input channels if it fits.
__global__ void convtranspose3d_shared_kernel(
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr when has_bias == 0
    float* __restrict__ out,
    // Tensor shapes
    int N, int Cin, int Din, int Hin, int Win,
    int Cout, int kD, int kH, int kW,
    int Dout, int Hout, int Wout,
    // Flags
    int has_bias)
{
    extern __shared__ float s_w[]; // size = Cin * kD * kH * kW floats

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ox = blockIdx.x * blockDim.x + tx;
    const int oy = blockIdx.y * blockDim.y + ty;

    if (ox >= Wout || oy >= Hout) return;

    // grid.z is N * Cout
    const int z_pack = blockIdx.z;
    const int n  = z_pack / Cout;
    const int oc = z_pack % Cout;
    if (n >= N || oc >= Cout) return;

    // Flattened strides for input/output
    const long in_batch_stride    = (long)Cin * Din * Hin * Win;
    const long in_channel_stride  = (long)Din * Hin * Win;
    const long in_depth_stride    = (long)Hin * Win;
    const long in_row_stride      = (long)Win;

    const long out_batch_stride   = (long)Cout * Dout * Hout * Wout;
    const long out_channel_stride = (long)Dout * Hout * Wout;
    const long out_depth_stride   = (long)Hout * Wout;
    const long out_row_stride     = (long)Wout;

    const int kVol = kD * kH * kW;

    // Load weights for this (oc) into shared memory: s_w[ci * kVol + kd*kH*kW + kh*kW + kw]
    // Global weight index: ((((ci * Cout) + oc) * kD + kd) * kH + kh) * kW + kw
    const int threads_per_block = blockDim.x * blockDim.y;
    const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int idx = linear_tid; idx < Cin * kVol; idx += threads_per_block) {
        int tmp = idx;
        const int ci = tmp / kVol;
        tmp -= ci * kVol;
        const int kd = tmp / (kH * kW);
        tmp -= kd * (kH * kW);
        const int kh = tmp / kW;
        const int kw = tmp - kh * kW;

        const long w_idx = ((((long)ci * Cout + oc) * kD + kd) * kH + kh) * kW + kw;
        s_w[idx] = weight[w_idx];
    }
    __syncthreads();

    // Loop over output depth slices
    for (int oz = 0; oz < Dout; ++oz) {
        // Compute valid kernel index ranges to avoid per-iteration bounds checks
        // iz = oz - kd in [0, Din-1]  => kd in [max(0, oz-(Din-1)) .. min(kD-1, oz)]
        int kd_lo = oz - (Din - 1);
        if (kd_lo < 0) kd_lo = 0;
        int kd_hi = oz;
        if (kd_hi > kD - 1) kd_hi = kD - 1;

        // iy = oy - kh in [0, Hin-1] => kh in [max(0, oy-(Hin-1)) .. min(kH-1, oy)]
        int kh_lo = oy - (Hin - 1);
        if (kh_lo < 0) kh_lo = 0;
        int kh_hi = oy;
        if (kh_hi > kH - 1) kh_hi = kH - 1;

        // ix = ox - kw in [0, Win-1] => kw in [max(0, ox-(Win-1)) .. min(kW-1, ox)]
        int kw_lo = ox - (Win - 1);
        if (kw_lo < 0) kw_lo = 0;
        int kw_hi = ox;
        if (kw_hi > kW - 1) kw_hi = kW - 1;

        float sum = 0.0f;
        if (kd_lo <= kd_hi && kh_lo <= kh_hi && kw_lo <= kw_hi) {
            const long in_n_base = (long)n * in_batch_stride;

            // Accumulate over input channels and kernel volume
            for (int ci = 0; ci < Cin; ++ci) {
                const long in_nc_base = in_n_base + (long)ci * in_channel_stride;

                // s_w base offset for this ci
                const int s_ci_base = ci * kVol;

                for (int kd = kd_lo; kd <= kd_hi; ++kd) {
                    const int iz = oz - kd;
                    const long in_z_base = in_nc_base + (long)iz * in_depth_stride;
                    const int s_kd_base = s_ci_base + kd * (kH * kW);

                    for (int kh = kh_lo; kh <= kh_hi; ++kh) {
                        const int iy = oy - kh;
                        const long in_y_base = in_z_base + (long)iy * in_row_stride;
                        const int s_kh_base = s_kd_base + kh * kW;

                        // Iterate kw as innermost loop for coalesced input access along x
                        // ix = ox - kw
                        for (int kw = kw_lo; kw <= kw_hi; ++kw) {
                            const int ix = ox - kw;
                            const float in_val = in[in_y_base + ix];
                            const float w_val  = s_w[s_kh_base + kw];
                            sum += in_val * w_val;
                        }
                    }
                }
            }
        }

        if (has_bias) {
            sum += bias[oc];
        }

        const long out_idx = ((long)n * out_batch_stride) +
                             ((long)oc * out_channel_stride) +
                             ((long)oz * out_depth_stride) +
                             ((long)oy * out_row_stride) +
                             ox;
        out[out_idx] = sum;
    }
}

// Fallback kernel without shared memory (direct global weight loads)
__global__ void convtranspose3d_nosmem_kernel(
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr when has_bias == 0
    float* __restrict__ out,
    // Tensor shapes
    int N, int Cin, int Din, int Hin, int Win,
    int Cout, int kD, int kH, int kW,
    int Dout, int Hout, int Wout,
    // Flags
    int has_bias)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ox = blockIdx.x * blockDim.x + tx;
    const int oy = blockIdx.y * blockDim.y + ty;

    if (ox >= Wout || oy >= Hout) return;

    // grid.z is N * Cout
    const int z_pack = blockIdx.z;
    const int n  = z_pack / Cout;
    const int oc = z_pack % Cout;
    if (n >= N || oc >= Cout) return;

    // Flattened strides for input/output
    const long in_batch_stride    = (long)Cin * Din * Hin * Win;
    const long in_channel_stride  = (long)Din * Hin * Win;
    const long in_depth_stride    = (long)Hin * Win;
    const long in_row_stride      = (long)Win;

    const long out_batch_stride   = (long)Cout * Dout * Hout * Wout;
    const long out_channel_stride = (long)Dout * Hout * Wout;
    const long out_depth_stride   = (long)Hout * Wout;
    const long out_row_stride     = (long)Wout;

    // Loop over output depth slices
    for (int oz = 0; oz < Dout; ++oz) {
        int kd_lo = oz - (Din - 1);
        if (kd_lo < 0) kd_lo = 0;
        int kd_hi = oz;
        if (kd_hi > kD - 1) kd_hi = kD - 1;

        int kh_lo = oy - (Hin - 1);
        if (kh_lo < 0) kh_lo = 0;
        int kh_hi = oy;
        if (kh_hi > kH - 1) kh_hi = kH - 1;

        int kw_lo = ox - (Win - 1);
        if (kw_lo < 0) kw_lo = 0;
        int kw_hi = ox;
        if (kw_hi > kW - 1) kw_hi = kW - 1;

        float sum = 0.0f;
        if (kd_lo <= kd_hi && kh_lo <= kh_hi && kw_lo <= kw_hi) {
            const long in_n_base = (long)n * in_batch_stride;

            for (int ci = 0; ci < Cin; ++ci) {
                const long in_nc_base = in_n_base + (long)ci * in_channel_stride;

                for (int kd = kd_lo; kd <= kd_hi; ++kd) {
                    const int iz = oz - kd;
                    const long in_z_base = in_nc_base + (long)iz * in_depth_stride;

                    for (int kh = kh_lo; kh <= kh_hi; ++kh) {
                        const int iy = oy - kh;
                        const long in_y_base = in_z_base + (long)iy * in_row_stride;

                        // Iterate kw as innermost loop for coalesced input access along x
                        for (int kw = kw_lo; kw <= kw_hi; ++kw) {
                            const int ix = ox - kw;
                            const long w_idx = ((((long)ci * Cout + oc) * kD + kd) * kH + kh) * kW + kw;
                            sum += in[in_y_base + ix] * weight[w_idx];
                        }
                    }
                }
            }
        }

        if (has_bias) {
            sum += bias[oc];
        }

        const long out_idx = ((long)n * out_batch_stride) +
                             ((long)oc * out_channel_stride) +
                             ((long)oz * out_depth_stride) +
                             ((long)oy * out_row_stride) +
                             ox;
        out[out_idx] = sum;
    }
}

// Host wrapper: run(input, weight) and run(input, weight, bias)
// Only tensor parameters are accepted; configuration inferred from shapes and defaults.
static at::Tensor run_impl(at::Tensor input, at::Tensor weight, c10::optional<at::Tensor> bias_opt) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA/HIP device");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA/HIP device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Only float32 supported");

    // Ensure contiguous tensors
    input = input.contiguous();
    weight = weight.contiguous();
    at::Tensor bias;
    const bool has_bias = bias_opt.has_value();
    if (has_bias) {
        bias = bias_opt.value().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias must be on CUDA/HIP device");
        TORCH_CHECK(bias.scalar_type() == at::kFloat, "Bias must be float32");
    }

    // Shapes
    const int64_t N   = input.size(0);
    const int64_t Cin = input.size(1);
    const int64_t Din = input.size(2);
    const int64_t Hin = input.size(3);
    const int64_t Win = input.size(4);

    // Weight: [Cin, Cout, kD, kH, kW] for groups=1
    TORCH_CHECK(weight.dim() == 5, "Weight must be 5D [Cin, Cout, kD, kH, kW]");
    const int64_t wCin = weight.size(0);
    const int64_t Cout = weight.size(1);
    const int64_t kD   = weight.size(2);
    const int64_t kH   = weight.size(3);
    const int64_t kW   = weight.size(4);
    TORCH_CHECK(wCin == Cin, "Weight in_channels must match input channels (groups=1)");

    if (has_bias) {
        TORCH_CHECK(bias.numel() == Cout, "Bias size must match out_channels");
    }

    // Defaults: stride=1, padding=0, dilation=1, output_padding=0, groups=1
    const int64_t Dout = Din + kD - 1;
    const int64_t Hout = Hin + kH - 1;
    const int64_t Wout = Win + kW - 1;

    auto out = at::empty({N, Cout, Dout, Hout, Wout}, input.options());

    // Launch config
    dim3 block(TILE_X, TILE_Y, 1);
    dim3 grid((Wout + TILE_X - 1) / TILE_X,
              (Hout + TILE_Y - 1) / TILE_Y,
              N * Cout);

    // Decide whether to use shared memory kernel
    const size_t kVol = (size_t)kD * (size_t)kH * (size_t)kW;
    const size_t smem_floats = (size_t)Cin * kVol;
    const size_t smem_bytes  = smem_floats * sizeof(float);

    // Limit shared memory usage to preserve occupancy (e.g., <= 48KB)
    const size_t smem_threshold = 48 * 1024;

    hipStream_t stream = at::cuda::getCurrentHIPStream();

    if (smem_bytes > 0 && smem_bytes <= smem_threshold) {
        hipLaunchKernelGGL(
            convtranspose3d_shared_kernel,
            grid, block, smem_bytes, stream,
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            has_bias ? bias.data_ptr<float>() : nullptr,
            out.data_ptr<float>(),
            (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win,
            (int)Cout, (int)kD, (int)kH, (int)kW,
            (int)Dout, (int)Hout, (int)Wout,
            has_bias ? 1 : 0
        );
    } else {
        hipLaunchKernelGGL(
            convtranspose3d_nosmem_kernel,
            grid, block, 0, stream,
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            has_bias ? bias.data_ptr<float>() : nullptr,
            out.data_ptr<float>(),
            (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win,
            (int)Cout, (int)kD, (int)kH, (int)kW,
            (int)Dout, (int)Hout, (int)Wout,
            has_bias ? 1 : 0
        );
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

// Overloads to match parameter presence
at::Tensor run(at::Tensor input, at::Tensor weight) {
    return run_impl(std::move(input), std::move(weight), c10::nullopt);
}
at::Tensor run(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    return run_impl(std::move(input), std::move(weight), std::move(bias));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", py::overload_cast<at::Tensor, at::Tensor>(&run),
          "ConvTranspose3d (stride=1,padding=0,dilation=1,output_padding=0,groups=1) - no bias");
    m.def("run", py::overload_cast<at::Tensor, at::Tensor, at::Tensor>(&run),
          "ConvTranspose3d (stride=1,padding=0,dilation=1,output_padding=0,groups=1) - with bias");
}