// ConvTranspose3d (deconvolution) HIP kernel for PyTorch (ROCm)
// Assumptions (match the provided model defaults):
//   - stride = (1,1,1)
//   - padding = (0,0,0)
//   - output_padding = (0,0,0)
//   - dilation = (1,1,1)
//   - groups = 1
// Weight layout for ConvTranspose3d in PyTorch: [Cin, Cout, kD, kH, kW]
// Input  layout: [N, Cin, D, H, W]
// Output layout: [N, Cout, D + kD - 1, H + kH - 1, W + kW - 1]
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Tunables
#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 4
#define THREADS_PER_BLOCK (BLOCK_X * BLOCK_Y * BLOCK_Z)

// Naive kernel: reads weights from global memory
__global__ void convt3d_naive_kernel(
    const float* __restrict__ input,   // [N, Cin, D, H, W]
    const float* __restrict__ weight,  // [Cin, Cout, kD, kH, kW]
    const float* __restrict__ bias,    // [Cout] or nullptr
    float* __restrict__ output,        // [N, Cout, Do, Ho, Wo]
    int N, int Cin, int D, int H, int W,
    int Cout, int kD, int kH, int kW,
    int Do, int Ho, int Wo,
    int has_bias)
{
    // Tile decomposition:
    // grid.x covers Wo in BLOCK_X
    // grid.y covers Ho in BLOCK_Y
    // grid.z packs: tile over Do (BLOCK_Z) x Cout x N
    const int tiles_d = (Do + BLOCK_Z - 1) / BLOCK_Z;

    int bz = blockIdx.z;
    int oc = (bz / tiles_d) % Cout;
    int n  = (bz / tiles_d) / Cout;
    int tile_d = bz % tiles_d;

    if (n >= N) return;

    int ox = blockIdx.x * BLOCK_X + threadIdx.x;
    int oy = blockIdx.y * BLOCK_Y + threadIdx.y;
    int oz = tile_d * BLOCK_Z + threadIdx.z;

    if (ox >= Wo || oy >= Ho || oz >= Do) return;

    // Accumulate
    float acc = (has_bias && bias != nullptr) ? bias[oc] : 0.0f;

    // For each input channel and kernel element, scatter-add contribution
    // Mapping (stride=1, pad=0, dil=1):
    //   iz = oz - kd, iy = oy - kh, ix = ox - kw
    // Valid only if iz in [0, D), iy in [0, H), ix in [0, W)
    for (int ic = 0; ic < Cin; ++ic) {
        for (int kd = 0; kd < kD; ++kd) {
            int iz = oz - kd;
            if ((unsigned)iz >= (unsigned)D) continue;
            for (int kh = 0; kh < kH; ++kh) {
                int iy = oy - kh;
                if ((unsigned)iy >= (unsigned)H) continue;
                for (int kw_ = 0; kw_ < kW; ++kw_) {
                    int ix = ox - kw_;
                    if ((unsigned)ix >= (unsigned)W) continue;

                    // input index
                    int in_idx = ((((n * Cin + ic) * D + iz) * H + iy) * W + ix);

                    // weight index: [ic, oc, kd, kh, kw_]
                    int w_idx = (((((ic * Cout) + oc) * kD + kd) * kH + kh) * kW + kw_);

                    acc += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    // write output
    int out_idx = ((((n * Cout + oc) * Do + oz) * Ho + oy) * Wo + ox);
    output[out_idx] = acc;
}

// Shared-memory optimized kernel: cache weight slice for fixed oc into LDS
// Use when Cin*kD*kH*kW fits comfortably into LDS.
__global__ void convt3d_sharedw_kernel(
    const float* __restrict__ input,   // [N, Cin, D, H, W]
    const float* __restrict__ weight,  // [Cin, Cout, kD, kH, kW]
    const float* __restrict__ bias,    // [Cout] or nullptr
    float* __restrict__ output,        // [N, Cout, Do, Ho, Wo]
    int N, int Cin, int D, int H, int W,
    int Cout, int kD, int kH, int kW,
    int Do, int Ho, int Wo,
    int has_bias,
    int slice_elems)                    // Cin*kD*kH*kW
{
    extern __shared__ float s_w[]; // size = slice_elems

    // Tile decomposition similar to naive kernel
    const int tiles_d = (Do + BLOCK_Z - 1) / BLOCK_Z;

    int bz = blockIdx.z;
    int oc = (bz / tiles_d) % Cout;
    int n  = (bz / tiles_d) / Cout;
    int tile_d = bz % tiles_d;

    if (n >= N) return;

    // Cooperative load of weight slice for this oc
    // Global base index into weight tensor for this oc slice
    // Layout: ((((ic * Cout) + oc) * kD + kd) * kH + kh) * kW + kw
    int tcount = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.x + BLOCK_X * (threadIdx.y + BLOCK_Y * threadIdx.z);

    for (int idx = tid; idx < slice_elems; idx += tcount) {
        // Decompose idx back to (ic, kd, kh, kw)
        int tmp = idx;
        int kw_ = tmp % kW; tmp /= kW;
        int kh  = tmp % kH; tmp /= kH;
        int kd  = tmp % kD; tmp /= kD;
        int ic  = tmp;      // [0..Cin)

        int w_idx = (((((ic * Cout) + oc) * kD + kd) * kH + kh) * kW + kw_);
        s_w[idx] = weight[w_idx];
    }
    __syncthreads();

    int ox = blockIdx.x * BLOCK_X + threadIdx.x;
    int oy = blockIdx.y * BLOCK_Y + threadIdx.y;
    int oz = tile_d * BLOCK_Z + threadIdx.z;

    if (ox >= Wo || oy >= Ho || oz >= Do) return;

    float acc = (has_bias && bias != nullptr) ? bias[oc] : 0.0f;

    // Consume cached weights
    for (int ic = 0; ic < Cin; ++ic) {
        for (int kd = 0; kd < kD; ++kd) {
            int iz = oz - kd;
            if ((unsigned)iz >= (unsigned)D) continue;
            for (int kh = 0; kh < kH; ++kh) {
                int iy = oy - kh;
                if ((unsigned)iy >= (unsigned)H) continue;
                for (int kw_ = 0; kw_ < kW; ++kw_) {
                    int ix = ox - kw_;
                    if ((unsigned)ix >= (unsigned)W) continue;

                    int in_idx = ((((n * Cin + ic) * D + iz) * H + iy) * W + ix);
                    int s_idx = (((ic * kD + kd) * kH + kh) * kW + kw_);

                    acc += input[in_idx] * s_w[s_idx];
                }
            }
        }
    }

    int out_idx = ((((n * Cout + oc) * Do + oz) * Ho + oy) * Wo + ox);
    output[out_idx] = acc;
}

static inline void check_inputs(const at::Tensor& input, const at::Tensor& weight, const at::Tensor* bias_opt) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 supported for input");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Only float32 supported for weight");
    TORCH_CHECK(input.dim() == 5, "Input must be [N, Cin, D, H, W]");
    TORCH_CHECK(weight.dim() == 5, "Weight must be [Cin, Cout, kD, kH, kW]");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(weight.size(0) == input.size(1),
                "Assuming groups=1: weight.size(0) must equal input Cin");
    if (bias_opt) {
        const at::Tensor& bias = *bias_opt;
        TORCH_CHECK(bias.is_cuda(), "Bias must be on GPU");
        TORCH_CHECK(bias.scalar_type() == at::kFloat, "Only float32 supported for bias");
        TORCH_CHECK(bias.dim() == 1, "Bias must be 1D [Cout]");
    }
}

// Internal implementation: bias may be nullptr
static at::Tensor run_impl(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    // Defaults as per model: stride=1, padding=0, dilation=1, output_padding=0, groups=1
    check_inputs(input, weight, bias.defined() ? &bias : nullptr);

    // Make sure tensors are contiguous (no-op if already)
    input  = input.contiguous();
    weight = weight.contiguous();
    if (bias.defined()) bias = bias.contiguous();

    int N   = static_cast<int>(input.size(0));
    int Cin = static_cast<int>(input.size(1));
    int D   = static_cast<int>(input.size(2));
    int H   = static_cast<int>(input.size(3));
    int W   = static_cast<int>(input.size(4));

    int Cin_w = static_cast<int>(weight.size(0));
    int Cout  = static_cast<int>(weight.size(1));
    int kD    = static_cast<int>(weight.size(2));
    int kH    = static_cast<int>(weight.size(3));
    int kW    = static_cast<int>(weight.size(4));
    TORCH_CHECK(Cin_w == Cin, "Weight Cin must equal input Cin for groups=1");

    // Output spatial dims for conv transpose with stride=1, pad=0, dil=1, out_pad=0
    int Do = D + kD - 1;
    int Ho = H + kH - 1;
    int Wo = W + kW - 1;

    auto options = input.options();
    at::Tensor output = at::zeros({N, Cout, Do, Ho, Wo}, options);

    // Launch configuration
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid((Wo + BLOCK_X - 1) / BLOCK_X,
              (Ho + BLOCK_Y - 1) / BLOCK_Y,
              ((Do + BLOCK_Z - 1) / BLOCK_Z) * Cout * N);

    const float* in_ptr  = input.data_ptr<float>();
    const float* w_ptr   = weight.data_ptr<float>();
    const float* b_ptr   = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* out_ptr       = output.data_ptr<float>();

    // Decide kernel path: cache per-oc weight slice in LDS if it fits
    // Slice size = Cin * kD * kH * kW floats
    int64_t slice_elems_ll = static_cast<int64_t>(Cin) * kD * kH * kW;
    // Keep a conservative cap (<= 48 KB) to allow good occupancy
    const int64_t max_shared_bytes = 48 * 1024;
    const int64_t slice_bytes = slice_elems_ll * static_cast<int64_t>(sizeof(float));

    if (slice_bytes > 0 && slice_bytes <= max_shared_bytes) {
        int slice_elems = static_cast<int>(slice_elems_ll);
        hipLaunchKernelGGL(
            convt3d_sharedw_kernel,
            grid, block,
            static_cast<size_t>(slice_bytes), 0,
            in_ptr, w_ptr, b_ptr, out_ptr,
            N, Cin, D, H, W,
            Cout, kD, kH, kW,
            Do, Ho, Wo,
            bias.defined() ? 1 : 0,
            slice_elems
        );
    } else {
        hipLaunchKernelGGL(
            convt3d_naive_kernel,
            grid, block,
            0, 0,
            in_ptr, w_ptr, b_ptr, out_ptr,
            N, Cin, D, H, W,
            Cout, kD, kH, kW,
            Do, Ho, Wo,
            bias.defined() ? 1 : 0
        );
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return output;
}

// Overload: no bias (matches provided model defaults)
at::Tensor run(at::Tensor input, at::Tensor weight) {
    at::Tensor empty_bias;
    return run_impl(std::move(input), std::move(weight), empty_bias);
}

// Overload: with bias (if a model instance happens to include bias)
at::Tensor run(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    return run_impl(std::move(input), std::move(weight), std::move(bias));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run",
          (at::Tensor (*)(at::Tensor, at::Tensor)) &run,
          "ConvTranspose3d (deconvolution) without bias (HIP)");
    m.def("run",
          (at::Tensor (*)(at::Tensor, at::Tensor, at::Tensor)) &run,
          "ConvTranspose3d (deconvolution) with bias (HIP)");
}