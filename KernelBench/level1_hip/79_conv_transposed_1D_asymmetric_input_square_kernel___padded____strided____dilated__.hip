// ConvTranspose1d (deconvolution) optimized HIP kernel for AMD MI300X (gfx942)
// Target model settings (from provided PyTorch code):
//   stride = 2, padding = 1, dilation = 2, groups = 1, bias optional
// Weight layout for ConvTranspose1d: [Cin, Cout, K]
// Input layout: [B, Cin, Lin]
// Output layout: [B, Cout, Lout], where
//   Lout = (Lin - 1) * stride - 2*padding + dilation*(K - 1) + 1
// For the given settings: Lout = (Lin - 1)*2 - 2 + 2*(K - 1) + 1

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256  // multiple of 64 for MI300X
#endif

// Specialized gather kernel for stride=2, padding=1, dilation=2
// Each thread computes one output element: (b, oc, pos)
// Accumulate over input channels and kernel taps without atomics.
template <bool HasBias>
__global__ __launch_bounds__(BLOCK_SIZE)
void convtrans1d_s2_p1_d2_kernel(
    const float* __restrict__ x,      // [B, Cin, Lin]
    const float* __restrict__ w,      // [Cin, Cout, K]
    const float* __restrict__ bias,   // [Cout] or nullptr
    float* __restrict__ y,            // [B, Cout, Lout]
    int B, int Cin, int Lin, int Cout, int K, int Lout)
{
    // Hard-coded params for this model
    const int stride = 2;
    const int padding = 1;
    const int dilation = 2;

    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int b    = blockIdx.y;
    int oc   = blockIdx.z;

    if (b >= B || oc >= Cout || pos >= Lout) return;

    // Fast parity rejection for stride=2 and even dilation=2:
    // t = pos + padding - k*dilation
    // t % 2 == (pos + padding) % 2 (since dilation is even)
    // If (pos + padding) is odd -> no valid input index for any k.
    if (((pos + padding) & (stride - 1)) != 0) {
        float outv = HasBias ? bias[oc] : 0.0f;
        y[((b * Cout + oc) * Lout) + pos] = outv;
        return;
    }

    float acc = 0.0f;

    // Index base helpers
    const int x_batch_stride = Cin * Lin;
    const int y_batch_stride = Cout * Lout;
    const int w_ic_stride    = Cout * K;

    // For this fixed config: when valid, i = (pos + padding - k*dilation) / 2
    // With stride=2, dilation=2, padding=1:
    // i = ((pos + 1) - 2*k) / 2 = ((pos + 1) >> 1) - k when numerator even.
    const int base_i = (pos + padding) >> 1;  // (pos + 1)/2

    // Loop over input channels
    for (int ic = 0; ic < Cin; ++ic) {
        int x_base = (b * Cin + ic) * Lin;
        int w_base = ic * w_ic_stride + oc * K;

        // Unroll over small kernel size
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            // Compute input index candidate
            // For stride=2/dilation=2/padding=1, parity already matched above.
            int i = base_i - k;

            if (0 <= i && i < Lin) {
                float xv = x[x_base + i];
                float wv = w[w_base + k];
                acc += xv * wv;
            }
        }
    }

    if (HasBias) acc += bias[oc];

    y[((b * Cout + oc) * Lout) + pos] = acc;
}

// Host launcher: no scalar configuration arguments are accepted.
// We infer all configuration from tensor shapes and hard-code model settings
// (stride=2, padding=1, dilation=2) per the provided model.
static at::Tensor run_internal(const at::Tensor& input,
                               const at::Tensor& weight,
                               const c10::optional<at::Tensor>& bias_opt)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input shape must be [B, Cin, Lin]");
    TORCH_CHECK(weight.dim() == 3, "weight shape must be [Cin, Cout, K] for ConvTranspose1d");

    at::Tensor x = input.contiguous();
    at::Tensor w = weight.contiguous();

    int64_t B    = x.size(0);
    int64_t Cin  = x.size(1);
    int64_t Lin  = x.size(2);
    int64_t CinW = w.size(0);
    int64_t Cout = w.size(1);
    int64_t K    = w.size(2);

    TORCH_CHECK(Cin == CinW, "weight.size(0) must match input channels");

    // Hard-coded from model (see target description)
    const int stride   = 2;
    const int padding  = 1;
    const int dilation = 2;
    const int output_padding = 0;

    // Compute output length
    int64_t Lout = (Lin - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;
    TORCH_CHECK(Lout > 0, "Computed output length must be positive");

    auto options = x.options();
    at::Tensor y = at::empty({B, Cout, Lout}, options);

    dim3 block(BLOCK_SIZE);
    dim3 grid( (static_cast<int>(Lout) + BLOCK_SIZE - 1) / BLOCK_SIZE,
               static_cast<unsigned>(B),
               static_cast<unsigned>(Cout) );

    hipError_t err;

    if (bias_opt.has_value() && bias_opt.value().defined()) {
        at::Tensor b = bias_opt.value().contiguous();
        TORCH_CHECK(b.is_cuda(), "bias must be on GPU");
        TORCH_CHECK(b.scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(b.dim() == 1 && b.size(0) == Cout, "bias shape must be [Cout]");

        hipLaunchKernelGGL(
            convtrans1d_s2_p1_d2_kernel<true>,
            grid, block, 0, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            b.data_ptr<float>(),
            y.data_ptr<float>(),
            static_cast<int>(B),
            static_cast<int>(Cin),
            static_cast<int>(Lin),
            static_cast<int>(Cout),
            static_cast<int>(K),
            static_cast<int>(Lout)
        );

    } else {
        hipLaunchKernelGGL(
            convtrans1d_s2_p1_d2_kernel<false>,
            grid, block, 0, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            nullptr,
            y.data_ptr<float>(),
            static_cast<int>(B),
            static_cast<int>(Cin),
            static_cast<int>(Lin),
            static_cast<int>(Cout),
            static_cast<int>(K),
            static_cast<int>(Lout)
        );
    }

    err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

// PyTorch-visible wrappers: only tensor arguments (no scalars)
at::Tensor run(at::Tensor input, at::Tensor weight) {
    return run_internal(input, weight, c10::nullopt);
}

at::Tensor run_bias(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    return run_internal(input, weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "ConvTranspose1d (stride=2, padding=1, dilation=2) - HIP");
    m.def("run", &run_bias, "ConvTranspose1d with bias (stride=2, padding=1, dilation=2) - HIP");
}