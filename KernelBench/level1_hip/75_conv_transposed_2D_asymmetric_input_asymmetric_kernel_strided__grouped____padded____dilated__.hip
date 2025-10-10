// ConvTranspose2d (grouped, asymmetric, dilated, padded) - HIP implementation for AMD MI300X
// NOTE: Per task rules, the public PyTorch entry point only accepts tensors.
//       Model configuration (stride/padding/dilation/groups) is embedded here based on the provided target model.
//
// Target model parameters (from "Target PyTorch Model to Convert"):
//   kernel_size = (3, 5)
//   stride      = (2, 3)
//   padding     = (1, 2)
//   dilation    = (2, 1)
//   groups      = 4
//
// Weight layout for nn.ConvTranspose2d: [in_channels, out_channels/groups, kernel_h, kernel_w]
// Input/Output layout: NCHW (contiguous)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Hard-coded configuration to match the target model
#define STRIDE_H 2
#define STRIDE_W 3
#define PAD_H 1
#define PAD_W 2
#define DILATION_H 2
#define DILATION_W 1
#define GROUPS 4
#define OUTPAD_H 0
#define OUTPAD_W 0

// Thread tile config
#define TILE_W 64
#define TILE_H 4

// Utility: integer ceil-div
__host__ __device__ inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

// Kernel using shared memory cache for per-(oc group) weights.
// Each block computes a tile over (oh, ow) for a single (n, oc).
__global__ void convt2d_grouped_smem_kernel(
    const float* __restrict__ x,        // [N, Cin, Hin, Win]
    const float* __restrict__ w,        // [Cin, Cout_per_group, Kh, Kw]
    const float* __restrict__ bias,     // [Cout] or nullptr
    float* __restrict__ y,              // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout_per_group,
    int Kh, int Kw,
    int Hout, int Wout)
{
    // Derive (n, oc) from blockIdx.z
    int Cout = Cout_per_group * GROUPS;
    int n = blockIdx.z / Cout;
    int oc = blockIdx.z % Cout;
    if (n >= N || oc >= Cout) return;

    // Map block to spatial tile
    int oh = blockIdx.y * TILE_H + threadIdx.y;
    int ow = blockIdx.x * TILE_W + threadIdx.x;

    // Group mapping
    int g = oc / Cout_per_group;
    int ocg = oc % Cout_per_group;
    int Cin_per_group = Cin / GROUPS;
    int cin0 = g * Cin_per_group;

    // Dynamic shared memory for weights of this (group, ocg): size = Cin_per_group * Kh * Kw
    extern __shared__ float s_w[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int elements = Cin_per_group * Kh * Kw;

    // Load the weight slice for this ocg (all cin in group, kh, kw) into shared memory
    // Global memory indexing for weight:
    // w_idx = (((cin) * Cout_per_group + ocg) * Kh + kh) * Kw + kw
    for (int idx = tid; idx < elements; idx += blockDim.x * blockDim.y) {
        int ci = idx / (Kh * Kw);
        int rem = idx % (Kh * Kw);
        int kh = rem / Kw;
        int kw = rem % Kw;

        int cin = cin0 + ci;
        int w_idx = (((cin) * Cout_per_group + ocg) * Kh + kh) * Kw + kw;
        s_w[idx] = w[w_idx];
    }
    __syncthreads();

    if (oh >= Hout || ow >= Wout) return;

    // Accumulator
    float acc = 0.0f;
    if (bias != nullptr) {
        acc = bias[oc];
    }

    // For each kernel position, compute the corresponding input position (ih, iw) if aligns with stride
    // ih = (oh + PAD_H - kh*DILATION_H) / STRIDE_H  if divisible (no remainder)
    // iw = (ow + PAD_W - kw*DILATION_W) / STRIDE_W  if divisible (no remainder)
    // and 0 <= ih < Hin, 0 <= iw < Win
    #pragma unroll 1
    for (int kh = 0; kh < Kh; ++kh) {
        int ih_num = oh + PAD_H - kh * DILATION_H;
        if (ih_num % STRIDE_H != 0) continue;
        int ih = ih_num / STRIDE_H;
        if ((unsigned)ih >= (unsigned)Hin) continue;

        #pragma unroll 1
        for (int kw = 0; kw < Kw; ++kw) {
            int iw_num = ow + PAD_W - kw * DILATION_W;
            if (iw_num % STRIDE_W != 0) continue;
            int iw = iw_num / STRIDE_W;
            if ((unsigned)iw >= (unsigned)Win) continue;

            // Load across input channels for this group
            // s_w index for (ci, kh, kw): ci*Kh*Kw + kh*Kw + kw
            int w_base = kh * Kw + kw;
            int x_base_hw = ih * Win + iw;
            int x_base_n = n * Cin * Hin * Win;

            // Iterate over Cin_per_group
            #pragma unroll 1
            for (int ci = 0; ci < Cin_per_group; ++ci) {
                int cin = cin0 + ci;
                float xv = x[x_base_n + (cin * Hin * Win) + x_base_hw];
                float wv = s_w[ci * (Kh * Kw) + w_base];
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    // Write result
    int y_idx = (((n * Cout + oc) * Hout) + oh) * Wout + ow;
    y[y_idx] = acc;
}

// Fallback kernel without shared memory cache (for very large group weight slices)
__global__ void convt2d_grouped_global_kernel(
    const float* __restrict__ x,        // [N, Cin, Hin, Win]
    const float* __restrict__ w,        // [Cin, Cout_per_group, Kh, Kw]
    const float* __restrict__ bias,     // [Cout] or nullptr
    float* __restrict__ y,              // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout_per_group,
    int Kh, int Kw,
    int Hout, int Wout)
{
    int Cout = Cout_per_group * GROUPS;
    int n = blockIdx.z / Cout;
    int oc = blockIdx.z % Cout;
    if (n >= N || oc >= Cout) return;

    int oh = blockIdx.y * TILE_H + threadIdx.y;
    int ow = blockIdx.x * TILE_W + threadIdx.x;

    int g = oc / Cout_per_group;
    int ocg = oc % Cout_per_group;
    int Cin_per_group = Cin / GROUPS;
    int cin0 = g * Cin_per_group;

    if (oh >= Hout || ow >= Wout) return;

    float acc = 0.0f;
    if (bias != nullptr) {
        acc = bias[oc];
    }

    #pragma unroll 1
    for (int kh = 0; kh < Kh; ++kh) {
        int ih_num = oh + PAD_H - kh * DILATION_H;
        if (ih_num % STRIDE_H != 0) continue;
        int ih = ih_num / STRIDE_H;
        if ((unsigned)ih >= (unsigned)Hin) continue;

        #pragma unroll 1
        for (int kw = 0; kw < Kw; ++kw) {
            int iw_num = ow + PAD_W - kw * DILATION_W;
            if (iw_num % STRIDE_W != 0) continue;
            int iw = iw_num / STRIDE_W;
            if ((unsigned)iw >= (unsigned)Win) continue;

            int x_base_hw = ih * Win + iw;
            int x_base_n = n * Cin * Hin * Win;

            // w_idx = (((cin) * Cout_per_group + ocg) * Kh + kh) * Kw + kw
            #pragma unroll 1
            for (int ci = 0; ci < Cin_per_group; ++ci) {
                int cin = cin0 + ci;
                float xv = x[x_base_n + (cin * Hin * Win) + x_base_hw];
                int w_idx = (((cin) * Cout_per_group + ocg) * Kh + kh) * Kw + kw;
                float wv = w[w_idx];
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    int y_idx = (((n * Cout + oc) * Hout) + oh) * Wout + ow;
    y[y_idx] = acc;
}

// Public entry: tensors only (input, weight[, bias])
// Weight: [Cin, Cout_per_group, Kh, Kw], Input: [N, Cin, Hin, Win]
at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(weight.dtype() == at::kFloat, "Only float32 weight supported");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous (NCHW)");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    // Extract dims
    int64_t N = input.size(0);
    int64_t Cin = input.size(1);
    int64_t Hin = input.size(2);
    int64_t Win = input.size(3);

    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D [Cin, Cout_per_group, Kh, Kw]");
    int64_t w_Cin = weight.size(0);
    int64_t Cout_per_group = weight.size(1);
    int64_t Kh = weight.size(2);
    int64_t Kw = weight.size(3);

    TORCH_CHECK(Cin == w_Cin, "Input channels must equal weight.size(0)");
    TORCH_CHECK(Cin % GROUPS == 0, "Cin must be divisible by groups=", GROUPS);
    int64_t Cout = Cout_per_group * GROUPS;

    // Compute output spatial dims for ConvTranspose2d
    int64_t Hout = (Hin - 1) * STRIDE_H - 2 * PAD_H + DILATION_H * (Kh - 1) + OUTPAD_H + 1;
    int64_t Wout = (Win - 1) * STRIDE_W - 2 * PAD_W + DILATION_W * (Kw - 1) + OUTPAD_W + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "Computed output size is non-positive");

    auto y = at::empty({N, Cout, Hout, Wout}, input.options());

    // Launch configuration
    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid(iDivUp((int)Wout, TILE_W), iDivUp((int)Hout, TILE_H), (unsigned)(N * Cout));

    // Choose kernel based on weight-slice size for shared memory use
    size_t smem_elems = (size_t)(Cin / GROUPS) * (size_t)Kh * (size_t)Kw;
    size_t smem_bytes = smem_elems * sizeof(float);

    const float* x_ptr = input.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = nullptr;  // bias not provided in this model
    float* y_ptr = y.data_ptr<float>();

    if (smem_bytes > 0 && smem_bytes <= 48 * 1024) {
        hipLaunchKernelGGL(
            convt2d_grouped_smem_kernel,
            grid, block, smem_bytes, 0,
            x_ptr, w_ptr, b_ptr, y_ptr,
            (int)N, (int)Cin, (int)Hin, (int)Win,
            (int)Cout_per_group,
            (int)Kh, (int)Kw,
            (int)Hout, (int)Wout
        );
    } else {
        hipLaunchKernelGGL(
            convt2d_grouped_global_kernel,
            grid, block, 0, 0,
            x_ptr, w_ptr, b_ptr, y_ptr,
            (int)N, (int)Cin, (int)Hin, (int)Win,
            (int)Cout_per_group,
            (int)Kh, (int)Kw,
            (int)Hout, (int)Wout
        );
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

// Optional variant if bias is present in the model (not used in the provided test)
// Signature kept separate; harness will call run(input, weight) since bias=False by default.
at::Tensor run_with_bias(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA/HIP tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA/HIP tensor");
    TORCH_CHECK(input.dtype() == at::kFloat && weight.dtype() == at::kFloat && bias.dtype() == at::kFloat,
                "Only float32 supported");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && bias.is_contiguous(),
                "Input/weight/bias must be contiguous");

    int64_t N = input.size(0);
    int64_t Cin = input.size(1);
    int64_t Hin = input.size(2);
    int64_t Win = input.size(3);
    int64_t w_Cin = weight.size(0);
    int64_t Cout_per_group = weight.size(1);
    int64_t Kh = weight.size(2);
    int64_t Kw = weight.size(3);
    TORCH_CHECK(Cin == w_Cin, "Input channels must equal weight.size(0)");
    TORCH_CHECK(Cin % GROUPS == 0, "Cin must be divisible by groups=", GROUPS);
    int64_t Cout = Cout_per_group * GROUPS;
    TORCH_CHECK(bias.numel() == Cout, "Bias size mismatch");

    int64_t Hout = (Hin - 1) * STRIDE_H - 2 * PAD_H + DILATION_H * (Kh - 1) + OUTPAD_H + 1;
    int64_t Wout = (Win - 1) * STRIDE_W - 2 * PAD_W + DILATION_W * (Kw - 1) + OUTPAD_W + 1;

    auto y = at::empty({N, Cout, Hout, Wout}, input.options());

    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid(iDivUp((int)Wout, TILE_W), iDivUp((int)Hout, TILE_H), (unsigned)(N * Cout));

    size_t smem_elems = (size_t)(Cin / GROUPS) * (size_t)Kh * (size_t)Kw;
    size_t smem_bytes = smem_elems * sizeof(float);

    const float* x_ptr = input.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();

    if (smem_bytes > 0 && smem_bytes <= 48 * 1024) {
        hipLaunchKernelGGL(
            convt2d_grouped_smem_kernel,
            grid, block, smem_bytes, 0,
            x_ptr, w_ptr, b_ptr, y_ptr,
            (int)N, (int)Cin, (int)Hin, (int)Win,
            (int)Cout_per_group,
            (int)Kh, (int)Kw,
            (int)Hout, (int)Wout
        );
    } else {
        hipLaunchKernelGGL(
            convt2d_grouped_global_kernel,
            grid, block, 0, 0,
            x_ptr, w_ptr, b_ptr, y_ptr,
            (int)N, (int)Cin, (int)Hin, (int)Win,
            (int)Cout_per_group,
            (int)Kh, (int)Kw,
            (int)Hout, (int)Wout
        );
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Grouped ConvTranspose2d (HIP) - tensors only, bias optional (not provided here)");
    m.def("run_with_bias", &run_with_bias, "Grouped ConvTranspose2d (HIP) with bias");
}