// Conv3D forward HIP kernel optimized for AMD MI300X (gfx942)
// - Inputs: input (N, C_in, D, H, W), weight (C_out, C_in/groups, kD, kH, kW), [optional bias (C_out)]
// - Defaults used: stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1)
// - Groups inferred from weight tensor: groups = C_in / weight.size(1)
// - Output: (N, C_out, D_out, H_out, W_out) where D_out = D - kD + 1, etc.

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#ifndef CEILDIV
#define CEILDIV(x,y) (((x) + (y) - 1) / (y))
#endif

// Block tiling (multiple of 64 for AMD wavefront alignment)
constexpr int TILE_W = 8;
constexpr int TILE_H = 8;
constexpr int TILE_D = 4;
constexpr int THREADS_PER_BLOCK = TILE_W * TILE_H * TILE_D; // 256

// Shared weight cache threshold per output channel (floats)
constexpr int MAX_SHARED_WEIGHTS = 2048; // 8KB

// Flattened index helpers (NCDHW layout)
__device__ __forceinline__ int idx_input(int n, int c, int d, int h, int w,
                                         int C, int D, int H, int W) {
    return (((n * C + c) * D + d) * H + h) * W + w;
}
__device__ __forceinline__ int idx_output(int n, int oc, int d, int h, int w,
                                          int OC, int D, int H, int W) {
    return (((n * OC + oc) * D + d) * H + h) * W + w;
}
__device__ __forceinline__ int idx_weight(int oc, int icg, int kd, int kh, int kw,
                                          int CinG, int kD, int kH, int kW) {
    return ((((oc * CinG) + icg) * kD + kd) * kH + kh) * kW + kw;
}

// Kernel: direct conv3d, optional bias, no shared weight cache
__global__ __launch_bounds__(THREADS_PER_BLOCK, 2)
void conv3d_forward_kernel(
    const float* __restrict__ input,    // [N, C, D, H, W]
    const float* __restrict__ weight,   // [OC, CinG, kD, kH, kW]
    const float* __restrict__ bias,     // [OC] or nullptr
    float* __restrict__ output,         // [N, OC, Do, Ho, Wo]
    int N, int C, int D, int H, int W,
    int OC, int CinG, int kD, int kH, int kW,
    int Do, int Ho, int Wo,
    int groups, int outCperGroup,
    int nDoBlocks,
    int has_bias)
{
    const int ow = blockIdx.x * TILE_W + threadIdx.x;
    const int oh = blockIdx.y * TILE_H + threadIdx.y;

    const int do_block = blockIdx.z % nDoBlocks;
    const int tmp = blockIdx.z / nDoBlocks;
    const int oc = tmp % OC;
    const int n = tmp / OC;

    const int od = do_block * TILE_D + threadIdx.z;

    if (n >= N || oc >= OC || od >= Do || oh >= Ho || ow >= Wo) return;

    // Group index for this output channel
    const int g = oc / outCperGroup;

    float acc = 0.0f;

    // Accumulate over input channels of this group and kernel volume
    #pragma unroll 1
    for (int icg = 0; icg < CinG; ++icg) {
        const int c = g * CinG + icg;
        // For stride=1, pad=0, dil=1: id = od + kd, etc.
        #pragma unroll 1
        for (int kd_i = 0; kd_i < kD; ++kd_i) {
            const int id = od + kd_i;
            #pragma unroll 1
            for (int kh_i = 0; kh_i < kH; ++kh_i) {
                const int ih = oh + kh_i;
                #pragma unroll 1
                for (int kw_i = 0; kw_i < kW; ++kw_i) {
                    const int iw = ow + kw_i;

                    const int in_idx = idx_input(n, c, id, ih, iw, C, D, H, W);
                    const int w_idx  = idx_weight(oc, icg, kd_i, kh_i, kw_i, CinG, kD, kH, kW);

                    acc += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    if (has_bias) {
        acc += bias[oc];
    }

    const int out_idx = idx_output(n, oc, od, oh, ow, OC, Do, Ho, Wo);
    output[out_idx] = acc;
}


// Kernel: conv3d with per-OC shared weight cache (when small enough)
__global__ __launch_bounds__(THREADS_PER_BLOCK, 2)
void conv3d_forward_kernel_sharedW(
    const float* __restrict__ input,    // [N, C, D, H, W]
    const float* __restrict__ weight,   // [OC, CinG, kD, kH, kW]
    const float* __restrict__ bias,     // [OC] or nullptr
    float* __restrict__ output,         // [N, OC, Do, Ho, Wo]
    int N, int C, int D, int H, int W,
    int OC, int CinG, int kD, int kH, int kW,
    int Do, int Ho, int Wo,
    int groups, int outCperGroup,
    int nDoBlocks,
    int has_bias,
    int weight_elems_per_oc)
{
    const int ow = blockIdx.x * TILE_W + threadIdx.x;
    const int oh = blockIdx.y * TILE_H + threadIdx.y;

    const int do_block = blockIdx.z % nDoBlocks;
    const int tmp = blockIdx.z / nDoBlocks;
    const int oc = tmp % OC;
    const int n = tmp / OC;

    const int od = do_block * TILE_D + threadIdx.z;

    if (n >= N || oc >= OC || od >= Do || oh >= Ho || ow >= Wo) return;

    // Cache weights for this oc into LDS
    extern __shared__ float s_w[]; // size = weight_elems_per_oc floats
    const int tid_linear = threadIdx.z * (TILE_W * TILE_H) + threadIdx.y * TILE_W + threadIdx.x;
    const int threads_lin = blockDim.x * blockDim.y * blockDim.z;

    // Offsets to the first element of this oc's weight slice
    const int w_oc_base = oc * CinG * kD * kH * kW;

    for (int idx = tid_linear; idx < weight_elems_per_oc; idx += threads_lin) {
        s_w[idx] = weight[w_oc_base + idx];
    }
    __syncthreads();

    // Group index for this output channel
    const int g = oc / outCperGroup;

    float acc = 0.0f;

    // Flattened index in s_w: (((icg * kD + kd) * kH + kh) * kW + kw)
    #pragma unroll 1
    for (int icg = 0; icg < CinG; ++icg) {
        const int c = g * CinG + icg;
        const int s_icg_base = icg * (kD * kH * kW);

        #pragma unroll 1
        for (int kd_i = 0; kd_i < kD; ++kd_i) {
            const int id = od + kd_i;
            const int s_kd_base = s_icg_base + kd_i * (kH * kW);
            #pragma unroll 1
            for (int kh_i = 0; kh_i < kH; ++kh_i) {
                const int ih = oh + kh_i;
                const int s_kh_base = s_kd_base + kh_i * kW;
                #pragma unroll 1
                for (int kw_i = 0; kw_i < kW; ++kw_i) {
                    const int iw = ow + kw_i;

                    const int in_idx = idx_input(n, c, id, ih, iw, C, D, H, W);
                    const float wv = s_w[s_kh_base + kw_i];

                    acc += input[in_idx] * wv;
                }
            }
        }
    }

    if (has_bias) {
        acc += bias[oc];
    }

    const int out_idx = idx_output(n, oc, od, oh, ow, OC, Do, Ho, Wo);
    output[out_idx] = acc;
}


// Host wrappers (PyTorch extension)

// No-bias variant
static at::Tensor run_nobias(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "Input must be on HIP device");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on HIP device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Only float32 is supported");

    auto x = input.contiguous();
    auto w = weight.contiguous();

    TORCH_CHECK(x.dim() == 5, "Input must be NCDHW");
    TORCH_CHECK(w.dim() == 5, "Weight must be [OC, CinG, kD, kH, kW]");

    const int N  = x.size(0);
    const int C  = x.size(1);
    const int D  = x.size(2);
    const int H  = x.size(3);
    const int W  = x.size(4);

    const int OC  = w.size(0);
    const int CinG= w.size(1);
    const int kD  = w.size(2);
    const int kH  = w.size(3);
    const int kW  = w.size(4);

    TORCH_CHECK(CinG > 0 && kD > 0 && kH > 0 && kW > 0, "Invalid weight shape");

    // Infer groups and per-group outputs
    TORCH_CHECK(C % CinG == 0, "Input channels not divisible by weight Cin/group");
    const int groups = C / CinG;
    TORCH_CHECK(OC % groups == 0, "Out channels must be divisible by groups");
    const int outCperGroup = OC / groups;

    // Defaults: stride=1, pad=0, dilation=1
    const int Do = D - kD + 1;
    const int Ho = H - kH + 1;
    const int Wo = W - kW + 1;

    TORCH_CHECK(Do >= 0 && Ho >= 0 && Wo >= 0, "Output size would be negative; adjust kernel or input");

    auto out = at::empty({N, OC, Do, Ho, Wo}, x.options());

    // Launch config
    dim3 block(TILE_W, TILE_H, TILE_D);
    const int nDoBlocks = CEILDIV(Do, TILE_D);
    dim3 grid(CEILDIV(Wo, TILE_W), CEILDIV(Ho, TILE_H), nDoBlocks * N * OC);

    const int weight_elems_per_oc = CinG * kD * kH * kW;

    if (weight_elems_per_oc <= MAX_SHARED_WEIGHTS) {
        size_t shmem_bytes = static_cast<size_t>(weight_elems_per_oc) * sizeof(float);
        hipLaunchKernelGGL(
            conv3d_forward_kernel_sharedW,
            grid, block, shmem_bytes, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            nullptr,
            out.data_ptr<float>(),
            N, C, D, H, W,
            OC, CinG, kD, kH, kW,
            Do, Ho, Wo,
            groups, outCperGroup,
            nDoBlocks,
            0, // has_bias = 0
            weight_elems_per_oc
        );
    } else {
        hipLaunchKernelGGL(
            conv3d_forward_kernel,
            grid, block, 0, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            nullptr,
            out.data_ptr<float>(),
            N, C, D, H, W,
            OC, CinG, kD, kH, kW,
            Do, Ho, Wo,
            groups, outCperGroup,
            nDoBlocks,
            0 // has_bias = 0
        );
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

// Bias variant
static at::Tensor run_bias(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be on HIP device");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on HIP device");
    TORCH_CHECK(bias.is_cuda(), "Bias must be on HIP device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "Only float32 is supported");

    auto x = input.contiguous();
    auto w = weight.contiguous();
    auto b = bias.contiguous();

    TORCH_CHECK(x.dim() == 5, "Input must be NCDHW");
    TORCH_CHECK(w.dim() == 5, "Weight must be [OC, CinG, kD, kH, kW]");
    TORCH_CHECK(b.dim() == 1, "Bias must be 1D [OC]");

    const int N  = x.size(0);
    const int C  = x.size(1);
    const int D  = x.size(2);
    const int H  = x.size(3);
    const int W  = x.size(4);

    const int OC  = w.size(0);
    const int CinG= w.size(1);
    const int kD  = w.size(2);
    const int kH  = w.size(3);
    const int kW  = w.size(4);

    TORCH_CHECK(b.size(0) == OC, "Bias size must equal out_channels");

    TORCH_CHECK(CinG > 0 && kD > 0 && kH > 0 && kW > 0, "Invalid weight shape");

    TORCH_CHECK(C % CinG == 0, "Input channels not divisible by weight Cin/group");
    const int groups = C / CinG;
    TORCH_CHECK(OC % groups == 0, "Out channels must be divisible by groups");
    const int outCperGroup = OC / groups;

    // Defaults: stride=1, pad=0, dilation=1
    const int Do = D - kD + 1;
    const int Ho = H - kH + 1;
    const int Wo = W - kW + 1;

    TORCH_CHECK(Do >= 0 && Ho >= 0 && Wo >= 0, "Output size would be negative; adjust kernel or input");

    auto out = at::empty({N, OC, Do, Ho, Wo}, x.options());

    // Launch config
    dim3 block(TILE_W, TILE_H, TILE_D);
    const int nDoBlocks = CEILDIV(Do, TILE_D);
    dim3 grid(CEILDIV(Wo, TILE_W), CEILDIV(Ho, TILE_H), nDoBlocks * N * OC);

    const int weight_elems_per_oc = CinG * kD * kH * kW;

    if (weight_elems_per_oc <= MAX_SHARED_WEIGHTS) {
        size_t shmem_bytes = static_cast<size_t>(weight_elems_per_oc) * sizeof(float);
        hipLaunchKernelGGL(
            conv3d_forward_kernel_sharedW,
            grid, block, shmem_bytes, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            b.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, D, H, W,
            OC, CinG, kD, kH, kW,
            Do, Ho, Wo,
            groups, outCperGroup,
            nDoBlocks,
            1, // has_bias = 1
            weight_elems_per_oc
        );
    } else {
        hipLaunchKernelGGL(
            conv3d_forward_kernel,
            grid, block, 0, 0,
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            b.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, D, H, W,
            OC, CinG, kD, kH, kW,
            Do, Ho, Wo,
            groups, outCperGroup,
            nDoBlocks,
            1 // has_bias = 1
        );
    }

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run",
          (at::Tensor(*)(at::Tensor, at::Tensor)) &run_nobias,
          "Conv3D forward (stride=1,pad=0,dil=1), NCDHW x OI_kDkHkW (HIP)");
    m.def("run",
          (at::Tensor(*)(at::Tensor, at::Tensor, at::Tensor)) &run_bias,
          "Conv3D forward with bias (stride=1,pad=0,dil=1), NCDHW x OI_kDkHkW (HIP)");
}