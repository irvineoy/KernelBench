// Conv3D (NCDHW) with square kernel in two spatial dims and width=1 along last dim supported,
// but implemented generically for any Kd x Kh x Kw. Optimized for MI300X with tiling and
// per-block weight caching in LDS. PyTorch extension entrypoint: run(input, weight).

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#define TILE_X 16  // ow tile
#define TILE_Y 16  // flattened (od, oh) tile

// Flattened helpers
__device__ __forceinline__ int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Kernel computes y[n, co, od, oh, ow] for a tile of (ow, {od,oh} flattened).
// Data layout is NCDHW and weights are [Cout, Cin/groups, Kd, Kh, Kw] (PyTorch default).
extern "C"
__global__ void conv3d_ncdhw_tiled_kernel(
    const float* __restrict__ x,      // [N, Cin, D, H, W]
    const float* __restrict__ w,      // [Cout, CinPg, Kd, Kh, Kw]
    float* __restrict__ y,            // [N, Cout, Od, Oh, Ow]
    // Scalar metadata
    int N, int Cin, int Din, int Hin, int Win,
    int Cout, int Kd, int Kh, int Kw,
    int Od, int Oh, int Ow,
    int groups,
    int use_smem)                     // 1 if LDS has weights cached, 0 else
{
    extern __shared__ float s_w[];    // size = CinPg * Kd * Kh * Kw floats when use_smem==1

    // Block/thread coordinates
    const int tx = threadIdx.x;       // ow within tile
    const int ty = threadIdx.y;       // flattened (od, oh) within tile
    const int tid = ty * blockDim.x + tx;

    const int ow = blockIdx.x * blockDim.x + tx;
    const int flat_y = blockIdx.y * blockDim.y + ty; // 0..(Od*Oh-1)
    if (flat_y >= (Od * Oh) || ow >= Ow) return;

    const int od = flat_y / Oh;
    const int oh = flat_y - od * Oh;

    // Map blockIdx.z to (n, co)
    const int co = blockIdx.z % Cout;
    const int n  = blockIdx.z / Cout;
    if (n >= N) return;

    const int CinPg = Cin / groups;
    const int CoutPg = Cout / groups;
    const int g = co / CoutPg;  // group id for this co

    // Tensor sizes (flattened strides)
    const int x_stride_c = Din * Hin * Win;
    const int x_stride_d = Hin * Win;
    const int x_stride_h = Win;
    const int y_stride_c = Od * Oh * Ow;
    const int y_stride_d = Oh * Ow;
    const int y_stride_h = Ow;

    // Base offsets
    const int x_base_n = n * Cin * x_stride_c;
    const int y_base_n = n * Cout * y_stride_c;
    const int y_index = y_base_n + co * y_stride_c + od * y_stride_d + oh * y_stride_h + ow;

    // Weight slice for this co is contiguous:
    // w_idx_base = co * (CinPg * Kd * Kh * Kw)
    const int w_slice_elems = CinPg * Kd * Kh * Kw;
    const int w_idx_base = co * w_slice_elems;

    // Optionally cache weights into LDS once per block
    if (use_smem) {
        for (int idx = tid; idx < w_slice_elems; idx += blockDim.x * blockDim.y) {
            s_w[idx] = w[w_idx_base + idx];
        }
    }
    __syncthreads();

    float acc = 0.0f;

    // Iterate input channels in this group
    const int cin_start = g * CinPg;
    // Unroll small loops to leverage ILP; Kw often 1 for this model
    for (int cig = 0; cig < CinPg; ++cig) {
        const int ci = cin_start + cig;

        // For default stride=1, pad=0, dil=1, valid conv: indices are within bounds
        // id = od + kd, ih = oh + kh, iw = ow + kw
        // Compute base pointer for input[n, ci, od, oh, ow] at kd=kh=kw=0
        const int x_ci_base = x_base_n + ci * x_stride_c;

        #pragma unroll 4
        for (int kd = 0; kd < Kd; ++kd) {
            const int id = od + kd;
            const int x_d_base = x_ci_base + id * x_stride_d;

            #pragma unroll 4
            for (int kh = 0; kh < Kh; ++kh) {
                const int ih = oh + kh;
                const int x_h_base = x_d_base + ih * x_stride_h;

                // Inner loop over Kw kept generic, typically Kw==1 in this model
                if (Kw == 1) {
                    const int iw = ow;
                    const float xv = x[x_h_base + iw];

                    // weight index for [cig, kd, kh, 0]
                    const int w_local_idx = (((cig * Kd + kd) * Kh + kh) * Kw + 0);
                    const float wv = use_smem ? s_w[w_local_idx] : w[w_idx_base + w_local_idx];

                    acc += xv * wv;
                } else {
                    #pragma unroll 2
                    for (int kw = 0; kw < Kw; ++kw) {
                        const int iw = ow + kw;
                        const float xv = x[x_h_base + iw];

                        const int w_local_idx = (((cig * Kd + kd) * Kh + kh) * Kw + kw);
                        const float wv = use_smem ? s_w[w_local_idx] : w[w_idx_base + w_local_idx];

                        acc += xv * wv;
                    }
                }
            }
        }
    }

    y[y_index] = acc;
}

// PyTorch extension entry point: only tensor parameters (input, weight).
// No stride/padding/dilation/groups scalars are accepted; we assume defaults:
// stride = 1, padding = 0, dilation = 1. Groups inferred from weight shape.
at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

    // Ensure contiguous NCDHW
    auto x = input.contiguous();
    auto w = weight.contiguous();

    // Extract dimensions from input [N, Cin, D, H, W]
    TORCH_CHECK(x.dim() == 5, "input must be 5D (N, C, D, H, W)");
    const int N   = static_cast<int>(x.size(0));
    const int Cin = static_cast<int>(x.size(1));
    const int Din = static_cast<int>(x.size(2));
    const int Hin = static_cast<int>(x.size(3));
    const int Win = static_cast<int>(x.size(4));

    // Weight shape: [Cout, CinPg, Kd, Kh, Kw]
    TORCH_CHECK(w.dim() == 5, "weight must be 5D (Cout, Cin/groups, Kd, Kh, Kw)");
    const int Cout = static_cast<int>(w.size(0));
    const int CinPg = static_cast<int>(w.size(1));
    const int Kd    = static_cast<int>(w.size(2));
    const int Kh    = static_cast<int>(w.size(3));
    const int Kw    = static_cast<int>(w.size(4));

    TORCH_CHECK(CinPg > 0 && Kd > 0 && Kh > 0 && Kw > 0, "Invalid weight dimensions");
    TORCH_CHECK(Cin % CinPg == 0, "Cin must be divisible by Cin/group from weight");
    const int groups = Cin / CinPg;
    TORCH_CHECK(Cout % groups == 0, "Cout must be divisible by groups");

    // Assume stride=1, padding=0, dilation=1 (as per model defaults)
    const int Od = Din - Kd + 1;
    const int Oh = Hin - Kh + 1;
    const int Ow = Win - Kw + 1;

    TORCH_CHECK(Od > 0 && Oh > 0 && Ow > 0, "Output dimensions must be positive; check kernel sizes vs input");

    auto y = at::empty({N, Cout, Od, Oh, Ow}, x.options());

    // Launch configuration
    dim3 block(TILE_X, TILE_Y, 1);
    const int flat_y_len = Od * Oh;
    dim3 grid(
        (Ow + TILE_X - 1) / TILE_X,
        (flat_y_len + TILE_Y - 1) / TILE_Y,
        N * Cout
    );

    // Dynamic shared memory for per-co weight slice
    const size_t w_slice_elems = static_cast<size_t>(CinPg) * Kd * Kh * Kw;
    const size_t smem_bytes = w_slice_elems * sizeof(float);
    // MI300X LDS per CU is 64KB; per-block allocation must not exceed it
    const bool use_smem = smem_bytes > 0 && smem_bytes <= 64 * 1024;

    hipLaunchKernelGGL(
        conv3d_ncdhw_tiled_kernel,
        grid, block,
        use_smem ? smem_bytes : 0, 0,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        y.data_ptr<float>(),
        N, Cin, Din, Hin, Win,
        Cout, Kd, Kh, Kw,
        Od, Oh, Ow,
        groups,
        use_smem ? 1 : 0
    );

    // Error checking
    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Optimized Conv3D (NCDHW) with square Kd x Kh and general Kw (HIP)");
}