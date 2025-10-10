// AvgPool3d (kernel=3, stride=2, padding=1, count_include_pad=True) optimized HIP kernel for AMD MI300X
// Single .hip file containing kernels and PyTorch extension binding.
//
// Notes:
// - Entry point 'run' accepts only tensors (no scalar config), per requirement.
// - Configuration is specialized for the given target model:
//     kernel_size = 3, stride = 2, padding = 1, count_include_pad = True, ceil_mode = False
// - Layout: NCDHW (contiguous)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Specialization for the provided model
#define KERNEL_SIZE 3
#define STRIDE_D 2
#define STRIDE_H 2
#define STRIDE_W 2
#define PAD_D 1
#define PAD_H 1
#define PAD_W 1
#define COUNT_INCLUDE_PAD 1  // True
#define CEIL_MODE 0          // False

// Tiling parameters (2D tile over output HxW for a single output depth slice)
#ifndef TILE_W
#define TILE_W 16
#endif
#ifndef TILE_H
#define TILE_H 16
#endif

// Derived input tile extents needed per kd plane
#define TILE_IN_W (TILE_W * STRIDE_W + (KERNEL_SIZE - 1))
#define TILE_IN_H (TILE_H * STRIDE_H + (KERNEL_SIZE - 1))
// Add +1 padding on the fast dimension to mitigate bank conflicts
#define TILE_IN_W_PAD (TILE_IN_W + 1)

// Fast index computation for NCDHW
__device__ __forceinline__ int idx_ncdhw(int n, int c, int d, int h, int w,
                                         int C, int D, int H, int W) {
    return (((n * C + c) * D + d) * H + h) * W + w;
}

// Naive per-output-element kernel (baseline correctness, decent performance)
__global__ void avgpool3d_naive_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int outD, int outH, int outW)
{
    // Map z-dimension to (n, c, od)
    int z = blockIdx.z;
    if (z >= N * C * outD) return;

    int od = z % outD;
    z /= outD;
    int c = z % C;
    int n = z / C;

    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;

    if (oh >= outH || ow >= outW) return;

    // Base input coordinates (top-left-front) for this output element
    const int in_d_base = od * STRIDE_D - PAD_D;
    const int in_h_base = oh * STRIDE_H - PAD_H;
    const int in_w_base = ow * STRIDE_W - PAD_W;

    float sum = 0.0f;
    // Accumulate valid values; denominator is fixed to kernel volume when COUNT_INCLUDE_PAD == 1
    #pragma unroll
    for (int kd = 0; kd < KERNEL_SIZE; ++kd) {
        int id = in_d_base + kd;
        if (id < 0 || id >= D) continue;
        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            int ih = in_h_base + kh;
            if (ih < 0 || ih >= H) continue;
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                int iw = in_w_base + kw;
                if (iw < 0 || iw >= W) continue;
                int iidx = idx_ncdhw(n, c, id, ih, iw, C, D, H, W);
                sum += input[iidx];
            }
        }
    }

    const float inv_kernel_vol = 1.0f / float(KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE);
    float avg = sum * inv_kernel_vol;

    int oidx = idx_ncdhw(n, c, od, oh, ow, C, outD, outH, outW);
    output[oidx] = avg;
}

// Tiled kernel: cooperatively stages the needed HxW slice per kd plane into LDS (shared memory)
__global__ void avgpool3d_tiled_hw_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int outD, int outH, int outW)
{
    // Shared tile for a single kd slice
    __shared__ float s_tile[TILE_IN_H][TILE_IN_W_PAD];

    // Map z-dimension to (n, c, od)
    int z = blockIdx.z;
    if (z >= N * C * outD) return;

    int od = z % outD;
    z /= outD;
    int c = z % C;
    int n = z / C;

    // Output tile origin
    const int ow0 = blockIdx.x * TILE_W;
    const int oh0 = blockIdx.y * TILE_H;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // The output element this thread computes
    const int ow = ow0 + tx;
    const int oh = oh0 + ty;

    // Fixed denominator for count_include_pad=True
    const float inv_kernel_vol = 1.0f / float(KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE);

    // Base in-coordinates for the staged tile (per kd we add depth offset)
    const int in_w_base = ow0 * STRIDE_W - PAD_W;
    const int in_h_base = oh0 * STRIDE_H - PAD_H;
    const int in_d_base = od  * STRIDE_D - PAD_D;

    float sum = 0.0f;

    // Loop over depth kernel
    #pragma unroll
    for (int kd = 0; kd < KERNEL_SIZE; ++kd) {
        const int id = in_d_base + kd;
        const bool d_valid = (id >= 0 && id < D);

        // Tile load: cover TILE_IN_H x TILE_IN_W cooperatively
        for (int th = ty; th < TILE_IN_H; th += blockDim.y) {
            int ih = in_h_base + th;
            bool h_valid = (ih >= 0 && ih < H);
            // Load scalars with coalesced-ish pattern along W
            for (int tw = tx; tw < TILE_IN_W; tw += blockDim.x) {
                int iw = in_w_base + tw;
                float v = 0.0f;
                if (d_valid && h_valid && iw >= 0 && iw < W) {
                    int iidx = idx_ncdhw(n, c, id, ih, iw, C, D, H, W);
                    v = input[iidx];
                }
                s_tile[th][tw] = v;
            }
        }
        __syncthreads();

        // Compute this thread's partial sum for this kd slice if output coord is valid
        if (oh < outH && ow < outW) {
            // Starting indices inside the staged tile for this output
            const int th0 = ty * STRIDE_H;
            const int tw0 = tx * STRIDE_W;

            // 3x3 window
            float partial = 0.0f;
            #pragma unroll
            for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    partial += s_tile[th0 + kh][tw0 + kw];
                }
            }
            sum += partial;
        }
        __syncthreads();
    }

    if (oh < outH && ow < outW) {
        int oidx = idx_ncdhw(n, c, od, oh, ow, C, outD, outH, outW);
        output[oidx] = sum * inv_kernel_vol;
    }
}

// Host wrapper: accepts only tensors (input), infers/sets configuration internally.
// Returns a new output tensor.
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 input is supported");
    TORCH_CHECK(input.dim() == 5, "Input must be NCDHW (5D)");

    auto x = input.contiguous();

    const int N = x.size(0);
    const int C = x.size(1);
    const int D = x.size(2);
    const int H = x.size(3);
    const int W = x.size(4);

    // Compute output sizes using floor mode (ceil_mode=False)
    const int outD = (D + 2 * PAD_D - KERNEL_SIZE) / STRIDE_D + 1;
    const int outH = (H + 2 * PAD_H - KERNEL_SIZE) / STRIDE_H + 1;
    const int outW = (W + 2 * PAD_W - KERNEL_SIZE) / STRIDE_W + 1;

    auto out = at::empty({N, C, outD, outH, outW}, x.options());

    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid((outW + TILE_W - 1) / TILE_W,
              (outH + TILE_H - 1) / TILE_H,
              N * C * outD);

    // Choose tiled kernel for small fixed kernel size (3x3x3)
    hipLaunchKernelGGL(
        avgpool3d_tiled_hw_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, D, H, W, outD, outH, outW
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));

    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "AvgPool3d (k=3, s=2, p=1, count_include_pad=True) - HIP optimized");
}