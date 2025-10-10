// MaxPool3d (dilated) HIP kernel for PyTorch (MI300X-optimized)
// Model specifics (hard-coded from construction):
//   kernel_size = 3
//   stride      = 2
//   padding     = 1
//   dilation    = 3
//   ceil_mode   = False
//
// Input  tensor: (N, C, D, H, W) - float32, contiguous
// Output tensor: (N, C, D_out, H_out, W_out)
// No learnable parameters.
//
// Build example:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 maxpool3d_dilated.hip -o maxpool3d_dilated
//
// PyTorch extension is provided via PYBIND11_MODULE at the bottom.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cfloat>
#include <iostream>

#ifndef TILE_W
#define TILE_W 64   // Multiple of wavefront (64)
#endif

#ifndef TILE_H
#define TILE_H 4    // 64*4 = 256 threads per block
#endif

// Pooling configuration (hard-coded for the provided model)
constexpr int KSIZE = 3;   // kernel_size
constexpr int STRIDE = 2;  // stride
constexpr int PAD = 1;     // padding
constexpr int DIL = 3;     // dilation

// Utility: index computation helpers
__device__ __forceinline__ int64_t idx5(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w,
                                        int64_t C, int64_t D, int64_t H, int64_t W) {
    return (((n * C + c) * D + d) * H + h) * W + w;
}

// Kernel: each thread computes one output element (ow, oh) within a tile for a fixed (b,c,od)
__global__ __launch_bounds__(TILE_W * TILE_H, 2)
void maxpool3d_dilated_kernel(
    const float* __restrict__ input,   // [N, C, D, H, W]
    float* __restrict__ output,        // [N, C, D_out, H_out, W_out]
    int N, int C, int D, int H, int W,
    int D_out, int H_out, int W_out)
{
    // Map grid.z to (b, c, od)
    int bc_od_lin = blockIdx.z;
    if (bc_od_lin >= N * C * D_out) return;

    int od = bc_od_lin % D_out;
    int tmp = bc_od_lin / D_out;
    int c   = tmp % C;
    int b   = tmp / C;

    // Tile start in output space
    int ow = blockIdx.x * TILE_W + threadIdx.x;  // X dimension maps to W_out (fast varying)
    int oh = blockIdx.y * TILE_H + threadIdx.y;  // Y dimension maps to H_out

    if (ow >= W_out || oh >= H_out) return;

    // Compute starting input coordinates for this output position
    // start = out_idx * STRIDE - PAD
    const int in_z_start = od * STRIDE - PAD;
    const int in_y_start = oh * STRIDE - PAD;
    const int in_x_start = ow * STRIDE - PAD;

    // Precompute strides
    const int64_t HW = static_cast<int64_t>(H) * W;
    const int64_t DHW = static_cast<int64_t>(D) * HW;
    const int64_t base_nc = (static_cast<int64_t>(b) * C + c) * DHW;

    float max_val = -FLT_MAX;

    // Iterate over the 3x3x3 pooling window with dilation
    #pragma unroll
    for (int kz = 0; kz < KSIZE; ++kz) {
        int iz = in_z_start + kz * DIL;
        if (iz < 0 || iz >= D) continue;

        #pragma unroll
        for (int ky = 0; ky < KSIZE; ++ky) {
            int iy = in_y_start + ky * DIL;
            if (iy < 0 || iy >= H) continue;

            #pragma unroll
            for (int kx = 0; kx < KSIZE; ++kx) {
                int ix = in_x_start + kx * DIL;
                if (ix < 0 || ix >= W) continue;

                int64_t in_index = base_nc + static_cast<int64_t>(iz) * HW + static_cast<int64_t>(iy) * W + ix;
                float v = input[in_index];
                // fmaxf avoids branches and is efficient on GPU
                max_val = fmaxf(max_val, v);
            }
        }
    }

    // Store result
    const int64_t outHW = static_cast<int64_t>(H_out) * W_out;
    const int64_t outDHW = static_cast<int64_t>(D_out) * outHW;
    int64_t out_index = (((static_cast<int64_t>(b) * C + c) * D_out + od) * H_out + oh) * W_out + ow;
    output[out_index] = max_val;
}

// Host wrapper: PyTorch entry point
// Signature matches benchmarking harness: run(input)
// The model has no learnable parameters.
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA/ROCm device");
    TORCH_CHECK(input.dtype() == at::kFloat, "Input must be float32");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (N, C, D, H, W)");

    // Ensure contiguous memory for coalesced access
    auto x = input.contiguous();

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t D = x.size(2);
    const int64_t H = x.size(3);
    const int64_t W = x.size(4);

    // Output sizes using floor mode (ceil_mode=False)
    // Effective kernel: K_eff = DIL*(KSIZE-1)+1
    const int64_t K_eff = static_cast<int64_t>(DIL) * (KSIZE - 1) + 1;
    TORCH_CHECK(K_eff > 0, "Invalid effective kernel size");

    const int64_t D_out = (D + 2 * PAD - K_eff) / STRIDE + 1;
    const int64_t H_out = (H + 2 * PAD - K_eff) / STRIDE + 1;
    const int64_t W_out = (W + 2 * PAD - K_eff) / STRIDE + 1;

    TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0, "Computed non-positive output size");

    auto y = at::empty({N, C, D_out, H_out, W_out}, x.options());

    // Launch configuration
    dim3 block(TILE_W, TILE_H, 1);  // 64 x 4 = 256 threads/block
    dim3 grid(
        static_cast<unsigned int>((W_out + TILE_W - 1) / TILE_W),
        static_cast<unsigned int>((H_out + TILE_H - 1) / TILE_H),
        static_cast<unsigned int>(N * C * D_out)
    );

    // Guard grid.z to hardware limits (typical max 65535). Fallback tiling in depth if needed.
    if (grid.z > 65535u) {
        // Tile depth across grid.y to respect grid.z limit
        const int64_t max_z = 65535;
        // We split (N*C*D_out) into chunks processed via multiple launches
        const int64_t total = N * C * D_out;
        int64_t processed = 0;
        hipError_t err;

        while (processed < total) {
            int64_t chunk = std::min<int64_t>(total - processed, max_z);
            dim3 grid_chunk(grid.x, grid.y, static_cast<unsigned int>(chunk));

            hipLaunchKernelGGL(
                maxpool3d_dilated_kernel,
                grid_chunk, block, 0, 0,
                x.data_ptr<float>(),
                y.data_ptr<float>(),
                static_cast<int>(N), static_cast<int>(C), static_cast<int>(D),
                static_cast<int>(H), static_cast<int>(W),
                static_cast<int>(D_out), static_cast<int>(H_out), static_cast<int>(W_out)
            );

            err = hipDeviceSynchronize();
            TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
            err = hipGetLastError();
            TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

            processed += chunk;
        }
        return y;
    }

    // Single launch path
    hipLaunchKernelGGL(
        maxpool3d_dilated_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(D),
        static_cast<int>(H), static_cast<int>(W),
        static_cast<int>(D_out), static_cast<int>(H_out), static_cast<int>(W_out)
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Dilated MaxPool3d (K=3, S=2, P=1, DIL=3) - HIP");
}