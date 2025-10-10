#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

// Tiling parameters tuned for MI300X (gfx942)
#ifndef TILE_W
#define TILE_W 32
#endif
#ifndef TILE_H
#define TILE_H 8
#endif
#ifndef CIN_BLK
#define CIN_BLK 8
#endif
#ifndef COUT_BLK
#define COUT_BLK 8
#endif

// Assumptions for this model (per provided PyTorch definition):
// stride = (1,1), padding = (0,0), dilation = (1,1), groups = 1, bias = False
// NCHW layout, float32 tensors, contiguous

__global__ __launch_bounds__(256)
void conv2d_nchw_tiled_kernel(
    const float* __restrict__ x,        // [N, Cin, H, W]
    const float* __restrict__ w,        // [Cout, Cin, Kh, Kw]
    float* __restrict__ y,              // [N, Cout, Hout, Wout]
    int N, int Cin, int H, int W,
    int Cout, int Kh, int Kw,
    int Hout, int Wout)
{
    // Block/thread coordinates
    const int tx = threadIdx.x;   // 0..TILE_W-1
    const int ty = threadIdx.y;   // 0..TILE_H-1
    const int tid = ty * blockDim.x + tx;
    const int numThreads = blockDim.x * blockDim.y;

    // Grid mapping: x->width tiles, y->height tiles, z->(N * OC tiles)
    const int out_w0 = blockIdx.x * TILE_W;
    const int out_h0 = blockIdx.y * TILE_H;

    const int numOcTiles = (Cout + COUT_BLK - 1) / COUT_BLK;
    const int n  = blockIdx.z / numOcTiles;
    const int ocTileIdx = blockIdx.z % numOcTiles;

    if (n >= N) return;

    const int oc0 = ocTileIdx * COUT_BLK;
    const int ocActual = min(COUT_BLK, Cout - oc0);

    // Effective sizes for edge tiles
    const int STW = max(0, min(TILE_W, Wout - out_w0));
    const int STH = max(0, min(TILE_H, Hout - out_h0));

    // If this block is completely outside output range, we still participate in syncs but can return early after to avoid work.
    // However dynamic shared memory is per-block regardless, so keep structure consistent.
    // We'll guard actual compute/write with (oh < Hout && ow < Wout).

    // Shared memory layout (dynamic):
    // sIn:  CIN_BLK x (TILE_H + Kh - 1) x (TILE_W + Kw - 1 + 1 padding)
    // sW:   COUT_BLK x CIN_BLK x Kh x Kw
    extern __shared__ float smem[];
    const int inTileH = TILE_H + Kh - 1;
    const int inTileW = TILE_W + Kw - 1;
    const int sInPitch = inTileW + 1; // +1 to reduce bank conflicts
    const int sInPlane = inTileH * sInPitch;

    float* sIn = smem;
    float* sW  = sIn + CIN_BLK * sInPlane;

    // Per-thread output accumulators for this spatial position across a small tile of output channels
    float acc[COUT_BLK];
    #pragma unroll
    for (int i = 0; i < COUT_BLK; ++i) acc[i] = 0.0f;

    // Compute global output location for this thread
    const int ow = out_w0 + tx;
    const int oh = out_h0 + ty;

    // Loop over input channels in blocks
    for (int cin0 = 0; cin0 < Cin; cin0 += CIN_BLK) {
        const int cinActual = min(CIN_BLK, Cin - cin0);

        // Load weight tile into LDS: [ocActual, cinActual, Kh, Kw]
        const int totalW = ocActual * cinActual * Kh * Kw;
        for (int idx = tid; idx < totalW; idx += numThreads) {
            int tmp = idx;
            const int kw = tmp % Kw; tmp /= Kw;
            const int kh = tmp % Kh; tmp /= Kh;
            const int ci = tmp % cinActual; tmp /= cinActual;
            const int ocs = tmp; // 0..ocActual-1

            const int oc = oc0 + ocs;
            const int ciGlob = cin0 + ci;
            sW[idx] = w[ ((oc * Cin + ciGlob) * Kh + kh) * Kw + kw ];
        }

        // Load input tiles for cinActual channels into LDS
        // We only need to cover the valid output sub-tile sizes (STH, STW), plus halo (Kh-1, Kw-1)
        const int loadTileH = STH + Kh - 1;
        const int loadTileW = STW + Kw - 1;
        const int totalIn = cinActual * loadTileH * loadTileW;

        for (int idx = tid; idx < totalIn; idx += numThreads) {
            int tmp = idx;
            const int tw = tmp % loadTileW; tmp /= loadTileW;
            const int th = tmp % loadTileH; tmp /= loadTileH;
            const int ci = tmp; // 0..cinActual-1

            const int gh = out_h0 + th;    // global input h
            const int gw = out_w0 + tw;    // global input w
            const int ciGlob = cin0 + ci;

            float v = 0.0f;
            // Since padding = 0, simply check bounds
            if (gh >= 0 && gh < H && gw >= 0 && gw < W) {
                v = x[ ((n * Cin + ciGlob) * H + gh) * W + gw ];
            }
            sIn[ci * sInPlane + th * sInPitch + tw] = v;
        }

        __syncthreads();

        // Compute for valid output threads
        if (oh < Hout && ow < Wout) {
            // For each input channel in this chunk and kernel window
            #pragma unroll 1
            for (int ci = 0; ci < cinActual; ++ci) {
                #pragma unroll 1
                for (int kh = 0; kh < Kh; ++kh) {
                    #pragma unroll 1
                    for (int kw = 0; kw < Kw; ++kw) {
                        // Read input value for this thread's output position
                        const float in_v = sIn[ci * sInPlane + (ty + kh) * sInPitch + (tx + kw)];
                        // Multiply-accumulate across oc sub-tile
                        // sW layout linearized same way as in load loop
                        const int wBase = ((0 * cinActual + ci) * Kh + kh) * Kw + kw; // ocSub offset added in-loop
                        #pragma unroll
                        for (int ocs = 0; ocs < ocActual; ++ocs) {
                            const float wv = sW[wBase + ocs * (cinActual * Kh * Kw)];
                            acc[ocs] += in_v * wv;
                        }
                    }
                }
            }
        }

        __syncthreads(); // ensure we don't overwrite LDS before all threads are done
    }

    // Write output
    if (oh < Hout && ow < Wout) {
        #pragma unroll
        for (int ocs = 0; ocs < ocActual; ++ocs) {
            const int oc = oc0 + ocs;
            y[ ((n * Cout + oc) * Hout + oh) * Wout + ow ] = acc[ocs];
        }
    }
}

at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA/ROCm device");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA/ROCm device");
    TORCH_CHECK(input.dtype() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(weight.dtype() == at::kFloat, "Only float32 weights supported");
    TORCH_CHECK(input.dim() == 4, "Input must be NCHW");
    TORCH_CHECK(weight.dim() == 4, "Weight must be [Cout, Cin, Kh, Kw]");

    // Ensure contiguous
    auto x = input.contiguous();
    auto w = weight.contiguous();

    const int N  = x.size(0);
    const int Cin = x.size(1);
    const int H  = x.size(2);
    const int W  = x.size(3);

    const int Cout = w.size(0);
    const int Cin_w = w.size(1);
    const int Kh = w.size(2);
    const int Kw = w.size(3);

    // groups=1 (from model), so Cin must match weight Cin
    TORCH_CHECK(Cin == Cin_w, "Weight Cin must match input Cin (groups=1 assumed)");
    TORCH_CHECK(Kh >= 1 && Kw >= 1, "Kernel sizes must be >= 1");

    // Defaults: stride=1, padding=0, dilation=1
    const int Hout = H - Kh + 1;
    const int Wout = W - Kw + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "Invalid output size; ensure stride=1, padding=0, dilation=1");

    auto y = at::empty({N, Cout, Hout, Wout}, x.options());

    // Launch config
    dim3 block(TILE_W, TILE_H, 1);            // 32x8 = 256 threads/block (multiple of wavefront 64)
    dim3 grid((Wout + TILE_W - 1) / TILE_W,
              (Hout + TILE_H - 1) / TILE_H,
              N * ((Cout + COUT_BLK - 1) / COUT_BLK));

    // Dynamic shared memory size (worst-case tile sizes used uniformly)
    const int inTileH = TILE_H + Kh - 1;
    const int inTileW = TILE_W + Kw - 1;
    const int sInPitch = inTileW + 1; // padded
    const size_t sInSize = CIN_BLK * inTileH * sInPitch;
    const size_t sWSize  = COUT_BLK * CIN_BLK * Kh * Kw;
    const size_t sharedBytes = (sInSize + sWSize) * sizeof(float);

    hipLaunchKernelGGL(
        conv2d_nchw_tiled_kernel,
        grid, block, sharedBytes, 0,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        y.data_ptr<float>(),
        N, Cin, H, W,
        Cout, Kh, Kw,
        Hout, Wout
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Optimized Conv2D (stride=1, pad=0, dil=1, groups=1) - HIP");
}