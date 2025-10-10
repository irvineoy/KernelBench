// MaxPool2d HIP kernel for AMD GPUs (MI300X-optimized)
// Implements nn.MaxPool2d with kernel_size=4, stride=1, padding=1, dilation=1
// Single .hip file containing kernels and PyTorch extension bindings.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cfloat>

#ifndef TILE_W
#define TILE_W 16
#endif
#ifndef TILE_H
#define TILE_H 16
#endif

// Hard-coded pooling configuration to match the provided PyTorch model
// Model config: kernel_size=4, stride=1, padding=1, dilation=1
constexpr int KERNEL_H = 4;
constexpr int KERNEL_W = 4;
constexpr int STRIDE_H = 1;
constexpr int STRIDE_W = 1;
constexpr int PAD_H = 1;
constexpr int PAD_W = 1;
constexpr int DIL_H = 1;
constexpr int DIL_W = 1;

constexpr int K_EFF_H = (KERNEL_H - 1) * DIL_H + 1; // 4
constexpr int K_EFF_W = (KERNEL_W - 1) * DIL_W + 1; // 4

// Tiled shared-memory maxpool: one thread computes one output (n,c,oh,ow)
// Each block processes a TILE_H x TILE_W output tile for a specific (n,c)
__global__ __launch_bounds__(TILE_W * TILE_H)
void maxpool2d_k4s1p1_kernel(const float* __restrict__ x,
                             float* __restrict__ y,
                             int N, int C, int H, int W,
                             int OH, int OW) {
    // Block coordinates in output space
    const int out_w0 = blockIdx.x * TILE_W;
    const int out_h0 = blockIdx.y * TILE_H;

    // Flattened (batch, channel) in grid.z
    const int bc = blockIdx.z;
    const int b = bc / C;
    const int c = bc - b * C;
    if (b >= N || c >= C) return;

    // Thread coordinates within the output tile
    const int tx = threadIdx.x; // maps to out_w within tile
    const int ty = threadIdx.y; // maps to out_h within tile

    // Global output coordinates this thread will produce
    const int out_h = out_h0 + ty;
    const int out_w = out_w0 + tx;

    // Input origin for this tile (top-left of receptive field for (out_h0, out_w0))
    const int in_h0 = out_h0 * STRIDE_H - PAD_H; // stride=1 => out_h0 - PAD_H
    const int in_w0 = out_w0 * STRIDE_W - PAD_W;

    // Shared memory tile:
    // We need TILE_H + K_EFF_H - 1 rows and TILE_W + K_EFF_W - 1 cols
    // Add +1 padding on the second dimension to reduce bank conflicts
    __shared__ float s_in[TILE_H + K_EFF_H - 1][TILE_W + K_EFF_W - 1 + 1];

    // Base pointers for this (b,c)
    const int HW = H * W;
    const int OHW = OH * OW;
    const int x_base = (b * C + c) * HW;
    const int y_base = (b * C + c) * OHW;

    // Populate shared memory tile, covering halo
    // Each thread loads multiple elements in strides of blockDim.{x,y}
    for (int dy = ty; dy < TILE_H + K_EFF_H - 1; dy += blockDim.y) {
        const int in_h = in_h0 + dy * DIL_H; // DIL_H = 1
        for (int dx = tx; dx < TILE_W + K_EFF_W - 1; dx += blockDim.x) {
            const int in_w = in_w0 + dx * DIL_W; // DIL_W = 1
            float val = 0.0f; // padding value for MaxPool2d is 0
            if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                val = x[x_base + in_h * W + in_w];
            }
            s_in[dy][dx] = val;
        }
    }
    __syncthreads();

    // Compute output if within bounds
    if (out_h < OH && out_w < OW) {
        // The window for (out_h, out_w) starts at local (ty, tx) in shared memory
        float m = -FLT_MAX;

        // Unroll 4x4 window (kernel 4x4, dilation 1)
        #pragma unroll
        for (int kh = 0; kh < KERNEL_H; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < KERNEL_W; ++kw) {
                float v = s_in[ty + kh][tx + kw];
                m = v > m ? v : m;
            }
        }

        y[y_base + out_h * OW + out_w] = m;
    }
}

// Host wrapper: accepts only tensors (input), infers output shape from hard-coded config
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be NCHW (4D)");

    auto x = input.contiguous();

    const int N = static_cast<int>(x.size(0));
    const int C = static_cast<int>(x.size(1));
    const int H = static_cast<int>(x.size(2));
    const int W = static_cast<int>(x.size(3));

    // Output dimensions for MaxPool2d with K=4, S=1, P=1, D=1
    const int OH = (H + 2 * PAD_H - K_EFF_H) / STRIDE_H + 1;
    const int OW = (W + 2 * PAD_W - K_EFF_W) / STRIDE_W + 1;
    TORCH_CHECK(OH > 0 && OW > 0, "Computed non-positive output size");

    auto y = at::empty({N, C, OH, OW}, x.options());

    // Launch configuration
    dim3 block(TILE_W, TILE_H, 1); // 16x16 = 256 threads/block (multiple of wavefront=64)
    dim3 grid((OW + TILE_W - 1) / TILE_W,
              (OH + TILE_H - 1) / TILE_H,
              N * C);

    hipLaunchKernelGGL(
        maxpool2d_k4s1p1_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        OH, OW
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "MaxPool2d (kernel=4, stride=1, padding=1, dilation=1) - HIP optimized");
}