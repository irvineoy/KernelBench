// Optimized HIP kernel for 1D Average Pooling (AvgPool1d) on AMD GPUs (MI300X/gfx942)
// Target PyTorch model configuration: kernel_size=8, stride=1, padding=4, count_include_pad=True, ceil_mode=False
// Only tensor parameters are accepted at runtime, as required.

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#ifndef BLOCK_THREADS
#define BLOCK_THREADS 256         // Multiple of 64 (wavefront size)
#endif

#ifndef TILE_OUT
#define TILE_OUT 1024             // Number of output elements computed per block along length dimension
#endif

// Hard-coded model parameters inferred from the provided model
// AvgPool1d(kernel_size=8, stride=1, padding=4), count_include_pad=True (PyTorch default)
#define KERNEL_SIZE 8
#define STRIDE 1
#define PADDING 4

// Kernel computes a tiled 1D average pooling across the length dimension for each (batch, channel) row.
// Uses LDS (shared memory) to reuse inputs across overlapping windows for stride=1.
__global__ void avgpool1d_tiled_kernel(
    const float* __restrict__ x,   // [N, C, L]
    float* __restrict__ y,         // [N, C, out_L]
    int N,
    int C,
    int L,
    int out_L)
{
    // Each block processes a tile of outputs along the length dimension for a specific (n, c) row.
    const int row = blockIdx.y;  // 0..(N*C-1)
    if (row >= N * C) return;

    const int tile_out_start = blockIdx.x * TILE_OUT;
    if (tile_out_start >= out_L) return;

    const int tile_out_count = min(TILE_OUT, out_L - tile_out_start);
    const int tile_load_count = tile_out_count + KERNEL_SIZE - 1;  // inputs to load in padded domain

    // Base pointers for the current row
    const float* __restrict__ x_row = x + static_cast<long long>(row) * L;
    float* __restrict__ y_row = y + static_cast<long long>(row) * out_L;

    // Shared memory to hold needed input slice (from padded domain) for this tile
    __shared__ float s_data[TILE_OUT + KERNEL_SIZE - 1 + 1]; // +1 to mitigate LDS bank conflicts

    // Load the required slice from the padded domain into shared memory
    // Padded domain index -> original index: orig = pos - PADDING
    // If orig out of [0, L-1], value is 0 (count_include_pad=True).
    for (int i = threadIdx.x; i < tile_load_count; i += blockDim.x) {
        int padded_pos = tile_out_start + i;
        int orig = padded_pos - PADDING;

        float v = 0.0f;
        if ((unsigned)orig < (unsigned)L) {
            v = x_row[orig];
        }
        s_data[i] = v;
    }
    __syncthreads();

    // Compute outputs for this tile; each output = average over KERNEL_SIZE window
    const float inv_k = 1.0f / float(KERNEL_SIZE);
    for (int o = threadIdx.x; o < tile_out_count; o += blockDim.x) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < KERNEL_SIZE; ++k) {
            sum += s_data[o + k];
        }
        y_row[tile_out_start + o] = sum * inv_k;
    }
}

// PyTorch extension entry point
// Signature only accepts tensors, as required. No scalar config parameters are passed.
// Input: x [N, C, L], float32
// Output: y [N, C, out_L], where out_L = floor((L + 2*PADDING - KERNEL_SIZE)/STRIDE) + 1
at::Tensor run(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 tensors are supported");

    // Ensure contiguous memory layout
    x = x.contiguous();

    const int64_t N64 = x.size(0);
    const int64_t C64 = x.size(1);
    const int64_t L64 = x.size(2);

    TORCH_CHECK(N64 > 0 && C64 > 0 && L64 > 0, "Invalid input dimensions");

    // Hard-coded AvgPool1d config from model: K=8, S=1, P=4, count_include_pad=True, ceil_mode=False
    const int K = KERNEL_SIZE;
    const int S = STRIDE;
    const int P = PADDING;

    const int64_t padded_len = L64 + 2LL * P;
    TORCH_CHECK(padded_len >= K, "Invalid configuration leading to negative output length");

    // out_L = floor((L + 2*P - K) / S) + 1
    const int64_t out_L64 = (padded_len - K) / S + 1;

    auto y = at::empty({N64, C64, out_L64}, x.options());

    const int N = static_cast<int>(N64);
    const int C = static_cast<int>(C64);
    const int L = static_cast<int>(L64);
    const int out_L = static_cast<int>(out_L64);

    dim3 block(BLOCK_THREADS, 1, 1);
    dim3 grid((out_L + TILE_OUT - 1) / TILE_OUT, N * C, 1);

    hipLaunchKernelGGL(
        avgpool1d_tiled_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, L, out_L
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "1D Average Pooling (AvgPool1d) kernel with K=8, S=1, P=4 (HIP, optimized)");
}