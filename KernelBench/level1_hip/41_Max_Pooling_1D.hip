// MaxPool1d (kernel=8, stride=1, padding=4, dilation=3) optimized HIP kernel for AMD MI300X
// Single-file PyTorch extension with HIP kernels and bindings.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <cstdint>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Tunables (chosen for MI300X - wavefront=64)
#define TILE_O 256                 // outputs per block along sequence dim (multiple of 64)
#define KERNEL_SIZE 8              // from target model
#define STRIDE 1                   // from target model
#define PADDING 4                  // from target model
#define DILATION 3                 // from target model

// Shared segment length needed per block: outputs tile plus dilation span
#define SHARED_SPAN (TILE_O + DILATION * (KERNEL_SIZE - 1))
// Add +1 padding to avoid shared bank conflicts
#define SHARED_SIZE (SHARED_SPAN + 1)

// Kernel: Each block computes a tile of outputs along the sequence dimension for one (n,c)
// Grid:
//   grid.x = ceil_div(out_L, TILE_O)
//   grid.y = N*C
// Block:
//   block.x = TILE_O
template <typename T>
__global__ __launch_bounds__(TILE_O, 8)
void maxpool1d_shared_kernel(
    const T* __restrict__ x,   // [N, C, L] contiguous
    T* __restrict__ y,         // [N, C, out_L] contiguous
    int N, int C, int L, int out_L)
{
    __shared__ T s_in[SHARED_SIZE];

    const int tid = threadIdx.x;
    const int out_block_start = blockIdx.x * TILE_O;

    // Map blockIdx.y to (n, c)
    const int nc = blockIdx.y;
    const int n = nc / C;
    const int c = nc - n * C;

    if (n >= N || c >= C) return;

    // Limit last block span to valid outputs
    const int last_out = min(out_block_start + TILE_O - 1, out_L - 1);

    // Compute union input span this block needs to cover
    const int min_in = out_block_start * STRIDE - PADDING;                 // start base
    const int max_in = (last_out * STRIDE - PADDING) + DILATION * (KERNEL_SIZE - 1); // end inclusive

    const int union_len = max(0, max_in - min_in + 1);
    // Pointers to current (n,c) row
    const size_t in_row_base  = (static_cast<size_t>(n) * C + static_cast<size_t>(c)) * static_cast<size_t>(L);
    const size_t out_row_base = (static_cast<size_t>(n) * C + static_cast<size_t>(c)) * static_cast<size_t>(out_L);

    // Cooperative load into shared memory, pad out-of-range with -inf
    for (int i = tid; i < union_len; i += blockDim.x) {
        const int g = min_in + i;
        T v;
        if (g >= 0 && g < L) {
            v = x[in_row_base + static_cast<size_t>(g)];
        } else {
            v = -INFINITY;
        }
        // +1 padding in shared array to mitigate bank conflicts
        s_in[i] = v;
    }
    // If union_len < SHARED_SIZE, the remainder is unused; no need to initialize
    __syncthreads();

    // Compute this thread's output (if within range)
    const int out_o = out_block_start + tid;
    if (out_o < out_L) {
        const int base = out_o * STRIDE - PADDING;
        T m = -INFINITY;

        #pragma unroll
        for (int k = 0; k < KERNEL_SIZE; ++k) {
            const int g = base + k * DILATION;
            const int si = g - min_in; // index into shared
            if (static_cast<unsigned>(si) < static_cast<unsigned>(union_len)) {
                T v = s_in[si];
                m = fmaxf(m, v);
            }
        }
        y[out_row_base + static_cast<size_t>(out_o)] = m;
    }
}

// Host wrapper: accepts only runtime tensors (input). No scalar config in signature.
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(input.dim() == 3, "Input must have shape [N, C, L]");

    auto x = input.contiguous();

    const int N = static_cast<int>(x.size(0));
    const int C = static_cast<int>(x.size(1));
    const int L = static_cast<int>(x.size(2));

    // Output length using PyTorch's pooling formula:
    // out_L = floor((L + 2*P - D*(K-1) - 1)/S + 1)
    const int out_L = (L + 2 * PADDING - DILATION * (KERNEL_SIZE - 1) - 1) / STRIDE + 1;
    TORCH_CHECK(out_L > 0, "Computed negative/zero output length; check input size and pooling params");

    auto y = at::empty({N, C, out_L}, x.options());

    dim3 block(TILE_O);
    dim3 grid((out_L + TILE_O - 1) / TILE_O, N * C);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(maxpool1d_shared_kernel<float>),
        grid, block, 0, 0,
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, L, out_L
    );

    hipError_t err_sync = hipDeviceSynchronize();
    TORCH_CHECK(err_sync == hipSuccess, "HIP error (sync): ", hipGetErrorString(err_sync));
    hipError_t err_async = hipGetLastError();
    TORCH_CHECK(err_async == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err_async));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "MaxPool1d (k=8, s=1, p=4, d=3) optimized HIP");
}