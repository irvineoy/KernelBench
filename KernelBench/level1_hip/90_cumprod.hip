// Cumprod along dim=1 for contiguous tensors
// Optimized for AMD MI300X (gfx942) using HIP
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256  // Must be a multiple of 64 (wavefront size)
#endif

// Tile-based inclusive scan (product) per sequence with carry across tiles
// Sequences are defined over a flattened 3D view: (outer, length, inner)
// We compute cumprod along the "length" dimension (dim=1 in the original tensor)
__global__ void __launch_bounds__(BLOCK_SIZE)
cumprod_dim1_kernel(const float* __restrict__ x,
                    float* __restrict__ y,
                    long long outer,
                    long long length,
                    long long inner) {
    const long long seq = static_cast<long long>(blockIdx.x);
    if (seq >= outer * inner) return;

    // Map flattened sequence index back to (outer_idx, inner_idx)
    const long long outer_idx = seq / inner;
    const long long inner_idx = seq % inner;

    // Base address for this sequence in contiguous memory
    const long long base = (outer_idx * length * inner) + inner_idx;

    // Shared memory for tile scan (+1 padding to mitigate bank conflicts)
    __shared__ float sdata[BLOCK_SIZE + 1];
    __shared__ float s_carry;

    const int tid = threadIdx.x;

    if (tid == 0) s_carry = 1.0f;
    __syncthreads();

    // Process the sequence in tiles of size BLOCK_SIZE
    for (long long tile_base = 0; tile_base < length; tile_base += BLOCK_SIZE) {
        int n = static_cast<int>(length - tile_base);
        if (n > BLOCK_SIZE) n = BLOCK_SIZE;

        // Load tile
        if (tid < n) {
            sdata[tid] = x[base + (tile_base + tid) * inner];
        }
        __syncthreads();

        // Inclusive scan (product) within the tile using Hillis-Steele
        // Unroll to reduce loop overhead; guarded by n
        #pragma unroll 8
        for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
            float val = 1.0f;
            if (tid < n && tid >= offset) {
                val = sdata[tid - offset];
            }
            __syncthreads();
            if (tid < n) {
                sdata[tid] *= val;
            }
            __syncthreads();
        }

        // Multiply by carry from previous tiles
        float carry = s_carry;
        if (tid < n) {
            sdata[tid] *= carry;
        }
        __syncthreads();

        // Write results
        if (tid < n) {
            y[base + (tile_base + tid) * inner] = sdata[tid];
        }

        // Update carry to last element of this tile's output
        if (tid == n - 1) {
            s_carry = sdata[n - 1];
        }
        __syncthreads();
    }
}

// Host wrapper: accepts ONLY tensors (per requirements)
at::Tensor run(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA/HIP device");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported");

    // We implement cumprod along dim=1 (as per target model)
    constexpr int dim = 1;

    // Ensure contiguous layout for predictable indexing
    auto x_contig = x.contiguous();
    const auto sizes = x_contig.sizes();
    TORCH_CHECK(sizes.size() >= 2, "cumprod dim=1 requires tensor with at least 2 dimensions");

    // Flatten to (outer, length, inner) with cumprod along 'length'
    long long outer = 1;
    for (int i = 0; i < dim; ++i) outer *= static_cast<long long>(sizes[i]);
    const long long length = static_cast<long long>(sizes[dim]);
    long long inner = 1;
    for (int i = dim + 1; i < static_cast<int>(sizes.size()); ++i)
        inner *= static_cast<long long>(sizes[i]);

    // Allocate output
    auto y = at::empty_like(x_contig);

    // Launch configuration
    const dim3 block(BLOCK_SIZE);
    const long long num_sequences = outer * inner;
    TORCH_CHECK(num_sequences > 0, "Invalid number of sequences to process");
    TORCH_CHECK(length >= 0, "Invalid length along dim=1");

    // Grid size capped to 32-bit as required by HIP launch; split if needed
    // For typical use (including target), this is sufficient.
    TORCH_CHECK(num_sequences <= static_cast<long long>(std::numeric_limits<int>::max()),
                "Number of sequences too large for single kernel launch");

    const dim3 grid(static_cast<unsigned int>(num_sequences));

    hipLaunchKernelGGL(
        cumprod_dim1_kernel,
        grid, block, 0, 0,
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        outer, length, inner
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Cumulative product along dim=1 (HIP, tile-scan)");
}