// Matrix-Vector Multiplication (C = A * B) optimized for AMD MI300X (gfx942)
// - A: (M, K) contiguous float32
// - B: (K, 1) contiguous float32 (treated as length-K vector)
// - C: (M, 1) float32
//
// Tiling strategy:
//   - Tile along K (columns) into TILE_K chunks
//   - Cache each B tile into LDS once per block
//   - Each wavefront computes one output row
//   - Block processes ROWS_PER_BLOCK rows in parallel
//
// Build: hipcc -O3 --offload-arch=gfx9-4-generic matvec.hip -o matvec
//        (or -march=gfx942 for MI300X)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <cstdint>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Tunables
#ifndef ROWS_PER_BLOCK
#define ROWS_PER_BLOCK 4   // waves per block
#endif

#ifndef TILE_K
#define TILE_K 1024        // elements of K cached per tile (4KB of LDS)
#endif

// Threads layout: (x=64 threads per wave, y=ROWS_PER_BLOCK waves per block)
#define BLOCK_DIM_X WAVE_SIZE
#define BLOCK_DIM_Y ROWS_PER_BLOCK
#define THREADS_PER_BLOCK (BLOCK_DIM_X * BLOCK_DIM_Y)

__device__ inline float dot4(const float4& a, const float bx, const float by, const float bz, const float bw) {
    return a.x * bx + a.y * by + a.z * bz + a.w * bw;
}

__device__ inline float warp_reduce_sum(float v) {
    // wavefront size is 64 on AMD; use width=64
    for (int offset = WAVE_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, WAVE_SIZE);
    }
    return v;
}

// Each block computes up to ROWS_PER_BLOCK rows (one per wavefront).
// K dimension is processed in tiles of TILE_K, caching B into shared memory.
__global__ __launch_bounds__(THREADS_PER_BLOCK, 4)
void matvec_tiled_kernel(
    const float* __restrict__ A,  // (M, K)
    const float* __restrict__ B,  // (K)
    float* __restrict__ C,        // (M)
    int M,
    int K)
{
    __shared__ float sB[TILE_K];  // cached B tile

    const int lane = threadIdx.x;         // 0..63
    const int wave = threadIdx.y;         // 0..ROWS_PER_BLOCK-1
    const int rows_per_block = blockDim.y;

    const int row = blockIdx.x * rows_per_block + wave;
    if (row >= M) {
        return;
    }

    // Pointer to the start of this A row
    const float* __restrict__ Arow = A + static_cast<long long>(row) * K;

    float sum = 0.0f;

    // Loop over K tiles
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const int tile_elems = min(TILE_K, K - k0);
        const int n_threads = blockDim.x * blockDim.y;          // total threads per block
        const int tid = wave * blockDim.x + lane;               // 0..THREADS_PER_BLOCK-1

        // 1) Load B tile from global to shared memory
        // Vectorized float4 loads when possible
        const int vec4_count = tile_elems >> 2;  // number of float4
        const int rem = tile_elems & 3;

        // Load vectorized part
        for (int i4 = tid; i4 < vec4_count; i4 += n_threads) {
            float4 vb = reinterpret_cast<const float4*>(B + k0)[i4];
            const int base = (i4 << 2);
            sB[base + 0] = vb.x;
            sB[base + 1] = vb.y;
            sB[base + 2] = vb.z;
            sB[base + 3] = vb.w;
        }
        // Load remainder scalars (at most 3)
        for (int i = vec4_count * 4 + tid; i < vec4_count * 4 + rem; i += n_threads) {
            sB[i] = B[k0 + i];
        }

        __syncthreads();

        // 2) Compute partial dot product for this row and K tile
        // Use float4 vectorized loads for A; read B from shared as scalars
        // Each lane processes strided chunks
        for (int j4 = lane; j4 < vec4_count; j4 += WAVE_SIZE) {
            float4 a4 = reinterpret_cast<const float4*>(Arow + k0)[j4];
            const int base = (j4 << 2);
            sum += dot4(a4, sB[base + 0], sB[base + 1], sB[base + 2], sB[base + 3]);
        }
        // Remainder for this tile
        for (int j = (vec4_count << 2) + lane; j < tile_elems; j += WAVE_SIZE) {
            sum += Arow[k0 + j] * sB[j];
        }

        __syncthreads();
    }

    // 3) Reduce within wavefront (64 lanes) and write result
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        C[row] = sum;
    }
}

at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.dim() == 2, "A must be 2D (M, K)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (K, 1)");
    TORCH_CHECK(B.size(1) == 1, "B must have shape (K, 1)");
    TORCH_CHECK(A.size(1) == B.size(0), "A.size(1) must equal B.size(0)");

    // Ensure contiguous for coalesced access
    if (!A.is_contiguous()) A = A.contiguous();
    if (!B.is_contiguous()) B = B.contiguous();

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);

    // Output shape (M, 1)
    auto C = at::empty({M, 1}, A.options());

    // Flatten B to length-K vector view
    at::Tensor Bv = B.view({K});

    // Launch configuration
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 grid((static_cast<int>(M) + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    hipStream_t stream = c10::hip::getCurrentHIPStream();
    hipLaunchKernelGGL(
        matvec_tiled_kernel,
        grid, block, 0, stream,
        A.data_ptr<float>(),
        Bv.data_ptr<float>(),
        C.data_ptr<float>(),
        static_cast<int>(M),
        static_cast<int>(K)
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));
    err = hipStreamSynchronize(stream);
    TORCH_CHECK(err == hipSuccess, "HIP kernel execution failed: ", hipGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Optimized matrix-vector multiplication (C = A * B) using HIP");
}