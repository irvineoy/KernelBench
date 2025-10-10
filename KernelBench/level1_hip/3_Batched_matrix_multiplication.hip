// Batched Matrix Multiplication (BMM) optimized HIP kernel for AMD MI300X (gfx942)
// Implements C[b] = A[b] * B[b] for b in [0, batch), with A:[B,M,K], B:[B,K,N], C:[B,M,N]
// Single .hip file containing both kernel and PyTorch bindings.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

// Tile configuration
#ifndef BLOCK_M
#define BLOCK_M 64
#endif
#ifndef BLOCK_N
#define BLOCK_N 64
#endif
#ifndef BLOCK_K
#define BLOCK_K 16
#endif

// Per-thread micro-tile
#ifndef THREAD_TILE_M
#define THREAD_TILE_M 4
#endif
#ifndef THREAD_TILE_N
#define THREAD_TILE_N 4
#endif

// Derived thread block dimensions (must be integers)
static_assert(BLOCK_M % THREAD_TILE_M == 0, "BLOCK_M must be divisible by THREAD_TILE_M");
static_assert(BLOCK_N % THREAD_TILE_N == 0, "BLOCK_N must be divisible by THREAD_TILE_N");
#define BLOCK_THREADS_Y (BLOCK_M / THREAD_TILE_M)  // 64/4 = 16
#define BLOCK_THREADS_X (BLOCK_N / THREAD_TILE_N)  // 64/4 = 16
#define THREADS_PER_BLOCK (BLOCK_THREADS_X * BLOCK_THREADS_Y) // 256

// Launch bounds hint: 256 threads/block
__global__ __launch_bounds__(THREADS_PER_BLOCK, 2)
void bmm_tiled_kernel(
    const float* __restrict__ A,    // [B, M, K]
    const float* __restrict__ B,    // [B, K, N]
    float* __restrict__ C,          // [B, M, N]
    int BATCH, int M, int N, int K)
{
    // Shared memory tiles with +1 padding on the fastest dimension to avoid bank conflicts
    __shared__ float As[BLOCK_M][BLOCK_K + 1];
    __shared__ float Bs[BLOCK_K][BLOCK_N + 1];

    // 2D thread indices within the block
    const int tx = threadIdx.x; // 0..BLOCK_THREADS_X-1
    const int ty = threadIdx.y; // 0..BLOCK_THREADS_Y-1

    // Block coordinates map to output C tile
    const int block_row = blockIdx.y;  // along M
    const int block_col = blockIdx.x;  // along N
    const int batch_id  = blockIdx.z;  // batch

    // Starting indices for this tile in output matrix
    const int row_base = block_row * BLOCK_M + ty * THREAD_TILE_M; // starting output row for this thread's micro-tile
    const int col_base = block_col * BLOCK_N + tx * THREAD_TILE_N; // starting output col for this thread's micro-tile

    // Pointers to this batch's matrices
    const size_t strideA = static_cast<size_t>(M) * static_cast<size_t>(K);
    const size_t strideB = static_cast<size_t>(K) * static_cast<size_t>(N);
    const size_t strideC = static_cast<size_t>(M) * static_cast<size_t>(N);
    const float* __restrict__ Ap = A + static_cast<size_t>(batch_id) * strideA;
    const float* __restrict__ Bp = B + static_cast<size_t>(batch_id) * strideB;
    float* __restrict__ Cp       = C + static_cast<size_t>(batch_id) * strideC;

    // Accumulator registers for a THREAD_TILE_M x THREAD_TILE_N micro-tile
    float acc[THREAD_TILE_M][THREAD_TILE_N];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Iterate over K dimension by BLOCK_K tiles
    const int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * BLOCK_K;

        // Load A tile [BLOCK_M x BLOCK_K]
        // Each thread loads THREAD_TILE_M elements from A:
        // rows: row_base + i, column: k0 + tx
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_M; ++i) {
            const int a_row = row_base + i;
            const int a_col = k0 + tx; // tx in [0, BLOCK_THREADS_X-1], but we need [0, BLOCK_K-1].
            // Map tx from [0, BLOCK_THREADS_X-1] to [0, BLOCK_K-1]
            // Here BLOCK_THREADS_X == BLOCK_K (16), so this holds.
            if (a_row < M && a_col < K) {
                As[ty * THREAD_TILE_M + i][tx] = Ap[static_cast<size_t>(a_row) * K + a_col];
            } else {
                As[ty * THREAD_TILE_M + i][tx] = 0.0f;
            }
        }

        // Load B tile [BLOCK_K x BLOCK_N]
        // Each thread loads THREAD_TILE_N elements from B:
        // row: k0 + ty, columns: col_base + j
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            const int b_row = k0 + ty; // ty in [0, BLOCK_THREADS_Y-1], but we need [0, BLOCK_K-1].
            // Map ty from [0, BLOCK_THREADS_Y-1] to [0, BLOCK_K-1]
            // Here BLOCK_THREADS_Y == BLOCK_K (16), so this holds as well.
            const int b_col = col_base + j;
            if (b_row < K && b_col < N) {
                Bs[ty][tx * THREAD_TILE_N + j] = Bp[static_cast<size_t>(b_row) * N + b_col];
            } else {
                Bs[ty][tx * THREAD_TILE_N + j] = 0.0f;
            }
        }

        __syncthreads();

        // Compute this K-tile contribution
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            float a_frag[THREAD_TILE_M];
            float b_frag[THREAD_TILE_N];

            // Load A fragment for this thread
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                a_frag[i] = As[ty * THREAD_TILE_M + i][kk];
            }
            // Load B fragment for this thread
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                b_frag[j] = Bs[kk][tx * THREAD_TILE_N + j];
            }

            // FMA into accumulators
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results to C
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        const int row = row_base + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                const int col = col_base + j;
                if (col < N) {
                    Cp[static_cast<size_t>(row) * N + col] = acc[i][j];
                }
            }
        }
    }
}

at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "A and B must be 3D tensors: [B, M, K] and [B, K, N]");

    // Dimensions
    const int64_t BATCH = A.size(0);
    const int64_t M = A.size(1);
    const int64_t K = A.size(2);
    TORCH_CHECK(B.size(0) == BATCH, "Batch size mismatch between A and B");
    TORCH_CHECK(B.size(1) == K, "Inner dimension K mismatch between A and B");
    const int64_t N = B.size(2);

    // Ensure contiguous memory
    auto A_c = A.contiguous();
    auto B_c = B.contiguous();

    // Allocate output
    auto C = at::empty({BATCH, M, N}, A.options());

    if (BATCH == 0 || M == 0 || N == 0 || K == 0) {
        // Nothing to compute
        return C;
    }

    // Launch configuration
    dim3 block(BLOCK_THREADS_X, BLOCK_THREADS_Y, 1); // 16x16=256 threads
    dim3 grid(
        static_cast<unsigned int>((N + BLOCK_N - 1) / BLOCK_N),
        static_cast<unsigned int>((M + BLOCK_M - 1) / BLOCK_M),
        static_cast<unsigned int>(BATCH)
    );

    hipLaunchKernelGGL(
        bmm_tiled_kernel,
        grid,
        block,
        0, 0,
        A_c.data_ptr<float>(),
        B_c.data_ptr<float>(),
        C.data_ptr<float>(),
        static_cast<int>(BATCH),
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K)
    );

    // Error checking and synchronization
    hipError_t err_sync = hipDeviceSynchronize();
    TORCH_CHECK(err_sync == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err_sync));
    hipError_t err_launch = hipGetLastError();
    TORCH_CHECK(err_launch == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err_launch));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Batched Matrix Multiplication (HIP, tiled, shared memory)");
}