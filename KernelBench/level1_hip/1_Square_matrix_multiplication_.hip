// Optimized HIP kernel for square matrix multiplication C = A * B
// Supports general MxK times KxN -> MxN (float32), optimized for AMD MI300X (gfx942)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Tile configuration
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

// Threads per block
#define BLOCK_Y 16
#define BLOCK_X 16
#define THREADS_PER_BLOCK (BLOCK_X * BLOCK_Y)

// Per-thread register tile (micro-tile)
#define TM 4
#define TN 4

// Tiled SGEMM kernel: C[M,N] = A[M,K] * B[K,N]
// Row-major layout, contiguous tensors
__global__ __launch_bounds__(THREADS_PER_BLOCK, 2)
void sgemm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc)
{
    // Shared memory tiles with +1 padding to reduce bank conflicts
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    // Block tile origin in output C
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    // Thread coordinates within block
    const int tx = threadIdx.x; // [0, BLOCK_X)
    const int ty = threadIdx.y; // [0, BLOCK_Y)

    // Registers to accumulate a TMxTN micro-tile per thread
    float c_reg[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            c_reg[i][j] = 0.0f;
        }
    }

    // Loop over K in tiles
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A tile: size TILE_M x TILE_K
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            int a_row = block_row + ty * TM + i;
            int a_col = k0 + tx;
            float a_val = 0.0f;
            if (a_row < M && a_col < K) {
                a_val = A[a_row * lda + a_col];
            }
            As[ty * TM + i][tx] = a_val;
        }

        // Load B tile: size TILE_K x TILE_N
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int b_row = k0 + ty;
            int b_col = block_col + tx * TN + j;
            float b_val = 0.0f;
            if (b_row < K && b_col < N) {
                b_val = B[b_row * ldb + b_col];
            }
            Bs[ty][tx * TN + j] = b_val;
        }

        __syncthreads();

        // Compute on the loaded tiles
#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a_frag[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = As[ty * TM + i][kk];
            }
            float b_frag[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = Bs[kk][tx * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    c_reg[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    // Write back results
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = block_row + ty * TM + i;
        if (row >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int col = block_col + tx * TN + j;
            if (col < N) {
                C[row * ldc + col] = c_reg[i][j];
            }
        }
    }
}

// PyTorch entry point: only tensors as arguments (A, B) -> returns C
at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.dtype() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.dtype() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");

    // Ensure contiguous for coalesced access
    auto A_ = A.contiguous();
    auto B_ = B.contiguous();

    const int64_t M64 = A_.size(0);
    const int64_t K64 = A_.size(1);
    TORCH_CHECK(B_.size(0) == K64, "Inner dimensions must match: A(M,K) x B(K,N)");
    const int64_t N64 = B_.size(1);

    TORCH_CHECK(M64 > 0 && N64 > 0 && K64 > 0, "Dimensions must be positive");

    // Allocate output
    auto C = at::empty({M64, N64}, A_.options());

    // Cast sizes to int (safe for typical sizes; 2^31-1 limit)
    const int M = static_cast<int>(M64);
    const int K = static_cast<int>(K64);
    const int N = static_cast<int>(N64);

    const float* A_ptr = A_.data_ptr<float>();
    const float* B_ptr = B_.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Leading dimensions for row-major contiguous tensors
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // Launch configuration
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, 1);

    hipLaunchKernelGGL(
        sgemm_tiled_kernel,
        grid, block, 0, 0,
        A_ptr, B_ptr, C_ptr,
        M, N, K, lda, ldb, ldc
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Optimized matrix multiplication C = A * B (HIP, tiled SGEMM)");
}