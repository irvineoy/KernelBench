// High-performance GEMM (C = A @ B) optimized for tall-skinny K on AMD MI300X (gfx942)
// Input tensors: A [M, K] (row-major), B [K, N] (row-major); Output: C [M, N]
// This file includes both the HIP kernel and the PyTorch binding.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

// Tile sizes tuned for K small (e.g., 16-64) and large M,N
#ifndef BM
#define BM 128   // Block tile size along M (rows of C)
#endif
#ifndef BN
#define BN 64    // Block tile size along N (cols of C)
#endif
#ifndef BK
#define BK 32    // K tile size
#endif

// Threads per block: 16x16 = 256 (multiple of 64 wavefront)
#ifndef TB_X
#define TB_X 16
#endif
#ifndef TB_Y
#define TB_Y 16
#endif

// Per-thread micro-tile: (TM x TN) computed by each thread
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 4
#endif

// Sanity: TB_X * TN == BN and TB_Y * TM == BM
static_assert(TB_X * TN == BN, "TB_X * TN must equal BN");
static_assert(TB_Y * TM == BM, "TB_Y * TM must equal BM");

__global__ __launch_bounds__(TB_X * TB_Y, 4)
void gemm_tall_skinny_kernel(
    const float* __restrict__ A,  // [M, K] row-major
    const float* __restrict__ B,  // [K, N] row-major
    float* __restrict__ C,        // [M, N] row-major
    int M, int K, int N) {

    // Shared memory tiles with +1 padding in the minor dimension to mitigate bank conflicts
    __shared__ float sA[BM][BK + 1];
    __shared__ float sB[BK][BN + 1];

    const int tidx = threadIdx.x;   // [0, TB_X)
    const int tidy = threadIdx.y;   // [0, TB_Y)

    const int block_row = blockIdx.y;  // tile along M
    const int block_col = blockIdx.x;  // tile along N

    const int row_start = block_row * BM;
    const int col_start = block_col * BN;

    // Per-thread offset in the block tile
    const int lrow = tidy * TM;  // local row in sA tile
    const int lcol = tidx * TN;  // local col in sB tile

    // Accumulator registers
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Loop over K dimension in tiles of BK
    for (int kk = 0; kk < K; kk += BK) {
        const int Ktile = min(BK, K - kk);

        // Cooperative load of A tile: [BM x Ktile]
        for (int i = tidy; i < BM; i += TB_Y) {
            const int m = row_start + i;
            // Load along Ktile (minor dim) with coalesced access across x
            for (int j = tidx; j < Ktile; j += TB_X) {
                const int k = kk + j;
                float v = 0.0f;
                if (m < M && k < K) {
                    v = A[m * K + k];
                }
                sA[i][j] = v;
            }
        }

        // Cooperative load of B tile: [Ktile x BN]
        for (int i = tidy; i < Ktile; i += TB_Y) {
            const int k = kk + i;
            for (int j = tidx; j < BN; j += TB_X) {
                const int n = col_start + j;
                float v = 0.0f;
                if (k < K && n < N) {
                    v = B[k * N + n];
                }
                sB[i][j] = v;
            }
        }

        __syncthreads();

        // Compute micro-tile: accumulate over Ktile
        #pragma unroll
        for (int k = 0; k < Ktile; ++k) {
            // Load TM elements from sA for this thread's micro-row block
            float a_reg[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_reg[i] = sA[lrow + i][k];
            }
            // Load TN elements from sB for this thread's micro-col block
            float b_reg[TN];
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_reg[j] = sB[k][lcol + j];
            }
            // FMA into accumulators
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
    }

    // Write back results to C with bounds checks
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int m = row_start + lrow + i;
        if (m < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int n = col_start + lcol + j;
                if (n < N) {
                    C[m * N + n] = acc[i][j];
                }
            }
        }
    }
}

// PyTorch wrapper - ONLY tensor parameters, extract dimensions internally
at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() || A.is_hip(), "A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda() || B.is_hip(), "B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "Only float32 supported for A");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "Only float32 supported for B");

    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();

    TORCH_CHECK(A_contig.dim() == 2 && B_contig.dim() == 2, "Inputs must be 2D matrices");
    const int64_t M64 = A_contig.size(0);
    const int64_t K64 = A_contig.size(1);
    TORCH_CHECK(B_contig.size(0) == K64, "Inner dimensions must match: A[M,K] x B[K,N]");
    const int64_t N64 = B_contig.size(1);

    TORCH_CHECK(M64 <= std::numeric_limits<int>::max(), "M too large");
    TORCH_CHECK(K64 <= std::numeric_limits<int>::max(), "K too large");
    TORCH_CHECK(N64 <= std::numeric_limits<int>::max(), "N too large");

    const int M = static_cast<int>(M64);
    const int K = static_cast<int>(K64);
    const int N = static_cast<int>(N64);

    auto options = A_contig.options();
    auto C = at::empty({M64, N64}, options);

    const float* dA = A_contig.data_ptr<float>();
    const float* dB = B_contig.data_ptr<float>();
    float* dC = C.data_ptr<float>();

    dim3 block(TB_X, TB_Y, 1);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

    hipLaunchKernelGGL(
        gemm_tall_skinny_kernel,
        grid, block, 0, 0,
        dA, dB, dC, M, K, N
    );

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        TORCH_CHECK(false, "HIP kernel failed: ", hipGetErrorString(err));
    }
    err = hipGetLastError();
    if (err != hipSuccess) {
        TORCH_CHECK(false, "HIP kernel launch failed: ", hipGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Tall-skinny GEMM: C = A @ B (HIP optimized for MI300X)");
}