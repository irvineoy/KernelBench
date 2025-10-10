// Copyright (c) 2025
// High-performance HIP GEMM kernel for C = A(MxK) * B(KxN)
// Optimized for AMD MI300X (gfx942), wavefront=64

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

// Block-tiled GEMM parameters
// 64x64 output tile per block, K is processed in tiles of 32
// 16x16 threads per block, each thread computes a 4x4 micro-tile
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int TM = 4;
constexpr int TN = 4;

__global__ void gemm_tiled_kernel_f32(
    const float* __restrict__ A,   // [M, K]
    const float* __restrict__ B,   // [K, N]
    float* __restrict__ C,         // [M, N]
    int M, int N, int K,
    int lda, int ldb, int ldc) {

    // Shared memory tiles with +1 padding to avoid LDS bank conflicts
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    // Block tile origin in C
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int m0 = block_row * BM;
    const int n0 = block_col * BN;

    // Thread indices
    const int tx = threadIdx.x;  // 0..15
    const int ty = threadIdx.y;  // 0..15
    const int tid = ty * blockDim.x + tx; // 0..255

    // Registers for the per-thread output micro-tile
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Global row/col bases for this thread's micro-tile
    const int row_base = m0 + ty * TM;
    const int col_base = n0 + tx * TN;

    // Loop over K dimension in tiles of BK
    for (int k0 = 0; k0 < K; k0 += BK) {
        // 1) Load A tile: size BM x BK
        // Each of the 256 threads loads 8 elements (since 64*32 / 256 = 8)
        for (int idx = tid; idx < BM * BK; idx += blockDim.x * blockDim.y) {
            int a_r = idx / BK;  // 0..BM-1
            int a_c = idx % BK;  // 0..BK-1
            int g_r = m0 + a_r;
            int g_c = k0 + a_c;
            float v = 0.0f;
            if (g_r < M && g_c < K) {
                v = A[g_r * lda + g_c];
            }
            As[a_r][a_c] = v;
        }

        // 2) Load B tile: size BK x BN
        for (int idx = tid; idx < BK * BN; idx += blockDim.x * blockDim.y) {
            int b_r = idx / BN;  // 0..BK-1
            int b_c = idx % BN;  // 0..BN-1
            int g_r = k0 + b_r;
            int g_c = n0 + b_c;
            float v = 0.0f;
            if (g_r < K && g_c < N) {
                v = B[g_r * ldb + g_c];
            }
            Bs[b_r][b_c] = v;
        }

        __syncthreads();

        // 3) Compute on the loaded tiles
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // Load one column of A micro-tile and one row of B micro-tile into registers
            float a_reg[TM];
            float b_reg[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int r = ty * TM + i; // 0..BM-1
                a_reg[i] = As[r][kk];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = tx * TN + j; // 0..BN-1
                b_reg[j] = Bs[kk][c];
            }

            // FMA on the micro-tile
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

    // 4) Write back the results with bounds checks
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = row_base + i;
        if (r >= M) break;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int c_col = col_base + j;
            if (c_col < N) {
                C[r * ldc + c_col] = acc[i][j];
            }
        }
    }
}

// Host wrapper: accepts only tensors (A, B). No scalar config is passed.
at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");

    // Make contiguous for coalesced access
    auto A_c = A.contiguous();
    auto B_c = B.contiguous();

    const int64_t M64 = A_c.size(0);
    const int64_t K64 = A_c.size(1);
    const int64_t Kb  = B_c.size(0);
    const int64_t N64 = B_c.size(1);

    TORCH_CHECK(K64 == Kb, "Inner dimensions must match: A(MxK) * B(KxN)");

    // Cast to int (sizes fit in int range for this problem)
    const int M = static_cast<int>(M64);
    const int K = static_cast<int>(K64);
    const int N = static_cast<int>(N64);

    // Allocate output
    auto C = at::empty({M64, N64}, A_c.options());

    const float* A_ptr = A_c.data_ptr<float>();
    const float* B_ptr = B_c.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Leading dimensions for row-major layout
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    dim3 block(16, 16, 1); // 256 threads (multiple of wavefront=64)
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

    hipLaunchKernelGGL(
        gemm_tiled_kernel_f32,
        grid, block, 0, 0,
        A_ptr, B_ptr, C_ptr,
        M, N, K, lda, ldb, ldc
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err));

    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "GEMM: C = A * B (HIP, tiled, f32)");
}