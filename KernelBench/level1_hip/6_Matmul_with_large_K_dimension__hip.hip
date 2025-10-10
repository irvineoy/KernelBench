// High-performance HIP GEMM (C = A * B) for large-K matrices on AMD MI300X (gfx942)
// - Input: A [M x K], B [K x N]
// - Output: C [M x N]
// - Optimized for large K using block tiling, LDS (shared memory), and register blocking
// - Only tensor parameters are accepted at the PyTorch boundary (run), shapes inferred from tensors

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <iostream>

// Tiling configuration
#ifndef BM
#define BM 64      // Block tile size in M dimension
#endif
#ifndef BN
#define BN 64      // Block tile size in N dimension
#endif
#ifndef TK
#define TK 32      // K tile depth loaded into LDS per iteration
#endif

// Threads per block (must be multiples of 64 on AMD)
#ifndef THREADS_X
#define THREADS_X 16
#endif
#ifndef THREADS_Y
#define THREADS_Y 16
#endif
static_assert(THREADS_X * THREADS_Y == 256, "Block must have 256 threads");
static_assert(BM % THREADS_Y == 0 && BN % THREADS_X == 0, "BM/BN must be divisible by THREADS_Y/THREADS_X");

// Accumulator micro-tile per thread
#ifndef RM
#define RM (BM / THREADS_Y)   // Rows per thread (e.g., 64/16 = 4)
#endif
#ifndef RN
#define RN (BN / THREADS_X)   // Cols per thread (e.g., 64/16 = 4)
#endif

// Kernel: Tiled SGEMM with LDS and register blocking
__global__
void __launch_bounds__(256)
sgemm_tiled_kernel(
    const float* __restrict__ A,  // [M x K], row-major
    const float* __restrict__ B,  // [K x N], row-major
    float* __restrict__ C,        // [M x N], row-major
    int M, int N, int K,
    int lda, int ldb, int ldc)    // leading dimensions (K, N, N)
{
    // Shared memory tiles with padding to reduce bank conflicts
    __shared__ float As[BM][TK + 1];
    __shared__ float Bs[TK][BN + 1];

    // Thread and block coordinates
    const int tx = threadIdx.x;               // 0..THREADS_X-1
    const int ty = threadIdx.y;               // 0..THREADS_Y-1
    const int tid = ty * blockDim.x + tx;     // 0..255

    const int block_col = blockIdx.x * BN;    // starting N index for this block
    const int block_row = blockIdx.y * BM;    // starting M index for this block

    // Each thread computes a RM x RN micro-tile of C starting at (row_start, col_start)
    const int row_start = block_row + ty * RM;
    const int col_start = block_col + tx * RN;

    // Register accumulator
    float acc[RM][RN];
    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Main loop over K in chunks of TK
    for (int k0 = 0; k0 < K; k0 += TK) {
        // Load A tile [BM x TK] into As
        const int A_elems = BM * TK;  // 64 * 32 = 2048
        for (int idx = tid; idx < A_elems; idx += blockDim.x * blockDim.y) {
            int ai = idx / TK;       // 0..BM-1
            int ak = idx % TK;       // 0..TK-1
            int g_row = block_row + ai;
            int g_k   = k0 + ak;
            float v = 0.0f;
            if (g_row < M && g_k < K) {
                v = A[(size_t)g_row * lda + g_k];
            }
            As[ai][ak] = v;
        }

        // Load B tile [TK x BN] into Bs
        const int B_elems = TK * BN; // 32 * 64 = 2048
        for (int idx = tid; idx < B_elems; idx += blockDim.x * blockDim.y) {
            int bk = idx / BN;       // 0..TK-1 (K index within tile)
            int bj = idx % BN;       // 0..BN-1 (N index within tile)
            int g_k   = k0 + bk;
            int g_col = block_col + bj;
            float v = 0.0f;
            if (g_k < K && g_col < N) {
                v = B[(size_t)g_k * ldb + g_col];
            }
            Bs[bk][bj] = v;
        }

        __syncthreads();

        // Compute on the loaded tiles
        #pragma unroll 4
        for (int kk = 0; kk < TK; ++kk) {
            // Load RM elements of A from shared for this thread's rows
            float a_frag[RM];
            #pragma unroll
            for (int i = 0; i < RM; ++i) {
                int ar = ty * RM + i;  // local row inside As
                a_frag[i] = As[ar][kk];
            }

            // Load RN elements of B from shared for this thread's cols
            float b_frag[RN];
            #pragma unroll
            for (int j = 0; j < RN; ++j) {
                int bc = tx * RN + j;  // local col inside Bs
                b_frag[j] = Bs[kk][bc];
            }

            // FMA accumulate
            #pragma unroll
            for (int i = 0; i < RM; ++i) {
                #pragma unroll
                for (int j = 0; j < RN; ++j) {
                    acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Write back the results
    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        int r = row_start + i;
        if (r < M) {
            #pragma unroll
            for (int j = 0; j < RN; ++j) {
                int c = col_start + j;
                if (c < N) {
                    C[(size_t)r * ldc + c] = acc[i][j];
                }
            }
        }
    }
}

// PyTorch wrapper: accepts only tensor parameters (A, B); shapes inferred
at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA/ROCm tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA/ROCm tensor");
    TORCH_CHECK(A.dtype() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.dtype() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");

    auto A_c = A.contiguous();
    auto B_c = B.contiguous();

    int M = static_cast<int>(A_c.size(0));
    int K = static_cast<int>(A_c.size(1));
    TORCH_CHECK(B_c.size(0) == K, "Inner dimensions must match: A[M,K], B[K,N]");

    int N = static_cast<int>(B_c.size(1));

    auto options = A_c.options();
    at::Tensor C = at::empty({M, N}, options);

    // Leading dimensions for row-major storage
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Launch configuration
    dim3 block(THREADS_X, THREADS_Y, 1);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

    // Launch kernel on the current stream
    hipStream_t stream = c10::hip::getCurrentHIPStream();

    hipLaunchKernelGGL(
        sgemm_tiled_kernel,
        grid, block, 0, stream,
        A_c.data_ptr<float>(),
        B_c.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K, lda, ldb, ldc
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    err = hipStreamSynchronize(stream);
    TORCH_CHECK(err == hipSuccess, "HIP kernel execution failed: ", hipGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Optimized SGEMM (C = A * B) for large-K on AMD MI300X (HIP)");
}