// Matmul A @ B^T for tensors A[M,K], B[N,K] -> C[M,N]
// Optimized tiled HIP kernel for AMD MI300X (gfx942)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef BM
#define BM 64    // Tile size in M
#endif
#ifndef BN
#define BN 64    // Tile size in N
#endif
#ifndef BK
#define BK 16    // Tile depth in K
#endif

#ifndef TM
#define TM 4     // Per-thread rows
#endif
#ifndef TN
#define TN 4     // Per-thread cols
#endif

// Ensure tile configuration is consistent with thread-block of (16,16)
static_assert(BM == 16 * TM, "BM must equal 16 * TM");
static_assert(BN == 16 * TN, "BN must equal 16 * TN");

// Kernel: computes C = A * B^T
// A: [M,K] row-major contiguous
// B: [N,K] row-major contiguous (we logically use B^T in compute)
// C: [M,N] row-major contiguous
__global__ __launch_bounds__(256)
void matmul_at_bt_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int M, int N, int K)
{
    // Thread/block indices
    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15

    const int block_m = blockIdx.y * BM; // starting row of C tile
    const int block_n = blockIdx.x * BN; // starting col of C tile

    // Shared memory tiles
    // Add +1 padding on the minor dimension to reduce bank conflicts
    __shared__ float sA[BM][BK + 1]; // [64][17]
    __shared__ float sB[BK][BN + 1]; // [16][65] - stores B tile as [BK][BN] (transposed on load)

    // Per-thread accumulator for a TM x TN micro-tile
    float acc[TM][TN];
    #pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < TN; ++ni) {
            acc[mi][ni] = 0.0f;
        }
    }

    // Base output indices for this thread's micro-tile
    const int row0 = block_m + ty * TM;
    const int col0 = block_n + tx * TN;

    // Number of iterations over K
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile [BM x BK] from global to shared:
        // Cover BM*BK = 64*16 = 1024 elements with 256 threads -> 4 loads/thread
        #pragma unroll
        for (int i = 0; i < (BM * BK) / (blockDim.x * blockDim.y); ++i) {
            int linear = (ty * blockDim.x + tx) + i * (blockDim.x * blockDim.y);
            int r = linear / BK;  // 0..BM-1
            int c = linear % BK;  // 0..BK-1

            int g_r = block_m + r;
            int g_c = k0 + c;

            float val = 0.0f;
            if (g_r < M && g_c < K) {
                val = A[g_r * K + g_c];
            }
            sA[r][c] = val;
        }

        // Load B tile (as transposed) [BK x BN] from global to shared:
        // We read from B[j, k] (j in N, k in K), and store into sB[k_local][n_local]
        // Cover BK*BN = 16*64 = 1024 elements with 256 threads -> 4 loads/thread
        #pragma unroll
        for (int i = 0; i < (BK * BN) / (blockDim.x * blockDim.y); ++i) {
            int linear = (ty * blockDim.x + tx) + i * (blockDim.x * blockDim.y);
            int r = linear / BN;  // 0..BK-1  (k local)
            int c = linear % BN;  // 0..BN-1  (n local)

            int g_k = k0 + r;
            int g_j = block_n + c;

            float val = 0.0f;
            if (g_j < N && g_k < K) {
                // B is [N,K] row-major, index: B[j, k] = B[j*K + k]
                val = B[g_j * K + g_k];
            }
            // Store transposed into shared: sB[k_local][n_local]
            sB[r][c] = val;
        }

        __syncthreads();

        // Compute this K block: accumulate into acc[TM][TN]
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // Load a vector of A values for this thread's TM rows
            float a_reg[TM];
            #pragma unroll
            for (int mi = 0; mi < TM; ++mi) {
                int rr = ty * TM + mi; // 0..BM-1
                a_reg[mi] = sA[rr][kk];
            }

            // Load a vector of B values for this thread's TN cols
            float b_reg[TN];
            #pragma unroll
            for (int ni = 0; ni < TN; ++ni) {
                int cc = tx * TN + ni; // 0..BN-1
                b_reg[ni] = sB[kk][cc];
            }

            // Fused multiply-add for micro-tile
            #pragma unroll
            for (int mi = 0; mi < TM; ++mi) {
                float a_val = a_reg[mi];
                #pragma unroll
                for (int ni = 0; ni < TN; ++ni) {
                    acc[mi][ni] = fmaf(a_val, b_reg[ni], acc[mi][ni]);
                }
            }
        }

        __syncthreads();
    }

    // Write back results with bounds checks
    #pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
        int r = row0 + mi;
        if (r >= M) break;
        #pragma unroll
        for (int ni = 0; ni < TN; ++ni) {
            int c = col0 + ni;
            if (c < N) {
                C[r * N + c] = acc[mi][ni];
            }
        }
    }
}

// PyTorch wrapper: inputs are A (M,K) and B (N,K); output is C (M,N) = A @ B^T
at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");

    // Shapes
    int64_t M = A.size(0);
    int64_t K_A = A.size(1);
    int64_t N = B.size(0);
    int64_t K_B = B.size(1);
    TORCH_CHECK(K_A == K_B, "Inner dimension mismatch: A.shape[1] must equal B.shape[1]");
    int64_t K = K_A;

    // Make tensors contiguous (row-major)
    auto A_c = A.contiguous();
    auto B_c = B.contiguous();

    // Allocate output
    auto C = at::empty({M, N}, A.options());

    // Launch config
    dim3 block(16, 16, 1); // 256 threads, multiple of wavefront size (64)
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

    // Launch
    hipLaunchKernelGGL(matmul_at_bt_kernel,
                       grid, block, 0, 0,
                       A_c.data_ptr<float>(),
                       B_c.data_ptr<float>(),
                       C.data_ptr<float>(),
                       static_cast<int>(M),
                       static_cast<int>(N),
                       static_cast<int>(K));

    // Error checking and sync
    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Matmul: C = A @ B^T (HIP, tiled, MI300X-optimized)");
}