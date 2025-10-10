// Optimized HIP kernel for C = A^T * B^T where
// A is (K, M), B is (N, K), output C is (M, N)

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define BM 128   // Block tile size in M dimension
#define BN 128   // Block tile size in N dimension
#define BK 32    // Block tile size in K dimension (depth)
#define TM 8     // Per-thread micro-tile in M
#define TN 8     // Per-thread micro-tile in N

// 16x16 threads per block, each computes 8x8 micro-tile -> 128x128 C tile
#define TB_X 16
#define TB_Y 16
#define THREADS_PER_BLOCK (TB_X * TB_Y)

// Kernel computes C = A^T * B^T without explicitly materializing transposes
// A: [K, M] (row-major), B: [N, K], C: [M, N]
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void gemm_atbt_kernel(const float* __restrict__ A,
                      const float* __restrict__ B,
                      float* __restrict__ C,
                      int M, int N, int K) {
    // Shared memory with +1 padding to reduce bank conflicts
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tid = tidy * blockDim.x + tidx;

    // Tile origin in output C
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    // Per-thread micro-tile origin within the block tile
    const int tm_off = tidy * TM;  // [0, BM)
    const int tn_off = tidx * TN;  // [0, BN)

    // Register accumulators
    float c_reg[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            c_reg[i][j] = 0.0f;
        }
    }

    // Loop over K dimension in tiles of BK
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperative load of As (BM x BK) from A[k, m] with k in [k0, k0+BK), m in [m0, m0+BM)
        // Linearize so that consecutive threads load contiguous memory from A (m varies fastest)
        for (int p = tid; p < BM * BK; p += THREADS_PER_BLOCK) {
            int k_local = p / BM;          // [0, BK)
            int m_local = p - k_local * BM;// [0, BM)
            int k_glob = k0 + k_local;
            int m_glob = m0 + m_local;
            float a_val = 0.0f;
            if (k_glob < K && m_glob < M) {
                // A is [K, M], row-major: index = k_glob * M + m_glob
                a_val = A[k_glob * M + m_glob];
            }
            As[m_local][k_local] = a_val;
        }

        // Cooperative load of Bs (BK x BN) from B[n, k] with k in [k0, k0+BK), n in [n0, n0+BN)
        // Linearize so that consecutive threads load contiguous memory from B (k varies fastest)
        for (int q = tid; q < BK * BN; q += THREADS_PER_BLOCK) {
            int n_local = q / BK;          // [0, BN)
            int k_local = q - n_local * BK;// [0, BK)
            int n_glob = n0 + n_local;
            int k_glob = k0 + k_local;
            float b_val = 0.0f;
            if (n_glob < N && k_glob < K) {
                // B is [N, K], row-major: index = n_glob * K + k_glob
                b_val = B[n_glob * K + k_glob];
            }
            Bs[k_local][n_local] = b_val;
        }

        __syncthreads();

        // Compute on the tile: C[m0:m0+BM, n0:n0+BN] += As * Bs
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // Load one column of As and one row of Bs into registers
            float a_reg[TM];
            float b_reg[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_reg[i] = As[tm_off + i][kk];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_reg[j] = Bs[kk][tn_off + j];
            }

            // Outer product update
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                float a_val = a_reg[i];
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    c_reg[i][j] += a_val * b_reg[j];
                }
            }
        }

        __syncthreads();
    }

    // Write back to C (M x N), guard boundaries
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int m = m0 + tm_off + i;
        if (m < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int n = n0 + tn_off + j;
                if (n < N) {
                    C[m * N + n] = c_reg[i][j];
                }
            }
        }
    }
}

// PyTorch entry point: run(A, B) -> C where
// A: [K, M], B: [N, K], returns C: [M, N]
at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");

    // Ensure contiguous layout
    auto A_ = A.contiguous();
    auto B_ = B.contiguous();

    TORCH_CHECK(A_.dim() == 2 && B_.dim() == 2, "A and B must be 2D");
    int64_t K = A_.size(0);
    int64_t M = A_.size(1);
    int64_t N = B_.size(0);
    TORCH_CHECK(B_.size(1) == K, "Incompatible shapes: A is (K,M), B is (N,K)");

    auto options = A_.options();
    auto C = at::empty({M, N}, options);

    dim3 block(TB_X, TB_Y, 1);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

    hipLaunchKernelGGL(
        gemm_atbt_kernel,
        grid, block, 0, 0,
        A_.data_ptr<float>(),
        B_.data_ptr<float>(),
        C.data_ptr<float>(),
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K)
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));
    err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel execution failed: ", hipGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Matrix multiply C = A^T * B^T (HIP, optimized for MI300X)");
}