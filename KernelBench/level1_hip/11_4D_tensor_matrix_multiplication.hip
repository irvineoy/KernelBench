// 4D tensor-matrix multiplication HIP kernel for PyTorch:
// Computes C[b,i,j,k] = sum_l A[b,i,j,l] * B[l,k]
//
// Implementation detail:
// - Reshape to GEMM: (M = b*i*j, K = l, N = k)
// - C[M, N] = A[M, K] * B[K, N]
// - Tiled, shared-memory (LDS) optimized GEMM with 8x8 register tiling per thread

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_ASSERTM
#endif

// Tile configuration (multiples of wavefront-friendly sizes)
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 32
#define TM 8   // per-thread rows
#define TN 8   // per-thread cols

// Kernel: GEMM with shared-memory tiling
__global__ void gemm_4d_tensor_mat_kernel(
    const float* __restrict__ A,   // [M, K]
    const float* __restrict__ B,   // [K, N]
    float* __restrict__ C,         // [M, N]
    int M, int N, int K)
{
    // Thread-block coordinates
    int bx = blockIdx.x; // along N
    int by = blockIdx.y; // along M

    // Thread coordinates within block
    int tx = threadIdx.x; // 0..(BLOCK_N/TN - 1)
    int ty = threadIdx.y; // 0..(BLOCK_M/TM - 1)
    int tid = ty * blockDim.x + tx;

    // Starting indices for this block's tile
    int m_block = by * BLOCK_M;
    int n_block = bx * BLOCK_N;

    // Shared memory tiles
    __shared__ float As[BLOCK_M][BLOCK_K + 1]; // +1 to reduce bank conflicts
    __shared__ float Bs[BLOCK_K][BLOCK_N + 1];

    // Register tile for accumulation
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Base local offsets in the block tile for this thread
    const int local_m0 = ty * TM; // 0..BLOCK_M-1
    const int local_n0 = tx * TN; // 0..BLOCK_N-1

    // Loop over K dimension in tiles of BLOCK_K
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Load A tile [BLOCK_M x BLOCK_K] into shared memory
        // Each thread loads multiple elements in a strided loop
        int totalA = BLOCK_M * BLOCK_K; // number of elements
        for (int idx = tid; idx < totalA; idx += blockDim.x * blockDim.y) {
            int row = idx / BLOCK_K; // 0..BLOCK_M-1
            int col = idx % BLOCK_K; // 0..BLOCK_K-1
            int gm = m_block + row;
            int gk = k0 + col;
            float v = 0.0f;
            if (gm < M && gk < K) {
                v = A[gm * K + gk];
            }
            As[row][col] = v;
        }

        // Load B tile [BLOCK_K x BLOCK_N] into shared memory
        int totalB = BLOCK_K * BLOCK_N;
        for (int idx = tid; idx < totalB; idx += blockDim.x * blockDim.y) {
            int row = idx / BLOCK_N; // 0..BLOCK_K-1
            int col = idx % BLOCK_N; // 0..BLOCK_N-1
            int gk = k0 + row;
            int gn = n_block + col;
            float v = 0.0f;
            if (gk < K && gn < N) {
                v = B[gk * N + gn];
            }
            Bs[row][col] = v;
        }

        __syncthreads();

        // Compute on the loaded tiles
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            // Load a strip of A and B for this thread's micro-tile
            float a_reg[TM];
            float b_reg[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_reg[i] = As[local_m0 + i][kk];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_reg[j] = Bs[kk][local_n0 + j];
            }

            // FMA on registers
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                float av = a_reg[i];
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += av * b_reg[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results back to global memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gm = m_block + local_m0 + i;
        if (gm < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int gn = n_block + local_n0 + j;
                if (gn < N) {
                    C[gm * N + gn] = acc[i][j];
                }
            }
        }
    }
}

// Host wrapper: receives tensors only (no scalar config), extracts shapes, launches kernel
at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.dim() == 4, "A must be 4D: (b,i,j,l)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D: (l,k)");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch: A[..., l] must equal B[l, k]");

    // Make sure memory is contiguous for coalesced access
    auto A_ = A.contiguous();
    auto B_ = B.contiguous();

    // Extract shapes
    const int64_t b = A_.size(0);
    const int64_t i = A_.size(1);
    const int64_t j = A_.size(2);
    const int64_t l = A_.size(3); // K dimension of GEMM
    const int64_t k = B_.size(1); // N dimension of GEMM

    // GEMM dims
    const int64_t M64 = b * i * j;
    const int64_t K64 = l;
    const int64_t N64 = k;

    TORCH_CHECK(M64 <= INT_MAX && K64 <= INT_MAX && N64 <= INT_MAX, "Tensor sizes too large for int indexing");
    const int M = static_cast<int>(M64);
    const int K = static_cast<int>(K64);
    const int N = static_cast<int>(N64);

    // Output tensor (b, i, j, k)
    auto C = at::empty({b, i, j, k}, A_.options());

    // Launch configuration
    dim3 block(BLOCK_N / TN, BLOCK_M / TM, 1); // (16,16,1) -> 256 threads
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M,
              1);

    // Flattened pointers for GEMM-style indexing
    const float* A_ptr = A_.data_ptr<float>();
    const float* B_ptr = B_.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    hipLaunchKernelGGL(
        gemm_4d_tensor_mat_kernel,
        grid, block, 0, 0,
        A_ptr, B_ptr, C_ptr, M, N, K
    );

    hipError_t errSync = hipDeviceSynchronize();
    TORCH_CHECK(errSync == hipSuccess, "HIP sync error: ", hipGetErrorString(errSync));
    hipError_t errAsync = hipGetLastError();
    TORCH_CHECK(errAsync == hipSuccess, "HIP launch error: ", hipGetErrorString(errAsync));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "4D tensor-matrix multiplication: C[b,i,j,k] = sum_l A[b,i,j,l] * B[l,k] (HIP)");
}