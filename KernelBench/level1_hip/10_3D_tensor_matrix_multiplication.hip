#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef BM
#define BM 64     // Tile size in output rows (R dimension)
#endif
#ifndef BN
#define BN 64     // Tile size in output columns (L dimension)
#endif
#ifndef BK
#define BK 16     // Tile size along K dimension
#endif

#ifndef THREADS_X
#define THREADS_X 16
#endif
#ifndef THREADS_Y
#define THREADS_Y 16
#endif

// 3D tensor-matrix multiplication kernel:
// A: (R, K) where R = N * M (flattened 3D tensor (N, M, K))
// B: (K, L)
// C: (R, L)
__global__ __launch_bounds__(THREADS_X * THREADS_Y)
void tensor3d_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int R,   // total rows = N*M
    int K,   // inner dimension
    int L)   // output columns
{
    // Shared memory tiles with +1 padding to avoid bank conflicts
    __shared__ float sA[BM][BK + 1];
    __shared__ float sB[BK][BN + 1];

    const int tx = threadIdx.x; // [0, THREADS_X-1] = [0,15]
    const int ty = threadIdx.y; // [0, THREADS_Y-1] = [0,15]

    const int blockCol = blockIdx.x; // tile index along L
    const int blockRow = blockIdx.y; // tile index along R

    // Each thread computes a 4x4 micro-tile
    float acc[4][4];
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        #pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
            acc[rr][cc] = 0.0f;
        }
    }

    // Loop over tiles along K
    for (int kt = 0; kt < K; kt += BK) {
        // Load A tile: shape (BM x BK) = (64 x 16)
        // Map 256 threads to load all 1024 elements: 4 phases over rows
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = ty + i * THREADS_Y;      // 0..63
            int kcol = tx;                     // 0..15
            int global_r = blockRow * BM + row;
            int global_k = kt + kcol;
            if (global_r < R && global_k < K) {
                sA[row][kcol] = A[global_r * K + global_k];
            } else {
                sA[row][kcol] = 0.0f;
            }
        }

        // Load B tile: shape (BK x BN) = (16 x 64)
        // Map 256 threads to load all 1024 elements: 4 phases over columns
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int col = tx * 4 + i;              // 0..63
            int krow = ty;                     // 0..15
            int global_c = blockCol * BN + col;
            int global_k = kt + krow;
            if (global_k < K && global_c < L) {
                sB[krow][col] = B[global_k * L + global_c];
            } else {
                sB[krow][col] = 0.0f;
            }
        }

        __syncthreads();

        // Compute on the loaded tiles
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_reg[4];
            a_reg[0] = sA[ty * 4 + 0][kk];
            a_reg[1] = sA[ty * 4 + 1][kk];
            a_reg[2] = sA[ty * 4 + 2][kk];
            a_reg[3] = sA[ty * 4 + 3][kk];

            float b_reg[4];
            b_reg[0] = sB[kk][tx * 4 + 0];
            b_reg[1] = sB[kk][tx * 4 + 1];
            b_reg[2] = sB[kk][tx * 4 + 2];
            b_reg[3] = sB[kk][tx * 4 + 3];

            // FMA into accumulators
            #pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                float aval = a_reg[rr];
                #pragma unroll
                for (int cc = 0; cc < 4; ++cc) {
                    acc[rr][cc] += aval * b_reg[cc];
                }
            }
        }

        __syncthreads();
    }

    // Write back results
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        int global_r = blockRow * BM + ty * 4 + rr;
        if (global_r < R) {
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                int global_c = blockCol * BN + tx * 4 + cc;
                if (global_c < L) {
                    C[global_r * L + global_c] = acc[rr][cc];
                }
            }
        }
    }
}

// PyTorch-visible wrapper. Only tensor parameters are accepted.
// A: [N, M, K], B: [K, L]  -> returns C: [N, M, L]
at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() || A.is_hip(), "Input A must be on HIP device");
    TORCH_CHECK(B.is_cuda() || B.is_hip(), "Input B must be on HIP device");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "Input A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "Input B must be float32");
    TORCH_CHECK(A.dim() == 3, "A must be 3D tensor of shape (N, M, K)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D tensor of shape (K, L)");

    // Ensure contiguous layout
    at::Tensor Acontig = A.contiguous();
    at::Tensor Bcontig = B.contiguous();

    int64_t N = Acontig.size(0);
    int64_t M = Acontig.size(1);
    int64_t K = Acontig.size(2);
    int64_t Kb = Bcontig.size(0);
    int64_t L = Bcontig.size(1);
    TORCH_CHECK(K == Kb, "Inner dimensions must match: A[..., K] vs B[K, ...]");

    int64_t R = N * M;

    auto options = Acontig.options();
    at::Tensor Cout = at::empty({N, M, L}, options);

    const float* A_ptr = Acontig.data_ptr<float>();
    const float* B_ptr = Bcontig.data_ptr<float>();
    float* C_ptr = Cout.data_ptr<float>();

    dim3 block(THREADS_X, THREADS_Y, 1); // 16x16 = 256 threads (multiple of 64)
    dim3 grid(
        static_cast<unsigned int>((L + BN - 1) / BN),
        static_cast<unsigned int>((R + BM - 1) / BM),
        1
    );

    hipLaunchKernelGGL(
        tensor3d_matmul_kernel,
        grid, block, 0, 0,
        A_ptr, B_ptr, C_ptr,
        static_cast<int>(R),
        static_cast<int>(K),
        static_cast<int>(L)
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return Cout;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "3D tensor-matrix multiplication (A[N,M,K] @ B[K,L] -> C[N,M,L]) [HIP]");
}