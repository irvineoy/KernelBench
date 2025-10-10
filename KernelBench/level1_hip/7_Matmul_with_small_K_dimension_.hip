// High-performance GEMM (C = A * B) for small K on AMD MI300X (gfx942)
// Inputs: A [M, K] (row-major), B [K, N] (row-major)
// Output: C [M, N] (row-major)
//
// Build (generic):
//   hipcc -O3 --offload-arch=gfx9-4-generic gemm_smallk.hip -shared -fPIC -o gemm_smallk.so
//
// Build (MI300X specific):
//   hipcc -O3 -march=gfx942 gemm_smallk.hip -shared -fPIC -o gemm_smallk.so

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif

// Block tile sizes optimized for small K
#define BM 64
#define BN 64
#define BK 64

// Per-thread micro-tile
#define TM 4
#define TN 4

// Threads per block (must match launch)
#define TBX 16
#define TBY 16
#define TPB (TBX * TBY)

// Shared memory tiles with padding to reduce bank conflicts
// Padding on the "K" interacting dimension
__global__ void gemm_smallk_kernel(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [M, N]
    int M, int N, int K)
{
    // Block indices (tile coordinates)
    const int block_col = blockIdx.x; // along N
    const int block_row = blockIdx.y; // along M

    // Thread indices
    const int tx = threadIdx.x; // [0, TBX)
    const int ty = threadIdx.y; // [0, TBY)
    const int tid = ty * blockDim.x + tx;

    // Compute starting coordinates for this block in output matrix
    const int row_start = block_row * BM;
    const int col_start = block_col * BN;

    // Register tile for accumulators
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Shared memory tiles
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    // Determine effective K slicing
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int KB = min(BK, K - k0);

        // Load A tile: [BM x KB]
        // Linearized cooperative load
        const int a_tile_elems = BM * KB;
        for (int idx = tid; idx < a_tile_elems; idx += TPB) {
            int r = idx / KB;   // [0, BM)
            int c = idx % KB;   // [0, KB)
            int gr = row_start + r;
            int gc = k0 + c;
            float val = 0.0f;
            if (gr < M && gc < K) {
                val = A[gr * K + gc];
            }
            As[r][c] = val;
        }

        // Load B tile: [KB x BN]
        const int b_tile_elems = KB * BN;
        for (int idx = tid; idx < b_tile_elems; idx += TPB) {
            int r = idx / BN;   // [0, KB)
            int c = idx % BN;   // [0, BN)
            int gr = k0 + r;
            int gc = col_start + c;
            float val = 0.0f;
            if (gr < K && gc < N) {
                val = B[gr * N + gc];
            }
            Bs[r][c] = val;
        }

        __syncthreads();

        // Thread's micro-tile starting coordinates within the block tile
        const int local_row = ty * TM;  // [0, BM)
        const int local_col = tx * TN;  // [0, BN)

        // Compute on the tile
        // Reuse As[local_row..local_row+TM-1][k] and Bs[k][local_col..local_col+TN-1]
        #pragma unroll 8
        for (int k = 0; k < KB; ++k) {
            float aReg[TM];
            float bReg[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                aReg[i] = As[local_row + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                bReg[j] = Bs[k][local_col + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fmaf(aReg[i], bReg[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Store results back to C
    const int global_row_base = row_start + ty * TM;
    const int global_col_base = col_start + tx * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = global_row_base + i;
        if (r < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = global_col_base + j;
                if (c < N) {
                    C[r * N + c] = acc[i][j];
                }
            }
        }
    }
}

// PyTorch-facing API: only tensors allowed as parameters
at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.dtype() == at::kFloat, "Input A must be float32");
    TORCH_CHECK(B.dtype() == at::kFloat, "Input B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match: A[M,K], B[K,N]");

    // Ensure contiguous layout
    A = A.contiguous();
    B = B.contiguous();

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);

    auto C = at::empty({M, N}, A.options());

    dim3 block(TBX, TBY, 1);  // 256 threads/block (multiple of wavefront size 64)
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

    hipLaunchKernelGGL(
        gemm_smallk_kernel,
        grid, block, 0, 0,
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K)
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "GEMM C = A * B optimized for small K (HIP)");
}