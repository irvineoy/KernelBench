// tril_matmul_hip.hip
// High-performance lower-triangular GEMM C = tril(A @ B) for square matrices
// Optimized for AMD MI300X (gfx942) using tiled shared-memory GEMM with triangular-aware pruning.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef TILE_M
#define TILE_M 64
#endif

#ifndef TILE_N
#define TILE_N 64
#endif

#ifndef TILE_K
#define TILE_K 32
#endif

// Ensure block size is a multiple of 64 (wavefront size on AMD)
#define BLOCK_THREADS_X 16
#define BLOCK_THREADS_Y 16
#define BLOCK_THREADS   (BLOCK_THREADS_X * BLOCK_THREADS_Y)

// Shared memory tiles with +1 padding on the second dimension to avoid LDS bank conflicts
// As: [TILE_M x TILE_K], Bs: [TILE_K x TILE_N]
template<typename T>
__global__ __launch_bounds__(BLOCK_THREADS)
void tril_gemm_kernel(const T* __restrict__ A,
                      const T* __restrict__ B,
                      T* __restrict__ C,
                      int N) {
    // Block and thread indices
    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15

    // Tile origin in global coordinates
    const int row0 = blockIdx.y * TILE_M;
    const int col0 = blockIdx.x * TILE_N;

    // If tile is completely out of bounds, exit early
    if (row0 >= N || col0 >= N) return;

    // Triangular pruning: if entire tile lies strictly above the diagonal, skip.
    // C is initialized to zeros on the host side.
    if (col0 > row0 + (TILE_M - 1)) return;

    __shared__ T As[TILE_M][TILE_K + 1];
    __shared__ T Bs[TILE_K][TILE_N + 1];

    // Each thread computes a 4x4 register tile
    T acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = T(0);
        }
    }

    // K tile range pruning based on triangular structure:
    // Non-zero contributions require j <= k <= i.
    // For the entire output tile [row0, row0+TILE_M) x [col0, col0+TILE_N),
    // the only k that can contribute are in [col0, row0+TILE_M-1].
    const int k_begin = max(0, (col0 / TILE_K) * TILE_K); // align down
    const int k_end   = min(N, ((row0 + TILE_M - 1) / TILE_K) * TILE_K + TILE_K); // align up

    for (int k0 = k_begin; k0 < k_end; k0 += TILE_K) {
        // Load As[TILE_M][TILE_K]: each thread loads 8 elements (4 rows x 2 cols)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int ar = ty * 4 + i; // 0..63
            const int gr = row0 + ar;  // global row
            #pragma unroll
            for (int v = 0; v < 2; ++v) {
                const int ak = tx * 2 + v;    // 0..31
                const int gk = k0 + ak;       // global k
                T val = T(0);
                if (gr < N && gk < N) {
                    val = A[gr * N + gk];
                }
                As[ar][ak] = val;
            }
        }

        // Load Bs[TILE_K][TILE_N]: each thread loads 8 elements (2 rows x 4 cols)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int bk = ty * 2 + i; // 0..31
            const int gk = k0 + bk;    // global k
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int bc = tx * 4 + v; // 0..63
                const int gc = col0 + bc;  // global col
                T val = T(0);
                if (gk < N && gc < N) {
                    val = B[gk * N + gc];
                }
                Bs[bk][bc] = val;
            }
        }

        __syncthreads();

        // Compute 4x4 outputs per thread using the loaded tiles
        // Compute coordinates of this thread's 4x4 micro-tile
        const int r_base = ty * 4; // local base row within tile
        const int c_base = tx * 4; // local base col within tile

        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            T a_val[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a_val[i] = As[r_base + i][kk];
            }

            T b_val[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                b_val[j] = Bs[kk][c_base + j];
            }

            // FMA for 4x4 block
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const T a = a_val[i];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a * b_val[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory, masking upper-triangular outputs to zero
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int gr = row0 + ty * 4 + i;
        if (gr >= N) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int gc = col0 + tx * 4 + j;
            if (gc >= N) continue;
            // Lower triangular mask: keep only gc <= gr
            if (gc <= gr) {
                C[gr * N + gc] = acc[i][j];
            }
            // else: do nothing; C was zero-initialized on host
        }
    }
}

// Host wrapper: accepts the same runtime tensors as the PyTorch module (A, B)
at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(), "Inputs must be on the same CUDA/HIP device");
    TORCH_CHECK(A.device().index() == B.device().index(), "Inputs must be on the same device");

    // Enforce float32 and contiguous layout
    auto A_c = A.contiguous().to(torch::kFloat32);
    auto B_c = B.contiguous().to(torch::kFloat32);

    const int64_t N64 = A_c.size(0);
    TORCH_CHECK(N64 <= std::numeric_limits<int>::max(), "Matrix dimension too large");
    const int N = static_cast<int>(N64);

    // Output tensor initialized to zeros so we can skip writing upper-triangular region
    auto C = at::zeros({N64, N64}, A_c.options());

    // Launch configuration
    dim3 block(BLOCK_THREADS_X, BLOCK_THREADS_Y, 1);
    dim3 grid((N + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M, 1);

    // Launch kernel
    hipLaunchKernelGGL(
        (tril_gemm_kernel<float>),
        grid, block, 0, 0,
        A_c.data_ptr<float>(),
        B_c.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err));

    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Lower-triangular matrix multiplication C = tril(A @ B) (HIP, tiled shared memory)");
}