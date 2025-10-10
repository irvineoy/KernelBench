// Triangular MatMul with upper-triangular mask: C = triu(A @ B)
// Optimized for AMD MI300X (gfx942) using tiled LDS and masked K-range.
// Single file HIP + PyTorch extension.

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

#ifndef BLOCK_THREADS_X
#define BLOCK_THREADS_X 16
#endif
#ifndef BLOCK_THREADS_Y
#define BLOCK_THREADS_Y 16
#endif

static inline int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

// Tiled upper-triangular matmul kernel
// A, B: [N, N] upper triangular (data outside triangle may be arbitrary, we mask)
// C: [N, N], result = triu(A @ B)
__global__ void triu_matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N)
{
    // Tile coordinates
    const int block_row = blockIdx.y;  // tile along rows (i)
    const int block_col = blockIdx.x;  // tile along cols (j)

    const int i0 = block_row * TILE_M; // starting row index of this tile
    const int j0 = block_col * TILE_N; // starting col index of this tile

    if (i0 >= N || j0 >= N) return;

    // Entire tile strictly below diagonal? Then C[i,j]=0 for all (i,j) in tile; skip.
    if (i0 > j0 + (TILE_N - 1)) return;

    // Determine contributing K-range for this output tile:
    // For upper triangular A and B, for any (i in [i0, i0+M), j in [j0, j0+N)):
    // valid k ∈ [i, j]. Thus union over the tile is [i0, j0+TILE_N-1], intersect [0,N-1]
    int k_start = i0;
    int k_end_excl = j0 + TILE_N;
    if (k_end_excl > N) k_end_excl = N;
    if (k_start < 0) k_start = 0;
    if (k_start >= k_end_excl) {
        // No valid K for this tile (happens if tile is strictly below diagonal)
        return;
    }

    // Shared memory (LDS) tiles with padding to avoid bank conflicts
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int tx = threadIdx.x;  // 0..(BLOCK_THREADS_X-1)
    const int ty = threadIdx.y;  // 0..(BLOCK_THREADS_Y-1)
    const int tid_linear = ty * BLOCK_THREADS_X + tx; // 0..(BLOCK_THREADS_X*BLOCK_THREADS_Y-1) == 0..255

    // Register tile: each thread computes a 4x4 sub-block
    float acc[4][4];
    #pragma unroll
    for (int ii = 0; ii < 4; ++ii) {
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            acc[ii][jj] = 0.0f;
        }
    }

    // Base indices for this thread's 4x4 C-subblock within the tile
    const int row_base_local = ty * 4;  // 0..(TILE_M-1)
    const int col_base_local = tx * 4;  // 0..(TILE_N-1)

    // Loop over K tiles
    for (int kb = k_start; kb < k_end_excl; kb += TILE_K) {
        // Current K tile width (handle tail)
        const int cur_k = (kb + TILE_K <= k_end_excl) ? TILE_K : (k_end_excl - kb);

        // Load A tile: [TILE_M x cur_k] from A[i0:i0+TILE_M, kb:kb+cur_k]
        // Layout in shared: As[row][kk]
        for (int l = tid_linear; l < TILE_M * TILE_K; l += BLOCK_THREADS_X * BLOCK_THREADS_Y) {
            int r = l / TILE_K;     // 0..(TILE_M-1)
            int kk = l % TILE_K;    // 0..(TILE_K-1)
            float val = 0.0f;
            if (kk < cur_k) {
                int i = i0 + r;
                int k = kb + kk;
                // Bounds and upper-triangular mask for A (A[i,k] is non-zero only if k >= i)
                if (i < N && k < N && k >= i) {
                    val = A[i * N + k];
                }
            }
            As[r][kk] = val;
        }

        // Load B tile: [cur_k x TILE_N] from B[kb:kb+cur_k, j0:j0+TILE_N]
        // Layout in shared: Bs[kk][c]
        for (int l = tid_linear; l < TILE_K * TILE_N; l += BLOCK_THREADS_X * BLOCK_THREADS_Y) {
            int kk = l / TILE_N;    // 0..(TILE_K-1)
            int c  = l % TILE_N;    // 0..(TILE_N-1)
            float val = 0.0f;
            if (kk < cur_k) {
                int k = kb + kk;
                int j = j0 + c;
                // Bounds and upper-triangular mask for B (B[k,j] is non-zero only if j >= k)
                if (k < N && j < N && j >= k) {
                    val = B[k * N + j];
                }
            }
            Bs[kk][c] = val;
        }

        __syncthreads();

        // Compute this K tile contribution
        // Local row/col indices within the tile for this thread's 4x4 block
        int r0 = row_base_local + 0;
        int r1 = row_base_local + 1;
        int r2 = row_base_local + 2;
        int r3 = row_base_local + 3;

        int c0 = col_base_local + 0;
        int c1 = col_base_local + 1;
        int c2 = col_base_local + 2;
        int c3 = col_base_local + 3;

        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            if (kk >= cur_k) break;  // uniform across the block

            float a0 = As[r0][kk];
            float a1 = As[r1][kk];
            float a2 = As[r2][kk];
            float a3 = As[r3][kk];

            float b0 = Bs[kk][c0];
            float b1 = Bs[kk][c1];
            float b2 = Bs[kk][c2];
            float b3 = Bs[kk][c3];

            // FMA updates for 4x4 micro-tile
            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[0][2] += a0 * b2;
            acc[0][3] += a0 * b3;

            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
            acc[1][2] += a1 * b2;
            acc[1][3] += a1 * b3;

            acc[2][0] += a2 * b0;
            acc[2][1] += a2 * b1;
            acc[2][2] += a2 * b2;
            acc[2][3] += a2 * b3;

            acc[3][0] += a3 * b0;
            acc[3][1] += a3 * b1;
            acc[3][2] += a3 * b2;
            acc[3][3] += a3 * b3;
        }

        __syncthreads();
    }

    // Write back results with upper-triangular mask (i <= j)
    // Compute global indices
    int gi0 = i0 + row_base_local + 0;
    int gi1 = i0 + row_base_local + 1;
    int gi2 = i0 + row_base_local + 2;
    int gi3 = i0 + row_base_local + 3;

    int gj0 = j0 + col_base_local + 0;
    int gj1 = j0 + col_base_local + 1;
    int gj2 = j0 + col_base_local + 2;
    int gj3 = j0 + col_base_local + 3;

    // Helper lambda to store with mask and bounds
    auto store = [&](int gi, int gj, float v) {
        if (gi < N && gj < N) {
            // Keep only upper triangle
            C[gi * N + gj] = (gi <= gj) ? v : 0.0f;
        }
    };

    store(gi0, gj0, acc[0][0]);
    store(gi0, gj1, acc[0][1]);
    store(gi0, gj2, acc[0][2]);
    store(gi0, gj3, acc[0][3]);

    store(gi1, gj0, acc[1][0]);
    store(gi1, gj1, acc[1][1]);
    store(gi1, gj2, acc[1][2]);
    store(gi1, gj3, acc[1][3]);

    store(gi2, gj0, acc[2][0]);
    store(gi2, gj1, acc[2][1]);
    store(gi2, gj2, acc[2][2]);
    store(gi2, gj3, acc[2][3]);

    store(gi3, gj0, acc[3][0]);
    store(gi3, gj1, acc[3][1]);
    store(gi3, gj2, acc[3][2]);
    store(gi3, gj3, acc[3][3]);
}

// PyTorch entry: ONLY tensor arguments (A, B). Shapes: [N, N], float32
at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same size");

    // Make contiguous views (no extra allocation if already contiguous)
    at::Tensor A_c = A.contiguous();
    at::Tensor B_c = B.contiguous();
    const int64_t N64 = A_c.size(0);
    TORCH_CHECK(N64 <= std::numeric_limits<int>::max(), "Matrix too large");
    const int N = static_cast<int>(N64);

    // Allocate output
    auto C = at::empty({N64, N64}, A_c.options());

    // Launch config
    dim3 block(BLOCK_THREADS_X, BLOCK_THREADS_Y, 1);
    dim3 grid(ceil_div_int(N, TILE_N), ceil_div_int(N, TILE_M), 1);

    // Launch kernel
    hipLaunchKernelGGL(
        triu_matmul_tiled_kernel,
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
    m.def("run", &run, "Upper-triangular matrix multiplication: triu(A @ B) (HIP, tiled)");
}