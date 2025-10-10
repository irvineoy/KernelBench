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
#define TILE_K 16
#endif

// Wavefront-aligned block: 16x16 = 256 threads (multiple of 64)
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// Tiled SGEMM kernel: C = A * B, all matrices are N x N, row-major, float32
// Each block computes a TILE_M x TILE_N tile of C
// Each thread computes a 4x4 micro-tile using registers
__global__ void sgemm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N)
{
    // Shared memory tiles with padding to avoid bank conflicts
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int tx = threadIdx.x;  // 0..15
    const int ty = threadIdx.y;  // 0..15
    const int block_col = blockIdx.x * TILE_N;
    const int block_row = blockIdx.y * TILE_M;

    // Each thread will compute a 4x4 micro tile
    const int row_base = block_row + ty * 4;
    const int col_base = block_col + tx * 4;

    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
    float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
    float c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
    float c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;

    // Loop over tiles of K dimension
    for (int k0 = 0; k0 < N; k0 += TILE_K) {
        // Load A tile into shared memory (coalesced across threads)
        #pragma unroll
        for (int i = 0; i < TILE_M / BLOCK_DIM_Y; ++i) {  // 4 iterations
            int a_row = block_row + ty + i * BLOCK_DIM_Y; // covers 64 rows
            int a_col = k0 + tx;                          // covers 16 cols
            float a_val = 0.0f;
            if (a_row < N && a_col < N) {
                a_val = A[a_row * N + a_col];
            }
            As[ty + i * BLOCK_DIM_Y][tx] = a_val;
        }

        // Load B tile into shared memory (coalesced across threads)
        #pragma unroll
        for (int j = 0; j < TILE_N / BLOCK_DIM_X; ++j) {  // 4 iterations
            int b_row = k0 + ty;                           // covers 16 rows
            int b_col = block_col + tx + j * BLOCK_DIM_X; // covers 64 cols
            float b_val = 0.0f;
            if (b_row < N && b_col < N) {
                b_val = B[b_row * N + b_col];
            }
            Bs[ty][tx + j * BLOCK_DIM_X] = b_val;
        }

        __syncthreads();

        // Compute on the loaded tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            // Load A fragment (4 rows for this thread)
            float a0 = As[ty * 4 + 0][kk];
            float a1 = As[ty * 4 + 1][kk];
            float a2 = As[ty * 4 + 2][kk];
            float a3 = As[ty * 4 + 3][kk];

            // Load B fragment (4 cols for this thread)
            float b0 = Bs[kk][tx * 4 + 0];
            float b1 = Bs[kk][tx * 4 + 1];
            float b2 = Bs[kk][tx * 4 + 2];
            float b3 = Bs[kk][tx * 4 + 3];

            // Outer-product update (FMA-friendly)
            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
        }

        __syncthreads();
    }

    // Write results back to C with bounds checks
    if (row_base + 0 < N) {
        if (col_base + 0 < N) C[(row_base + 0) * N + (col_base + 0)] = c00;
        if (col_base + 1 < N) C[(row_base + 0) * N + (col_base + 1)] = c01;
        if (col_base + 2 < N) C[(row_base + 0) * N + (col_base + 2)] = c02;
        if (col_base + 3 < N) C[(row_base + 0) * N + (col_base + 3)] = c03;
    }
    if (row_base + 1 < N) {
        if (col_base + 0 < N) C[(row_base + 1) * N + (col_base + 0)] = c10;
        if (col_base + 1 < N) C[(row_base + 1) * N + (col_base + 1)] = c11;
        if (col_base + 2 < N) C[(row_base + 1) * N + (col_base + 2)] = c12;
        if (col_base + 3 < N) C[(row_base + 1) * N + (col_base + 3)] = c13;
    }
    if (row_base + 2 < N) {
        if (col_base + 0 < N) C[(row_base + 2) * N + (col_base + 0)] = c20;
        if (col_base + 1 < N) C[(row_base + 2) * N + (col_base + 1)] = c21;
        if (col_base + 2 < N) C[(row_base + 2) * N + (col_base + 2)] = c22;
        if (col_base + 3 < N) C[(row_base + 2) * N + (col_base + 3)] = c23;
    }
    if (row_base + 3 < N) {
        if (col_base + 0 < N) C[(row_base + 3) * N + (col_base + 0)] = c30;
        if (col_base + 1 < N) C[(row_base + 3) * N + (col_base + 1)] = c31;
        if (col_base + 2 < N) C[(row_base + 3) * N + (col_base + 2)] = c32;
        if (col_base + 3 < N) C[(row_base + 3) * N + (col_base + 3)] = c33;
    }
}

// PyTorch entry point: ONLY tensor parameters (A, B) as required.
// Performs C = A @ B for square float32 tensors (N x N).
at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    // Ensure contiguous
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();

    const int64_t N64 = A_contig.size(0);
    TORCH_CHECK(N64 <= std::numeric_limits<int>::max(), "Matrix size too large");
    const int N = static_cast<int>(N64);

    auto options = A_contig.options();
    auto C = at::empty({N64, N64}, options);

    const float* A_ptr = A_contig.data_ptr<float>();
    const float* B_ptr = B_contig.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 grid(
        static_cast<unsigned int>((N + TILE_N - 1) / TILE_N),
        static_cast<unsigned int>((N + TILE_M - 1) / TILE_M),
        1
    );

    hipLaunchKernelGGL(
        sgemm_tiled_kernel,
        grid, block, 0, 0,
        A_ptr, B_ptr, C_ptr, N
    );

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        TORCH_CHECK(false, "HIP device sync failed: ", hipGetErrorString(err));
    }
    err = hipGetLastError();
    if (err != hipSuccess) {
        TORCH_CHECK(false, "HIP kernel launch failed: ", hipGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Optimized SGEMM (C = A @ B) for square matrices (HIP)");
}