// High-performance HIP GEMM kernel for PyTorch: C = A * B
// Optimized for AMD MI300X (gfx942)
// Single file contains kernel, wrapper, and PyTorch bindings

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>
#include <stdint.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Tile configuration
#define TILE_M 128
#define TILE_N 128
#define TILE_K 32

// Thread block configuration (must be multiple of 64)
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define THREADS_PER_BLOCK (BLOCK_DIM_X * BLOCK_DIM_Y)

// Shared memory tiles with padding (+1) to reduce LDS bank conflicts
// Total LDS per block ~33 KB
__global__ void gemm_tiled_128x128x32_kernel(
    const float* __restrict__ A,  // [M, K], row-major
    const float* __restrict__ B,  // [K, N], row-major
    float* __restrict__ C,        // [M, N], row-major
    int M, int N, int K) {

    // Block and thread indices
    const int tx = threadIdx.x;         // [0, BLOCK_DIM_X)
    const int ty = threadIdx.y;         // [0, BLOCK_DIM_Y)
    const int tid = ty * BLOCK_DIM_X + tx;

    const int block_row = blockIdx.y;   // tile along M
    const int block_col = blockIdx.x;   // tile along N

    const int row_start = block_row * TILE_M;
    const int col_start = block_col * TILE_N;

    // Shared memory tiles (LDS)
    __shared__ float As[TILE_M][TILE_K + 1];  // [128][33]
    __shared__ float Bs[TILE_K][TILE_N + 1];  // [32][129]

    // Register tile (micro-tile 8x8 per thread)
    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Compute coordinates of this thread's micro-tile within the 128x128 C-block
    const int c_row_base = ty * 8;  // 0..120 step 8
    const int c_col_base = tx * 8;  // 0..120 step 8

    // Iterate over K dimension tiles
    for (int k0 = 0; k0 < K; k0 += TILE_K) {

        // Fast path: full tiles entirely within bounds
        const bool fullTile = (row_start + TILE_M <= M) &&
                              (col_start + TILE_N <= N) &&
                              (k0 + TILE_K   <= K);

        // Cooperative load of A tile: size TILE_M x TILE_K (128x32) into As
        // 4096 values => 1024 float4 vectors => 4 vector loads per thread
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            int a_vec_idx = tid + it * THREADS_PER_BLOCK; // 0..1023
            int a_row = a_vec_idx / (TILE_K / 4);         // /8 => 0..127
            int a_kvec = a_vec_idx % (TILE_K / 4);        // 0..7
            int g_row = row_start + a_row;
            int g_col = k0 + a_kvec * 4;

            if (fullTile) {
                // Vectorized load from global A and scatter into shared
                const float4 va = *reinterpret_cast<const float4 const*>(
                    &A[g_row * K + g_col]);
                As[a_row][a_kvec * 4 + 0] = va.x;
                As[a_row][a_kvec * 4 + 1] = va.y;
                As[a_row][a_kvec * 4 + 2] = va.z;
                As[a_row][a_kvec * 4 + 3] = va.w;
            } else {
                // Bounds-checked scalar loads (handles ragged tiles)
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    int gc = g_col + t;
                    float v = 0.0f;
                    if (g_row < M && gc < K) {
                        v = A[g_row * K + gc];
                    }
                    As[a_row][a_kvec * 4 + t] = v;
                }
            }
        }

        // Cooperative load of B tile: size TILE_K x TILE_N (32x128) into Bs
        // 4096 values => 1024 float4 vectors => 4 vector loads per thread
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            int b_vec_idx = tid + it * THREADS_PER_BLOCK; // 0..1023
            int b_row = b_vec_idx / (TILE_N / 4);         // /32 => 0..31
            int b_cvec = b_vec_idx % (TILE_N / 4);        // 0..31
            int g_row = k0 + b_row;
            int g_col = col_start + b_cvec * 4;

            if (fullTile) {
                // Vectorized load from global B and scatter into shared
                const float4 vb = *reinterpret_cast<const float4 const*>(
                    &B[g_row * N + g_col]);
                Bs[b_row][b_cvec * 4 + 0] = vb.x;
                Bs[b_row][b_cvec * 4 + 1] = vb.y;
                Bs[b_row][b_cvec * 4 + 2] = vb.z;
                Bs[b_row][b_cvec * 4 + 3] = vb.w;
            } else {
                // Bounds-checked scalar loads
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    int gc = g_col + t;
                    float v = 0.0f;
                    if (g_row < K && gc < N) {
                        v = B[g_row * N + gc];
                    }
                    Bs[b_row][b_cvec * 4 + t] = v;
                }
            }
        }

        __syncthreads();

        // Compute on the loaded tiles
        // Each thread computes an 8x8 micro-tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float aFrag[8];
            float bFrag[8];

            // Load A fragment (8 rows) for this thread's micro-tile
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                aFrag[i] = As[c_row_base + i][kk];
            }
            // Load B fragment (8 cols) for this thread's micro-tile
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                bFrag[j] = Bs[kk][c_col_base + j];
            }

            // FMA: 8x8 outer-product
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float a_val = aFrag[i];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[i][j] += a_val * bFrag[j];
                }
            }
        }

        __syncthreads();
    }

    // Store results back to C with bounds checks
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int gr = row_start + c_row_base + i;
        if (gr >= M) break;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int gc = col_start + c_col_base + j;
            if (gc >= N) break;
            C[gr * N + gc] = acc[i][j];
        }
    }
}

// PyTorch entry point: only tensor parameters (A, B) matching get_inputs()
at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

    // Make contiguous for coalesced/vectorized access
    auto Acontig = A.contiguous();
    auto Bcontig = B.contiguous();

    const int64_t M64 = Acontig.size(0);
    const int64_t K64 = Acontig.size(1);
    const int64_t Kb64 = Bcontig.size(0);
    const int64_t N64 = Bcontig.size(1);

    TORCH_CHECK(K64 == Kb64, "Inner dimensions must match: A[M,K] * B[K,N]");

    // Use 32-bit indices internally (sizes here are well within int range)
    const int M = static_cast<int>(M64);
    const int K = static_cast<int>(K64);
    const int N = static_cast<int>(N64);

    auto options = Acontig.options();
    at::Tensor C = at::empty({M64, N64}, options);

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M,
              1);

    hipLaunchKernelGGL(
        gemm_tiled_128x128x32_kernel,
        grid, block, 0, 0,
        Acontig.data_ptr<float>(),
        Bcontig.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K);

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        TORCH_CHECK(false, "HIP kernel failed: ", hipGetErrorString(err));
    }
    err = hipGetLastError();
    if (err != hipSuccess) {
        TORCH_CHECK(false, "HIP kernel launch failed: ", hipGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Optimized GEMM C = A * B (HIP, MI300X)");
}