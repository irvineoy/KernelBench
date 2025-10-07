#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_SIZE 16

// Simple tiled matrix multiplication kernel
__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tiles into shared memory
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        As[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // Compute partial result
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

at::Tensor matmul_hip(at::Tensor A, at::Tensor B) {
    int N = A.size(0);
    auto C = at::zeros_like(A);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    hipLaunchKernelGGL(matmul_kernel, grid, block, 0, 0,
                       A.data_ptr<float>(), B.data_ptr<float>(),
                       C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul_hip, "Square matrix multiplication (HIP)");
}
