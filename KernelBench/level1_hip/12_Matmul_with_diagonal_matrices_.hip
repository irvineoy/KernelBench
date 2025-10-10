// diagmm_hip.hip
// Row-wise scaling: C = diag(A) @ B = (A[:, None] * B)
// Optimized for AMD MI300X (gfx942)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Ceil-div utility
static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

// Scalar kernel: each block processes one row, threads cover columns with stride
__global__ void diagmm_row_scale_scalar_kernel(
    const float* __restrict__ A,   // [N]
    const float* __restrict__ B,   // [N, M]
    float* __restrict__ C,         // [N, M]
    int N,
    int M) {

    int row = blockIdx.y;
    if (row >= N) return;

    float s = A[row];  // scale for this row

    int tid = threadIdx.x;
    int col_start = blockIdx.x * blockDim.x + tid;
    int col_stride = blockDim.x * gridDim.x;

    int base = row * M;

    for (int j = col_start; j < M; j += col_stride) {
        C[base + j] = s * B[base + j];
    }
}

// Vectorized kernel (float4): requires M % 4 == 0 and 16B alignment
__global__ void diagmm_row_scale_vec4_kernel(
    const float* __restrict__ A,   // [N]
    const float* __restrict__ B,   // [N, M]
    float* __restrict__ C,         // [N, M]
    int N,
    int M4 /* M/4 */) {

    int row = blockIdx.y;
    if (row >= N) return;

    float s = A[row];

    int tid = threadIdx.x;
    int vec_start = blockIdx.x * blockDim.x + tid;  // index in float4 units
    int vec_stride = blockDim.x * gridDim.x;

    // Row base in float elements
    int base_elems = row * (M4 * 4);

    // Cast the row pointers to float4* (assumes 16B alignment when used)
    const float4* __restrict__ B4 = reinterpret_cast<const float4*>(B + base_elems);
    float4* __restrict__ C4 = reinterpret_cast<float4*>(C + base_elems);

    for (int v = vec_start; v < M4; v += vec_stride) {
        float4 x = B4[v];
        x.x *= s;
        x.y *= s;
        x.z *= s;
        x.w *= s;
        C4[v] = x;
    }
}

at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.dim() == 1, "A must be 1D (N,)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (N, M)");
    TORCH_CHECK(A.size(0) == B.size(0), "A.size(0) must equal B.size(0)");

    // Ensure dtype float32 and contiguous layout
    if (A.scalar_type() != at::kFloat) A = A.to(at::kFloat);
    if (B.scalar_type() != at::kFloat) B = B.to(at::kFloat);
    if (!A.is_contiguous()) A = A.contiguous();
    if (!B.is_contiguous()) B = B.contiguous();

    int N = static_cast<int>(A.size(0));
    int M = static_cast<int>(B.size(1));

    // Allocate output
    auto C = at::empty_like(B);

    // Launch configuration
    const int block_x = 256;  // multiple of wavefront size (64)
    dim3 block(block_x, 1, 1);
    dim3 grid_scalar(ceil_div_int(M, block_x), N, 1);

    // Try vectorized path if possible
    bool can_vec4 = (M % 4 == 0);
    if (can_vec4) {
        // Check 16-byte alignment for B and C base pointers
        uintptr_t b_addr = reinterpret_cast<uintptr_t>(B.data_ptr<float>());
        uintptr_t c_addr = reinterpret_cast<uintptr_t>(C.data_ptr<float>());
        can_vec4 = ((b_addr % 16) == 0) && ((c_addr % 16) == 0);
    }

    hipError_t err;

    if (can_vec4) {
        int M4 = M / 4;
        dim3 grid_vec(ceil_div_int(M4, block_x), N, 1);

        hipLaunchKernelGGL(
            diagmm_row_scale_vec4_kernel,
            grid_vec, block, 0, 0,
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            M4
        );

        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP launch failed (vec4): ", hipGetErrorString(err));
        err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP kernel failed (vec4): ", hipGetErrorString(err));
    } else {
        hipLaunchKernelGGL(
            diagmm_row_scale_scalar_kernel,
            grid_scalar, block, 0, 0,
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            M
        );

        err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP launch failed (scalar): ", hipGetErrorString(err));
        err = hipDeviceSynchronize();
        TORCH_CHECK(err == hipSuccess, "HIP kernel failed (scalar): ", hipGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "C = diag(A) @ B (row-wise scaling) [HIP]");
}