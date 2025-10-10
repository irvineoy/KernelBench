// Matrix multiplication: C = A.T @ B
// A: (K, M), B: (K, N), C: (M, N)
#include <hip/hip_runtime.h>
#include <torch/extension.h>

at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "Only float32 supported");

    // Use PyTorch's optimized matmul with transpose
    return at::matmul(A.t(), B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Matrix multiplication C = A^T @ B");
}
