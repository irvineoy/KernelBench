// Matrix multiplication: C = A * B  
// A: (M, K), B: (K, N), C: (M, N)
// Large K dimension version
#include <hip/hip_runtime.h>
#include <torch/extension.h>

at::Tensor run(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "Only float32 supported");

    // Use PyTorch's optimized matmul
    return at::matmul(A, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Matrix multiplication (large K)");
}
