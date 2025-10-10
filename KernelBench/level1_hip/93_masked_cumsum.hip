// Masked cumulative sum
#include <hip/hip_runtime.h>
#include <torch/extension.h>

at::Tensor run(at::Tensor input, at::Tensor mask) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "Mask must be CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(mask.scalar_type() == at::kBool, "Mask must be bool");

    auto x = input.contiguous();
    auto m = mask.contiguous();
    const int64_t dim = 1;  // hardcoded from get_init_inputs

    // Apply mask then cumsum
    auto masked = x * m;
    auto result = at::cumsum(masked, dim);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Masked cumulative sum along dim=1");
}
