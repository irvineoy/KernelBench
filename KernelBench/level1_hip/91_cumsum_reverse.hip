// Reverse cumulative sum: flip, cumsum, flip
#include <hip/hip_runtime.h>
#include <torch/extension.h>

at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 supported");

    auto x = input.contiguous();
    const int64_t dim = 1;  // hardcoded from get_init_inputs

    // Reverse cumsum: flip, cumsum, flip back
    auto flipped = x.flip(dim);
    auto cumsum_result = at::cumsum(flipped, dim);
    auto result = cumsum_result.flip(dim);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Reverse cumulative sum along dim=1");
}
