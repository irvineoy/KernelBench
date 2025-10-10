// Batch Normalization 2D
#include <hip/hip_runtime.h>
#include <torch/extension.h>

at::Tensor run(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 supported");

    // BatchNorm2d expects 4D input: (N, C, H, W)
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");

    int64_t num_features = input.size(1);

    // Create running mean/var (initialized to 0 and 1 for inference)
    auto running_mean = at::zeros({num_features}, input.options());
    auto running_var = at::ones({num_features}, input.options());

    // Use PyTorch's native batch_norm
    // training=true, momentum=0.1, eps=1e-5
    auto result = at::batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        /*training=*/true,
        /*momentum=*/0.1,
        /*eps=*/1e-5,
        /*cudnn_enabled=*/false  // Use native implementation
    );

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Batch Normalization 2D");
}
