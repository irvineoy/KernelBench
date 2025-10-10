// matrix_scalar_mul.hip
// High-performance matrix-scalar multiplication C = A * s for AMD GPUs (MI300X/gfx942)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <cstdint>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Grid-stride, coalesced elementwise scaling kernel
__global__ void scale_kernel_f32(const float* __restrict__ A,
                                 float* __restrict__ C,
                                 long long size,
                                 float alpha) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    // Unroll by 4 to reduce loop overhead
    for (long long i = idx; i < size; i += stride * 4) {
        long long i1 = i;
        long long i2 = i + stride;
        long long i3 = i + 2 * stride;
        long long i4 = i + 3 * stride;

        if (i1 < size) C[i1] = A[i1] * alpha;
        if (i2 < size) C[i2] = A[i2] * alpha;
        if (i3 < size) C[i3] = A[i3] * alpha;
        if (i4 < size) C[i4] = A[i4] * alpha;
    }
}

// Host wrapper: accepts tensor and scalar parameters.
// Inputs: A (Tensor [M, N] on HIP device), s (scalar float value)
at::Tensor run(at::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA/HIP tensor");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "Only float32 tensors are supported");

    // Ensure contiguous memory for coalesced access
    at::Tensor A_c = A.contiguous();

    // Use the scalar value directly
    float alpha = s;

    auto out = at::empty_like(A_c);

    const float* A_ptr = A_c.data_ptr<float>();
    float* C_ptr = out.data_ptr<float>();

    long long size = A_c.numel();

    // Launch configuration: multiple of 64 threads per block
    int threads = 256;  // Good balance for MI300X (4 waves/block)
    // Cap grid to avoid oversubscription; grid-stride loop covers the rest
    long long blocks_ll = (size + threads - 1) / threads;
    int blocks = static_cast<int>(blocks_ll > 65535 ? 65535 : blocks_ll);

    hipStream_t stream = at::hip::getCurrentHIPStream();

    hipLaunchKernelGGL(scale_kernel_f32,
                       dim3(blocks),
                       dim3(threads),
                       0,
                       stream,
                       A_ptr,
                       C_ptr,
                       size,
                       alpha);

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel execution failed: ", hipGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Matrix-scalar multiplication C = A * s (HIP, float32)");
}