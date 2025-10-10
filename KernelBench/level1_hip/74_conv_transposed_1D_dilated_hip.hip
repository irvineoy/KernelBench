// ConvTranspose1d (stride=1, padding=0, dilation=3) optimized HIP kernel for AMD MI300X (gfx942)
//
// Notes:
// - Runtime API receives only tensors (input, weight[, bias]). No scalar config is passed in.
// - We implement ConvTranspose1d with stride=1, padding=0 and dilation=3 (as used in the provided model).
// - Weight layout for ConvTranspose1d in PyTorch: [in_channels, out_channels, kernel_size].
// - Output length = L_in + dilation * (kernel_size - 1).
//
// Build:
//   hipcc -O3 --offload-arch=gfx9-4-generic -I$(python -c "import torch,site; print(site.getsitepackages()[0])")/torch/include -I$(python -c "import torch,site; print(site.getsitepackages()[0])")/torch/include/torch/csrc/api/include -I$(python -c "import torch,site; print(site.getsitepackages()[0])")/torch/include/TH -I$(python -c "import torch,site; print(site.getsitepackages()[0])")/torch/include/THC -fPIC -shared conv_transpose1d_hip.hip -o conv_transpose1d_hip.so
//
// Python usage (via torch.utils.cpp_extension.load):
//   mod = load(name="conv_t1d", sources=["conv_transpose1d_hip.hip"], extra_cuda_cflags=["-O3"], extra_cflags=["-O3"])
//   y = mod.run(x, weight)  # bias is False in the provided model.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Use 256 threads per block (multiple of 64 for MI300X)
#ifndef BLOCK_THREADS
#define BLOCK_THREADS 256
#endif

// Kernel: One thread computes one output element y[n, co, lo].
// We iterate over all input channels ci and kernel taps k. We avoid atomics by
// assigning a unique output element per thread and accumulating locally.
// We cache the K weights for (ci, co, :) into shared memory once per block and reuse.
__global__ __launch_bounds__(BLOCK_THREADS)
void conv_transpose1d_stride1_pad0_dil3_kernel(
    const float* __restrict__ x,        // [N, Cin, Lin]
    const float* __restrict__ w,        // [Cin, Cout, K]
    float* __restrict__ y,              // [N, Cout, Lout]
    int N,
    int Cin,
    int Cout,
    int Lin,
    int K,
    int Lout,
    int dilation)  // fixed to 3 by caller, but kept as a local scalar for flexibility inside
{
    extern __shared__ float s_w[];  // size = K floats

    const int lo = blockIdx.x * blockDim.x + threadIdx.x;  // output position
    if (lo >= Lout) return;

    // Map blockIdx.z to (n, co)
    int linear = blockIdx.z;
    const int n = linear / Cout;
    const int co = linear % Cout;
    if (n >= N) return;

    float acc = 0.0f;

    // Base pointers to current batch within x and y
    const int64_t x_batch_base = static_cast<int64_t>(n) * Cin * (int64_t)Lin;
    const int64_t y_idx = ((int64_t)n * Cout + co) * (int64_t)Lout + lo;

    // Loop over input channels
    for (int ci = 0; ci < Cin; ++ci) {
        // Load weights for this (ci, co, :)
        // weight layout: [Cin, Cout, K]
        const int64_t w_base = ((int64_t)ci * Cout + co) * (int64_t)K;

        // Distribute loading of K weights across threads in the block
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            s_w[k] = w[w_base + k];
        }
        __syncthreads();

        // Accumulate contributions from this input channel
        // y[n,co,lo] += sum_k x[n,ci, li] * w[ci,co,k], where li = lo - k*dilation and 0 <= li < Lin
        // For stride=1, padding=0.
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            int li = lo - k * dilation;
            if (li >= 0 && li < Lin) {
                const int64_t x_idx = x_batch_base + (int64_t)ci * Lin + li;
                acc += x[x_idx] * s_w[k];
            }
        }
        __syncthreads();
    }

    y[y_idx] = acc;
}

// Wrapper visible to PyTorch. IMPORTANT: only tensor parameters are accepted.
// This matches the harness: run(input, weight[, bias]) – here bias=False in the model, so no bias tensor is passed.
at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA/HIP device");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA/HIP device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "This kernel supports float32 tensors only for input");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "This kernel supports float32 tensors only for weight");

    // Ensure contiguous layout
    auto x = input.contiguous();
    auto w = weight.contiguous();

    // Shapes:
    // x: [N, Cin, Lin]
    // w: [Cin, Cout, K]  (PyTorch ConvTranspose1d weight layout)
    TORCH_CHECK(x.dim() == 3, "Input must be 3D: [N, Cin, Lin]");
    TORCH_CHECK(w.dim() == 3, "Weight must be 3D: [Cin, Cout, K]");

    const int N = static_cast<int>(x.size(0));
    const int Cin = static_cast<int>(x.size(1));
    const int Lin = static_cast<int>(x.size(2));
    const int Cin_w = static_cast<int>(w.size(0));
    const int Cout = static_cast<int>(w.size(1));
    const int K = static_cast<int>(w.size(2));
    TORCH_CHECK(Cin_w == Cin, "Weight Cin must match input Cin");

    // Model configuration (not passed as arguments):
    // Based on the target model snippet: stride=1, padding=0, dilation=3, bias=False.
    const int stride = 1;
    const int padding = 0;
    const int dilation = 3;
    (void)stride;
    (void)padding;

    // Output length for ConvTranspose1d: Lout = (Lin - 1) * stride - 2*padding + dilation*(K - 1) + 1
    // Here stride=1, padding=0 -> Lout = Lin + dilation*(K - 1)
    const int Lout = Lin + dilation * (K - 1);

    auto y = at::zeros({N, Cout, Lout}, x.options());

    if (N == 0 || Cout == 0 || Cin == 0 || Lin == 0 || K == 0) {
        // Nothing to do
        return y;
    }

    // Launch configuration
    const int threads = BLOCK_THREADS;  // multiple of 64 (wavefront)
    const dim3 block(threads, 1, 1);
    const dim3 grid((Lout + threads - 1) / threads, 1, static_cast<unsigned int>(N * Cout));

    // Shared memory: K floats per block
    const size_t shared_bytes = static_cast<size_t>(K) * sizeof(float);

    // Launch
    hipStream_t stream = c10::hip::getCurrentHIPStream();
    hipLaunchKernelGGL(
        conv_transpose1d_stride1_pad0_dil3_kernel,
        grid, block, shared_bytes, stream,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        y.data_ptr<float>(),
        N, Cin, Cout, Lin, K, Lout, dilation
    );

    // Error checks
    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    err = hipStreamSynchronize(stream);
    TORCH_CHECK(err == hipSuccess, "HIP kernel execution failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "ConvTranspose1d (stride=1, pad=0, dilation=3) - HIP optimized");
}