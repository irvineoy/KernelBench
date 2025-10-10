// MSE loss HIP kernel for AMD MI300X (gfx942)
// Computes torch.mean((predictions - targets) ** 2) over all elements
// Single file contains kernels and PyTorch extension bindings.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <iostream>
#include <limits>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256  // multiple of 64 (wavefront)
#endif

// Scalar grid-stride MSE reducer: per-block partial reduced into a single scalar via atomicAdd
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void mse_reduce_scalar_kernel(const float* __restrict__ pred,
                              const float* __restrict__ targ,
                              float* __restrict__ out_mean,  // single-element device tensor
                              long long N,
                              float invN) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    long long idx = blockIdx.x * blockDim.x + tid;
    long long stride = (long long)blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    // Grid-stride loop over all elements
    for (long long i = idx; i < N; i += stride) {
        float d = pred[i] - targ[i];
        local_sum = fmaf(d, d, local_sum);  // local_sum += d*d
    }

    // Block reduction
    sdata[tid] = local_sum;
    __syncthreads();

    // Standard power-of-two reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // One atomic per block, directly accumulating mean contribution
    if (tid == 0) {
        atomicAdd(out_mean, sdata[0] * invN);
    }
}

// Vectorized (float4) MSE reducer with tail handling
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void mse_reduce_vec4_kernel(const float* __restrict__ pred,
                            const float* __restrict__ targ,
                            float* __restrict__ out_mean,  // single-element device tensor
                            long long N,
                            float invN) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    long long idx = blockIdx.x * blockDim.x + tid;
    long long stride = (long long)blockDim.x * gridDim.x;

    const long long N4 = N >> 2;       // N / 4
    const long long tail_start = N4 << 2; // N4 * 4

    const float4* __restrict__ p4 = reinterpret_cast<const float4*>(pred);
    const float4* __restrict__ t4 = reinterpret_cast<const float4*>(targ);

    float local_sum = 0.0f;

    // Vectorized grid-stride loop
    for (long long i4 = idx; i4 < N4; i4 += stride) {
        float4 pv = p4[i4];
        float4 tv = t4[i4];

        float dx0 = pv.x - tv.x;
        float dx1 = pv.y - tv.y;
        float dx2 = pv.z - tv.z;
        float dx3 = pv.w - tv.w;

        // Accumulate using FMA when possible
        local_sum = fmaf(dx0, dx0, local_sum);
        local_sum = fmaf(dx1, dx1, local_sum);
        local_sum = fmaf(dx2, dx2, local_sum);
        local_sum = fmaf(dx3, dx3, local_sum);
    }

    // Tail handling (scalar)
    for (long long i = tail_start + idx; i < N; i += stride) {
        float d = pred[i] - targ[i];
        local_sum = fmaf(d, d, local_sum);
    }

    // Block reduction
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_mean, sdata[0] * invN);
    }
}

// Host wrapper: receives only tensors (predictions, targets), returns scalar tensor (mean MSE)
at::Tensor run(at::Tensor predictions, at::Tensor targets) {
    TORCH_CHECK(predictions.device().is_cuda(), "predictions must be on CUDA/ROCm device");
    TORCH_CHECK(targets.device().is_cuda(), "targets must be on CUDA/ROCm device");
    TORCH_CHECK(predictions.scalar_type() == at::kFloat, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == at::kFloat, "targets must be float32");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "predictions and targets must have the same shape");

    // Ensure contiguous memory for coalesced access
    auto pred = predictions.contiguous();
    auto targ = targets.contiguous();

    const long long N = pred.numel();
    TORCH_CHECK(N > 0, "Input tensors must have at least one element");

    // Output scalar (mean), initialized to 0 on device
    auto options = pred.options();
    at::Tensor out_mean = at::zeros({}, options); // 0-dim scalar tensor on device

    const float invN = 1.0f / static_cast<float>(N);

    // Determine launch parameters
    int device_id = pred.get_device();
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device_id);
    const int numCUs = props.multiProcessorCount > 0 ? props.multiProcessorCount : 304;

    const int threads = BLOCK_SIZE;
    // Aim for ~16-20 blocks per CU, but not exceeding a reasonable bound
    long long target_blocks = static_cast<long long>(numCUs) * 20LL;
    // Use grid-stride loop, so cap blocks sensibly
    long long blocks = std::min(target_blocks, std::max(1LL, (N + threads - 1) / (long long)threads));

    // Choose vectorized kernel if alignment permits and N >= 4
    bool can_vec4 = false;
    {
        uintptr_t p_addr = reinterpret_cast<uintptr_t>(pred.data_ptr<float>());
        uintptr_t t_addr = reinterpret_cast<uintptr_t>(targ.data_ptr<float>());
        can_vec4 = ((p_addr % 16u) == 0u) && ((t_addr % 16u) == 0u) && (N >= 4);
    }

    hipStream_t stream = c10::hip::getCurrentHIPStream();

    if (can_vec4) {
        hipLaunchKernelGGL(
            mse_reduce_vec4_kernel,
            dim3((unsigned int)blocks),
            dim3(threads),
            0, stream,
            pred.data_ptr<float>(),
            targ.data_ptr<float>(),
            out_mean.data_ptr<float>(),
            N,
            invN
        );
    } else {
        hipLaunchKernelGGL(
            mse_reduce_scalar_kernel,
            dim3((unsigned int)blocks),
            dim3(threads),
            0, stream,
            pred.data_ptr<float>(),
            targ.data_ptr<float>(),
            out_mean.data_ptr<float>(),
            N,
            invN
        );
    }

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));
    err = hipStreamSynchronize(stream);
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));

    return out_mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Mean Squared Error (MSE) loss - HIP optimized");
}