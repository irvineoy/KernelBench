// Hinge Loss Mean kernel for AMD GPUs (HIP, optimized for MI300X)
// Computes: mean( clamp(1 - predictions * targets, min=0) )
// Inputs:
//   predictions: float tensor of shape (B, N_total_per_sample...) [flattened to (B, N)]
//   targets:     float tensor of shape (B,), with values in {-1, 1}
// Output:
//   scalar tensor (0-dim) with the mean hinge loss
//
// Build:
//   hipcc -O3 --offload-arch=gfx942 hinge_loss_mean.hip -shared -fPIC -o hinge_loss_mean.so
//
// PyTorch extension entrypoint: run(predictions, targets)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <algorithm>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256  // multiple of 64 (wavefront size)
#endif

// Block-level reduction into a single float using LDS
template <int BS>
__device__ inline float block_reduce_sum(float val) {
    __shared__ float sdata[BS];
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    // Standard tree reduction
    for (int s = BS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    return sdata[0];  // valid only for tid == 0
}

// Kernel: compute sum of hinge loss over all elements, then scale by 1/total on host or via scaling factor
__global__ void hinge_loss_mean_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ out_sum,    // single-element device buffer (accumulator)
    long long total_elems,          // total elements in predictions (B*N)
    long long N,                    // elements per batch row
    float inv_total                 // 1.0f / (B*N)
) {
    const long long global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;

    float thread_acc = 0.0f;

    // Grid-stride loop over flattened predictions
    for (long long idx = global_tid; idx < total_elems; idx += stride) {
        // Determine sample index b and read its target
        long long b = idx / N;  // 64-bit division; acceptable relative to memory traffic
        float t = targets[b];

        float p = predictions[idx];
        float v = 1.0f - p * t;
        // clamp min=0
        thread_acc += (v > 0.0f) ? v : 0.0f;
    }

    // Block reduction
    float block_sum = block_reduce_sum<BLOCK_SIZE>(thread_acc);

    if (threadIdx.x == 0) {
        // Accumulate normalized contribution to the single-element output
        atomicAdd(out_sum, block_sum * inv_total);
    }
}

at::Tensor run(at::Tensor predictions, at::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA/HIP tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA/HIP tensor");
    TORCH_CHECK(predictions.scalar_type() == at::kFloat, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == at::kFloat, "targets must be float32");
    TORCH_CHECK(predictions.dim() >= 1, "predictions must have at least 1 dimension");
    TORCH_CHECK(targets.dim() == 1, "targets must be 1D of shape (B,)");

    // Ensure contiguous memory for coalesced access
    auto preds = predictions.contiguous();
    auto targs = targets.contiguous();

    const long long B = preds.size(0);
    TORCH_CHECK(targs.size(0) == B, "targets.size(0) must match predictions.size(0)");
    const long long total_elems = preds.numel();
    TORCH_CHECK(B > 0 && total_elems > 0, "empty tensors are not supported");

    // Flatten predictions to (B, N)
    const long long N = total_elems / B;
    TORCH_CHECK(B * N == total_elems, "predictions size(0) must evenly divide numel()");

    // Output scalar on same device/dtype
    auto options = preds.options();
    auto out = at::zeros({}, options);  // 0-dim scalar initialized to 0

    const float inv_total = 1.0f / static_cast<float>(total_elems);

    // Launch configuration
    const int block = BLOCK_SIZE;
    // Limit grid size to a reasonable maximum for good occupancy; grid-stride loop will cover all data
    const long long max_blocks = 16384;  // tuned upper bound for MI300X; avoids oversubscription overhead
    long long suggested = (total_elems + block - 1) / block;
    int grid = static_cast<int>(std::min<long long>(suggested, max_blocks));
    if (grid < 1) grid = 1;

    hipLaunchKernelGGL(
        hinge_loss_mean_kernel,
        dim3(grid), dim3(block), 0, 0,
        preds.data_ptr<float>(),
        targs.data_ptr<float>(),
        out.data_ptr<float>(),
        total_elems,
        N,
        inv_total
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Hinge Loss Mean (HIP, optimized for MI300X)");
}