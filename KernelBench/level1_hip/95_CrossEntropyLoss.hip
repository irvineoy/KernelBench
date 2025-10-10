// Cross Entropy Loss (mean reduction) for 2D inputs (N x C)
// Optimized for AMD MI300X (gfx942)
// Build: hipcc -O3 --offload-arch=gfx9-4-generic cross_entropy.hip -shared -fPIC -o cross_entropy.so

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cfloat>
#include <cstdint>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256  // Must be multiple of 64 (wavefront size on AMD)
#endif

// Each block processes one row (one sample) of logits
// Computes numerically-stable -log_softmax(logits[row])[target[row]]
// and atomically accumulates (loss / N) into out[0].
// logits: [N, C], row-major, float32
// targets: [N], int64
// out: scalar buffer (size=1), float32
__global__ void __launch_bounds__(BLOCK_SIZE)
cross_entropy_mean_kernel(const float* __restrict__ logits,
                          const int64_t* __restrict__ targets,
                          float* __restrict__ out,
                          int N, int C, float invN) {
    int row = blockIdx.x;
    if (row >= N) return;

    // Shared memory for reductions
    __shared__ float sdata[BLOCK_SIZE];
    __shared__ float row_max_shared;

    int tid = threadIdx.x;

    // 1) Compute row-wise max for numerical stability
    float local_max = -FLT_MAX;
    // Strided loop over classes for this row
    for (int j = tid; j < C; j += blockDim.x) {
        float v = logits[row * C + j];
        local_max = fmaxf(local_max, v);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Reduce max
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        row_max_shared = sdata[0];
    }
    __syncthreads();

    const float row_max = row_max_shared;

    // 2) Compute sum(exp(logits - row_max))
    float local_sum = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        float v = logits[row * C + j] - row_max;
        local_sum += expf(v);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 3) Thread 0 computes logsumexp and per-row loss, then atomically adds mean contribution
    if (tid == 0) {
        float lse = logf(sdata[0]) + row_max;  // logsumexp for this row
        int64_t t = targets[row];

        // Guard against invalid target values (not expected in this benchmark)
        if (t >= 0 && t < (int64_t)C) {
            float chosen = logits[row * C + (int)t];
            float loss = -(chosen - lse);      // -log_softmax
            atomicAdd(out, loss * invN);       // accumulate mean in-place
        }
        // If invalid, skip contribution (matches ignore behavior for out-of-range)
    }
}

// C++/PyTorch wrapper: accepts only runtime tensors (inputs), no scalars
// Signature must match get_inputs(): predictions, targets
at::Tensor run(at::Tensor predictions, at::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be on CUDA/HIP device");
    TORCH_CHECK(targets.is_cuda(), "targets must be on CUDA/HIP device");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D [N, C]");
    TORCH_CHECK(targets.dim() == 1, "targets must be 1D [N]");
    TORCH_CHECK(predictions.size(0) == targets.size(0), "Batch size mismatch between predictions and targets");
    TORCH_CHECK(predictions.scalar_type() == at::kFloat, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == at::kLong, "targets must be int64");

    // Extract dimensions
    const int N = static_cast<int>(predictions.size(0));
    const int C = static_cast<int>(predictions.size(1));

    // Allocate scalar output on device, initialized to 0
    auto out = at::zeros({}, predictions.options());

    // Launch configuration
    dim3 block(BLOCK_SIZE);
    dim3 grid(N);  // One block per row/sample

    const float invN = 1.0f / static_cast<float>(N);

    // Launch kernel on default stream
    hipLaunchKernelGGL(
        cross_entropy_mean_kernel,
        grid, block, 0, 0,
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        N, C, invN
    );

    // Error checking and sync
    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Cross Entropy Loss (mean) for 2D logits (HIP, MI300X-optimized)");
}