// hip_kl_div_batchmean.hip
// Computes KLDiv(target || prediction) with reduction='batchmean'.
// Input tensors are probabilities (softmax outputs). We compute log on-the-fly.
//
// Build example:
// hipcc -O3 --offload-arch=gfx942 -I$(python -c "import torch; import os; print(os.path.dirname(torch.__file__)+'/include')") \
//      -I$(python -c "import torch; import os; print(os.path.dirname(torch.__file__)+'/include/torch/csrc/api/include')") \
//      -fPIC -shared hip_kl_div_batchmean.hip -o kl_divhip.so

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256  // multiple of 64
#endif

// Block reduction into shared memory (double)
__device__ inline double block_reduce_sum_double(double val) {
    __shared__ double sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    // Binary tree reduction
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    return sdata[0]; // Only valid for tid==0
}

// Scalar path: grid-stride loop over all elements
__global__ void kl_div_sum_scalar_kernel(
    const float* __restrict__ preds,   // probabilities
    const float* __restrict__ targets, // probabilities
    double* __restrict__ out_sum,      // single double accumulator
    int64_t N                           // total elements
) {
    int64_t gid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t gstride = (int64_t)blockDim.x * gridDim.x;

    double local = 0.0;

    for (int64_t i = gid; i < N; i += gstride) {
        float p = preds[i];
        float t = targets[i];
        // KL component: t * (log(t) - log(p))
        // input to log is > 0 for softmax outputs; no clamping to match PyTorch
        local += static_cast<double>(t) * (logf(t) - logf(p));
    }

    // Reduce within block
    double block_sum = block_reduce_sum_double(local);
    if (threadIdx.x == 0) {
        atomicAdd(out_sum, block_sum);
    }
}

// Vectorized path with float4 loads (requires 16B alignment and N % 4 == 0)
__global__ void kl_div_sum_vec4_kernel(
    const float4* __restrict__ preds4,
    const float4* __restrict__ targets4,
    double* __restrict__ out_sum,
    int64_t N4,      // number of float4 elements
    const float* __restrict__ preds_tail,   // for possible scalar tail (unused when N%4==0)
    const float* __restrict__ targets_tail,
    int64_t tailN    // tail length in floats
) {
    int64_t gid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t gstride = (int64_t)blockDim.x * gridDim.x;

    double local = 0.0;

    // Vectorized main loop
    for (int64_t i4 = gid; i4 < N4; i4 += gstride) {
        float4 pv = preds4[i4];
        float4 tv = targets4[i4];
        // Manually expand to avoid any potential vector math surprises
        local += (double)tv.x * (logf(tv.x) - logf(pv.x));
        local += (double)tv.y * (logf(tv.y) - logf(pv.y));
        local += (double)tv.z * (logf(tv.z) - logf(pv.z));
        local += (double)tv.w * (logf(tv.w) - logf(pv.w));
    }

    // Scalar tail in case it's used (here tailN is expected 0 for our selection path)
    for (int64_t i = gid; i < tailN; i += gstride) {
        float p = preds_tail[i];
        float t = targets_tail[i];
        local += (double)t * (logf(t) - logf(p));
    }

    // Reduce within block
    double block_sum = block_reduce_sum_double(local);
    if (threadIdx.x == 0) {
        atomicAdd(out_sum, block_sum);
    }
}

// Finalize: divide by batch size and cast to float
__global__ void finalize_kernel(const double* __restrict__ sum, float* __restrict__ out, int64_t batch_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double val = *sum / (double)batch_size;
        out[0] = static_cast<float>(val);
    }
}

static inline void hip_check_last_error(const char* msg) {
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        TORCH_CHECK(false, msg, " - ", hipGetErrorString(err));
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        TORCH_CHECK(false, "hipDeviceSynchronize failed: ", hipGetErrorString(err));
    }
}

// PyTorch entry point: only tensor parameters allowed
at::Tensor run(at::Tensor predictions, at::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda(), "Inputs must be CUDA/HIP tensors.");
    TORCH_CHECK(predictions.dtype() == at::kFloat && targets.dtype() == at::kFloat, "Inputs must be float32.");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Inputs must be contiguous.");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Predictions and targets must have the same shape.");
    TORCH_CHECK(predictions.dim() == 2, "Expected 2D tensors [batch, dim].");

    const int64_t batch = predictions.size(0);
    const int64_t dim   = predictions.size(1);
    const int64_t N     = predictions.numel();

    // Allocate device accumulator in FP64 for accuracy
    auto sum_options = predictions.options().dtype(at::kDouble);
    at::Tensor d_sum = at::zeros({1}, sum_options);

    // Choose grid size: cap to limit atomicAdd calls; use grid-stride loop to cover all elements
    int block = BLOCK_SIZE;
    // Cap blocks to 65535 to keep atomic traffic reasonable; kernel uses grid-stride loop
    int64_t max_blocks = 65535;
    int64_t est_blocks = (N + block - 1) / block;
    int grid = static_cast<int>(std::min<int64_t>(est_blocks, max_blocks));
    if (grid < 1) grid = 1;

    const float* p_ptr = predictions.data_ptr<float>();
    const float* t_ptr = targets.data_ptr<float>();
    double* sum_ptr = d_sum.data_ptr<double>();

    // Check 16-byte alignment for vectorized path
    uintptr_t p_addr = reinterpret_cast<uintptr_t>(p_ptr);
    uintptr_t t_addr = reinterpret_cast<uintptr_t>(t_ptr);
    bool aligned16 = ((p_addr % 16u) == 0u) && ((t_addr % 16u) == 0u);
    bool vec_ok = aligned16 && (N % 4 == 0);

    if (vec_ok) {
        int64_t N4 = N / 4;
        const float4* p4 = reinterpret_cast<const float4*>(p_ptr);
        const float4* t4 = reinterpret_cast<const float4*>(t_ptr);
        const float*  p_tail = nullptr;
        const float*  t_tail = nullptr;
        int64_t tailN = 0;

        hipLaunchKernelGGL(
            kl_div_sum_vec4_kernel,
            dim3(grid), dim3(block), 0, 0,
            p4, t4, sum_ptr, N4, p_tail, t_tail, tailN
        );
        hip_check_last_error("kl_div_sum_vec4_kernel launch failed");
    } else {
        hipLaunchKernelGGL(
            kl_div_sum_scalar_kernel,
            dim3(grid), dim3(block), 0, 0,
            p_ptr, t_ptr, sum_ptr, N
        );
        hip_check_last_error("kl_div_sum_scalar_kernel launch failed");
    }

    // Finalize: divide by batch size and cast to float32 scalar tensor
    at::Tensor out = at::empty({}, predictions.options().dtype(at::kFloat));
    hipLaunchKernelGGL(
        finalize_kernel,
        dim3(1), dim3(1), 0, 0,
        sum_ptr, out.data_ptr<float>(), batch
    );
    hip_check_last_error("finalize_kernel launch failed");

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "KL Divergence (batchmean) between two probability distributions (HIP optimized)");
}