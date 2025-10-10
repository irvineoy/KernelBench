// Smooth L1 (Huber) Loss - HIP optimized kernel for AMD MI300X (gfx942)
// Computes F.smooth_l1_loss(predictions, targets) with default beta=1.0 and reduction='mean'.
//
// Build example:
//   hipcc -O3 --offload-arch=gfx9-4-generic smooth_l1_loss.hip -shared -fPIC -o smooth_l1_loss.so
//
// Usage from Python (PyTorch):
//   import torch
//   import smooth_l1_loss
//   out = smooth_l1_loss.run(pred, target)  # returns a scalar tensor (mean Smooth L1 loss)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <iostream>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256  // Multiple of 64 (wavefront size on AMD)
#endif

// Per-element Smooth L1 with beta=1.0f, branchless select
__device__ inline float smooth_l1_elem(float d) {
    float ad = fabsf(d);
    // beta = 1.0f -> piecewise:
    // 0.5 * d^2 / 1.0 if |d| < 1, else |d| - 0.5
    float quad = 0.5f * d * d;
    float lin  = ad - 0.5f;
    // Select without divergent branch
    return (ad < 1.0f) ? quad : lin;
}

// Kernel: compute partial sums of Smooth L1 loss using grid-stride loops,
// vectorized (float4) path when input pointers are 16B-aligned and N % 4 == 0.
// One atomicAdd (double) per block to a global accumulator.
template<bool Vec4>
__global__ /*__launch_bounds__(BLOCK_SIZE)*/ void smooth_l1_partial_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ targ,
    double* __restrict__ g_accum,
    long long N)
{
    __shared__ double ssum[BLOCK_SIZE];

    const long long tid    = threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    double local = 0.0;

    if (Vec4) {
        // Vectorized path
        const long long N4 = N >> 2; // N / 4
        const float4* __restrict__ p4 = reinterpret_cast<const float4*>(pred);
        const float4* __restrict__ t4 = reinterpret_cast<const float4*>(targ);
        for (long long i = blockIdx.x * blockDim.x + tid; i < N4; i += stride) {
            float4 a = p4[i];
            float4 b = t4[i];
            float d0 = a.x - b.x; local += (double)smooth_l1_elem(d0);
            float d1 = a.y - b.y; local += (double)smooth_l1_elem(d1);
            float d2 = a.z - b.z; local += (double)smooth_l1_elem(d2);
            float d3 = a.w - b.w; local += (double)smooth_l1_elem(d3);
        }
        // Tail (should be zero when N % 4 == 0, but safe to handle)
        const long long base = N4 << 2;
        for (long long i = base + blockIdx.x * blockDim.x + tid; i < N; i += stride) {
            float d = pred[i] - targ[i];
            local += (double)smooth_l1_elem(d);
        }
    } else {
        // Scalar path
        for (long long i = blockIdx.x * blockDim.x + tid; i < N; i += stride) {
            float d = pred[i] - targ[i];
            local += (double)smooth_l1_elem(d);
        }
    }

    // Block reduction (in LDS)
    ssum[tid] = local;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x >> 1; s >= 1; s >>= 1) {
        if (tid < s) ssum[tid] += ssum[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(g_accum, ssum[0]);
    }
}

// Finalize: write mean = sum / N to output[0] (float)
__global__ void finalize_mean_kernel(const double* __restrict__ g_accum,
                                     float* __restrict__ out,
                                     long long N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double mean = (*g_accum) / (double)N;
        out[0] = static_cast<float>(mean);
    }
}

// PyTorch entry point: predictions, targets -> scalar tensor (mean Smooth L1 loss)
at::Tensor run(at::Tensor predictions, at::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA/HIP tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA/HIP tensor");
    TORCH_CHECK(predictions.scalar_type() == at::kFloat, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == at::kFloat, "targets must be float32");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "predictions and targets must have the same shape");
    TORCH_CHECK(predictions.is_contiguous(), "predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");

    const long long N = predictions.numel();
    TORCH_CHECK(N > 0, "Input tensors must contain at least one element");

    // Output scalar (float32)
    auto out = at::empty({}, predictions.options());

    // Global accumulator (double) initialized to zero
    auto accum = at::zeros({1}, predictions.options().dtype(at::kDouble));

    const float* pred_ptr = predictions.data_ptr<float>();
    const float* targ_ptr = targets.data_ptr<float>();
    double* accum_ptr     = accum.data_ptr<double>();
    float* out_ptr        = out.data_ptr<float>();

    // Launch configuration
    const int block_size = BLOCK_SIZE; // must match any launch_bounds if used
    // Limit grid size to a reasonable number to control number of atomics
    long long max_blocks = 65535;  // safe, widely supported
    long long grid_size = (N + block_size - 1) / block_size;
    if (grid_size > max_blocks) grid_size = max_blocks;

    // Decide vectorized path based on 16-byte alignment and N % 4 == 0
    bool vec4_ok = ((reinterpret_cast<std::uintptr_t>(pred_ptr) & 0xF) == 0) &&
                   ((reinterpret_cast<std::uintptr_t>(targ_ptr) & 0xF) == 0) &&
                   ((N & 3LL) == 0LL);

    if (vec4_ok) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(smooth_l1_partial_kernel<true>),
            dim3((unsigned int)grid_size), dim3(block_size), 0, 0,
            pred_ptr, targ_ptr, accum_ptr, N
        );
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(smooth_l1_partial_kernel<false>),
            dim3((unsigned int)grid_size), dim3(block_size), 0, 0,
            pred_ptr, targ_ptr, accum_ptr, N
        );
    }

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed (partial): ", hipGetErrorString(err));

    err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel (partial) failed: ", hipGetErrorString(err));

    // Finalize: compute mean on device
    hipLaunchKernelGGL(finalize_mean_kernel, dim3(1), dim3(1), 0, 0, accum_ptr, out_ptr, N);

    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed (finalize): ", hipGetErrorString(err));

    err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel (finalize) failed: ", hipGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Smooth L1 (Huber) Loss with reduction='mean' (HIP, MI300X optimized)");
}