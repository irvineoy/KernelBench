// Softmax over dim=1 for 2D tensor [rows, cols]
// Single HIP file with kernels and PyTorch binding

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cfloat>
#include <cmath>
#include <iostream>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Block size must be a multiple of 64 (wavefront size on AMD)
constexpr int BLOCK_SIZE = 256;

// Reduction utility: block-wide max using shared memory
__device__ inline float blockReduceMax(float val, float* sdata) {
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float v = sdata[tid + s];
            sdata[tid] = fmaxf(sdata[tid], v);
        }
        __syncthreads();
    }
    return sdata[0];
}

// Reduction utility: block-wide sum using shared memory
__device__ inline float blockReduceSum(float val, float* sdata) {
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    return sdata[0];
}

// Kernel: per-row softmax with numerically stable approach
// Phases per row (single kernel per row/block):
// 1) compute row max
// 2) compute exp(x - max), write to output, accumulate sum
// 3) scale output by inverse sum
__global__ void softmax_rows_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* __restrict__ row_in  = x + static_cast<size_t>(row) * cols;
    float* __restrict__       row_out = y + static_cast<size_t>(row) * cols;

    __shared__ float sdata[BLOCK_SIZE];

    // Check 16-byte alignment for vectorized float4 path
    bool aligned4 = ((reinterpret_cast<uintptr_t>(row_in) & 15) == 0) &&
                    ((reinterpret_cast<uintptr_t>(row_out) & 15) == 0) &&
                    (cols % 4 == 0);

    // Phase 1: compute row max
    float local_max = -FLT_MAX;

    if (aligned4) {
        int cols4 = cols >> 2;  // divided by 4
        const float4* __restrict__ in4 = reinterpret_cast<const float4*>(row_in);

        for (int i4 = threadIdx.x; i4 < cols4; i4 += blockDim.x) {
            float4 v4 = in4[i4];
            local_max = fmaxf(local_max, v4.x);
            local_max = fmaxf(local_max, v4.y);
            local_max = fmaxf(local_max, v4.z);
            local_max = fmaxf(local_max, v4.w);
        }
    } else {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float v = row_in[i];
            local_max = fmaxf(local_max, v);
        }
    }

    float row_max = blockReduceMax(local_max, sdata);
    __syncthreads();

    // Phase 2: compute exp(x - row_max), write to output, accumulate sum
    float local_sum = 0.0f;

    if (aligned4) {
        int cols4 = cols >> 2;
        const float4* __restrict__ in4 = reinterpret_cast<const float4*>(row_in);
        float4* __restrict__ out4 = reinterpret_cast<float4*>(row_out);

        for (int i4 = threadIdx.x; i4 < cols4; i4 += blockDim.x) {
            float4 v4 = in4[i4];
            v4.x = expf(v4.x - row_max);
            v4.y = expf(v4.y - row_max);
            v4.z = expf(v4.z - row_max);
            v4.w = expf(v4.w - row_max);
            local_sum += (v4.x + v4.y + v4.z + v4.w);
            out4[i4] = v4;
        }
    } else {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float ev = expf(row_in[i] - row_max);
            local_sum += ev;
            row_out[i] = ev;
        }
    }

    float row_sum = blockReduceSum(local_sum, sdata);
    __syncthreads();

    // Prevent division by zero in pathological cases
    float inv_sum = 1.0f / fmaxf(row_sum, 1e-20f);

    // Phase 3: scale outputs by inverse of sum
    if (aligned4) {
        int cols4 = cols >> 2;
        float4* __restrict__ out4 = reinterpret_cast<float4*>(row_out);

        for (int i4 = threadIdx.x; i4 < cols4; i4 += blockDim.x) {
            float4 v4 = out4[i4];
            v4.x *= inv_sum;
            v4.y *= inv_sum;
            v4.z *= inv_sum;
            v4.w *= inv_sum;
            out4[i4] = v4;
        }
    } else {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            row_out[i] *= inv_sum;
        }
    }
}

// PyTorch entry point: only tensors as parameters (matches get_inputs())
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU (ROCm device).");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D of shape [batch, dim].");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 is supported.");

    auto x = input.contiguous();
    const int rows = static_cast<int>(x.size(0));
    const int cols = static_cast<int>(x.size(1));

    auto y = at::empty_like(x);

    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);

    hipLaunchKernelGGL(
        softmax_rows_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        rows, cols
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Row-wise Softmax (dim=1) on HIP");
}