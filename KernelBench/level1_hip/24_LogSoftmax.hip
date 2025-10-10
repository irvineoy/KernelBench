// LogSoftmax(dim=1) optimized HIP kernel for AMD MI300X (gfx942)
// Single file: kernels + PyTorch C++ extension bindings

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cfloat>
#include <cstdint>
#include <cmath>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256  // multiple of 64 (wavefront size), good balance for MI300X
#endif

// Kernel: Compute log_softmax along last dimension (dim=1) for a 2D tensor [rows, cols]
// y[i, j] = x[i, j] - log(sum_k exp(x[i, k])) computed stably via max subtraction
__global__ void logsoftmax_rows_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long rows,
    long long cols) {

    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;

    const float* __restrict__ in_row = input + static_cast<long long>(row) * cols;
    float* __restrict__ out_row = output + static_cast<long long>(row) * cols;

    __shared__ float sdata[BLOCK_SIZE];

    // Fast path conditions for vectorized loads/stores
    const uintptr_t addr = reinterpret_cast<uintptr_t>(in_row);
    const bool aligned16 = (addr % 16) == 0;
    const bool use_vec4 = aligned16 && (cols % 4 == 0);
    const long long cols4 = cols >> 2;

    // Pass 1: compute row max
    float local_max = -FLT_MAX;

    if (use_vec4) {
        const float4* __restrict__ in4 = reinterpret_cast<const float4*>(in_row);
        for (long long j4 = tid; j4 < cols4; j4 += BLOCK_SIZE) {
            float4 v = in4[j4];
            local_max = fmaxf(local_max, v.x);
            local_max = fmaxf(local_max, v.y);
            local_max = fmaxf(local_max, v.z);
            local_max = fmaxf(local_max, v.w);
        }
    } else {
        for (long long j = tid; j < cols; j += BLOCK_SIZE) {
            float v = in_row[j];
            local_max = fmaxf(local_max, v);
        }
    }

    // Block reduction for max
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    const float row_max = sdata[0];

    // Pass 2: compute sum of exp(x - max)
    float local_sum = 0.0f;

    if (use_vec4) {
        const float4* __restrict__ in4 = reinterpret_cast<const float4*>(in_row);
        for (long long j4 = tid; j4 < cols4; j4 += BLOCK_SIZE) {
            float4 v = in4[j4];
            local_sum += expf(v.x - row_max);
            local_sum += expf(v.y - row_max);
            local_sum += expf(v.z - row_max);
            local_sum += expf(v.w - row_max);
        }
    } else {
        for (long long j = tid; j < cols; j += BLOCK_SIZE) {
            local_sum += expf(in_row[j] - row_max);
        }
    }

    // Block reduction for sum
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    const float row_sum = sdata[0];

    // Compute log denominator: log(sum(exp(x - max))) + max
    const float log_denom = logf(row_sum) + row_max;

    // Pass 3: write outputs
    if (use_vec4) {
        const float4* __restrict__ in4 = reinterpret_cast<const float4*>(in_row);
        float4* __restrict__ out4 = reinterpret_cast<float4*>(out_row);
        for (long long j4 = tid; j4 < cols4; j4 += BLOCK_SIZE) {
            float4 v = in4[j4];
            v.x = v.x - log_denom;
            v.y = v.y - log_denom;
            v.z = v.z - log_denom;
            v.w = v.w - log_denom;
            out4[j4] = v;
        }
    } else {
        for (long long j = tid; j < cols; j += BLOCK_SIZE) {
            out_row[j] = in_row[j] - log_denom;
        }
    }
}

// PyTorch wrapper: only tensor parameters, no scalar config
// Input: x [batch_size, dim], dtype float32
at::Tensor run(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be float32");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D [batch, dim]");

    auto x = input.contiguous();  // ensure row-major contiguous
    const long long rows = x.size(0);
    const long long cols = x.size(1);

    auto y = at::empty_like(x);

    const dim3 block(BLOCK_SIZE, 1, 1);
    const dim3 grid(static_cast<unsigned int>(rows), 1, 1);

    hipLaunchKernelGGL(
        logsoftmax_rows_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        rows, cols
    );

    hipError_t err_sync = hipDeviceSynchronize();
    TORCH_CHECK(err_sync == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err_sync));
    hipError_t err_async = hipGetLastError();
    TORCH_CHECK(err_async == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err_async));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "LogSoftmax(dim=1) (HIP, optimized for MI300X)");
}