// LayerNorm HIP kernel for PyTorch extension (AMD MI300X optimized)
// Implements nn.LayerNorm over the last 3 dims (features, H, W) with affine (weight/bias).
// Signature: run(input, weight, bias)
// Notes:
// - Only tensors are accepted by the public run() function (matches benchmarking harness).
// - Sizes/metadata are extracted in C++ and passed to device kernels as scalars.
// - Two-pass algorithm: (1) parallel reduction for sums/sumsq per sample, (2) normalization with affine.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Kernel 1: reduce across normalized dims to compute per-sample sum and sumsq
template<int VEC>
__global__ void reduce_sums_sumsq_kernel(
    const float* __restrict__ x,  // [B, N] flattened
    float* __restrict__ sums,     // [B]
    float* __restrict__ sumsq,    // [B]
    int64_t N)                    // number of elements per sample
{
    // Each block processes a tile of one sample (batch element)
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int64_t block_stride = (int64_t)blockDim.x * VEC;
    const int64_t grid_stride = (int64_t)gridDim.x * block_stride;

    const int64_t sample_offset = (int64_t)b * N;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    // Start index for this block and thread
    int64_t start = (int64_t)blockIdx.x * block_stride + (int64_t)tid * VEC;

    for (int64_t i = start; i + (VEC - 1) < N; i += grid_stride) {
        // Vectorized load
        if constexpr (VEC == 4) {
            const float4* p = reinterpret_cast<const float4*>(x + sample_offset + i);
            float4 v = *p;
            local_sum   += v.x + v.y + v.z + v.w;
            local_sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        } else { // VEC == 1
            float v = x[sample_offset + i];
            local_sum   += v;
            local_sumsq += v * v;
        }
    }

    // Handle tail for VEC==1 (shouldn't occur for VEC==4 path since N % 4 == 0 when chosen)
    if constexpr (VEC == 1) {
        for (int64_t i = start; i < N; i += grid_stride) {
            // already processed once above; ensure not double-process by skipping when (i + (VEC - 1) < N)
            // For VEC==1 this is equivalent to i < N
            // But to avoid double-work, we'll limit to the main loop alone
            // The above loop already covered all i < N. No extra tail loop needed here.
            // Keeping empty tail for clarity.
            break;
        }
    }

    // Block reduction in shared memory
    __shared__ float s_sum[256];
    __shared__ float s_sumsq[256];
    s_sum[tid] = local_sum;
    s_sumsq[tid] = local_sumsq;
    __syncthreads();

    // Reduce to one value per block
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid]   += s_sum[tid + s];
            s_sumsq[tid] += s_sumsq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&sums[b],  s_sum[0]);
        atomicAdd(&sumsq[b], s_sumsq[0]);
    }
}

// Kernel 2: compute mean and inv_std from sums and sumsq for each sample
__global__ void compute_stats_kernel(
    const float* __restrict__ sums,   // [B]
    const float* __restrict__ sumsq,  // [B]
    float* __restrict__ mean,         // [B]
    float* __restrict__ invstd,       // [B]
    int64_t N,
    float eps,
    int B)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    float s  = sums[b];
    float ss = sumsq[b];
    float m  = s / (float)N;
    float v  = ss / (float)N - m * m;
    // Clamp variance to avoid negative values due to numerical issues
    v = v > 0.f ? v : 0.f;
    float inv = rsqrtf(v + eps);
    mean[b]   = m;
    invstd[b] = inv;
}

// Kernel 3: normalize and apply affine
template<int VEC>
__global__ void normalize_affine_kernel(
    const float* __restrict__ x,       // [B, N]
    const float* __restrict__ gamma,   // [N]
    const float* __restrict__ beta,    // [N]
    const float* __restrict__ mean,    // [B]
    const float* __restrict__ invstd,  // [B]
    float* __restrict__ y,             // [B, N]
    int64_t N)
{
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int64_t block_stride = (int64_t)blockDim.x * VEC;
    const int64_t grid_stride = (int64_t)gridDim.x * block_stride;

    const float m  = mean[b];
    const float iv = invstd[b];

    const int64_t sample_offset = (int64_t)b * N;

    int64_t start = (int64_t)blockIdx.x * block_stride + (int64_t)tid * VEC;

    for (int64_t i = start; i + (VEC - 1) < N; i += grid_stride) {
        if constexpr (VEC == 4) {
            const float4* px = reinterpret_cast<const float4*>(x + sample_offset + i);
            const float4* pg = reinterpret_cast<const float4*>(gamma + i);
            const float4* pb = reinterpret_cast<const float4*>(beta + i);

            float4 xv = *px;
            float4 gv = *pg;
            float4 bv = *pb;

            xv.x = ((xv.x - m) * iv) * gv.x + bv.x;
            xv.y = ((xv.y - m) * iv) * gv.y + bv.y;
            xv.z = ((xv.z - m) * iv) * gv.z + bv.z;
            xv.w = ((xv.w - m) * iv) * gv.w + bv.w;

            float4* py = reinterpret_cast<float4*>(y + sample_offset + i);
            *py = xv;
        } else { // VEC == 1
            float v = x[sample_offset + i];
            float out = ((v - m) * iv) * gamma[i] + beta[i];
            y[sample_offset + i] = out;
        }
    }

    if constexpr (VEC == 1) {
        // No extra tail handling needed (loop condition i < N captured)
    }
}

// Public entry point: accepts only tensors (input, weight, bias)
at::Tensor run(at::Tensor x, at::Tensor weight, at::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA/HIP device");
    TORCH_CHECK(weight.is_cuda() && bias.is_cuda(), "Parameters must be on CUDA/HIP device");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(weight.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat, "Parameters must be float32");
    TORCH_CHECK(x.dim() >= 2, "Input must have at least 2 dimensions");

    // Ensure contiguous memory for coalesced/vectorized access
    auto x_contig = x.contiguous();
    auto w_contig = weight.contiguous();
    auto b_contig = bias.contiguous();

    // Infer normalization dims: normalized over last 3 dims (from model)
    // Shape is (B, F, H, W) given get_inputs()
    int64_t B = x_contig.size(0);
    int64_t N = 1;
    for (int d = 1; d < x_contig.dim(); ++d) {
        N *= x_contig.size(d);
    }
    TORCH_CHECK(w_contig.numel() == N && b_contig.numel() == N,
                "Weight/Bias must match the normalized_shape (product of last 3 dims)");

    auto y = at::empty_like(x_contig);

    // Allocate temporary buffers for reduction and stats
    auto opts_f = x_contig.options().dtype(at::kFloat);
    at::Tensor sums  = at::zeros({B}, opts_f);
    at::Tensor sumsq = at::zeros({B}, opts_f);
    at::Tensor mean  = at::empty({B}, opts_f);
    at::Tensor invstd= at::empty({B}, opts_f);

    // Launch configuration
    const int block_size = 256; // multiple of 64 (wavefront)
    // Choose vectorization (float4) if pointers are 16B aligned and N multiple of 4
    auto x_ptr = reinterpret_cast<uintptr_t>(x_contig.data_ptr<float>());
    auto w_ptr = reinterpret_cast<uintptr_t>(w_contig.data_ptr<float>());
    auto b_ptr = reinterpret_cast<uintptr_t>(b_contig.data_ptr<float>());
    auto y_ptr = reinterpret_cast<uintptr_t>(y.data_ptr<float>());

    bool align16 = ((x_ptr | w_ptr | b_ptr | y_ptr) & 0xF) == 0;
    bool can_vec4 = align16 && (N % 4 == 0);

    int vec = can_vec4 ? 4 : 1;
    int64_t block_stride = (int64_t)block_size * vec;

    int blocks_per_sample = (int)std::min<int64_t>(
        std::max<int64_t>(1, (N + block_stride - 1) / block_stride),
        (int64_t)65535
    );

    dim3 grid_reduce(blocks_per_sample, (unsigned)B, 1);
    dim3 block(block_size, 1, 1);

    // Kernel 1: reduction
    if (can_vec4) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(reduce_sums_sumsq_kernel<4>),
            grid_reduce, block, 0, 0,
            x_contig.data_ptr<float>(),
            sums.data_ptr<float>(),
            sumsq.data_ptr<float>(),
            N
        );
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(reduce_sums_sumsq_kernel<1>),
            grid_reduce, block, 0, 0,
            x_contig.data_ptr<float>(),
            sums.data_ptr<float>(),
            sumsq.data_ptr<float>(),
            N
        );
    }
    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP reduce kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP reduce kernel launch failed: ", hipGetErrorString(err));

    // Kernel 2: compute stats
    const float eps = 1e-5f;
    int threads_stats = 64;
    int blocks_stats = (int)((B + threads_stats - 1) / threads_stats);
    hipLaunchKernelGGL(
        compute_stats_kernel,
        dim3(blocks_stats, 1, 1), dim3(threads_stats, 1, 1), 0, 0,
        sums.data_ptr<float>(),
        sumsq.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        N, eps, (int)B
    );
    err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP stats kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP stats kernel launch failed: ", hipGetErrorString(err));

    // Kernel 3: normalize + affine
    dim3 grid_norm(blocks_per_sample, (unsigned)B, 1);
    if (can_vec4) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(normalize_affine_kernel<4>),
            grid_norm, block, 0, 0,
            x_contig.data_ptr<float>(),
            w_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            mean.data_ptr<float>(),
            invstd.data_ptr<float>(),
            y.data_ptr<float>(),
            N
        );
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(normalize_affine_kernel<1>),
            grid_norm, block, 0, 0,
            x_contig.data_ptr<float>(),
            w_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            mean.data_ptr<float>(),
            invstd.data_ptr<float>(),
            y.data_ptr<float>(),
            N
        );
    }
    err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP normalize kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP normalize kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "LayerNorm over last 3 dims with affine (HIP, MI300X-optimized)");
}