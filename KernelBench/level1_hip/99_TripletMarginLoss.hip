// Triplet Margin Loss (p=2, margin=1.0, eps=1e-6, reduction='mean', swap=False)
// Optimized HIP kernel for AMD MI300X (gfx942)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Constants to match PyTorch defaults
__device__ __constant__ float kMargin = 1.0f;
__device__ __constant__ float kEps    = 1e-6f;

#define THREADS_PER_BLOCK 256  // multiple of 64 (wavefront)

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
triplet_margin_loss_mean_kernel(
    const float* __restrict__ A,   // [B, D]
    const float* __restrict__ P,   // [B, D]
    const float* __restrict__ N,   // [B, D]
    float* __restrict__ out_sum,   // scalar (accumulates mean via per-block scaling)
    int D                           // feature dimension
) {
    int b = blockIdx.x;            // sample index
    int tid = threadIdx.x;

    const float* a = A + static_cast<size_t>(b) * D;
    const float* p = P + static_cast<size_t>(b) * D;
    const float* n = N + static_cast<size_t>(b) * D;

    float acc_ap = 0.0f;
    float acc_an = 0.0f;

    // Shared memory for block-level reduction (two arrays, padded to avoid bank conflicts)
    __shared__ float s_ap[THREADS_PER_BLOCK];
    __shared__ float s_an[THREADS_PER_BLOCK];

    // Vectorized path if pointers are 16-byte aligned and D % 4 == 0
    bool can_vec4 = false;
    {
        uintptr_t addr_a = reinterpret_cast<uintptr_t>(a);
        uintptr_t addr_p = reinterpret_cast<uintptr_t>(p);
        uintptr_t addr_n = reinterpret_cast<uintptr_t>(n);
        can_vec4 = ((D & 3) == 0) && ((addr_a | addr_p | addr_n) % 16 == 0);
    }

    if (can_vec4) {
        const float4* __restrict__ a4 = reinterpret_cast<const float4*>(a);
        const float4* __restrict__ p4 = reinterpret_cast<const float4*>(p);
        const float4* __restrict__ n4 = reinterpret_cast<const float4*>(n);
        int D4 = D >> 2;

        #pragma unroll 4
        for (int i = tid; i < D4; i += blockDim.x) {
            float4 va = a4[i];
            float4 vp = p4[i];
            float4 vn = n4[i];

            float d0 = va.x - vp.x; float e0 = va.x - vn.x;
            float d1 = va.y - vp.y; float e1 = va.y - vn.y;
            float d2 = va.z - vp.z; float e2 = va.z - vn.z;
            float d3 = va.w - vp.w; float e3 = va.w - vn.w;

            acc_ap += d0*d0 + d1*d1 + d2*d2 + d3*d3;
            acc_an += e0*e0 + e1*e1 + e2*e2 + e3*e3;
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < D; i += blockDim.x) {
            float da = a[i] - p[i];
            float dn = a[i] - n[i];
            acc_ap += da * da;
            acc_an += dn * dn;
        }
    }

    s_ap[tid] = acc_ap;
    s_an[tid] = acc_an;
    __syncthreads();

    // Parallel reduction within block
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_ap[tid] += s_ap[tid + s];
            s_an[tid] += s_an[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of the block computes the per-sample loss and atomically adds mean contribution
    if (tid == 0) {
        float d_ap = sqrtf(s_ap[0] + kEps);
        float d_an = sqrtf(s_an[0] + kEps);
        float loss = d_ap - d_an + kMargin;
        loss = fmaxf(loss, 0.0f);

        // Accumulate scaled by 1/B to produce mean directly
        float invB = 1.0f / static_cast<float>(gridDim.x);
        atomicAdd(out_sum, loss * invB);
    }
}

// PyTorch binding: run(anchor, positive, negative) -> scalar tensor (mean loss)
at::Tensor run(at::Tensor anchor, at::Tensor positive, at::Tensor negative) {
    TORCH_CHECK(anchor.is_cuda(), "anchor must be a CUDA/HIP tensor");
    TORCH_CHECK(positive.is_cuda(), "positive must be a CUDA/HIP tensor");
    TORCH_CHECK(negative.is_cuda(), "negative must be a CUDA/HIP tensor");
    TORCH_CHECK(anchor.dtype() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(positive.dtype() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(negative.dtype() == at::kFloat, "Only float32 is supported");

    TORCH_CHECK(anchor.dim() == 2, "anchor must be 2D [B, D]");
    TORCH_CHECK(positive.sizes() == anchor.sizes(), "positive must match anchor shape");
    TORCH_CHECK(negative.sizes() == anchor.sizes(), "negative must match anchor shape");

    auto A = anchor.contiguous();
    auto P = positive.contiguous();
    auto N = negative.contiguous();

    const int64_t B = A.size(0);
    const int64_t D = A.size(1);

    TORCH_CHECK(B > 0 && D > 0, "Batch and feature dimensions must be positive");
    TORCH_CHECK(B <= INT32_MAX && D <= INT32_MAX, "Dimensions exceed int32 range");

    // Output scalar on device
    auto out = at::zeros({}, A.options());

    // Launch configuration
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(static_cast<unsigned int>(B));

    hipLaunchKernelGGL(
        triplet_margin_loss_mean_kernel,
        grid, block, 0, 0,
        A.data_ptr<float>(),
        P.data_ptr<float>(),
        N.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(D)
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP device sync failed: ", hipGetErrorString(err));

    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Triplet Margin Loss (p=2, margin=1.0, reduction='mean') [HIP]");
}