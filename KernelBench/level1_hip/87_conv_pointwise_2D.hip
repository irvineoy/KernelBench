// 1x1 (pointwise) Conv2D on AMD GPUs (HIP), optimized as a tiled GEMM:
//   Out[N, C_out, H, W] = Weight[C_out, C_in, 1, 1] * In[N, C_in, H, W] (+ bias)
//
// Mapping:
//   Flatten spatial positions: S = N * H * W (NCHW layout).
//   Compute C_out x S = (C_out x C_in) * (C_in x S)
//
// Kernel tiles the matrix multiply across (M=C_out, N=S, K=C_in) using LDS.
//
// Build: hipcc -O3 --offload-arch=gfx942 pointwise_conv1x1.hip -shared -fPIC

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>

#ifndef CEIL_DIV
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#endif

// Tile sizes (tuned for MI300X)
constexpr int BM = 64;   // tile size along M (C_out)
constexpr int BN = 64;   // tile size along N (S = N*H*W)
constexpr int BK = 64;   // tile size along K (C_in)

// Thread arrangement inside a block
constexpr int TM = 16;   // threads along M
constexpr int TN = 16;   // threads along N
constexpr int THREADS_PER_BLOCK = TM * TN; // must be 256

// Each thread computes a small 4x4 output fragment inside the 64x64 tile
constexpr int RM = BM / TM; // 4
constexpr int RN = BN / TN; // 4

// Optimized 1x1 conv kernel implemented as tiled GEMM with LDS reuse.
// weight: [C_out, C_in, 1, 1] contiguous
// input:  [N, C_in, H, W]      contiguous (NCHW)
// output: [N, C_out, H, W]     contiguous (NCHW)
// bias:   [C_out] or nullptr if no bias
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void conv1x1_gemm_kernel(const float* __restrict__ input,
                         const float* __restrict__ weight,
                         const float* __restrict__ bias,
                         float* __restrict__ output,
                         int N, int C_in, int H, int W, int C_out) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ float sA[BM][BK + 1]; // [C_out tile][C_in tile]
    __shared__ float sB[BK][BN + 1]; // [C_in tile][S tile]

    // Flattened spatial size
    const int64_t plane = static_cast<int64_t>(H) * static_cast<int64_t>(W);
    const int64_t S_total = static_cast<int64_t>(N) * plane;

    // Block tile coordinates
    const int m0 = blockIdx.y * BM;                 // starting row in M (C_out)
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * BN; // starting col in N (S)

    // Effective tile sizes (handle edges)
    const int m_eff = (m0 + BM <= C_out) ? BM : (C_out - m0);
    const int64_t n_eff = (n0 + BN <= S_total) ? BN : (S_total - n0);

    // Thread coordinates within tile
    const int tx = threadIdx.x; // [0, TN)
    const int ty = threadIdx.y; // [0, TM)
    const int tid = ty * blockDim.x + tx;

    // Initialize accumulators for a 4x4 sub-tile
    float acc[RM][RN];
    #pragma unroll
    for (int im = 0; im < RM; ++im) {
        #pragma unroll
        for (int in = 0; in < RN; ++in) {
            acc[im][in] = 0.0f;
        }
    }

    // Tiled K-loop (C_in)
    for (int k0 = 0; k0 < C_in; k0 += BK) {
        const int k_eff = (k0 + BK <= C_in) ? BK : (C_in - k0);

        // Load sA tile: [m in 0..BM), [k in 0..k_eff)
        for (int idx = tid; idx < BM * k_eff; idx += THREADS_PER_BLOCK) {
            int m = idx / k_eff;
            int kk = idx % k_eff;
            int gm = m0 + m;
            int gk = k0 + kk;
            float val = 0.0f;
            if (gm < C_out && gk < C_in) {
                // weight layout: [C_out, C_in, 1, 1] contiguous
                val = weight[static_cast<int64_t>(gm) * C_in + gk];
            }
            sA[m][kk] = val;
        }

        // Load sB tile: [k in 0..k_eff), [n in 0..n_eff)
        // For each element at (kk, nn), we fetch input at channel (k0+kk) and spatial index (n0+nn)
        for (int idx = tid; idx < k_eff * static_cast<int>(n_eff); idx += THREADS_PER_BLOCK) {
            int kk = idx / static_cast<int>(n_eff);
            int nn = idx % static_cast<int>(n_eff);
            int gk = k0 + kk;
            int64_t gn = n0 + nn; // flattened spatial index in [0, S_total)

            // Map gn to base offset: base = n * (C_in*H*W) + (h*W + w)
            // We avoid explicit h,w by using remainder.
            int64_t n = gn / plane;
            int64_t rem = gn - n * plane; // gn % plane

            float val = 0.0f;
            if (gk < C_in && gn < S_total) {
                int64_t base_in = n * (static_cast<int64_t>(C_in) * plane) + rem;
                int64_t idx_in = base_in + static_cast<int64_t>(gk) * plane;
                val = input[idx_in];
            }
            sB[kk][nn] = val;
        }

        __syncthreads();

        // Compute on the loaded tiles
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            if (kk >= k_eff) break;

            // Each thread computes a 4x4 block:
            // rows: m = ty + im*TM, cols: n = tx + in*TN
            #pragma unroll
            for (int im = 0; im < RM; ++im) {
                int m_rel = ty + im * TM; // 0..BM-1
                float a = sA[m_rel][kk];  // broadcast along RN
                #pragma unroll
                for (int in = 0; in < RN; ++in) {
                    int n_rel = tx + in * TN; // 0..BN-1
                    float b = sB[kk][n_rel];
                    acc[im][in] += a * b;
                }
            }
        }

        __syncthreads();
    } // end K-loop

    // Store results with optional bias
    const bool has_bias = (bias != nullptr);
    #pragma unroll
    for (int im = 0; im < RM; ++im) {
        int m_rel = ty + im * TM;
        int oc = m0 + m_rel;
        if (m_rel >= m_eff) continue;

        float bval = has_bias ? bias[oc] : 0.0f;

        #pragma unroll
        for (int in = 0; in < RN; ++in) {
            int n_rel = tx + in * TN;
            if (n_rel >= n_eff) continue;

            int64_t gn = n0 + n_rel;  // flattened spatial index
            int64_t n = gn / plane;
            int64_t rem = gn - n * plane;

            // output idx: base_out + oc * (H*W)
            int64_t base_out = n * (static_cast<int64_t>(C_out) * plane) + rem;
            int64_t idx_out = base_out + static_cast<int64_t>(oc) * plane;

            output[idx_out] = acc[im][in] + bval;
        }
    }
}

// Host wrapper (no bias)
at::Tensor run_nobias(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA (HIP) tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA (HIP) tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be NCHW");
    TORCH_CHECK(weight.dim() == 4, "weight must be [C_out, C_in, 1, 1]");

    auto x = input.contiguous();
    auto w = weight.contiguous();

    const int64_t N = x.size(0);
    const int64_t C_in = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    const int64_t C_out = w.size(0);
    TORCH_CHECK(w.size(1) == C_in, "weight C_in mismatch");
    TORCH_CHECK(w.size(2) == 1 && w.size(3) == 1, "kernel must be 1x1");

    auto y = at::empty({N, C_out, H, W}, x.options());

    const int64_t S_total = N * H * W;

    dim3 block(TN, TM, 1); // 16x16 = 256 threads
    dim3 grid(CEIL_DIV(static_cast<int>(S_total), BN),
              CEIL_DIV(static_cast<int>(C_out), BM),
              1);

    hipStream_t stream = at::cuda::getCurrentHIPStream();
    hipLaunchKernelGGL(
        conv1x1_gemm_kernel,
        grid, block, 0, stream,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        /*bias*/ nullptr,
        y.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C_in),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(C_out)
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));
    err = hipStreamSynchronize(stream);
    TORCH_CHECK(err == hipSuccess, "HIP kernel execution failed: ", hipGetErrorString(err));

    return y;
}

// Host wrapper (with bias)
at::Tensor run_bias(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA (HIP) tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA (HIP) tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA (HIP) tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be NCHW");
    TORCH_CHECK(weight.dim() == 4, "weight must be [C_out, C_in, 1, 1]");
    TORCH_CHECK(bias.dim() == 1, "bias must be [C_out]");

    auto x = input.contiguous();
    auto w = weight.contiguous();
    auto b = bias.contiguous();

    const int64_t N = x.size(0);
    const int64_t C_in = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    const int64_t C_out = w.size(0);
    TORCH_CHECK(w.size(1) == C_in, "weight C_in mismatch");
    TORCH_CHECK(w.size(2) == 1 && w.size(3) == 1, "kernel must be 1x1");
    TORCH_CHECK(b.size(0) == C_out, "bias length must match C_out");

    auto y = at::empty({N, C_out, H, W}, x.options());

    const int64_t S_total = N * H * W;

    dim3 block(TN, TM, 1); // 16x16 = 256 threads
    dim3 grid(CEIL_DIV(static_cast<int>(S_total), BN),
              CEIL_DIV(static_cast<int>(C_out), BM),
              1);

    hipStream_t stream = at::cuda::getCurrentHIPStream();
    hipLaunchKernelGGL(
        conv1x1_gemm_kernel,
        grid, block, 0, stream,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C_in),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(C_out)
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));
    err = hipStreamSynchronize(stream);
    TORCH_CHECK(err == hipSuccess, "HIP kernel execution failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Overloads to handle models with/without bias
    m.def("run", &run_nobias, "Pointwise (1x1) Conv2D without bias (HIP)");
    m.def("run", &run_bias,   "Pointwise (1x1) Conv2D with bias (HIP)");
}