// Scaled Dot-Product Attention (FlashAttention-style streaming) for AMD MI300X (gfx942)
// Inputs: Q, K, V tensors of shape [B, H, S, D], dtype=float16, contiguous
// Output: O tensor of shape [B, H, S, D], dtype=float16, contiguous
// Implementation focuses on correctness and good performance using shared memory tiling,
// online softmax (running max/sum), and FP32 accumulation.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <cmath>
#include <cstdint>
#include <iostream>

#ifndef CEILDIV
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#endif

// Tile sizes tuned for MI300X wavefront=64
#define TQ 64         // Query tile (sequence) per block
#define TK 64         // Key tile (sequence) per step
#define DTILE 64      // Head dim tile
#define BLOCK_THREADS 256  // Must be multiple of 64

// Clamp exp inputs to avoid NaNs/Infs in extreme cases
__device__ inline float safe_exp(float x) {
    // Limits to avoid overflow in expf on FP32
    x = fminf(x, 80.0f);
    x = fmaxf(x, -80.0f);
    return __expf(x);
}

// Kernel: one block processes a block of TQ query positions for a single (b,h) head
// Uses an auxiliary FP32 global accumulator to carry unnormalized numerator across K-tiles.
// After all K-tiles are processed, normalizes by softmax denominator and writes FP16 output.
__global__ void sdpa_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    float* __restrict__ Accum,  // FP32 accumulator buffer [B,H,S,D]
    int B, int H, int S, int D) {

    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int q_block = blockIdx.x;
    const int q0 = q_block * TQ;

    if (b >= B || h >= H || q0 >= S) return;

    const int tid = threadIdx.x;

    // Derived strides assuming contiguous [B,H,S,D]
    const long SD = (long)S * (long)D;
    const long HSD = (long)H * SD;

    const long baseBH = (long)b * HSD + (long)h * SD;

    // Shared memory
    __shared__ __half sQ[TQ][DTILE + 1];   // +1 padding to reduce bank conflicts
    __shared__ __half sK[TK][DTILE + 1];
    __shared__ __half sV[TK][DTILE + 1];
    __shared__ float sS[TQ][TK];           // Scores (or probs after transform) for current K-tile
    __shared__ float s_m[TQ];              // running max per query
    __shared__ float s_l[TQ];              // running sum per query
    __shared__ float s_alpha[TQ];          // scaling factor per query for accumulator update

    const float inv_sqrt_d = rsqrtf((float)D);
    const int num_q = min(TQ, S - q0);
    const int num_k_tiles = CEILDIV(S, TK);
    const int num_d_tiles = CEILDIV(D, DTILE);

    // Zero the accumulator region for this (b,h,q-block)
    // Accum shape [B,H,S,D], FP32
    {
        long total = (long)num_q * (long)D;
        for (long idx = tid; idx < total; idx += BLOCK_THREADS) {
            int iq = idx / D;
            int d  = idx % D;
            long off = baseBH + (long)(q0 + iq) * D + d;
            Accum[off] = 0.0f;
        }
    }

    // Initialize running max and sum
    if (tid < num_q) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    __syncthreads();

    // Loop over K tiles
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * TK;
        const int cur_k = min(TK, S - k0);

        // Zero sS (partial score tile)
        {
            int total = TQ * TK;
            for (int idx = tid; idx < total; idx += BLOCK_THREADS) {
                sS[idx / TK][idx % TK] = 0.0f;
            }
        }
        __syncthreads();

        // Accumulate Q*K^T into sS over D tiles
        for (int dt = 0; dt < num_d_tiles; ++dt) {
            const int d0 = dt * DTILE;
            const int cur_d = min(DTILE, D - d0);

            // Load Q chunk [num_q x cur_d] into sQ
            {
                int total = num_q * cur_d;
                for (int idx = tid; idx < total; idx += BLOCK_THREADS) {
                    int iq = idx / cur_d;
                    int dc = idx % cur_d;
                    long g_off = baseBH + (long)(q0 + iq) * D + (d0 + dc);
                    sQ[iq][dc] = Q[g_off];
                }
                // If cur_d < DTILE, pad rest with 0
                int total_pad = num_q * (DTILE - cur_d);
                for (int idx = tid; idx < total_pad; idx += BLOCK_THREADS) {
                    int iq = idx / (DTILE - cur_d);
                    int dc = idx % (DTILE - cur_d);
                    sQ[iq][cur_d + dc] = __float2half(0.0f);
                }
            }

            // Load K chunk [cur_k x cur_d] into sK (keys tile)
            {
                int total = cur_k * cur_d;
                for (int idx = tid; idx < total; idx += BLOCK_THREADS) {
                    int jk = idx / cur_d;
                    int dc = idx % cur_d;
                    long g_off = baseBH + (long)(k0 + jk) * D + (d0 + dc);
                    sK[jk][dc] = K[g_off];
                }
                // Pad remaining rows/cols with zeros
                int row_pad = (TK - cur_k) * cur_d;
                for (int idx = tid; idx < row_pad; idx += BLOCK_THREADS) {
                    int jk = idx / cur_d;
                    int dc = idx % cur_d;
                    sK[cur_k + jk][dc] = __float2half(0.0f);
                }
                int col_pad = TK * (DTILE - cur_d);
                for (int idx = tid; idx < col_pad; idx += BLOCK_THREADS) {
                    int jk = idx / (DTILE - cur_d);
                    int dc = idx % (DTILE - cur_d);
                    sK[jk][cur_d + dc] = __float2half(0.0f);
                }
            }

            __syncthreads();

            // Compute partial dot products and accumulate into sS
            // Each thread accumulates multiple (iq, jk) pairs
            {
                int total = TQ * TK;
                for (int idx = tid; idx < total; idx += BLOCK_THREADS) {
                    int iq = idx / TK;
                    int jk = idx % TK;

                    // If iq beyond actual num_q, skip
                    if (iq >= num_q) continue;

                    float acc = 0.0f;
                    // DTILE loop
                    #pragma unroll
                    for (int dc = 0; dc < DTILE; ++dc) {
                        float qv = __half2float(sQ[iq][dc]);
                        float kv = __half2float(sK[jk][dc]);
                        acc += qv * kv;
                    }
                    sS[iq][jk] += acc; // still unscaled
                }
            }

            __syncthreads();
        } // end D tiles

        // Apply mask for out-of-range keys and compute row-wise max and sums
        // Convert sS to probabilities normalized by new running max (per query row)
        // Also compute alpha per row for accumulator scaling
        if (tid < num_q) {
            // Set out-of-range key columns to -inf
            for (int jk = cur_k; jk < TK; ++jk) {
                sS[tid][jk] = -INFINITY;
            }

            // Scale by 1/sqrt(D) and get tile local max
            float local_max = -INFINITY;
            for (int jk = 0; jk < TK; ++jk) {
                float v = sS[tid][jk] * inv_sqrt_d;
                sS[tid][jk] = v;
                local_max = fmaxf(local_max, v);
            }

            // Online softmax update
            float m_prev = s_m[tid];
            float m_new = fmaxf(m_prev, local_max);
            float alpha = safe_exp(m_prev - m_new);

            float p_sum = 0.0f;
            for (int jk = 0; jk < TK; ++jk) {
                float e = safe_exp(sS[tid][jk] - m_new);
                p_sum += e;
                sS[tid][jk] = e;  // store probabilities for this tile
            }

            float l_new = s_l[tid] * alpha + p_sum;

            s_m[tid] = m_new;
            s_l[tid] = l_new;
            s_alpha[tid] = alpha;
        }
        __syncthreads();

        // For each DV tile: update global FP32 accumulator using:
        // Accum = Accum * alpha_iq + sum_j p_ij * V_j
        for (int dv = 0; dv < num_d_tiles; ++dv) {
            const int d0 = dv * DTILE;
            const int cur_dv = min(DTILE, D - d0);

            // Load V tile [cur_k x cur_dv]
            {
                int total = cur_k * cur_dv;
                for (int idx = tid; idx < total; idx += BLOCK_THREADS) {
                    int jk = idx / cur_dv;
                    int dc = idx % cur_dv;
                    long g_off = baseBH + (long)(k0 + jk) * D + (d0 + dc);
                    sV[jk][dc] = V[g_off];
                }
                // Pad remaining rows/cols with zeros
                int row_pad = (TK - cur_k) * cur_dv;
                for (int idx = tid; idx < row_pad; idx += BLOCK_THREADS) {
                    int jk = idx / cur_dv;
                    int dc = idx % cur_dv;
                    sV[cur_k + jk][dc] = __float2half(0.0f);
                }
                int col_pad = TK * (DTILE - cur_dv);
                for (int idx = tid; idx < col_pad; idx += BLOCK_THREADS) {
                    int jk = idx / (DTILE - cur_dv);
                    int dc = idx % (DTILE - cur_dv);
                    sV[jk][cur_dv + dc] = __float2half(0.0f);
                }
            }
            __syncthreads();

            // Update accumulator for this dv tile
            long dv_tile_elems = (long)num_q * (long)cur_dv;
            for (long idx = tid; idx < dv_tile_elems; idx += BLOCK_THREADS) {
                int iq = idx / cur_dv;
                int dc = idx % cur_dv;

                float alpha = s_alpha[iq];

                // Compute sum_j p_ij * V_j[dc]
                float sum = 0.0f;
                #pragma unroll
                for (int jk = 0; jk < TK; ++jk) {
                    sum += sS[iq][jk] * __half2float(sV[jk][dc]);
                }

                int q = q0 + iq;
                int d = d0 + dc;
                if (q < S) {
                    long off = baseBH + (long)q * D + d;
                    float prev = Accum[off];
                    Accum[off] = prev * alpha + sum;
                }
            }
            __syncthreads();
        } // end dv tiles
    } // end k tiles

    // Normalize: O = Accum / l_i  (convert to FP16)
    for (int dv = 0; dv < num_d_tiles; ++dv) {
        const int d0 = dv * DTILE;
        const int cur_dv = min(DTILE, D - d0);
        long dv_tile_elems = (long)num_q * (long)cur_dv;

        for (long idx = tid; idx < dv_tile_elems; idx += BLOCK_THREADS) {
            int iq = idx / cur_dv;
            int dc = idx % cur_dv;

            float denom = s_l[iq];
            // Guard against zero denom (shouldn't happen, but safe)
            denom = denom > 0.0f ? denom : 1.0f;

            int q = q0 + iq;
            int d = d0 + dc;
            if (q < S) {
                long off = baseBH + (long)q * D + d;
                float val = Accum[off] / denom;
                O[off] = __float2half(val);
            }
        }
        __syncthreads();
    }
}

// =================== PyTorch Extension Wrapper ===================

at::Tensor run(at::Tensor Q, at::Tensor K, at::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA tensors (ROCm).");
    TORCH_CHECK(Q.scalar_type() == at::kHalf && K.scalar_type() == at::kHalf && V.scalar_type() == at::kHalf,
                "Q, K, V must be float16 (Half).");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q, K, V must be contiguous.");

    TORCH_CHECK(Q.sizes() == K.sizes() && K.sizes() == V.sizes(),
                "Q, K, V must have the same shape [B, H, S, D].");

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    // Allocate output (Half) and auxiliary FP32 accumulator
    auto O = at::empty_like(Q);
    auto Accum = at::zeros({B, H, S, D}, at::device(Q.device()).dtype(at::kFloat));

    dim3 block(BLOCK_THREADS, 1, 1);
    dim3 grid(CEILDIV((int)S, TQ), (unsigned int)H, (unsigned int)B);

    hipLaunchKernelGGL(
        sdpa_kernel,
        grid, block, 0, 0,
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
        Accum.data_ptr<float>(),
        (int)B, (int)H, (int)S, (int)D
    );

    hipError_t errSync = hipDeviceSynchronize();
    TORCH_CHECK(errSync == hipSuccess, "HIP device sync failed: ", hipGetErrorString(errSync));

    hipError_t errAsync = hipGetLastError();
    TORCH_CHECK(errAsync == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(errAsync));

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Scaled Dot-Product Attention (HIP, FlashAttention-style, FP16 inputs)");
}