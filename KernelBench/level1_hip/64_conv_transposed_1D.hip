// ConvTranspose1d (transpose convolution) for ROCm HIP, optimized for MI300X (gfx942)
// Assumes defaults used by the provided PyTorch model:
//   stride = 1, padding = 0, output_padding = 0, dilation = 1, groups = 1, bias = False
// Weight layout (PyTorch): [in_channels, out_channels, kernel_size] when groups == 1
// Input layout:  [batch, in_channels, length]
// Output layout: [batch, out_channels, length_out], where length_out = length + (kernel_size - 1)

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256  // Multiple of wavefront size (64) -> good occupancy
#endif

// Kernel: each block computes a 1D tile along output length for one (batch b, out_channel oc)
// Output-stationary accumulation to avoid atomics.
// We cache the per-(oc) filter weights [Ci x K] in LDS to reduce global reads.
__global__ void conv1d_transpose_kernel_shared(
    const float* __restrict__ x,   // [N, Ci, Lin]
    const float* __restrict__ w,   // [Ci, Co, K]  (groups=1)
    float* __restrict__ y,         // [N, Co, Lout]
    int N,
    int Ci,
    int Co,
    int Lin,
    int K,
    int Lout)
{
    extern __shared__ float s_w[]; // size = Ci * K floats, weight tile for current oc

    const int oc = blockIdx.y;  // [0, Co)
    const int b  = blockIdx.z;  // [0, N)
    const int tx = threadIdx.x;
    const int x_out = blockIdx.x * blockDim.x + tx;

    if (oc >= Co || b >= N || x_out >= Lout) {
        // Still need to populate shared weights for correctness across all threads in blockIdx.y/b,
        // but we can early return since no output write will occur. It's safe to return here.
        return;
    }

    // Load weights for this oc into shared memory (cooperative load across threads in the block)
    // w layout: [Ci, Co, K] -> linear index ((ic * Co + oc) * K + k)
    const int weight_elems = Ci * K;
    for (int idx = tx; idx < weight_elems; idx += blockDim.x) {
        int ic = idx / K;
        int k  = idx % K;
        s_w[idx] = w[(ic * Co + oc) * K + k];
    }
    __syncthreads();

    // Compute valid kernel window for given x_out to keep x_in in [0, Lin-1]
    // x_in = x_out - k
    // 0 <= x_out - k < Lin  =>  x_out - (Lin-1) <= k <= x_out
    int k_min = x_out - (Lin - 1);
    if (k_min < 0) k_min = 0;
    int k_max = x_out < (K - 1) ? x_out : (K - 1);

    // Accumulate
    float acc = 0.0f;

    // Base offsets
    const int x_b_base = b * Ci * Lin;
    // y index base: ((b * Co + oc) * Lout)
    const int y_base = (b * Co + oc) * Lout;

    // Loop over input channels and kernel taps
    // For fixed (b, ic, k), all threads in a wave read consecutive x positions -> coalesced
    for (int ic = 0; ic < Ci; ++ic) {
        const int x_b_ic_base = x_b_base + ic * Lin;

        // s_w is laid out as [ic][k] contiguous in k: s_w[ic*K + k]
        const float* __restrict__ w_ic = &s_w[ic * K];

        // Restrict kernel taps to the valid range (precomputed)
        #pragma unroll 4
        for (int k = k_min; k <= k_max; ++k) {
            int x_in = x_out - k; // guaranteed in [0, Lin-1] due to k_min/k_max
            float xv = x[x_b_ic_base + x_in];
            acc += xv * w_ic[k];
        }
    }

    // Store result
    y[y_base + x_out] = acc;
}

at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA/HIP device");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA/HIP device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Weight must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must have shape [N, Ci, L]");
    TORCH_CHECK(weight.dim() == 3, "Weight must have shape [Ci, Co, K] for groups=1 ConvTranspose1d default");

    // Enforce contiguous tensors for predictable strides
    auto x = input.contiguous();
    auto w = weight.contiguous();

    const int64_t N    = x.size(0);
    const int64_t Ci   = x.size(1);
    const int64_t Lin  = x.size(2);
    const int64_t Ci_w = w.size(0);
    const int64_t Co   = w.size(1);  // since groups=1, this equals out_channels
    const int64_t K    = w.size(2);

    TORCH_CHECK(Ci == Ci_w, "Weight in_channels must match input in_channels (groups=1). Got Ci=", Ci, " Ci_w=", Ci_w);

    // Defaults from the provided model: stride=1, padding=0, dilation=1, output_padding=0, groups=1
    // Lout = (Lin - 1) * stride - 2*padding + dilation*(K - 1) + output_padding + 1
    // With defaults => Lout = Lin + (K - 1)
    const int64_t Lout = Lin + (K - 1);

    auto y = at::empty({N, Co, Lout}, x.options());

    // Launch configuration
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid((Lout + BLOCK_SIZE - 1) / BLOCK_SIZE, Co, N);

    // Shared memory size per block: Ci * K floats
    size_t shared_bytes = static_cast<size_t>(Ci) * static_cast<size_t>(K) * sizeof(float);

    hipLaunchKernelGGL(
        conv1d_transpose_kernel_shared,
        grid, block, shared_bytes, 0,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(Ci),
        static_cast<int>(Co),
        static_cast<int>(Lin),
        static_cast<int>(K),
        static_cast<int>(Lout)
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "ConvTranspose1d (stride=1, padding=0, dilation=1, output_padding=0, groups=1) [HIP]");
}