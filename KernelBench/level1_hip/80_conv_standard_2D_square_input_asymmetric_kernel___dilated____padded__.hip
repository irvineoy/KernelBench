// Optimized HIP kernel for standard 2D convolution (NCHW) with asymmetric kernel, dilation, and padding.
// Target: AMD MI300X (gfx942), wavefront size 64
// Single file includes kernels and PyTorch extension bindings.

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_W 16
#define TILE_H 16

// Tiled direct convolution kernel with LDS caching of input tile and all weights for the current output channel.
// Grid: (ceil(out_W/TILE_W), ceil(out_H/TILE_H), N * C_out)
// Block: (TILE_W, TILE_H)
__global__ void conv2d_tiled_kernel(
    const float* __restrict__ input,     // [N, C_in, H, W]
    const float* __restrict__ weight,    // [C_out, C_in, K_h, K_w]
    const float* __restrict__ bias,      // [C_out] or nullptr
    float* __restrict__ output,          // [N, C_out, out_H, out_W]
    int N, int C_in, int H, int W,
    int C_out, int K_h, int K_w,
    int out_H, int out_W,
    int pad_h, int pad_w,
    int dil_h, int dil_w,
    int stride_h, int stride_w)
{
    // Derive tile origin in output space
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int ow0 = bx * TILE_W;
    const int oh0 = by * TILE_H;

    const int ow = ow0 + tx;
    const int oh = oh0 + ty;

    // Map block to (n, co)
    const int co = bz % C_out;
    const int n  = bz / C_out;

    // Dynamic shared memory layout:
    // s_in:  (tile_h*stride_h + (K_h-1)*dil_h) x (tile_w*stride_w + (K_w-1)*dil_w + 1 padding)
    // s_w:   C_in * K_h * K_w
    extern __shared__ float smem[];
    // Worst-case per-launch sizes (computed in host and passed via sharedMemBytes):
    // We'll compute local sizes for this block to allow edge tiles smaller than TILE_W/H
    // Still index within the max allocation.

    // Compute block tile sizes (may be smaller at edges)
    const int b_tile_h = min(TILE_H, out_H - oh0);
    const int b_tile_w = min(TILE_W, out_W - ow0);

    // Effective shared tile sizes for input considering stride and dilation
    const int s_in_h = b_tile_h * stride_h + (K_h - 1) * dil_h;
    const int s_in_w = b_tile_w * stride_w + (K_w - 1) * dil_w;
    // Padded width to reduce bank conflicts
    // Note: allocation used worst-case sizes; here we use per-block computed dims but bounded by allocation.
    // The host side allocates based on TILE dims (worst-case), so these indices remain within bounds.
    // We'll use the max padded width for indexing math to match allocation layout.
    // However, it's simpler to compute offsets using per-block s_in_w_pad provided allocation >= this.
    // Host allocated with s_in_w_max + 1, so using per-block pad is fine.
    const int s_in_w_pad = s_in_w + 1;
    const int s_in_elems = s_in_h * s_in_w_pad;

    // Compute base pointers in shared memory
    // Layout: [s_in_elems] then [C_in*K_h*K_w]
    float* s_in = smem;
    float* s_w_all = s_in + s_in_elems;

    // Preload all weights for this (co) into shared memory once
    const int block_threads = blockDim.x * blockDim.y;
    int tid = ty * blockDim.x + tx;

    int total_w_elems = C_in * K_h * K_w;
    for (int idx = tid; idx < total_w_elems; idx += block_threads) {
        int cin = idx / (K_h * K_w);
        int rem = idx % (K_h * K_w);
        int kh  = rem / K_w;
        int kw  = rem % K_w;

        // weight layout: [C_out, C_in, K_h, K_w]
        int w_idx = ((co * C_in + cin) * K_h + kh) * K_w + kw;
        s_w_all[idx] = weight[w_idx];
    }
    __syncthreads();

    // Compute base input coordinates for this output tile
    const int in_h_base = oh0 * stride_h - pad_h;
    const int in_w_base = ow0 * stride_w - pad_w;

    // Accumulator for this thread's output element
    float acc = 0.0f;
    if (bias != nullptr && oh < out_H && ow < out_W) {
        acc = bias[co];
    }

    // Iterate over input channels
    for (int cin = 0; cin < C_in; ++cin) {
        // Load input tile for this input channel into shared memory (with padding and dilation-aware extent)
        // s_in[y, x] corresponds to input[n, cin, in_h_base + y, in_w_base + x], zero if OOB
        for (int idx = tid; idx < s_in_h * s_in_w; idx += block_threads) {
            int y = idx / s_in_w;
            int x = idx % s_in_w;

            int in_h = in_h_base + y;
            int in_w = in_w_base + x;

            float v = 0.0f;
            if ((unsigned)in_h < (unsigned)H && (unsigned)in_w < (unsigned)W) {
                int in_idx = ((n * C_in + cin) * H + in_h) * W + in_w;
                v = input[in_idx];
            }
            s_in[y * s_in_w_pad + x] = v;
        }
        __syncthreads();

        // Compute partial sums for valid output threads
        if (oh < out_H && ow < out_W) {
            const float* __restrict__ wbase = s_w_all + cin * (K_h * K_w);

            // Access s_in at: y = ty*stride_h + kh*dil_h, x = tx*stride_w + kw*dil_w
            // Bounds guaranteed by construction of s_in_h/s_in_w
            #pragma unroll
            for (int kh = 0; kh < K_h; ++kh) {
                int y_in = ty * stride_h + kh * dil_h;
                int y_off = y_in * s_in_w_pad;
                #pragma unroll
                for (int kw = 0; kw < K_w; ++kw) {
                    int x_in = tx * stride_w + kw * dil_w;
                    float wv = wbase[kh * K_w + kw];
                    float iv = s_in[y_off + x_in];
                    acc += iv * wv;
                }
            }
        }
        __syncthreads(); // Reuse s_in for next channel
    }

    // Write output
    if (oh < out_H && ow < out_W) {
        int out_idx = ((n * C_out + co) * out_H + oh) * out_W + ow;
        output[out_idx] = acc;
    }
}

// Host wrapper: accepts only tensors (input, weight[, bias]); no scalar config params are passed.
// Stride, padding, dilation are derived from the model context and hard-coded for this specific target model.
static at::Tensor run_impl(at::Tensor input, at::Tensor weight, c10::optional<at::Tensor> bias_opt) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "Only float32 tensors are supported");
    TORCH_CHECK(weight.dtype() == at::kFloat, "Only float32 tensors are supported");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous (NCHW)");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous (OIHW)");

    // Extract dims
    int64_t N  = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H  = input.size(2);
    int64_t W  = input.size(3);

    int64_t C_out = weight.size(0);
    int64_t K_h   = weight.size(2);
    int64_t K_w   = weight.size(3);

    TORCH_CHECK(weight.size(1) == C_in, "weight C_in mismatch");

    // Hard-coded configuration for the target model:
    // stride = 1, padding = (2, 4), dilation = (2, 3)
    // This matches the provided PyTorch model initialization in the prompt.
    const int stride_h = 1;
    const int stride_w = 1;
    const int pad_h    = 2;
    const int pad_w    = 4;
    const int dil_h    = 2;
    const int dil_w    = 3;

    // Compute output dims (PyTorch formula)
    int64_t out_H = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) / stride_h + 1;
    int64_t out_W = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) / stride_w + 1;
    TORCH_CHECK(out_H > 0 && out_W > 0, "Computed output size is invalid");

    // Prepare bias
    const float* bias_ptr = nullptr;
    at::Tensor bias;
    if (bias_opt.has_value() && bias_opt->defined()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA/HIP tensor");
        TORCH_CHECK(bias.dtype() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C_out, "bias shape must be [C_out]");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        bias_ptr = bias.data_ptr<float>();
    }

    // Allocate output
    auto out = at::empty({N, C_out, out_H, out_W}, input.options());

    // Launch configuration
    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid(
        (out_W + TILE_W - 1) / TILE_W,
        (out_H + TILE_H - 1) / TILE_H,
        N * C_out
    );

    // Dynamic shared memory size (worst-case for any block)
    const int s_in_h_max = TILE_H * stride_h + (K_h - 1) * dil_h;
    const int s_in_w_max = TILE_W * stride_w + (K_w - 1) * dil_w;
    const int s_in_elems_max = s_in_h_max * (s_in_w_max + 1); // +1 padding to reduce bank conflicts
    const int w_elems = C_in * K_h * K_w;
    size_t shared_bytes = (s_in_elems_max + w_elems) * sizeof(float);

    // Launch
    hipLaunchKernelGGL(
        conv2d_tiled_kernel,
        grid, block, shared_bytes, 0,
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        out.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C_in), static_cast<int>(H), static_cast<int>(W),
        static_cast<int>(C_out), static_cast<int>(K_h), static_cast<int>(K_w),
        static_cast<int>(out_H), static_cast<int>(out_W),
        pad_h, pad_w, dil_h, dil_w, stride_h, stride_w
    );

    // Error checks
    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));
    err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel execution failed: ", hipGetErrorString(err));

    return out;
}

// PyTorch-visible entry points: only tensor parameters allowed
at::Tensor run_no_bias(at::Tensor input, at::Tensor weight) {
    return run_impl(input, weight, c10::nullopt);
}
at::Tensor run_with_bias(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    return run_impl(input, weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run_no_bias, "Conv2D tiled (input, weight)");
    m.def("run", &run_with_bias, "Conv2D tiled (input, weight, bias)");
}