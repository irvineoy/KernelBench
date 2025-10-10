// Conv1d forward HIP kernel optimized for AMD MI300X (gfx942)
// Implements: y = conv1d(x, weight, bias?; stride=3, dilation=4, padding=0, groups=1)
// Only tensor parameters are accepted by the exported 'run' function.

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Tiling parameters
#define TX 256            // threads across output length (multiple of 64)
#define TY 4              // threads across output channel tile
#define C_TILE 16         // input-channel tile size (fits LDS)

// Kernel: computes Conv1d forward with stride=3, dilation=4, padding=0
// Input:  x [N, Cin, Lin], weight [Cout, Cin, K], bias [Cout] (optional)
// Output: y [N, Cout, Lout]
__global__ void conv1d_fwd_tiled_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,  // can be nullptr
    float* __restrict__ y,
    int N, int Cin, int Lin,
    int Cout, int K, int Lout)
{
    __shared__ float s_input[C_TILE][TX + 1];     // +1 to avoid LDS bank conflicts
    __shared__ float s_weight[TY][C_TILE];        // per-oc tile weights for current kh

    const int tx = threadIdx.x;                   // [0, TX)
    const int ty = threadIdx.y;                   // [0, TY)

    const int n  = blockIdx.z;                    // batch index
    const int oc = blockIdx.y * TY + ty;          // output channel
    const int t0 = blockIdx.x * TX;               // base output position for tile
    const int t  = t0 + tx;                       // output position (length dim)

    const bool valid_oc = (oc < Cout);
    const bool valid_t  = (t  < Lout);
    const bool valid    = (n  < N) && valid_oc && valid_t;

    // Use hard-coded stride and dilation as per the model description/example
    const int stride   = 3;
    const int dilation = 4;
    const int padding  = 0;   // valid conv

    // Base offsets to reduce repeated multiplications
    const int x_batch_stride = Cin * Lin;
    const int y_batch_stride = Cout * Lout;
    const int w_out_stride   = Cin * K;

    float acc = 0.0f;

    // Loop over input channels in tiles of C_TILE
    for (int c_base = 0; c_base < Cin; c_base += C_TILE) {
        const int c_tile = min(C_TILE, Cin - c_base);

        // For each kernel position kh, cooperatively:
        // 1) Load input tile for kh into LDS (s_input[ci][tx])
        // 2) Load weight tile for current oc and kh into LDS (s_weight[ty][ci])
        // 3) Accumulate into acc using the tiles
        for (int kh = 0; kh < K; ++kh) {
            // 1) Load s_input: each thread loads some rows (channels) for its tx
            // Map threadIdx.y to channel rows to distribute work
            for (int ci = ty; ci < c_tile; ci += TY) {
                // Compute input position for this output t and kh
                // in_pos = t * stride - padding + kh * dilation
                int in_pos = t * stride - padding + kh * dilation;

                float val = 0.0f;
                if (t < Lout) { // only compute meaningful position if t is in range
                    if (in_pos >= 0 && in_pos < Lin) {
                        int c = c_base + ci;
                        int x_idx = (n * x_batch_stride) + (c * Lin) + in_pos;
                        val = x[x_idx];
                    }
                }
                s_input[ci][tx] = val;
            }

            // 2) Load s_weight: have tx==0 threads load per-oc weight tile rows
            if (tx == 0 && valid_oc) {
                for (int ci = 0; ci < c_tile; ++ci) {
                    int c = c_base + ci;
                    int w_idx = (oc * w_out_stride) + (c * K) + kh;
                    s_weight[ty][ci] = w[w_idx];
                }
            }

            __syncthreads();

            // 3) Accumulate using the tiles for this kh
            if (valid) {
                float sum_k = 0.0f;
                #pragma unroll
                for (int ci = 0; ci < C_TILE; ++ci) {
                    if (ci < c_tile) {
                        sum_k = fmaf(s_input[ci][tx], s_weight[ty][ci], sum_k);
                    }
                }
                acc += sum_k;
            }

            __syncthreads();
        }
    }

    // Add bias and write output
    if (valid) {
        if (b != nullptr) acc += b[oc];
        int y_idx = (n * y_batch_stride) + (oc * Lout) + t;
        y[y_idx] = acc;
    }
}

static inline dim3 compute_grid(int Lout, int Cout, int N) {
    dim3 block(TX, TY, 1);
    dim3 grid((Lout + TX - 1) / TX, (Cout + TY - 1) / TY, N);
    return grid;
}

// Host wrapper: run without bias
at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be [N, Cin, Lin]");
    TORCH_CHECK(weight.dim() == 3, "weight must be [Cout, Cin, K]");

    auto x = input.contiguous();
    auto w = weight.contiguous();

    int64_t N    = x.size(0);
    int64_t Cin  = x.size(1);
    int64_t Lin  = x.size(2);
    int64_t Cout = w.size(0);
    int64_t CinW = w.size(1);
    int64_t K    = w.size(2);

    TORCH_CHECK(Cin == CinW, "weight Cin must match input Cin");

    // Hard-coded stride/dilation/padding based on model description/example
    const int stride   = 3;
    const int dilation = 4;
    const int padding  = 0;

    // Compute output length
    int64_t Lout = 0;
    if (Lin > 0 && K > 0) {
        int64_t tmp = Lin + 2LL * padding - (int64_t)dilation * (K - 1) - 1;
        if (tmp >= 0) Lout = tmp / stride + 1;
    }
    TORCH_CHECK(Lout >= 0, "Invalid output length computed");

    auto y = at::empty({N, Cout, Lout}, x.options());

    dim3 block(TX, TY, 1);
    dim3 grid = compute_grid((int)Lout, (int)Cout, (int)N);

    hipLaunchKernelGGL(
        conv1d_fwd_tiled_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        nullptr,
        y.data_ptr<float>(),
        (int)N, (int)Cin, (int)Lin,
        (int)Cout, (int)K, (int)Lout
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

// Host wrapper: run with bias
at::Tensor run(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA/HIP tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA/HIP tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.dtype() == at::kFloat, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be [N, Cin, Lin]");
    TORCH_CHECK(weight.dim() == 3, "weight must be [Cout, Cin, K]");
    TORCH_CHECK(bias.dim() == 1, "bias must be [Cout]");

    auto x = input.contiguous();
    auto w = weight.contiguous();
    auto b = bias.contiguous();

    int64_t N    = x.size(0);
    int64_t Cin  = x.size(1);
    int64_t Lin  = x.size(2);
    int64_t Cout = w.size(0);
    int64_t CinW = w.size(1);
    int64_t K    = w.size(2);

    TORCH_CHECK(Cin == CinW, "weight Cin must match input Cin");
    TORCH_CHECK(b.size(0) == Cout, "bias size must match Cout");

    // Hard-coded stride/dilation/padding based on model description/example
    const int stride   = 3;
    const int dilation = 4;
    const int padding  = 0;

    int64_t Lout = 0;
    if (Lin > 0 && K > 0) {
        int64_t tmp = Lin + 2LL * padding - (int64_t)dilation * (K - 1) - 1;
        if (tmp >= 0) Lout = tmp / stride + 1;
    }
    TORCH_CHECK(Lout >= 0, "Invalid output length computed");

    auto y = at::empty({N, Cout, Lout}, x.options());

    dim3 block(TX, TY, 1);
    dim3 grid = compute_grid((int)Lout, (int)Cout, (int)N);

    hipLaunchKernelGGL(
        conv1d_fwd_tiled_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)N, (int)Cin, (int)Lin,
        (int)Cout, (int)K, (int)Lout
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run",
          static_cast<at::Tensor(*)(at::Tensor, at::Tensor)>(&run),
          "Conv1d forward (HIP, stride=3, dilation=4, no bias)");
    m.def("run",
          static_cast<at::Tensor(*)(at::Tensor, at::Tensor, at::Tensor)>(&run),
          "Conv1d forward (HIP, stride=3, dilation=4, with bias)");
}