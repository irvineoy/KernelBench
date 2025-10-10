// Transposed 3D Convolution (ConvTranspose3d) optimized HIP kernel for AMD MI300X (gfx942)
// Assumes stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), output_padding=(0,0,0), groups=1
// Weight layout: [in_channels, out_channels, kD, kH, kW]
// Input layout:  [N, in_channels, D_in, H_in, W_in]
// Output layout: [N, out_channels, D_out, H_out, W_out], where
// D_out = D_in + kD - 1, H_out = H_in + kH - 1, W_out = W_in + kW - 1

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <iostream>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Tile width along the output W dimension (multiple of 64 for MI300X wavefront)
#ifndef TILE_W
#define TILE_W 256
#endif

// Gather-form ConvTranspose3d kernel: each thread computes one output element along W
// Grid mapping:
//   blockIdx.x -> tile over W_out
//   blockIdx.y -> linearized (D_out, H_out) plane: oz = y / H_out, oy = y % H_out
//   blockIdx.z -> linearized (N, out_channels):   b  = z / Cout,   oc = z % Cout
__global__ void __launch_bounds__(TILE_W)
convtranspose3d_gather_kernel(
    const float* __restrict__ input,    // [N, Cin, D_in, H_in, W_in]
    const float* __restrict__ weight,   // [Cin, Cout, kD, kH, kW]
    const float* __restrict__ bias,     // [Cout] or nullptr
    float* __restrict__ output,         // [N, Cout, D_out, H_out, W_out]
    int N,
    int Cin,
    int Din, int Hin, int Win,
    int Cout,
    int kD, int kH, int kW,
    int Dout, int Hout, int Wout)
{
    // 1D block along output width for coalesced writes
    int tx = threadIdx.x;
    int ox = blockIdx.x * blockDim.x + tx;
    if (ox >= Wout) return;

    int yz = blockIdx.y;
    int oz = yz / Hout;
    int oy = yz - oz * Hout;

    int bz = blockIdx.z;
    int b  = bz / Cout;
    int oc = bz - b * Cout;

    // Shared cache for the kW weights of the current (ic, oc, kd, kh)
    // kW is usually small (e.g., 7). We cache up to 64; fallback to direct loads if larger.
    __shared__ float s_w[64];

    // Initialize accumulator with bias if provided
    float acc = 0.0f;
    if (bias != nullptr) {
        acc = bias[oc];
    }

    // Loop over input channels and kernel volume (gather form, no atomics)
    // For each output position (oz, oy, ox), we sum contributions from all valid input indices:
    // id = oz - kd, ih = oy - kh, iw = ox - kw
    for (int ic = 0; ic < Cin; ++ic) {
        const float* __restrict__ w_icoc = weight + (((ic * Cout + oc) * kD) * kH) * kW;

        #pragma unroll 1
        for (int kd = 0; kd < kD; ++kd) {
            int id = oz - kd;
            if ((unsigned)id >= (unsigned)Din) continue;

            #pragma unroll 1
            for (int kh = 0; kh < kH; ++kh) {
                int ih = oy - kh;
                if ((unsigned)ih >= (unsigned)Hin) continue;

                const float* __restrict__ w_slice = w_icoc + (kd * kH + kh) * kW;

                if (kW <= 64) {
                    // Load kW weights into shared memory once per (ic,kd,kh) triplet
                    if (tx < kW) {
                        s_w[tx] = w_slice[tx];
                    }
                    __syncthreads();

                    // Accumulate over kernel width
                    #pragma unroll 8
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = ox - kw;
                        if ((unsigned)iw < (unsigned)Win) {
                            int in_idx = (((b * Cin + ic) * Din + id) * Hin + ih) * Win + iw;
                            float v = input[in_idx];
                            acc += v * s_w[kw];
                        }
                    }
                    __syncthreads();
                } else {
                    // Fallback: directly read weights without caching
                    #pragma unroll 4
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = ox - kw;
                        if ((unsigned)iw < (unsigned)Win) {
                            int in_idx = (((b * Cin + ic) * Din + id) * Hin + ih) * Win + iw;
                            float v = input[in_idx];
                            float wv = w_slice[kw];
                            acc += v * wv;
                        }
                    }
                }
            }
        }
    }

    // Write result
    int out_idx = (((b * Cout + oc) * Dout + oz) * Hout + oy) * Wout + ox;
    output[out_idx] = acc;
}

// Wrapper: input, weight -> output
// Only tensor parameters are accepted by this wrapper per requirements.
at::Tensor run(at::Tensor input, at::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Weight must be float32");

    auto x = input.contiguous();
    auto w = weight.contiguous();

    // Extract dimensions
    int64_t N   = x.size(0);
    int64_t Cin = x.size(1);
    int64_t Din = x.size(2);
    int64_t Hin = x.size(3);
    int64_t Win = x.size(4);

    TORCH_CHECK(w.size(0) == Cin, "Weight shape[0] must equal in_channels (Cin). Got: ", w.size(0), " vs Cin=", Cin);
    int64_t Cout = w.size(1);
    int64_t kD   = w.size(2);
    int64_t kH   = w.size(3);
    int64_t kW   = w.size(4);

    // Assumed defaults: stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), output_padding=(0,0,0), groups=1
    int64_t Dout = Din + kD - 1;
    int64_t Hout = Hin + kH - 1;
    int64_t Wout = Win + kW - 1;

    auto out = at::empty({N, Cout, Dout, Hout, Wout}, x.options());

    // Launch configuration
    dim3 block(TILE_W, 1, 1);
    dim3 grid(
        (unsigned)((Wout + TILE_W - 1) / TILE_W),
        (unsigned)(Dout * Hout),
        (unsigned)(N * Cout)
    );

    hipLaunchKernelGGL(
        convtranspose3d_gather_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        nullptr, // no bias
        out.data_ptr<float>(),
        (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win,
        (int)Cout, (int)kD, (int)kH, (int)kW,
        (int)Dout, (int)Hout, (int)Wout
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

// Overload with bias support: input, weight, bias -> output
at::Tensor run(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA/HIP tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "Bias must be float32");

    auto x = input.contiguous();
    auto w = weight.contiguous();
    auto b = bias.contiguous();

    // Extract dimensions
    int64_t N   = x.size(0);
    int64_t Cin = x.size(1);
    int64_t Din = x.size(2);
    int64_t Hin = x.size(3);
    int64_t Win = x.size(4);

    TORCH_CHECK(w.size(0) == Cin, "Weight shape[0] must equal in_channels (Cin). Got: ", w.size(0), " vs Cin=", Cin);
    int64_t Cout = w.size(1);
    int64_t kD   = w.size(2);
    int64_t kH   = w.size(3);
    int64_t kW   = w.size(4);

    TORCH_CHECK(b.numel() == Cout, "Bias size must equal out_channels (Cout). Got: ", b.numel(), " vs Cout=", Cout);

    // Assumed defaults
    int64_t Dout = Din + kD - 1;
    int64_t Hout = Hin + kH - 1;
    int64_t Wout = Win + kW - 1;

    auto out = at::empty({N, Cout, Dout, Hout, Wout}, x.options());

    // Launch configuration
    dim3 block(TILE_W, 1, 1);
    dim3 grid(
        (unsigned)((Wout + TILE_W - 1) / TILE_W),
        (unsigned)(Dout * Hout),
        (unsigned)(N * Cout)
    );

    hipLaunchKernelGGL(
        convtranspose3d_gather_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win,
        (int)Cout, (int)kD, (int)kH, (int)kW,
        (int)Dout, (int)Hout, (int)Wout
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel failed: ", hipGetErrorString(err));
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Overloads to support models with/without bias
    m.def("run", (at::Tensor(*)(at::Tensor, at::Tensor)) &run,
          "ConvTranspose3d (stride=1, padding=0, dilation=1) - no bias (HIP)");
    m.def("run", (at::Tensor(*)(at::Tensor, at::Tensor, at::Tensor)) &run,
          "ConvTranspose3d (stride=1, padding=0, dilation=1) - with bias (HIP)");
}