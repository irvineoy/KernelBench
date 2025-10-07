#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_SIZE 16
#define WARP_SIZE 64

// Optimized depthwise convolution kernel for asymmetric (kernel_h, 1) kernel
// Each thread block handles a tile of the output
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation) {
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Output position
    int out_w = blockIdx.x * TILE_SIZE + tx;
    int out_h = blockIdx.y * TILE_SIZE + ty;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;
    
    if (b >= batch_size || c >= channels || out_h >= out_height || out_w >= out_width) {
        return;
    }
    
    // Compute convolution
    float sum = 0.0f;
    
    // Since kernel width is 1, we only need to iterate over kernel height
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        int in_h = out_h * stride - padding + kh * dilation;
        int in_w = out_w * stride - padding;  // kernel_w = 1, so kw = 0
        
        // Bounds check
        if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
            int input_idx = ((b * channels + c) * height + in_h) * width + in_w;
            int weight_idx = c * kernel_h + kh;  // kernel_w = 1
            sum += input[input_idx] * weight[weight_idx];
        }
    }
    
    // Write output
    int output_idx = ((b * channels + c) * out_height + out_h) * out_width + out_w;
    output[output_idx] = sum;
}

// Optimized version with shared memory for larger kernels
__global__ void depthwise_conv2d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation) {
    
    // Shared memory for input tile and weights
    __shared__ float s_input[TILE_SIZE + 16][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    __shared__ float s_weight[32];  // Cache weights for current channel
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * TILE_SIZE + tx;
    
    // Output position
    int out_w = blockIdx.x * TILE_SIZE + tx;
    int out_h = blockIdx.y * TILE_SIZE + ty;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;
    
    // Load weights into shared memory (one thread loads all)
    if (tid < kernel_h) {
        s_weight[tid] = weight[c * kernel_h + tid];
    }
    __syncthreads();
    
    if (b >= batch_size || c >= channels || out_h >= out_height || out_w >= out_width) {
        return;
    }
    
    // Compute convolution
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int kh = 0; kh < kernel_h; ++kh) {
        int in_h = out_h * stride - padding + kh * dilation;
        int in_w = out_w * stride - padding;
        
        if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
            int input_idx = ((b * channels + c) * height + in_h) * width + in_w;
            sum += input[input_idx] * s_weight[kh];
        }
    }
    
    // Write output
    int output_idx = ((b * channels + c) * out_height + out_h) * out_width + out_w;
    output[output_idx] = sum;
}

at::Tensor run(at::Tensor input, at::Tensor weight) {
    // Extract dimensions from input and weight tensors
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    // Weight shape: [channels, 1, kernel_h, kernel_w]
    // For depthwise conv with groups=channels: [channels, 1, kernel_h, 1]
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);  // Should be 1
    
    // Default configuration (since not passed at runtime)
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    // Allocate output tensor
    auto output = at::zeros({batch_size, channels, out_height, out_width}, input.options());
    
    // Launch configuration
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * channels
    );
    
    // Choose kernel based on kernel size
    if (kernel_h <= 5) {
        // Use simple kernel for small kernels
        hipLaunchKernelGGL(
            depthwise_conv2d_kernel,
            grid, block, 0, 0,
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels, height, width,
            kernel_h, out_height, out_width,
            stride, padding, dilation
        );
    } else {
        // Use shared memory kernel for larger kernels
        hipLaunchKernelGGL(
            depthwise_conv2d_shared_kernel,
            grid, block, 0, 0,
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels, height, width,
            kernel_h, out_height, out_width,
            stride, padding, dilation
        );
    }
    
    // Error checking
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        AT_ERROR("HIP kernel failed: ", hipGetErrorString(err));
    }
    
    err = hipGetLastError();
    if (err != hipSuccess) {
        AT_ERROR("HIP kernel launch failed: ", hipGetErrorString(err));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Depthwise 2D convolution with asymmetric kernel (HIP)");
}