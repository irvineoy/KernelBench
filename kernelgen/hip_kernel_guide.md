# HIP Kernel Development Guide for AMD MI300X

## Hardware Specifications (MI300X)

**Architecture**: gfx942 (CDNA3) - also supports gfx9-4-generic
- **Wavefront Size**: 64 threads
- **Compute Units**: 304 CUs
- **SIMDs per CU**: 4
- **Shader Engines**: 32
- **Max Clock**: 2100 MHz
- **LDS per CU**: 64 KB (GROUP segment)
- **Cache**: L1 32KB/CU (128-byte cacheline), L2 4096KB, L3 262144KB
- **HBM**: ~192GB (200998912 KB, 2MB granule recommended)
- **Max Threads/Block**: 1024
- **Max Waves/CU**: 32
- **Max Work-items/CU**: 2048
- **XNACK**: Disabled (non-pageable memory)
- **SRAMECC**: Enabled

## Required Code Structure

All HIP kernels must include:

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void kernel_name(const float* __restrict__ input,
                            float* __restrict__ output,
                            int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;  // Bounds check
    // Implementation
}

// C wrapper for PyTorch integration - REQUIRED
extern "C" void run_kernel(void* input, void* output, int size) {
    float* d_input = static_cast<float*>(input);
    float* d_output = static_cast<float*>(output);

    int block_size = 256;  // Multiple of 64
    int grid_size = (size + block_size - 1) / block_size;

    kernel_name<<<grid_size, block_size>>>(d_input, d_output, size);

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;
    }
}

int main() {
    // Test/profiling code
    return 0;
}
```

## Core Optimization Principles

### 1. Thread Configuration
- **Block size MUST be multiple of 64** (wavefront size)
- Recommended: 256 threads/block (good balance)
- Scale to 512 if no register pressure; reduce to 128 if spilling occurs

### 2. Memory Access Patterns

**Coalesced Access** (GOOD):
```cpp
dst[tid] = src[tid];  // Consecutive threads → consecutive memory
float4 vec = *reinterpret_cast<const float4*>(&data[idx]);
```

**Strided Access** (BAD):
```cpp
dst[tid] = src[tid * stride];  // stride > 1 causes poor bandwidth
```

**Key Rules**:
- Align to 128-byte cache lines (MI300X cacheline size)
- Use vectorized loads (float4, double2) when aligned
- All 64 threads in wavefront should access consecutive memory
- Memory allocation granule: 4KB min, 2MB recommended for large buffers

### 3. Shared Memory (LDS) Optimization

```cpp
// Add padding to avoid bank conflicts (32 banks, 4-byte wide)
__shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
```

**Trade-offs**:
- Budget: 64 KB per CU
- More LDS/block → fewer concurrent blocks → lower occupancy
- Calculate: `occupancy = min(32 waves, 64KB/LDS_per_block)`

### 4. Eliminate Thread Divergence

**BAD** (divergent branches):
```cpp
if (in[tid] > 0.0f)
    out[tid] = in[tid] * 2.0f;
else
    out[tid] = in[tid] * -2.0f;
```

**GOOD** (predication):
```cpp
float v = in[tid];
int p = v > 0.0f;
out[tid] = p * (v * 2.0f) + (1 - p) * (v * -2.0f);
```

### 5. Occupancy Optimization

```cpp
// Hint compiler for occupancy
__global__ __launch_bounds__(256, 8)  // 256 threads, min 8 blocks/CU
void kernel(...) { ... }
```

**Occupancy limited by**:
- VGPRs per thread (256 max per SIMD)
- LDS per block (64 KB per CU)
- Block size (max 1024 threads)

### 6. Advanced Techniques

**Loop Unrolling**:
```cpp
#pragma unroll 4
for (int i = 0; i < N; i++) { ... }
```

**Vectorized Operations**:
```cpp
float4 a = make_float4(x1, x2, x3, x4);
// Process 4 elements at once
```

**Wave-Level Reduction**:
```cpp
float sum = value;
for (int offset = 32; offset > 0; offset >>= 1) {
    sum += __shfl_down(sum, offset);
}
```

## Memory Management Rules

- **Input/output are device pointers** - already on GPU
- NO `hipMalloc`/`hipFree` in wrapper function
- For multiple inputs: pack consecutively in input buffer
- For multiple outputs: pack consecutively in output buffer
- ALWAYS check HIP API return values

## Correctness First, Performance Second

**Iteration 0 (Naive)**:
- Prioritize correctness over performance
- Use double precision if needed for accuracy
- Handle edge cases and boundary conditions
- Verify outputs match PyTorch exactly

**Iteration 1+ (Optimization)**:
- Maintain correctness while optimizing
- Apply targeted optimizations based on profiling
- Validate each optimization doesn't break correctness

## Common Performance Patterns

### Matrix Multiplication Tiling
```cpp
#define TILE 16
__shared__ float As[TILE][TILE + 1];
__shared__ float Bs[TILE][TILE + 1];
// Tile and reuse via LDS
```

### Reduction Pattern
```cpp
// Block-level reduction with LDS
__shared__ float sdata[256];
sdata[tid] = partial_sum;
__syncthreads();
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

## Performance Checklist

- [ ] Block size multiple of 64
- [ ] Coalesced memory access
- [ ] Vectorized loads where possible
- [ ] LDS padding to avoid bank conflicts
- [ ] Minimized thread divergence
- [ ] Proper bounds checking
- [ ] Error handling for HIP calls
- [ ] `__launch_bounds__` if occupancy critical

## Profiling

```bash
# Compile with optimization (choose appropriate arch)
hipcc -O3 -march=gfx942 kernel.cpp -o kernel
# OR for broader compatibility:
hipcc -O3 --offload-arch=gfx9-4-generic kernel.cpp -o kernel

# Profile
rocprof --stats ./kernel
rocprof-compute analyze --path profile_data
```

**Supported ISAs**:
- `amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-` (specific)
- `amdgcn-amd-amdhsa--gfx9-4-generic:sramecc+:xnack-` (generic)

## Common Bottlenecks

| Issue | Cause | Solution |
|-------|-------|----------|
| Low Occupancy | High VGPR/LDS | Reduce resources, smaller blocks |
| Poor Bandwidth | Uncoalesced access | Fix stride patterns, vectorize |
| Bank Conflicts | LDS access pattern | Add padding, reorder access |
| Divergence | Branching | Use predication |

## rocSOLVER-Specific Notes

When optimizing rocSOLVER functions:
- **Preserve function signatures** - do NOT create standalone kernels
- Keep all `#include` statements intact
- Maintain rocSOLVER API compatibility
- Optimize WITHIN existing functions
- Use `rocblas_device_malloc`, not `hipMalloc`
- Focus on micro-optimizations: loops, memory patterns, registers
- Do NOT add `extern "C"` wrappers or `main()` functions
