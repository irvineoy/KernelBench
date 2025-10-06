# KernelBench HIP Kernel Development

This directory contains HIP kernel implementations and benchmarking tools for KernelBench models.

## Environment Setup

### Prerequisites
- ROCm 6.4.1 installed at `/opt/rocm-6.4.1`
- AMD GPU (gfx90a or gfx942 architecture, e.g., MI250X, MI300X)
- `uv` package manager installed

### Setup Steps

1. **Create and activate virtual environment:**
   ```bash
   cd kernelgen
   make setup
   ```

2. **Activate the environment:**
   ```bash
   source .venv/bin/activate
   # Or use: make act
   ```

The `make setup` command will:
- Create a Python 3.12 virtual environment
- Install PyTorch with ROCm 6.4 support
- Install required dependencies (ninja, numpy)
- Configure ROCm environment variables

## Directory Structure

```
kernelgen/
├── Makefile                    # Environment setup automation
├── README.md                   # This file
├── bench_kernel.py             # Benchmark script
├── level1/                     # Level 1 kernel implementations
│   ├── 1_Square_matrix_multiplication_.cpp   # PyTorch binding
│   ├── 1_Square_matrix_multiplication_.hip   # HIP kernel
│   └── build_cache/            # Compilation cache (auto-generated)
└── .venv/                      # Virtual environment (created by make setup)
```

## Running Benchmarks

### Basic Usage

Benchmark a HIP kernel against its PyTorch implementation:

```bash
python bench_kernel.py --model <model_name>
```

### Example

```bash
# Benchmark square matrix multiplication
python bench_kernel.py --model 1_Square_matrix_multiplication_
```

### What the Benchmark Does

The benchmark script will:

1. **Load the Python model** from `KernelBench/level1/<model_name>.py`
2. **Compile the HIP kernel** from `kernelgen/level1/<model_name>.{cpp,hip}`
3. **Generate inputs** using the model's `get_inputs()` function
4. **Run correctness checks:**
   - Compares HIP kernel output vs PyTorch model output
   - Reports max/mean/relative differences
5. **Run performance benchmarks:**
   - Warmup iterations (5x)
   - Timed iterations (20x)
   - Reports average execution time for both implementations
   - Calculates speedup ratio

### Output Example

```
================================================================================
Testing: 1_Square_matrix_multiplication_
================================================================================

Loading Python model from /root/KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py...
✓ Python model loaded

Input shapes: [torch.Size([4096, 4096]), torch.Size([4096, 4096])]
Loading HIP kernel from /root/KernelBench/kernelgen/level1...
✓ HIP kernel loaded successfully

--------------------------------------------------------------------------------
Running PyTorch model...
Output shape: torch.Size([4096, 4096])

Running HIP kernel...
Output shape: torch.Size([4096, 4096])

--------------------------------------------------------------------------------
Correctness Check:
  Max absolute difference: 1.234e-03
  Mean absolute difference: 2.345e-05
  Max relative difference: 3.456e-04
  ✓ PASSED (threshold: 1e-02)

--------------------------------------------------------------------------------
Performance Benchmark:
  Warming up...

  PyTorch time: 12.345 ms
  HIP kernel time: 8.901 ms
  Speedup: 1.39x
  ✓ HIP kernel is 1.39x faster

================================================================================
Summary:
  Correctness: ✓ PASSED
  Performance: 1.39x speedup
================================================================================
```

## Writing HIP Kernels

### File Naming Convention

For a PyTorch model at `KernelBench/level1/<model_name>.py`, create:
- `kernelgen/level1/<model_name>.cpp` - PyTorch binding
- `kernelgen/level1/<model_name>.hip` - HIP kernel implementation

### Example: Square Matrix Multiplication

**C++ Binding (`1_Square_matrix_multiplication_.cpp`):**
```cpp
#include <torch/extension.h>

at::Tensor matmul_hip(at::Tensor A, at::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul_hip, "Square matrix multiplication (HIP)");
}
```

**HIP Kernel (`1_Square_matrix_multiplication_.hip`):**
```cpp
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/ATen.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Kernel implementation
}

at::Tensor matmul_hip(at::Tensor A, at::Tensor B) {
    // Launch kernel
    hipLaunchKernelGGL(matmul_kernel, ...);
    return C;
}
```

## Compilation Details

The benchmark script uses `torch.utils.cpp_extension.load()` with:
- **Compiler:** `hipcc` for `.hip` files, `c++` for `.cpp` files
- **Optimization:** `-O3` for both C++ and HIP code
- **Target architectures:** `gfx90a`, `gfx942`
- **ROCm includes:** `/opt/rocm/include`
- **ROCm libraries:** `libamdhip64.so`

## Makefile Commands

```bash
make help      # Show available commands
make setup     # Create venv and install dependencies
make clean     # Remove virtual environment
make act       # Activate virtual environment in new shell
```

## Troubleshooting

### Ninja not found
```bash
source .venv/bin/activate
pip install ninja
```

### Compilation errors
- Check that ROCm 6.4.1 is installed: `ls /opt/rocm-6.4.1`
- Verify GPU architecture: `rocminfo | grep gfx`
- Clear build cache: `rm -rf kernelgen/level1/build_cache`

### Import errors
- Ensure virtual environment is activated: `which python` should show `.venv/bin/python`
- Reinstall PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/rocm6.4`

## Performance Tips

1. **Use shared memory** for tile-based algorithms
2. **Optimize memory access patterns** (coalesced reads/writes)
3. **Tune block/grid dimensions** for your GPU
4. **Use `#pragma unroll`** for small loops
5. **Profile with rocprof:** `rocprof --stats python bench_kernel.py --model <name>`
