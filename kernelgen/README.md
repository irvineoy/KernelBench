# KernelBench HIP Kernel Development

KernelBench is an automated framework for generating and benchmarking optimized HIP kernels from PyTorch models using LLMs.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Workflow Overview](#workflow-overview)
- [Kernel Scoring System](#kernel-scoring-system)
- [HIP-PyTorch Interface](#hip-pytorch-interface)
- [LLM Service and Prompting](#llm-service-and-prompting)
- [Running Benchmarks](#running-benchmarks)
- [Writing HIP Kernels](#writing-hip-kernels)
- [Troubleshooting](#troubleshooting)

## Environment Setup

### Prerequisites
- ROCm 6.4.1 or 7.0.0 installed at `/opt/rocm-{version}`
- AMD GPU (gfx90a or gfx942 architecture, e.g., MI250X, MI300X)
- `uv` package manager installed
- Python 3.12

### Setup Steps

1. **Create and activate virtual environment:**
   ```bash
   cd kernelgen
   make setup  # Creates venv, installs PyTorch with ROCm support
   ```

2. **Activate the environment:**
   ```bash
   source .venv/bin/activate
   # Or use: make act
   ```

The `make setup` command automatically:
- Detects your ROCm installation (6.4 or 7.0)
- Creates a Python 3.12 virtual environment
- Installs PyTorch with appropriate ROCm support
- Installs required dependencies (ninja, numpy, httpx, pyyaml)
- Configures ROCm environment variables for MI250X/MI300X

## Quick Start

### Generate and benchmark all Level 1 kernels:
```bash
python main.py --levels level1
```

### Continue from previous results:
```bash
python main.py --levels level1 --continue-from logs/benchmark_results_20251007_042506.json
```

### Use isolated subprocess for stability (prevents crashes from killing main process):
```bash
python main.py --levels level1 --isolated
```

### Test a single kernel:
```bash
python bench_kernel.py --model 1_Square_matrix_multiplication_
```

## Project Structure

```
kernelgen/
├── main.py                          # Main orchestration script
├── bench_kernel.py                  # Kernel benchmarking logic
├── bench_kernel_isolated.py        # Crash-isolated benchmarking
├── generation_prompt.py            # LLM prompt generation
├── llm_service.py                  # Multi-provider LLM service
├── llm_config.yaml                 # LLM configuration
├── analyze_benchmark.py            # Results analysis tool
├── Makefile                        # Environment setup automation
├── README.md                       # This file
├── logs/                           # Benchmark results and logs
│   └── debug/                      # Debug JSON files for failed kernels
├── prompts/                        # Prompt templates and examples
│   ├── hip_kernel_guide.md        # HIP optimization guide
│   └── *.hip/py                   # Example kernels for prompting
├── level1/                         # Generated Level 1 HIP kernels
│   ├── *.hip                       # HIP kernel implementations
│   └── build_cache/                # Compilation cache
└── .venv/                          # Virtual environment
```

## Workflow Overview

The system follows a pipeline architecture:

1. **Model Discovery**: Scans KernelBench levels for PyTorch models
2. **Parallel LLM Generation**: Multiple workers generate HIP kernels concurrently
3. **Immediate Saving**: Kernels saved to disk as soon as generated
4. **Sequential Benchmarking**: Queue-based benchmarking to prevent GPU conflicts
5. **Real-time Results**: Results saved to JSON after each benchmark
6. **Crash Isolation**: Optional subprocess isolation for stability

### Pipeline Execution Flow:
```
Discover Models → LLM Workers (parallel) → Save HIP Files → Benchmark Queue (sequential) → Save Results
                        ↓                                            ↓
                  Generation Pool                             Isolated Subprocess
                   (8 workers)                                 (if --isolated)
```

## Kernel Scoring System

Each kernel is scored based on three components:

### Scoring Formula
```
Total Score = Compilation Score + Correctness Score + Performance Score
```

### Component Breakdown

1. **Compilation Score: 20 points**
   - Successfully compiles with `hipcc -O3` for gfx90a/gfx942
   - Uses `torch.utils.cpp_extension.load()` for JIT compilation
   - Failure = 0 points

2. **Correctness Score: 100 points**
   - Compares HIP kernel output vs PyTorch model output
   - Pass criteria: `max_absolute_difference < 1e-2`
   - Failure = 0 points (and no performance testing)

3. **Performance Score: 100 × speedup**
   - Measured over 20 iterations with 5 warmup runs
   - Score = `100 × (pytorch_time / hip_time)`
   - Examples:
     - 2x speedup = 200 points
     - 3x speedup = 300 points
     - 0.5x speedup (slower) = 50 points

### Level Multipliers
For multi-level benchmarking, scores are multiplied:
- Level 1: 1x multiplier
- Level 2: 10x multiplier
- Level 3: 100x multiplier

### Example Scores
- Perfect kernel with 2x speedup: 20 + 100 + 200 = **320 points**
- Correct but slower (0.8x): 20 + 100 + 80 = **200 points**
- Wrong output: 20 + 0 + 0 = **20 points**
- Compilation failure: **0 points**

## HIP-PyTorch Interface

### How Kernels Interface with PyTorch

1. **PyTorch Model Structure**:
   ```python
   class Model(nn.Module):
       def __init__(self, *init_inputs):  # Architecture configuration
           # init_inputs define layer sizes, channels, etc.

       def forward(self, *inputs):        # Runtime computation
           # inputs are actual tensors
   ```

2. **HIP Kernel Signature**:
   ```cpp
   at::Tensor run(/* inputs from get_inputs() */, /* model.parameters() */)
   ```

### Input Parameter Handling

The HIP kernel receives parameters in strict order:

1. **First**: All tensors from `get_inputs()` (runtime inputs)
2. **Then**: All tensors from `model.parameters()` (learned weights/biases)

**Critical Rule**: Configuration values (stride, padding, kernel_size) are NOT passed!

### Examples

**Matrix Multiplication** (no parameters):
```cpp
at::Tensor run(at::Tensor A, at::Tensor B) {
    // Just the two input matrices
}
```

**Conv2D** (with weights and bias):
```cpp
at::Tensor run(at::Tensor input, at::Tensor weight, at::Tensor bias) {
    // Extract config from tensor shapes:
    int out_channels = weight.size(0);
    int in_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    // Stride/padding must use defaults or be inferred
}
```

### Compilation Process

1. **JIT Compilation**: Uses `torch.utils.cpp_extension.load()`
2. **Compiler**: `hipcc` for `.hip` files
3. **Optimization**: `-O3` flag for both C++ and HIP
4. **Target GPUs**: Only gfx90a (MI250X) and gfx942 (MI300X)
5. **Build Cache**: Stored in `level*/build_cache/` for faster rebuilds

### PyTorch Extension Binding

Every HIP file must include PyTorch bindings:
```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run_kernel, "Kernel description");
}
```

## LLM Service and Prompting

### Supported LLM Providers

Configure in `llm_config.yaml`:

1. **OpenAI GPT-5**:
   - Latest model with extended context
   - Configurable effort levels (low/medium/high)
   - Max 272K output tokens

2. **Claude (Opus 4/Sonnet 4.5)**:
   - Anthropic's models via API
   - 64K max tokens

3. **vLLM Local Models**:
   - Self-hosted models via vLLM server
   - Run with: `make vllm` (starts Docker container)
   - Default: Qwen3-Coder-30B

### Prompt Generation Process

1. **Context Assembly**:
   - Target PyTorch model code
   - Example PyTorch-to-HIP translation
   - HIP optimization guide
   - Scoring criteria

2. **Key Instructions**:
   - Exact function signature matching
   - Parameter extraction from tensor shapes
   - Performance optimization for MI300X
   - Single-file implementation with bindings

3. **Prompt Template Structure**:
   ```
   1. Example translation (Square Matrix Multiplication)
   2. HIP optimization guide
   3. Verification and scoring explanation
   4. Target model to convert
   5. Specific task instructions
   ```

### LLM Configuration

Edit `llm_config.yaml`:
```yaml
provider: openai  # or claude, vllm

openai:
  api_key: your_key_here
  model: gpt-5
  effort: low
  max_tokens: 272000

retry:
  max_retries: 3
  retry_delay: 2
  exponential_backoff: true
```

## Running Benchmarks

### Main Script Options

```bash
python main.py [options]
```

**Options**:
- `--levels`: Specify levels to process (level1, level2, level3)
- `--config`: LLM config file (default: llm_config.yaml)
- `--workers`: Parallel generation workers (default: 8)
- `--isolated`: Use subprocess isolation for crash protection
- `--continue-from`: Resume from previous results JSON
- `--skip-generation`: Skip LLM generation, only benchmark
- `--skip-benchmark`: Skip benchmarking, only generate

### Individual Kernel Testing

```bash
python bench_kernel.py --model <model_name>
```

**What it does**:
1. Loads PyTorch model from `KernelBench/level1/<model_name>.py`
2. Compiles HIP kernel from `kernelgen/level1/<model_name>.hip`
3. Generates inputs using `get_inputs()`
4. Runs correctness check (comparing outputs)
5. Runs performance benchmark (if correct)
6. Reports score and performance metrics

### Isolated Benchmarking

For unstable kernels that might crash:
```bash
python bench_kernel_isolated.py --model <model_name> --timeout 120
```

Creates debug files in `logs/debug/` for failed kernels.

### Analyzing Results

```bash
python analyze_benchmark.py logs/benchmark_results_*.json
```

Provides statistics on:
- Success rates per level
- Performance distribution
- Common failure patterns
- Top performing kernels


## Writing HIP Kernels

### File Naming Convention

For model at `KernelBench/level1/<model_name>.py`:
- Create: `kernelgen/level1/<model_name>.hip`

### Kernel Template

```cpp
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Define tile sizes and configurations
#define TILE_SIZE 16

// HIP kernel implementation
__global__ void my_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}

// PyTorch interface function
at::Tensor run(at::Tensor input) {
    // Get dimensions
    int size = input.numel();

    // Allocate output
    auto output = torch::empty_like(input);

    // Launch kernel
    dim3 blocks((size + TILE_SIZE - 1) / TILE_SIZE);
    dim3 threads(TILE_SIZE);

    hipLaunchKernelGGL(my_kernel, blocks, threads, 0, 0,
                       input.data_ptr<float>(),
                       output.data_ptr<float>(),
                       size);

    return output;
}

// PyTorch extension binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "My kernel");
}
```

### Optimization Guidelines

1. **Memory Access**:
   - Use shared memory (LDS) for frequently accessed data
   - Ensure coalesced memory access patterns
   - Add padding to avoid bank conflicts

2. **Thread Organization**:
   - Use 2D/3D thread blocks for 2D/3D problems
   - Typical block size: 256 threads (16×16)
   - Compute multiple elements per thread

3. **AMD-Specific**:
   - Wavefront size: 64 threads
   - LDS size: 64KB per CU
   - Use `__launch_bounds__` for occupancy hints

4. **Performance Tips**:
   - Unroll small loops with `#pragma unroll`
   - Use vector loads (float4) when possible
   - Minimize divergent branches
   - Profile with `rocprof` for optimization


## Troubleshooting

### Common Issues

**Ninja not found**:
```bash
source .venv/bin/activate
pip install ninja
```

**Compilation errors**:
```bash
# Check ROCm installation
ls /opt/rocm-6.4.1

# Verify GPU architecture
rocminfo | grep gfx

# Clear build cache
rm -rf kernelgen/level1/build_cache
```

**Import errors**:
```bash
# Verify environment
which python  # Should show .venv/bin/python

# Reinstall PyTorch
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
```

**Kernel crashes**:
```bash
# Use isolated mode
python main.py --levels level1 --isolated

# Check debug files
ls logs/debug/
```

**Memory issues**:
```bash
# Reduce parallel workers
python main.py --levels level1 --workers 4
```

### Debug Information

Failed kernels generate debug JSON files in `logs/debug/` containing:
- Exit codes and error messages
- Stdout/stderr output
- Compilation logs
- Stack traces

### Performance Profiling

```bash
# Profile with rocprof
rocprof --stats python bench_kernel.py --model <name>

# View kernel metrics
rocprof --hsa-trace python bench_kernel.py --model <name>
```

## Makefile Commands

```bash
make help      # Show available commands
make setup     # Create venv and install dependencies
make clean     # Remove virtual environment
make act       # Activate venv in new shell
make vllm      # Start vLLM server (Docker)
```

## Contributing

When adding new features:
1. Test with isolated mode first
2. Verify scoring accuracy
3. Update prompts if needed
4. Document configuration changes

## License

See main KernelBench repository for license information.
