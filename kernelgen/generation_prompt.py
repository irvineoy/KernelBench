#!/usr/bin/env python3
"""
Generate prompts for LLM to create HIP kernels from PyTorch models.
"""

from pathlib import Path
from typing import Optional


def read_file_content(file_path: Path) -> str:
    """Read and return file content."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path.read_text()


def generate_kernel_prompt(model_file_path: str) -> str:
    """
    Generate a prompt for LLM to create a HIP kernel from a PyTorch model.

    Args:
        model_file_path: Path to PyTorch model file (e.g., "KernelBench/level1/1_Square_matrix_multiplication_.py")

    Returns:
        str: Complete prompt for LLM with context and instructions
    """
    # Setup paths
    project_root = Path(__file__).parent.parent
    model_path = project_root / model_file_path

    # Extract model name from path
    model_name = Path(model_file_path).stem

    # Define example and guide paths
    example_py_path = project_root / "KernelBench" / "level1" / "1_Square_matrix_multiplication_.py"
    example_hip_path = project_root / "kernelgen" / "1_Square_matrix_multiplication_example.hip"
    guide_path = project_root / "kernelgen" / "hip_kernel_guide.md"

    # Read all necessary files
    try:
        target_model_content = read_file_content(model_path)
        example_py_content = read_file_content(example_py_path)
        example_hip_content = read_file_content(example_hip_path)
        guide_content = read_file_content(guide_path)
    except FileNotFoundError as e:
        raise RuntimeError(f"Required file not found: {e}")

    # Build the prompt
    verification_section = """## 3. Verification and Performance Testing

Your kernel will be automatically tested and scored:

**Compilation (+20 points)**: Compiled with `hipcc -O3` for gfx90a/gfx942. Must compile without errors.

**Correctness (+100 points)**: Compared against PyTorch model using `get_inputs()`. Pass if max absolute error < 1e-2.

**Performance (+100 × speedup)**: Measured over 20 iterations with 5 warmup runs. Score = 100 × (pytorch_time / hip_time).

**Total Score** = 20 + (0 or 100) + (100 × speedup)

Examples: 2x speedup = 320 pts | Wrong output = 20 pts | Compile fail = 0 pts"""

    prompt = f"""# Task: Generate Optimized HIP Kernel for PyTorch Model

You are an expert in GPU programming and optimization for AMD GPUs. Your task is to generate a high-performance HIP kernel that replicates the functionality of a given PyTorch model.

## 1. Example: PyTorch Model to HIP Kernel Translation

### Example PyTorch Model (Square Matrix Multiplication):

```python
{example_py_content}
```

### Corresponding HIP Kernel Implementation:

```cpp
{example_hip_content}
```

**Key Points from Example:**
- The HIP kernel takes the same inputs as the PyTorch model's `forward()` method
- Uses `get_inputs()` function signature to understand input shapes and types
- Implements the same computation using optimized HIP/CUDA kernels
- Includes PyTorch C++ extension bindings (PYBIND11_MODULE)
- Single `.hip` file contains both kernel code and PyTorch bindings

---

## 2. HIP Kernel Optimization Guide

{guide_content}

---

{verification_section}

---

## 4. Target PyTorch Model to Convert

```python
{target_model_content}
```

---

## 5. Your Task

Generate a **complete, optimized HIP kernel** that:

1. **Replicates the exact functionality** of the PyTorch model's `forward()` method
2. **Takes the same inputs** as defined by `get_inputs()` function
3. **Produces identical outputs** (within numerical precision 1e-2)
4. **Optimizes for AMD MI300X (gfx942)** using techniques from the optimization guide
5. **Includes PyTorch bindings** in a single `.hip` file

### Requirements:

- Use `#include <hip/hip_runtime.h>` and `#include <torch/extension.h>`
- Implement kernel(s) using HIP (`__global__` functions)
- Create a C++ wrapper function that accepts `at::Tensor` arguments
- Add PyTorch bindings using `PYBIND11_MODULE`
- Optimize for:
  - Coalesced memory access
  - Shared memory usage (avoid bank conflicts)
  - Proper thread block dimensions (multiples of 64 for wavefront alignment)
  - Loop unrolling where appropriate
- If you use `__launch_bounds__`, the declared thread count **must exactly match** the actual `blockDim` used in `hipLaunchKernelGGL`; otherwise omit the annotation
- After every kernel launch, call `hipDeviceSynchronize()` and check `hipGetLastError()` (or use `TORCH_CHECK`) to surface runtime issues promptly
- Ensure total threads per block ≤ 1024 and update the grid/block logic accordingly
- Export a wrapper like `run_kernel(...)` via `m.def("run", &run_kernel, ...)`; the benchmarking harness will call `module.run(...)`

### Output Format:

Return your response in the following format:

```xml
<hip_kernel>
// Your complete HIP kernel code here
// Include all necessary headers, kernel functions, wrapper, and bindings
</hip_kernel>

<explanation>
Brief explanation of:
1. Key optimizations applied
2. Expected performance characteristics
3. Any assumptions or limitations
</explanation>
```

**Important**:
- The kernel must compile with `hipcc` using `-O3` optimization
- Must work with ROCm 6.4 and PyTorch's JIT compilation system
- Focus on correctness first, then performance optimization
- Use the optimization guide principles for AMD GPUs

Now, generate the optimized HIP kernel for the target model above.
"""

    return prompt


def extract_hip_kernel(llm_response: str) -> tuple[str, Optional[str]]:
    """
    Extract HIP kernel code and explanation from LLM response.

    Args:
        llm_response: Raw response from LLM

    Returns:
        tuple: (kernel_code, explanation)
            - kernel_code: Extracted HIP kernel code
            - explanation: Extracted explanation or None
    """
    kernel_code = None
    explanation = None

    # Extract kernel code
    if "<hip_kernel>" in llm_response and "</hip_kernel>" in llm_response:
        start = llm_response.find("<hip_kernel>") + len("<hip_kernel>")
        end = llm_response.find("</hip_kernel>")
        kernel_code = llm_response[start:end].strip()

    # Extract explanation
    if "<explanation>" in llm_response and "</explanation>" in llm_response:
        start = llm_response.find("<explanation>") + len("<explanation>")
        end = llm_response.find("</explanation>")
        explanation = llm_response[start:end].strip()

    if kernel_code is None:
        raise ValueError("Could not extract <hip_kernel> from LLM response")

    return kernel_code, explanation


def save_hip_kernel(kernel_code: str, output_path: str) -> None:
    """
    Save extracted HIP kernel code to a file.

    Args:
        kernel_code: HIP kernel source code
        output_path: Path where to save the .hip file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(kernel_code)
    print(f"✓ HIP kernel saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generation_prompt.py <model_file_path>")
        print("Example: python generation_prompt.py KernelBench/level1/1_Square_matrix_multiplication_.py")
        sys.exit(1)

    model_file = sys.argv[1]

    try:
        # Generate prompt
        prompt = generate_kernel_prompt(model_file)

        # Print prompt (can be piped to LLM API)
        print(prompt)

        # Optionally save prompt to file
        # with open("prompt.txt", "w") as f:
        #     f.write(prompt)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
