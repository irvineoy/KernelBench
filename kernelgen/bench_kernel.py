#!/usr/bin/env python3
"""
General test script for comparing PyTorch model performance with HIP kernel implementation.
Usage: python test_kernel.py --model <model_name>
Example: python test_kernel.py --model 1_Square_matrix_multiplication_
"""

import argparse
import sys
import os
import time
import importlib.util
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

HIP_ENTRYPOINT = "run"


def load_python_model(model_path: str):
    """Dynamically load the Python model module."""
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_hip_kernel(kernel_dir: Path, model_name: str):
    """Load the HIP kernel as a PyTorch extension."""
    hip_file = kernel_dir / f"{model_name}.hip"

    if not hip_file.exists():
        raise FileNotFoundError(f"HIP kernel file not found: {hip_file}")

    # Create build directory
    target_archs = "gfx90a;gfx942"
    build_dir = kernel_dir / "build_cache" / target_archs.replace(";", "_")
    build_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Speed up compilation by only targeting MI250X (gfx90a) and MI300X (gfx942)
    # Default PyTorch compiles for 11 architectures which takes 2-3 minutes!
    # This reduces compilation time from ~120s to ~20-30s
    old_cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    old_rocm_arch_list = os.environ.get("PYTORCH_ROCM_ARCH", None)
    os.environ["TORCH_CUDA_ARCH_LIST"] = target_archs
    os.environ["PYTORCH_ROCM_ARCH"] = target_archs

    try:
        kernel = load(
            name=model_name.replace("_", ""),
            sources=[str(hip_file)],
            extra_cflags=['-O3', '-I/opt/rocm/include'],
            extra_cuda_cflags=['-O3', '-I/opt/rocm/include', '-fno-gpu-rdc'],
            extra_ldflags=['-L/opt/rocm/lib', "-lamdhip64"],
            build_directory=str(build_dir),
            verbose=False  # Suppress compilation output
        )
    finally:
        # Restore original environment
        if old_cuda_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_cuda_arch_list

        if old_rocm_arch_list is None:
            os.environ.pop("PYTORCH_ROCM_ARCH", None)
        else:
            os.environ["PYTORCH_ROCM_ARCH"] = old_rocm_arch_list

    elapsed_time = time.time() - start_time
    print(f"  Compiled in {elapsed_time:.1f}s (gfx90a/gfx942)")
    return kernel


def benchmark(func, *args, warmup=5, iterations=20):
    """Benchmark a function with warmup and multiple iterations."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iterations


def bench_kernel(model_name: str):
    """Test HIP kernel against PyTorch implementation.

    Returns:
        int: Score based on compilation, correctness, and performance
            - Compilation success: 20 points
            - Correctness pass: 100 points
            - Performance: 100 * speedup (e.g., 2x = 200, 3x = 300)
    """
    score = 0

    # Setup paths
    project_root = Path(__file__).parent.parent
    model_path = project_root / "KernelBench" / "level1" / f"{model_name}.py"
    kernel_dir = project_root / "kernelgen" / "level1"

    print("=" * 80)
    print(f"Testing: {model_name}")
    print("=" * 80)

    # Load Python model
    if not model_path.exists():
        raise FileNotFoundError(f"Python model not found: {model_path}")

    py_module = load_python_model(str(model_path))

    # Initialize model with init_inputs if available
    if hasattr(py_module, 'get_init_inputs'):
        init_inputs = py_module.get_init_inputs()
        model = py_module.Model(*init_inputs)
    else:
        model = py_module.Model()

    model.eval()
    model = model.cuda()

    # Get inputs
    inputs = py_module.get_inputs()
    inputs_cuda = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in inputs]

    # Extract model parameters (weights) if they exist
    # These will be passed to HIP kernel for layers like Conv, Linear, etc.
    model_params = []
    for param in model.parameters():
        model_params.append(param.data)

    # Load HIP kernel
    print(f"Compiling HIP kernel...")
    try:
        hip_kernel = load_hip_kernel(kernel_dir, model_name)
        if not hasattr(hip_kernel, HIP_ENTRYPOINT):
            available = ", ".join(dir(hip_kernel))
            raise AttributeError(
                f"HIP module missing required '{HIP_ENTRYPOINT}' entry point. "
                f"Available attributes: {available}"
            )
        hip_entry = getattr(hip_kernel, HIP_ENTRYPOINT)
        score += 20
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        return score

    # Run correctness check
    print(f"Running correctness check...")
    torch.cuda.synchronize()
    with torch.no_grad():
        output_torch = model(*inputs_cuda)
    torch.cuda.synchronize()

    # Call HIP kernel with inputs + model parameters
    # HIP kernel should accept: (*inputs, *model_params)
    torch.cuda.synchronize()
    hip_inputs = inputs_cuda + model_params
    output_hip = hip_entry(*hip_inputs)
    torch.cuda.synchronize()

    # Correctness check
    diff = torch.abs(output_hip - output_torch)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()

    threshold = 1e-2
    if max_diff < threshold:
        print(f"  ✓ Correctness PASSED (max_diff: {max_diff:.2e})")
        correctness_passed = True
        score += 100
    else:
        print(f"  ✗ Correctness FAILED (max_diff: {max_diff:.2e}, threshold: {threshold})")
        correctness_passed = False

    # Performance benchmark (only if correctness passed)
    if correctness_passed:
        print(f"Running performance benchmark...")
        torch_time = benchmark(lambda: model(*inputs_cuda), warmup=5, iterations=20)
        hip_time = benchmark(lambda: hip_entry(*hip_inputs), warmup=5, iterations=20)

        speedup = torch_time / hip_time
        performance_score = int(100 * speedup)
        score += performance_score

        print(f"  PyTorch: {torch_time * 1000:.2f}ms | HIP: {hip_time * 1000:.2f}ms | Speedup: {speedup:.2f}x")
    else:
        speedup = 0.0
        performance_score = 0

    # Summary
    print("-" * 80)
    status = "✓" if correctness_passed else "✗"
    print(f"Score: {score} | Compile: 20 | Correct: {100 if correctness_passed else 0} | Perf: {performance_score} {status}")
    print("=" * 80)

    return score


def main():
    parser = argparse.ArgumentParser(description="Test HIP kernel against PyTorch implementation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., 1_Square_matrix_multiplication_)"
    )

    args = parser.parse_args()

    try:
        score = bench_kernel(args.model)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
