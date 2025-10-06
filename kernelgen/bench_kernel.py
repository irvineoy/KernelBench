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
    build_dir = kernel_dir / "build_cache"
    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HIP kernel from {kernel_dir}...")
    kernel = load(
        name=model_name.replace("_", ""),
        sources=[str(hip_file)],
        extra_cflags=['-O3', '-I/opt/rocm/include'],
        extra_cuda_cflags=['-O3', '-I/opt/rocm/include'],
        extra_ldflags=['-L/opt/rocm/lib', "-lamdhip64"],
        build_directory=str(build_dir),
        verbose=True
    )
    print("✓ HIP kernel loaded successfully")
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
    print(f"\nLoading Python model from {model_path}...")
    if not model_path.exists():
        raise FileNotFoundError(f"Python model not found: {model_path}")

    py_module = load_python_model(str(model_path))
    print("✓ Python model loaded")

    # Get model and inputs
    model = py_module.Model()
    model.eval()
    model = model.cuda()

    # Get inputs
    inputs = py_module.get_inputs()
    inputs_cuda = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in inputs]

    print(f"\nInput shapes: {[inp.shape if isinstance(inp, torch.Tensor) else type(inp) for inp in inputs_cuda]}")

    # Load HIP kernel
    try:
        hip_kernel = load_hip_kernel(kernel_dir, model_name)
        score += 20
        print(f"✓ Compilation successful (+20 points)")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        print(f"\nFinal Score: {score}/20")
        return score

    # Run PyTorch model
    print("\n" + "-" * 80)
    print("Running PyTorch model...")
    with torch.no_grad():
        output_torch = model(*inputs_cuda)
    print(f"Output shape: {output_torch.shape}")

    # Run HIP kernel
    print("\nRunning HIP kernel...")
    output_hip = hip_kernel.matmul(*inputs_cuda)
    print(f"Output shape: {output_hip.shape}")

    # Correctness check
    print("\n" + "-" * 80)
    print("Correctness Check:")
    diff = torch.abs(output_hip - output_torch)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    relative_diff = torch.max(diff / (torch.abs(output_torch) + 1e-8)).item()

    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  Max relative difference: {relative_diff:.6e}")

    threshold = 1e-2
    if max_diff < threshold:
        print(f"  ✓ PASSED (threshold: {threshold})")
        correctness_passed = True
        score += 100
        print(f"  Correctness score: +100 points")
    else:
        print(f"  ✗ FAILED (threshold: {threshold})")
        correctness_passed = False
        print(f"  Correctness score: +0 points")

    # Performance benchmark
    print("\n" + "-" * 80)
    print("Performance Benchmark:")
    print("  Warming up...")

    torch_time = benchmark(lambda: model(*inputs_cuda), warmup=5, iterations=20)
    hip_time = benchmark(lambda: hip_kernel.matmul(*inputs_cuda), warmup=5, iterations=20)

    speedup = torch_time / hip_time
    performance_score = int(100 * speedup)

    print(f"\n  PyTorch time: {torch_time * 1000:.3f} ms")
    print(f"  HIP kernel time: {hip_time * 1000:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Performance score: +{performance_score} points")

    score += performance_score

    if hip_time < torch_time:
        print(f"  ✓ HIP kernel is {speedup:.2f}x faster")
    else:
        print(f"  ⚠ HIP kernel is {1/speedup:.2f}x slower")

    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Compilation: ✓ PASSED (+20 points)")
    print(f"  Correctness: {'✓ PASSED (+100 points)' if correctness_passed else '✗ FAILED (+0 points)'}")
    print(f"  Performance: {speedup:.2f}x speedup (+{performance_score} points)")
    print(f"\n  Final Score: {score}")
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
