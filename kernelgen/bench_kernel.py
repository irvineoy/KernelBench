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

    print(f"Loading HIP kernel from {kernel_dir}...")
    print(f"  Source file: {hip_file}")
    print(f"  Build directory: {build_dir}")
    print(f"  Target architectures: {target_archs}")
    print(f"  Starting compilation (optimized for gfx90a+gfx942, ~20-30s)...")

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
            verbose=True
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
    print(f"✓ HIP kernel loaded successfully (compilation took {elapsed_time:.1f}s)")
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
    print(f"[DEBUG] bench_kernel called with model_name: {model_name}", flush=True)
    score = 0

    # Setup paths
    print("[DEBUG] Setting up paths...", flush=True)
    project_root = Path(__file__).parent.parent
    model_path = project_root / "KernelBench" / "level1" / f"{model_name}.py"
    kernel_dir = project_root / "kernelgen" / "level1"

    print("=" * 80, flush=True)
    print(f"Testing: {model_name}", flush=True)
    print("=" * 80, flush=True)

    # Load Python model
    print(f"\n[1/5] Loading Python model from {model_path}...", flush=True)
    if not model_path.exists():
        raise FileNotFoundError(f"Python model not found: {model_path}")

    print("[DEBUG] Calling load_python_model...", flush=True)
    py_module = load_python_model(str(model_path))
    print("  ✓ Python model loaded", flush=True)

    # Get model and inputs
    print("\n[2/5] Initializing model and inputs...", flush=True)
    print("[DEBUG] Creating model instance...", flush=True)
    model = py_module.Model()
    print("[DEBUG] Setting eval mode...", flush=True)
    model.eval()
    print("[DEBUG] Moving model to cuda...", flush=True)
    model = model.cuda()
    print("[DEBUG] Model on cuda complete", flush=True)

    # Get inputs
    inputs = py_module.get_inputs()
    inputs_cuda = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in inputs]

    print(f"  Input shapes: {[inp.shape if isinstance(inp, torch.Tensor) else type(inp) for inp in inputs_cuda]}")
    print("  ✓ Model and inputs ready")

    # Load HIP kernel
    print("\n[3/5] Compiling HIP kernel...")
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
        print(f"  ✓ Compilation successful (+20 points)")
        print(f"  ✓ Entry point '{HIP_ENTRYPOINT}' located")
    except Exception as e:
        print(f"  ✗ Compilation failed: {e}")
        print(f"\nFinal Score: {score}/20")
        return score

    # Run PyTorch model
    print("\n[4/5] Running correctness check...")
    print("  Running PyTorch model...")
    import time
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output_torch = model(*inputs_cuda)
    torch.cuda.synchronize()
    torch_inference_time = time.time() - start
    print(f"  PyTorch output shape: {output_torch.shape} (took {torch_inference_time:.3f}s)")

    # Run HIP kernel
    print(f"  Running HIP entry '{HIP_ENTRYPOINT}'...")
    torch.cuda.synchronize()
    start = time.time()
    output_hip = hip_entry(*inputs_cuda)
    torch.cuda.synchronize()
    hip_inference_time = time.time() - start
    print(f"  HIP output shape: {output_hip.shape} (took {hip_inference_time:.3f}s)")

    # Correctness check
    print("\n  Comparing outputs...")
    diff = torch.abs(output_hip - output_torch)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    relative_diff = torch.max(diff / (torch.abs(output_torch) + 1e-8)).item()

    print(f"    Max absolute difference: {max_diff:.6e}")
    print(f"    Mean absolute difference: {mean_diff:.6e}")
    print(f"    Max relative difference: {relative_diff:.6e}")

    threshold = 1e-2
    if max_diff < threshold:
        print(f"    ✓ PASSED (threshold: {threshold})")
        correctness_passed = True
        score += 100
        print(f"    Correctness score: +100 points")
    else:
        print(f"    ✗ FAILED (threshold: {threshold})")
        correctness_passed = False
        print(f"    Correctness score: +0 points")

    # Performance benchmark (only if correctness passed)
    if correctness_passed:
        print("\n[5/5] Running performance benchmark...")
        print("  Warming up (5 iterations)...")

        torch_time = benchmark(lambda: model(*inputs_cuda), warmup=5, iterations=20)
        print(f"  PyTorch benchmark complete")

        hip_time = benchmark(lambda: hip_entry(*inputs_cuda), warmup=5, iterations=20)
        print(f"  HIP kernel benchmark complete")

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
    else:
        print("\n[5/5] Skipping performance benchmark (correctness test failed)")
        print("  ⚠ Performance score: +0 points (requires correctness)")
        speedup = 0.0
        performance_score = 0

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
