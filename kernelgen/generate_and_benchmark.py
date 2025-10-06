#!/usr/bin/env python3
"""
Generate optimized HIP kernels for all KernelBench models and benchmark them.
Supports parallel LLM generation and compilation, sequential testing.
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import traceback

from generation_prompt import generate_kernel_prompt, extract_hip_kernel, save_hip_kernel
from llm_service import LLMService
from bench_kernel import bench_kernel


# Level multipliers for scoring
LEVEL_MULTIPLIERS = {
    "level1": 1,
    "level2": 10,
    "level3": 100,
}


class KernelGenerator:
    """Generate HIP kernels using LLM."""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_kernel(self, model_file_path: str) -> Tuple[str, str, str]:
        """
        Generate HIP kernel for a model.

        Args:
            model_file_path: Relative path to model (e.g., "KernelBench/level1/1_Model.py")

        Returns:
            tuple: (model_file_path, kernel_code, explanation)
        """
        print(f"\n{'='*80}")
        print(f"Generating kernel for: {model_file_path}")
        print('='*80)

        # Generate prompt
        prompt = generate_kernel_prompt(model_file_path)

        # Call LLM (uses config from llm_config.yaml)
        print("Sending request to LLM...")
        response = self.llm_service.simple_chat(prompt=prompt)

        # Extract kernel code
        print("Extracting kernel code...")
        kernel_code, explanation = extract_hip_kernel(response)

        print(f"✓ Kernel generated successfully")
        if explanation:
            print(f"\nExplanation:\n{explanation}\n")

        return model_file_path, kernel_code, explanation


def discover_models(levels: List[str] = None) -> Dict[str, List[Path]]:
    """
    Discover all Python model files in KernelBench.

    Args:
        levels: List of levels to process (e.g., ["level1", "level2"]), or None for all

    Returns:
        dict: {level_name: [model_paths]}
    """
    project_root = Path(__file__).parent.parent
    kernelbench_dir = project_root / "KernelBench"

    if not kernelbench_dir.exists():
        raise FileNotFoundError(f"KernelBench directory not found: {kernelbench_dir}")

    all_levels = ["level1", "level2", "level3"]
    levels_to_process = levels if levels else all_levels

    models = {}
    for level in levels_to_process:
        level_dir = kernelbench_dir / level
        if not level_dir.exists():
            print(f"Warning: {level} directory not found, skipping")
            continue

        # Find all Python files (excluding __pycache__, __init__.py)
        model_files = sorted([
            f for f in level_dir.glob("*.py")
            if f.name != "__init__.py" and not f.name.startswith("_")
        ])

        if model_files:
            models[level] = model_files
            print(f"Found {len(model_files)} models in {level}")

    return models


def generate_kernels_parallel(
    models: Dict[str, List[Path]],
    llm_service: LLMService,
    max_workers: int = 8
) -> Dict[str, Tuple[str, str]]:
    """
    Generate HIP kernels in parallel.

    Args:
        models: Dictionary of {level: [model_paths]}
        llm_service: LLM service instance
        max_workers: Maximum parallel workers

    Returns:
        dict: {model_name: (kernel_code, explanation)}
    """
    generator = KernelGenerator(llm_service)
    results = {}
    errors = {}

    # Flatten all models into a list of relative paths
    tasks = []
    for level, model_files in models.items():
        for model_file in model_files:
            rel_path = f"KernelBench/{level}/{model_file.name}"
            tasks.append(rel_path)

    print(f"\n{'='*80}")
    print(f"Starting parallel kernel generation ({len(tasks)} models, {max_workers} workers)")
    print('='*80)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(generator.generate_kernel, path): path
            for path in tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_path):
            model_path = future_to_path[future]
            model_name = Path(model_path).stem

            try:
                _, kernel_code, explanation = future.result()
                results[model_name] = (kernel_code, explanation)
                print(f"✓ [{len(results)}/{len(tasks)}] Completed: {model_name}")
            except Exception as e:
                errors[model_name] = str(e)
                print(f"✗ [{len(results)+len(errors)}/{len(tasks)}] Failed: {model_name}")
                print(f"  Error: {e}")

    if errors:
        print(f"\n⚠ {len(errors)} kernel generations failed:")
        for name, error in errors.items():
            print(f"  - {name}: {error}")

    return results


def save_kernels(
    results: Dict[str, Tuple[str, str]],
    models: Dict[str, List[Path]]
) -> Dict[str, Path]:
    """
    Save generated kernels to appropriate directories.

    Args:
        results: {model_name: (kernel_code, explanation)}
        models: {level: [model_paths]}

    Returns:
        dict: {model_name: saved_hip_path}
    """
    project_root = Path(__file__).parent.parent
    saved_paths = {}

    print(f"\n{'='*80}")
    print("Saving generated kernels")
    print('='*80)

    for level, model_files in models.items():
        output_dir = project_root / "kernelgen" / level
        output_dir.mkdir(parents=True, exist_ok=True)

        for model_file in model_files:
            model_name = model_file.stem

            if model_name not in results:
                continue

            kernel_code, _ = results[model_name]
            output_path = output_dir / f"{model_name}.hip"

            save_hip_kernel(kernel_code, str(output_path))
            saved_paths[model_name] = output_path

    return saved_paths


def benchmark_kernels_sequential(
    models: Dict[str, List[Path]]
) -> Dict[str, Dict[str, any]]:
    """
    Benchmark kernels sequentially (one at a time for accurate timing).

    Args:
        models: {level: [model_paths]}

    Returns:
        dict: {model_name: {level, score, status, error}}
    """
    results = {}

    print(f"\n{'='*80}")
    print("Starting sequential benchmarking")
    print('='*80)

    total_models = sum(len(files) for files in models.values())
    completed = 0

    for level, model_files in models.items():
        for model_file in model_files:
            model_name = model_file.stem
            completed += 1

            print(f"\n[{completed}/{total_models}] Benchmarking {level}/{model_name}...")

            try:
                score = bench_kernel(model_name)
                results[model_name] = {
                    "level": level,
                    "score": score,
                    "status": "success",
                    "error": None
                }
            except Exception as e:
                error_msg = str(e)
                print(f"✗ Benchmark failed: {error_msg}")
                results[model_name] = {
                    "level": level,
                    "score": 0,
                    "status": "failed",
                    "error": error_msg
                }

    return results


def print_results_table(results: Dict[str, Dict[str, any]]):
    """Print formatted results table with scores."""
    print(f"\n{'='*100}")
    print("BENCHMARK RESULTS")
    print('='*100)

    # Group by level
    by_level = {"level1": [], "level2": [], "level3": []}
    for model_name, result in sorted(results.items()):
        level = result["level"]
        if level in by_level:
            by_level[level].append((model_name, result))

    # Print header
    print(f"{'Model':<50} {'Level':<8} {'Score':<10} {'Weighted':<12} {'Status':<10}")
    print('-'*100)

    total_weighted_score = 0
    total_models = 0

    for level in ["level1", "level2", "level3"]:
        if not by_level[level]:
            continue

        multiplier = LEVEL_MULTIPLIERS[level]
        level_total = 0

        for model_name, result in by_level[level]:
            score = result["score"]
            weighted = score * multiplier
            status = result["status"]

            # Truncate long model names
            display_name = model_name[:48] + ".." if len(model_name) > 50 else model_name

            status_symbol = "✓" if status == "success" else "✗"
            print(f"{display_name:<50} {level:<8} {score:<10} {weighted:<12} {status_symbol} {status:<10}")

            level_total += weighted
            total_models += 1

        print(f"{'':<50} {level} Total: {level_total} (×{multiplier})")
        print('-'*100)

        total_weighted_score += level_total

    # Final summary
    print(f"\n{'TOTAL SCORE':<50} {total_models} models   {total_weighted_score}")
    print('='*100)

    # Print failures
    failures = [(name, res) for name, res in results.items() if res["status"] == "failed"]
    if failures:
        print(f"\n⚠ Failed Models ({len(failures)}):")
        for name, res in failures:
            print(f"  - {name}: {res['error']}")


def main():
    parser = argparse.ArgumentParser(description="Generate and benchmark HIP kernels")
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=["level1", "level2", "level3"],
        help="Specific levels to process (default: all)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for generation (default: 8)"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip kernel generation, only benchmark existing kernels"
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip benchmarking, only generate kernels"
    )
    parser.add_argument(
        "--config",
        default="llm_config.yaml",
        help="Path to LLM config file (default: llm_config.yaml)"
    )

    args = parser.parse_args()

    try:
        # Discover models
        models = discover_models(args.levels)
        if not models:
            print("No models found!")
            return 1

        total_models = sum(len(files) for files in models.values())
        print(f"\nTotal models to process: {total_models}")

        # Generate kernels (if not skipped)
        if not args.skip_generation:
            llm_service = LLMService(args.config)
            results = generate_kernels_parallel(models, llm_service, args.workers)

            # Save kernels
            saved_paths = save_kernels(results, models)
            print(f"\n✓ Saved {len(saved_paths)} HIP kernels")

        # Benchmark kernels (if not skipped)
        if not args.skip_benchmark:
            benchmark_results = benchmark_kernels_sequential(models)

            # Print results table
            print_results_table(benchmark_results)

        print("\n✓ All done!")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
