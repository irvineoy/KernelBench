#!/usr/bin/env python3
"""
Generate optimized HIP kernels for all KernelBench models and benchmark them.
Supports parallel LLM generation and compilation, sequential testing.
"""

import os
# Speed up HIP kernel compilation by only targeting MI250X/MI300X
# Default PyTorch compiles for 11 GPU architectures (120s+ per kernel)
# This reduces compilation time to ~20-30s per kernel
os.environ['TORCH_CUDA_ARCH_LIST'] = 'gfx90a;gfx942'

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import traceback
import logging
from datetime import datetime
import json
import queue
import threading
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from generation_prompt import generate_kernel_prompt, extract_hip_kernel, save_hip_kernel
from llm_service import LLMService
from bench_kernel import bench_kernel
from bench_kernel_isolated import bench_kernel_isolated


# Level multipliers for scoring
LEVEL_MULTIPLIERS = {
    "level1": 1,
    "level2": 10,
    "level3": 100,
}


def setup_logging() -> Tuple[logging.Logger, Path]:
    """
    Setup logging to both console and timestamped log file.

    Returns:
        tuple: (logger, log_file_path)
    """
    # Create logs directory
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"kernelgen_{timestamp}.log"

    # Setup logger
    logger = logging.getLogger("kernelgen")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


class KernelGenerator:
    """Generate HIP kernels using LLM."""

    def __init__(self, llm_service: LLMService, logger: logging.Logger):
        self.llm_service = llm_service
        self.logger = logger

    def generate_kernel(self, model_file_path: str, output_path: Path = None) -> Tuple[str, str, str]:
        """
        Generate HIP kernel for a model.

        Args:
            model_file_path: Relative path to model (e.g., "KernelBench/level1/1_Model.py")
            output_path: Path where HIP file will be saved (used to check if it exists)

        Returns:
            tuple: (model_file_path, kernel_code, explanation)
        """
        # Check if HIP kernel already exists
        if output_path and output_path.exists():
            self.logger.info(f"✓ HIP kernel already exists, skipping generation: {output_path.name}")
            # Read existing kernel
            kernel_code = output_path.read_text()
            return model_file_path, kernel_code, "Loaded from existing file"

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Generating kernel for: {model_file_path}")
        self.logger.info('='*80)

        # Generate prompt
        prompt = generate_kernel_prompt(model_file_path)

        # Call LLM (uses config from llm_config.yaml)
        self.logger.info("Sending request to LLM...")
        response = self.llm_service.simple_chat(prompt=prompt)

        # Extract kernel code
        self.logger.info("Extracting kernel code...")
        kernel_code, explanation = extract_hip_kernel(response)

        self.logger.info(f"✓ Kernel generated successfully")
        if explanation:
            self.logger.info(f"\nExplanation:\n{explanation}\n")

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


def benchmark_worker(benchmark_queue: queue.Queue, results_dict: dict, logger: logging.Logger,
                     expected_count: threading.Event, stop_event: threading.Event,
                     results_file: Path = None, use_isolated: bool = False):
    """
    Worker thread that continuously benchmarks kernels from the queue.
    Starts working as soon as items are available, doesn't wait for all generation to complete.

    Args:
        benchmark_queue: Queue containing (model_name, level) tuples to benchmark
        results_dict: Shared dictionary to store benchmark results
        logger: Logger instance
        expected_count: Event signaling total expected count is set in results_dict['_expected_total']
        stop_event: Event signaling generation is complete
        results_file: Path to JSON file for real-time results saving
    """
    benchmarked_count = 0
    expected_total = None

    # Initialize results file (include existing results if any)
    if results_file:
        results_file.parent.mkdir(parents=True, exist_ok=True)
        # Get initial results (pre-loaded successful ones)
        initial_results = {k: v for k, v in results_dict.items() if not k.startswith('_')}
        with open(results_file, 'w') as f:
            json.dump({
                "results": initial_results,
                "metadata": {
                    "status": "in_progress",
                    "previous_results_count": len(initial_results)
                }
            }, f, indent=2)

    while True:
        try:
            # Check if we know the expected total
            if expected_total is None and expected_count.is_set():
                expected_total = results_dict.get('_expected_total', None)

            # Try to get item from queue
            try:
                item = benchmark_queue.get(timeout=0.5)
            except queue.Empty:
                # Check if we're done
                if stop_event.is_set() and benchmark_queue.empty():
                    if expected_total is None or benchmarked_count >= expected_total:
                        break
                continue

            if item is None:  # Sentinel value
                benchmark_queue.task_done()
                break

            model_name, level = item
            benchmarked_count += 1

            progress = f"[{benchmarked_count}/{expected_total}]" if expected_total else f"[{benchmarked_count}/?]"
            logger.info(f"\n{progress} Benchmarking {level}/{model_name}...")

            try:
                if use_isolated:
                    # Use isolated subprocess benchmarking
                    score = bench_kernel_isolated(model_name, timeout=120, verbose=True)
                else:
                    # Use regular in-process benchmarking
                    # Capture stdout/stderr from bench_kernel (which uses print())
                    stdout_capture = StringIO()
                    stderr_capture = StringIO()

                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        score = bench_kernel(model_name)

                    # Log captured output
                    stdout_text = stdout_capture.getvalue()
                    stderr_text = stderr_capture.getvalue()

                    if stdout_text:
                        for line in stdout_text.strip().split('\n'):
                            logger.info(f"  {line}")

                    if stderr_text:
                        for line in stderr_text.strip().split('\n'):
                            logger.error(f"  {line}")

                results_dict[model_name] = {
                    "level": level,
                    "score": score,
                    "status": "success",
                    "error": None
                }
                logger.info(f"  ✓ Benchmark complete - Score: {score}")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"  ✗ Benchmark failed: {error_msg}")
                logger.error(traceback.format_exc())
                results_dict[model_name] = {
                    "level": level,
                    "score": 0,
                    "status": "failed",
                    "error": error_msg
                }

            # Real-time save to JSON file
            if results_file:
                try:
                    # Filter out metadata keys when saving
                    clean_results = {k: v for k, v in results_dict.items() if not k.startswith('_')}
                    with open(results_file, 'w') as f:
                        json.dump({
                            "results": clean_results,
                            "metadata": {
                                "status": "in_progress",
                                "completed": benchmarked_count,
                                "total": expected_total or "unknown"
                            }
                        }, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to save results to {results_file}: {e}")

            benchmark_queue.task_done()

        except Exception as e:
            logger.error(f"Benchmark worker error: {e}")
            if 'item' in locals():
                benchmark_queue.task_done()

    # Final save with completed status
    if results_file:
        try:
            clean_results = {k: v for k, v in results_dict.items() if not k.startswith('_')}
            with open(results_file, 'w') as f:
                json.dump({
                    "results": clean_results,
                    "metadata": {
                        "status": "completed",
                        "completed": benchmarked_count,
                        "total": expected_total or benchmarked_count
                    }
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")

    logger.info(f"\nBenchmark worker finished: {benchmarked_count} kernels processed")


def pipeline_generate_and_benchmark(
    models: Dict[str, List[Path]],
    llm_service: LLMService,
    logger: logging.Logger,
    max_workers: int = 8,
    previous_results: Dict = None,
    use_isolated: bool = False
) -> Dict[str, Dict[str, any]]:
    """
    Pipeline: Generate (parallel) -> Save (immediate) -> Benchmark (queue, sequential).
    Benchmark worker starts immediately and processes kernels as they arrive.

    Args:
        models: Dictionary of {level: [model_paths]}
        llm_service: LLM service instance
        logger: Logger instance
        max_workers: Maximum parallel workers for generation

    Returns:
        dict: {model_name: {level, score, status, error}}
    """
    generator = KernelGenerator(llm_service, logger)
    project_root = Path(__file__).parent.parent

    # Create benchmark queue and results dict
    benchmark_queue = queue.Queue()
    benchmark_results = previous_results.copy() if previous_results else {}
    expected_count_event = threading.Event()
    stop_event = threading.Event()

    # Flatten all models into tasks with level info
    tasks = []
    model_to_level = {}
    for level, model_files in models.items():
        output_dir = project_root / "kernelgen" / level
        output_dir.mkdir(parents=True, exist_ok=True)

        for model_file in model_files:
            rel_path = f"KernelBench/{level}/{model_file.name}"
            model_name = model_file.stem
            tasks.append((rel_path, level, output_dir, model_name))
            model_to_level[model_name] = level

    total_models = len(tasks)
    generated_count = 0
    failed_count = 0

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting pipeline: Generate -> Save -> Benchmark Queue")
    logger.info(f"Total models: {total_models}, Generation workers: {max_workers}")
    logger.info(f"Benchmark worker will start immediately and process as kernels arrive")
    logger.info('='*80)

    # Set expected total for benchmark worker
    benchmark_results['_expected_total'] = total_models
    expected_count_event.set()

    # Create real-time results file
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = log_dir / f"benchmark_results_{timestamp}.json"

    # Start benchmark worker thread (starts immediately, waits for queue items)
    # NOT daemon - we want it to complete even if main thread is done
    benchmark_thread = threading.Thread(
        target=benchmark_worker,
        args=(benchmark_queue, benchmark_results, logger, expected_count_event, stop_event, results_file, use_isolated),
        daemon=False
    )
    benchmark_thread.start()
    logger.info(f"✓ Benchmark worker started, saving results to: {results_file}\n")

    # Generate kernels in parallel, save immediately, queue for benchmark
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare output paths for each task
        future_to_task = {}
        for rel_path, level, output_dir, model_name in tasks:
            output_path = output_dir / f"{model_name}.hip"
            future = executor.submit(generator.generate_kernel, rel_path, output_path)
            future_to_task[future] = (rel_path, level, output_dir, model_name, output_path)

        for future in as_completed(future_to_task):
            rel_path, level, output_dir, model_name, output_path = future_to_task[future]

            try:
                # Get generated kernel
                _, kernel_code, explanation = future.result()
                generated_count += 1

                # Save to disk if not already saved (when skipped due to existing file)
                if not output_path.exists():
                    save_hip_kernel(kernel_code, str(output_path))
                    logger.info(f"✓ [Gen: {generated_count + failed_count}/{total_models}] Generated & Saved: {model_name}")
                else:
                    logger.info(f"✓ [Gen: {generated_count + failed_count}/{total_models}] Using existing: {model_name}")

                logger.info(f"  -> {output_path}")

                # Add to benchmark queue (benchmark worker will pick it up immediately)
                benchmark_queue.put((model_name, level))

            except Exception as e:
                failed_count += 1
                error_msg = str(e)
                logger.error(f"✗ [Gen: {generated_count + failed_count}/{total_models}] Generation failed: {model_name}")
                logger.error(f"  Error: {error_msg}")

                # Still record the failure in results (won't be benchmarked)
                benchmark_results[model_name] = {
                    "level": level,
                    "score": 0,
                    "status": "generation_failed",
                    "error": error_msg
                }

    # Signal generation is complete and wait for benchmark worker to finish
    logger.info(f"\n{'='*80}")
    logger.info(f"Generation complete: {generated_count} succeeded, {failed_count} failed")
    logger.info(f"Waiting for benchmark queue to finish... (Queue size: {benchmark_queue.qsize()})")
    logger.info('='*80)

    stop_event.set()  # Signal that generation is done
    benchmark_thread.join()  # Wait for all benchmarks to complete

    # Clean up metadata
    benchmark_results.pop('_expected_total', None)

    logger.info(f"\n✓ All benchmarks complete!")

    return benchmark_results


def print_results_table(results: Dict[str, Dict[str, any]], logger: logging.Logger):
    """Print formatted results table with scores."""
    logger.info(f"\n{'='*100}")
    logger.info("BENCHMARK RESULTS")
    logger.info('='*100)

    # Group by level
    by_level = {"level1": [], "level2": [], "level3": []}
    for model_name, result in sorted(results.items()):
        level = result["level"]
        if level in by_level:
            by_level[level].append((model_name, result))

    # Print header
    logger.info(f"{'Model':<50} {'Level':<8} {'Score':<10} {'Weighted':<12} {'Status':<10}")
    logger.info('-'*100)

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
            logger.info(f"{display_name:<50} {level:<8} {score:<10} {weighted:<12} {status_symbol} {status:<10}")

            level_total += weighted
            total_models += 1

        logger.info(f"{'':<50} {level} Total: {level_total} (×{multiplier})")
        logger.info('-'*100)

        total_weighted_score += level_total

    # Final summary
    logger.info(f"\n{'TOTAL SCORE':<50} {total_models} models   {total_weighted_score}")
    logger.info('='*100)

    # Print failures
    failures = [(name, res) for name, res in results.items() if res["status"] == "failed"]
    if failures:
        logger.warning(f"\n⚠ Failed Models ({len(failures)}):")
        for name, res in failures:
            logger.warning(f"  - {name}: {res['error']}")


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
        default=4,
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
    parser.add_argument(
        "--continue-from",
        type=str,
        help="Path to previous benchmark JSON file to continue from (skip successful models)"
    )
    parser.add_argument(
        "--no-isolated",
        dest="isolated",
        action="store_false",
        help="Disable isolated mode and run benchmarks in the main process (not recommended)"
    )

    # Set default value for isolated mode
    parser.set_defaults(isolated=True)

    args = parser.parse_args()

    try:
        # Setup logging
        logger, log_file = setup_logging()
        logger.info("="*80)
        logger.info(f"KernelGen - HIP Kernel Generation and Benchmarking")
        logger.info(f"Log file: {log_file}")
        logger.info("="*80)

        # Load previous results if continuing and filter out existing models
        previous_results_data = {}
        existing_models = set()
        if args.continue_from:
            if not Path(args.continue_from).exists():
                logger.error(f"Previous results file not found: {args.continue_from}")
                return 1

            logger.info(f"\nLoading previous results from: {args.continue_from}")
            with open(args.continue_from, 'r') as f:
                prev_data = json.load(f)
                previous_results_data = prev_data.get('results', prev_data)  # Handle both formats
                existing_models = set(previous_results_data.keys())

            logger.info(f"Loaded {len(previous_results_data)} previous results (will skip these models)")

        # Discover models
        models = discover_models(args.levels)
        if not models:
            logger.error("No models found!")
            return 1

        # Filter out models that already exist in previous results
        if existing_models:
            original_count = sum(len(files) for files in models.values())
            for level in models:
                models[level] = [f for f in models[level] if f.stem not in existing_models]
            filtered_count = sum(len(files) for files in models.values())
            logger.info(f"Skipping {original_count - filtered_count} models that already have results")

        total_models = sum(len(files) for files in models.values())
        if total_models == 0:
            logger.info("\nAll models already have results! Nothing to do.")
            logger.info(f"Results are in: {args.continue_from}")
            return 0

        logger.info(f"\nTotal models to process: {total_models}")

        # Use pipeline mode (generate -> save -> benchmark queue)
        if not args.skip_generation and not args.skip_benchmark:
            llm_service = LLMService(args.config)
            benchmark_results = pipeline_generate_and_benchmark(
                models, llm_service, logger, args.workers, previous_results_data, args.isolated
            )

            # Print results table
            print_results_table(benchmark_results, logger)

            # Results already saved in real-time by benchmark worker
            # The results_file path is created inside pipeline_generate_and_benchmark
            log_dir = Path(__file__).parent / "logs"
            logger.info(f"\n✓ Benchmark results saved in real-time to: {log_dir}/benchmark_results_*.json")

        # Legacy: Generate only
        elif not args.skip_generation and args.skip_benchmark:
            logger.warning("Note: Using legacy mode. For pipeline mode, don't use --skip-benchmark")
            llm_service = LLMService(args.config)
            generator = KernelGenerator(llm_service, logger)
            project_root = Path(__file__).parent.parent

            for level, model_files in models.items():
                output_dir = project_root / "kernelgen" / level
                output_dir.mkdir(parents=True, exist_ok=True)

                for model_file in model_files:
                    rel_path = f"KernelBench/{level}/{model_file.name}"
                    model_name = model_file.stem
                    try:
                        _, kernel_code, _ = generator.generate_kernel(rel_path)
                        output_path = output_dir / f"{model_name}.hip"
                        save_hip_kernel(kernel_code, str(output_path))
                        logger.info(f"✓ Saved: {output_path}")
                    except Exception as e:
                        logger.error(f"✗ Failed {model_name}: {e}")

        # Legacy: Benchmark only
        elif args.skip_generation and not args.skip_benchmark:
            logger.warning("Note: Benchmarking existing kernels only")
            # Start with previous results if continuing
            benchmark_results = previous_results_data.copy() if previous_results_data else {}
            total_models = sum(len(files) for files in models.values())
            completed = 0

            for level, model_files in models.items():
                for model_file in model_files:
                    model_name = model_file.stem
                    completed += 1
                    logger.info(f"\n[{completed}/{total_models}] Benchmarking {level}/{model_name}...")
                    try:
                        if args.isolated:
                            score = bench_kernel_isolated(model_name, timeout=120, verbose=True)
                        else:
                            score = bench_kernel(model_name)
                        benchmark_results[model_name] = {
                            "level": level,
                            "score": score,
                            "status": "success",
                            "error": None
                        }
                        logger.info(f"  Score: {score}")
                    except Exception as e:
                        benchmark_results[model_name] = {
                            "level": level,
                            "score": 0,
                            "status": "failed",
                            "error": str(e)
                        }

            print_results_table(benchmark_results, logger)

            log_dir = Path(__file__).parent / "logs"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = log_dir / f"benchmark_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            logger.info(f"\n✓ Benchmark results saved to: {results_file}")

        logger.info("\n✓ All done!")
        logger.info(f"Complete log saved to: {log_file}")
        return 0

    except Exception as e:
        if 'logger' in locals():
            logger.error(f"\n✗ Error: {e}")
            logger.error(traceback.format_exc())
        else:
            print(f"\n✗ Error: {e}", file=sys.stderr)
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
