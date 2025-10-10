#!/usr/bin/env python3
"""
Isolated kernel benchmarking with subprocess protection.
Runs each benchmark in a separate process to prevent crashes from killing the main process.
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


def clean_build_cache(model_name: str) -> None:
    """
    Delete the build_cache folder for the given model before benchmarking.

    Args:
        model_name: Model name in format "level/name" (e.g., "level1/1_Square_matrix_multiplication_")
                    or just "name" (defaults to level1 for backward compatibility)
    """
    # Parse level from model_name if provided
    if "/" in model_name:
        level, _ = model_name.split("/", 1)
    else:
        # Backward compatibility: default to level1
        level = "level1"

    # Determine kernel directory based on level
    kernel_dir = Path(__file__).parent / level
    build_cache_dir = kernel_dir / "build_cache"

    if build_cache_dir.exists():
        try:
            shutil.rmtree(build_cache_dir)
        except Exception:
            # Ignore errors if deletion fails
            pass


def run_benchmark_subprocess(model_name: str, timeout: int = 120) -> dict:
    """
    Run bench_kernel.py in a subprocess to isolate crashes.

    Args:
        model_name: Name of the model to benchmark
        timeout: Maximum time to wait for the benchmark (seconds)

    Returns:
        dict: {score, status, error, stdout, stderr, exit_code}
    """
    # Clean build cache before running benchmark
    clean_build_cache(model_name)

    cmd = [
        sys.executable,
        "bench_kernel.py",
        "--model", model_name
    ]

    result = {
        "score": 0,
        "status": "failed",
        "error": None,
        "stdout": "",
        "stderr": "",
        "exit_code": None
    }

    try:
        # Run in subprocess with timeout
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent
        )

        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["exit_code"] = proc.returncode

        if proc.returncode == 0:
            # Parse score from output
            # Look for "Score: X" or "FINAL SCORE: X" pattern
            score_patterns = [
                r"FINAL SCORE:\s*([-+]?\d+(?:\.\d+)?)",
                r"Score:\s*([-+]?\d+(?:\.\d+)?)",
            ]

            for line in proc.stdout.split('\n'):
                for pattern in score_patterns:
                    match = re.search(pattern, line)
                    if match:
                        try:
                            result["score"] = float(match.group(1))
                            result["status"] = "success"
                            break
                        except ValueError:
                            continue
                if result["status"] == "success":
                    break
        else:
            # Check for specific error patterns
            if "Memory access fault" in proc.stderr or "core dumped" in proc.stderr.lower():
                result["error"] = "GPU memory fault (core dumped)"
                result["status"] = "gpu_crash"
            elif "Segmentation fault" in proc.stderr:
                result["error"] = "Segmentation fault"
                result["status"] = "segfault"
            elif proc.returncode == -6:  # SIGABRT
                result["error"] = "Process aborted (likely GPU error)"
                result["status"] = "aborted"
            elif proc.returncode == -11:  # SIGSEGV
                result["error"] = "Segmentation fault (signal 11)"
                result["status"] = "segfault"
            else:
                # Extract error message from stderr or stdout
                error_lines = []
                for line in (proc.stderr + proc.stdout).split('\n'):
                    if 'error' in line.lower() or 'exception' in line.lower():
                        error_lines.append(line.strip())

                if error_lines:
                    result["error"] = ' | '.join(error_lines[:2])  # First 2 error lines
                else:
                    # Get last few lines of output for context
                    all_output = (proc.stderr + proc.stdout).strip()
                    if all_output:
                        last_lines = all_output.split('\n')[-3:]
                        result["error"] = f"Exit code {proc.returncode}: " + ' | '.join(last_lines)
                    else:
                        result["error"] = f"Process exited with code {proc.returncode}"

    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout after {timeout} seconds"
        result["status"] = "timeout"

    except Exception as e:
        result["error"] = f"Subprocess error: {str(e)}"
        result["status"] = "subprocess_error"

    return result


def bench_kernel_isolated(
    model_name: str,
    timeout: int = 120,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> float:
    """
    Benchmark a kernel with crash isolation.

    Args:
        model_name: Name of the model to benchmark
        timeout: Maximum time for the benchmark
        verbose: Print detailed output
        logger: Optional logger for structured logging

    Returns:
        float: Score (0 if failed)
    """
    def log_info(message: str):
        if logger:
            logger.info(message)
        elif verbose:
            print(message)

    def log_error(message: str):
        if logger:
            logger.error(message)
        elif verbose:
            print(message)

    if verbose:
        log_info(f"\n{'='*80}")
        log_info(f"Testing: {model_name} (isolated)")
        log_info('='*80)

    # Run benchmark in subprocess
    result = run_benchmark_subprocess(model_name, timeout)

    if result["status"] == "success":
        log_info(f"  ✓ Score: {result['score']}")
    else:
        log_error(f"  ✗ Failed: {result['error']}")
        if result["status"] in ["gpu_crash", "segfault", "aborted"]:
            log_error(f"    Status: {result['status']} (process isolated - main process safe)")

    # Save detailed results for debugging if failed
    if result["status"] != "success":
        # Create logs/debug directory if it doesn't exist
        debug_dir = Path("logs/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Replace '/' with '_' in model_name for valid filename
        safe_model_name = model_name.replace('/', '_')
        debug_file = debug_dir / f"debug_{safe_model_name}_{int(time.time())}.json"
        with open(debug_file, 'w') as f:
            json.dump(result, f, indent=2)
        log_info(f"    Debug info saved to: {debug_file}")

    return result["score"]


def main():
    parser = argparse.ArgumentParser(description="Isolated kernel benchmarking")
    parser.add_argument("--model", required=True, help="Model name to benchmark")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds (default: 120)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    score = bench_kernel_isolated(args.model, args.timeout, args.verbose)

    # Print result in a parseable format
    print(f"\nFINAL SCORE: {score}")

    # Exit with appropriate code
    sys.exit(0 if score > 0 else 1)


if __name__ == "__main__":
    main()
