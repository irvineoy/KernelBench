#!/usr/bin/env python3
"""
Isolated kernel benchmarking with subprocess protection.
Runs each benchmark in a separate process to prevent crashes from killing the main process.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_benchmark_subprocess(model_name: str, timeout: int = 120) -> dict:
    """
    Run bench_kernel.py in a subprocess to isolate crashes.

    Args:
        model_name: Name of the model to benchmark
        timeout: Maximum time to wait for the benchmark (seconds)

    Returns:
        dict: {score, status, error, stdout, stderr, exit_code}
    """
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


def bench_kernel_isolated(model_name: str, timeout: int = 120, verbose: bool = False) -> float:
    """
    Benchmark a kernel with crash isolation.

    Args:
        model_name: Name of the model to benchmark
        timeout: Maximum time for the benchmark
        verbose: Print detailed output

    Returns:
        float: Score (0 if failed)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing: {model_name} (isolated)")
        print('='*80)

    # Run benchmark in subprocess
    result = run_benchmark_subprocess(model_name, timeout)

    if verbose:
        if result["status"] == "success":
            print(f"  ✓ Score: {result['score']}")
        else:
            print(f"  ✗ Failed: {result['error']}")
            if result["status"] in ["gpu_crash", "segfault", "aborted"]:
                print(f"    Status: {result['status']} (process isolated - main process safe)")

    # Save detailed results for debugging if failed
    if result["status"] != "success":
        # Create logs/debug directory if it doesn't exist
        debug_dir = Path("logs/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

        debug_file = debug_dir / f"debug_{model_name}_{int(time.time())}.json"
        with open(debug_file, 'w') as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"    Debug info saved to: {debug_file}")

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
