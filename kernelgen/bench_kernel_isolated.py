#!/usr/bin/env python3
"""
Isolated kernel benchmarking with subprocess protection.
Runs each benchmark in a separate process to prevent crashes from killing the main process.
"""

import argparse
import subprocess
import sys
import json
import tempfile
from pathlib import Path
import time


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
            # Look for "Score: X" pattern (may have | after the number)
            for line in proc.stdout.split('\n'):
                if 'Score:' in line:
                    try:
                        # Extract the score part after "Score:"
                        score_part = line.split('Score:')[1].strip()
                        # Take the first part before any "|"
                        score_str = score_part.split('|')[0].strip()
                        result["score"] = float(score_str)
                        result["status"] = "success"
                        break
                    except Exception as e:
                        # Log parsing error but continue
                        result["error"] = f"Score parsing error: {str(e)}"
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


def bench_kernel_isolated(model_name: str, timeout: int = 120, verbose: bool = False, logger=None) -> float:
    """
    Benchmark a kernel with crash isolation.

    Args:
        model_name: Name of the model to benchmark
        timeout: Maximum time for the benchmark
        verbose: Print detailed output
        logger: Optional logger instance

    Returns:
        float: Score (0 if failed)
    """
    # Run benchmark in subprocess
    result = run_benchmark_subprocess(model_name, timeout)

    # Use logger if provided, otherwise print if verbose
    def log_info(msg):
        if logger:
            logger.info(msg)
        elif verbose:
            print(msg)

    def log_error(msg):
        if logger:
            logger.error(msg)
        elif verbose:
            print(f"ERROR: {msg}")

    if result["status"] == "success":
        log_info(f"  ✓ Score: {result['score']}")
    else:
        # Only save debug files for actual crashes/errors, not compilation failures
        if result["status"] in ["gpu_crash", "segfault", "aborted", "timeout"]:
            debug_file = Path(f"debug_{model_name}_{int(time.time())}.json")
            with open(debug_file, 'w') as f:
                json.dump(result, f, indent=2)
            log_error(f"  ✗ {result['status']}: {result['error']}")
            log_info(f"    Debug info saved to: {debug_file}")
        else:
            # Regular failure (compilation, correctness, etc)
            log_error(f"  ✗ Failed: {result['error'] if result['error'] else 'See output'}")
            # Extract key info from stdout
            if "Correctness FAILED" in result.get("stdout", ""):
                for line in result["stdout"].split('\n'):
                    if "Correctness FAILED" in line:
                        log_info(f"    {line.strip()}")
            elif "Compilation failed" in result.get("stdout", ""):
                log_info("    Compilation failed")

    return result["score"]


def main():
    parser = argparse.ArgumentParser(description="Isolated kernel benchmarking")
    parser.add_argument("--model", required=True, help="Model name to benchmark")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds (default: 120)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    score = bench_kernel_isolated(args.model, args.timeout, args.verbose)

    # Don't print FINAL SCORE here - it's already in the output
    # Just exit with appropriate code
    sys.exit(0 if score > 0 else 1)


if __name__ == "__main__":
    main()