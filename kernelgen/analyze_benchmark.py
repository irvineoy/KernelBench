#!/usr/bin/env python3
"""
Benchmark Results Analyzer

This script analyzes benchmark results from a JSON file and generates a comprehensive report.

Usage:
    python analyze_benchmark.py <path_to_json_file>
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_benchmark_data(json_path: str) -> Dict[str, Any]:
    """Load benchmark data from JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)


def analyze_results(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze benchmark results and compute statistics."""
    results = data.get('results', {})
    
    if not results:
        # Handle case where data is directly the results dict
        results = data
    
    total_score = 0
    success_count = 0
    failed_count = 0
    scores = []
    successful_tests = []
    failed_tests = []
    
    for key, value in results.items():
        if isinstance(value, dict) and 'score' in value:
            score = value['score']
            scores.append(score)
            total_score += score
            
            status = value.get('status', 'unknown')
            if status == 'success':
                success_count += 1
                successful_tests.append((key, score))
            elif status == 'failed':
                failed_count += 1
                failed_tests.append((key, score, value.get('error', 'Unknown error')))
    
    # Sort tests by score
    successful_tests.sort(key=lambda x: x[1], reverse=True)
    failed_tests.sort(key=lambda x: x[0])
    
    return {
        'total_score': total_score,
        'total_entries': len(results),
        'success_count': success_count,
        'failed_count': failed_count,
        'scores': scores,
        'successful_tests': successful_tests,
        'failed_tests': failed_tests,
        'metadata': data.get('metadata', {})
    }


def format_test_name(test_name: str) -> str:
    """Format test name for better readability."""
    # Remove leading number and underscore
    parts = test_name.split('_', 1)
    if len(parts) > 1 and parts[0].isdigit():
        return parts[1].replace('_', ' ').title()
    return test_name.replace('_', ' ').title()


def print_report(analysis: Dict[str, Any], json_path: str):
    """Print a comprehensive report of the benchmark results."""
    
    print("=" * 80)
    print("BENCHMARK RESULTS REPORT")
    print("=" * 80)
    print(f"\nFile: {json_path}")
    print(f"Date: {Path(json_path).stat().st_mtime}")
    
    # Metadata section
    if analysis['metadata']:
        print("\n" + "-" * 80)
        print("METADATA")
        print("-" * 80)
        for key, value in analysis['metadata'].items():
            print(f"  {key.capitalize()}: {value}")
    
    # Summary section
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  {'TOTAL SCORE:':<25} {analysis['total_score']:>10,}")
    print(f"  {'Total Test Cases:':<25} {analysis['total_entries']:>10}")
    print(f"  {'Successful:':<25} {analysis['success_count']:>10} ({analysis['success_count']/analysis['total_entries']*100:.1f}%)")
    print(f"  {'Failed:':<25} {analysis['failed_count']:>10} ({analysis['failed_count']/analysis['total_entries']*100:.1f}%)")
    
    if analysis['scores']:
        avg_score = analysis['total_score'] / analysis['total_entries']
        max_score = max(analysis['scores'])
        min_score = min(analysis['scores'])
        
        print(f"\n  {'Average Score:':<25} {avg_score:>10.2f}")
        print(f"  {'Maximum Score:':<25} {max_score:>10}")
        print(f"  {'Minimum Score:':<25} {min_score:>10}")
    
    # Top performing tests
    if analysis['successful_tests']:
        print("\n" + "-" * 80)
        print("TOP 10 PERFORMING TESTS")
        print("-" * 80)
        print(f"\n  {'Rank':<6} {'Test Name':<50} {'Score':>10}")
        print("  " + "-" * 76)
        
        for i, (test_name, score) in enumerate(analysis['successful_tests'][:10], 1):
            formatted_name = format_test_name(test_name)
            if len(formatted_name) > 48:
                formatted_name = formatted_name[:45] + "..."
            print(f"  {i:<6} {formatted_name:<50} {score:>10,}")
    
    # Failed tests section
    if analysis['failed_tests']:
        print("\n" + "-" * 80)
        print(f"FAILED TESTS ({len(analysis['failed_tests'])})")
        print("-" * 80)
        
        for test_name, score, error in analysis['failed_tests']:
            formatted_name = format_test_name(test_name)
            print(f"\n  • {formatted_name}")
            print(f"    Test ID: {test_name}")
            print(f"    Score: {score}")
            if error:
                # Truncate long error messages
                error_msg = error if len(error) <= 100 else error[:97] + "..."
                print(f"    Error: {error_msg}")
    
    # Score distribution
    if analysis['scores']:
        print("\n" + "-" * 80)
        print("SCORE DISTRIBUTION")
        print("-" * 80)
        
        # Create score ranges
        ranges = [
            (0, 0, "Zero (Failed)"),
            (1, 50, "Low (1-50)"),
            (51, 100, "Medium (51-100)"),
            (101, 200, "High (101-200)"),
            (201, 500, "Very High (201-500)"),
            (501, float('inf'), "Exceptional (500+)")
        ]
        
        print(f"\n  {'Range':<25} {'Count':>10} {'Percentage':>15}")
        print("  " + "-" * 50)
        
        for min_val, max_val, label in ranges:
            count = sum(1 for s in analysis['scores'] if min_val <= s <= max_val)
            percentage = (count / len(analysis['scores'])) * 100
            print(f"  {label:<25} {count:>10} {percentage:>14.1f}%")
    
    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_benchmark.py <path_to_json_file>")
        print("\nExample:")
        print("  python analyze_benchmark.py logs/benchmark_results_sonnect4.5_temp07_level1.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    # Load and analyze data
    data = load_benchmark_data(json_path)
    analysis = analyze_results(data)
    
    # Print report
    print_report(analysis, json_path)


if __name__ == "__main__":
    main()
