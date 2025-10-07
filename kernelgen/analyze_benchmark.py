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
from datetime import datetime


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

    # Level-based analysis
    levels_data = {
        'level1': {'tests': [], 'total': 0, 'compiled': 0, 'correct': 0, 'outperform': 0,
                   'outperform_tests': [], 'total_score': 0},
        'level2': {'tests': [], 'total': 0, 'compiled': 0, 'correct': 0, 'outperform': 0,
                   'outperform_tests': [], 'total_score': 0},
        'level3': {'tests': [], 'total': 0, 'compiled': 0, 'correct': 0, 'outperform': 0,
                   'outperform_tests': [], 'total_score': 0},
        'level4': {'tests': [], 'total': 0, 'compiled': 0, 'correct': 0, 'outperform': 0,
                   'outperform_tests': [], 'total_score': 0}
    }

    for key, value in results.items():
        if isinstance(value, dict) and 'score' in value:
            score = value['score']
            scores.append(score)
            total_score += score

            status = value.get('status', 'unknown')
            level = value.get('level', 'unknown')

            # Process level data
            if level in levels_data:
                level_data = levels_data[level]
                level_data['tests'].append((key, score))
                level_data['total'] += 1
                level_data['total_score'] += score

                if score >= 20:
                    level_data['compiled'] += 1
                if score >= 120:
                    level_data['correct'] += 1
                if score >= 220:
                    level_data['outperform'] += 1
                    level_data['outperform_tests'].append((key, score))

            if status == 'success':
                success_count += 1
                successful_tests.append((key, score))
            elif status in ['failed', 'generation_failed']:
                failed_count += 1
                failed_tests.append((key, score, value.get('error', 'Unknown error')))

    # Sort tests by score
    successful_tests.sort(key=lambda x: x[1], reverse=True)
    failed_tests.sort(key=lambda x: x[0])

    # Sort outperform tests for each level
    for level_data in levels_data.values():
        level_data['outperform_tests'].sort(key=lambda x: x[1], reverse=True)

    return {
        'total_score': total_score,
        'total_entries': len(results),
        'success_count': success_count,
        'failed_count': failed_count,
        'scores': scores,
        'successful_tests': successful_tests,
        'failed_tests': failed_tests,
        'metadata': data.get('metadata', {}),
        'levels_data': levels_data
    }


def format_test_name(test_name: str) -> str:
    """Format test name for better readability."""
    # Remove leading number and underscore
    parts = test_name.split('_', 1)
    if len(parts) > 1 and parts[0].isdigit():
        return parts[1].replace('_', ' ').title()
    return test_name.replace('_', ' ').title()


def generate_report(analysis: Dict[str, Any], json_path: str) -> str:
    """Generate a comprehensive report of the benchmark results."""

    lines = []

    lines.append("=" * 70)
    lines.append("BENCHMARK RESULTS REPORT")
    lines.append("=" * 70)
    lines.append(f"\nFile: {json_path}")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Metadata section - only show completed and total
    if analysis['metadata']:
        lines.append("\n" + "-" * 70)
        lines.append("METADATA")
        lines.append("-" * 70)
        # Only show specific metadata fields
        if 'completed' in analysis['metadata']:
            lines.append(f"  Completed: {analysis['metadata']['completed']}")
        if 'total' in analysis['metadata']:
            lines.append(f"  Total: {analysis['metadata']['total']}")

    levels_data = analysis['levels_data']
    level_weights = {'level1': 1, 'level2': 10, 'level3': 100, 'level4': 1000}

    # Calculate weighted total first
    weighted_total = 0
    for level_name in ['level1', 'level2', 'level3', 'level4']:
        if level_name in levels_data:
            weighted_total += levels_data[level_name]['total_score'] * level_weights[level_name]

    # Weighted final score (moved before level breakdown)
    lines.append("\n" + "=" * 70)
    lines.append("WEIGHTED FINAL SCORE")
    lines.append("=" * 70)

    # Show individual level scores
    lines.append("\n  Level Scores:")
    lines.append(f"    Level 1: {levels_data['level1']['total_score']:,} × 1 = {levels_data['level1']['total_score'] * 1:,}")
    lines.append(f"    Level 2: {levels_data['level2']['total_score']:,} × 10 = {levels_data['level2']['total_score'] * 10:,}")
    lines.append(f"    Level 3: {levels_data['level3']['total_score']:,} × 100 = {levels_data['level3']['total_score'] * 100:,}")
    lines.append(f"    Level 4: {levels_data['level4']['total_score']:,} × 1000 = {levels_data['level4']['total_score'] * 1000:,}")

    lines.append(f"\n  Weighted Total Score: {weighted_total:,}")
    lines.append("\n  Formula: (Level1_Score × 1) + (Level2_Score × 10) + (Level3_Score × 100) + (Level4_Score × 1000)")

    # Level-based breakdown
    lines.append("\n" + "=" * 70)
    lines.append("LEVEL-BASED BREAKDOWN")
    lines.append("=" * 70)

    for level_name in ['level1', 'level2', 'level3', 'level4']:
        if level_name not in levels_data:
            continue

        level = levels_data[level_name]
        weight = level_weights[level_name]

        if level['total'] > 0:
            lines.append(f"\n{'-' * 70}")
            lines.append(f"{level_name.upper()} (Weight: {weight})")
            lines.append(f"{'-' * 70}")

            lines.append(f"  Total Models: {level['total']}")
            lines.append(f"  Total Score for Level: {level['total_score']:,}")

            compiled_pct = (level['compiled'] / level['total'] * 100) if level['total'] > 0 else 0
            lines.append(f"  Models Passed Compilation (score >= 20): {level['compiled']} ({compiled_pct:.1f}%)")

            correct_pct = (level['correct'] / level['total'] * 100) if level['total'] > 0 else 0
            lines.append(f"  Models Passed Correctness (score >= 120): {level['correct']} ({correct_pct:.1f}%)")

            outperform_pct = (level['outperform'] / level['total'] * 100) if level['total'] > 0 else 0
            lines.append(f"  Models Outperform Original (score >= 220): {level['outperform']} ({outperform_pct:.1f}%)")

            if level['outperform_tests']:
                lines.append(f"\n  Outperforming Models:")
                for test_name, score in level['outperform_tests']:
                    formatted_name = format_test_name(test_name)
                    lines.append(f"    - {formatted_name}: {score:,}")

    # Top performing tests overall
    if analysis['successful_tests']:
        lines.append("\n" + "-" * 70)
        lines.append("TOP 10 PERFORMING TESTS (OVERALL)")
        lines.append("-" * 70)
        lines.append(f"\n  {'Rank':<6} {'Test Name':<50} {'Score':>10}")
        lines.append("  " + "-" * 66)

        for i, (test_name, score) in enumerate(analysis['successful_tests'][:10], 1):
            formatted_name = format_test_name(test_name)
            if len(formatted_name) > 48:
                formatted_name = formatted_name[:45] + "..."
            lines.append(f"  {i:<6} {formatted_name:<50} {score:>10,}")

    # Failed tests section
    if analysis['failed_tests']:
        lines.append("\n" + "-" * 70)
        lines.append(f"FAILED/ERROR TESTS ({len(analysis['failed_tests'])})")
        lines.append("-" * 70)

        for test_name, score, error in analysis['failed_tests']:
            formatted_name = format_test_name(test_name)
            lines.append(f"\n  • {formatted_name}")
            lines.append(f"    Test ID: {test_name}")
            lines.append(f"    Score: {score}")
            if error:
                # Truncate long error messages
                error_msg = error if len(error) <= 100 else error[:97] + "..."
                lines.append(f"    Error: {error_msg}")

    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return '\n'.join(lines)


def save_report(report_text: str, json_path: str):
    """Save the report to a text file with _report suffix."""
    # Create output path with _report suffix
    json_path_obj = Path(json_path)
    output_path = json_path_obj.parent / f"{json_path_obj.stem}_report.txt"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nReport saved to: {output_path}")
    return output_path


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

    # Generate report
    report_text = generate_report(analysis, json_path)

    # Print report to console
    print(report_text)

    # Save report to file
    save_report(report_text, json_path)


if __name__ == "__main__":
    main()