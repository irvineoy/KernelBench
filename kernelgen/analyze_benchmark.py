#!/usr/bin/env python3
"""
Benchmark Results Analyzer

This script analyzes benchmark results from a JSON file and generates a comprehensive report.
Supports both old format (direct results) and new format (with metadata.total_score).

Usage:
    python analyze_benchmark.py <path_to_json_file>
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
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
    """
    Analyze benchmark results and compute statistics.
    Supports both old format and new format with metadata.total_score.
    """
    results = data.get('results', {})

    if not results:
        # Handle case where data is directly the results dict (old format)
        results = data

    # Check if we have the new total_score metadata
    metadata = data.get('metadata', {})
    total_score_info = metadata.get('total_score', None)

    # If we have the new format with pre-calculated scores, use them
    if total_score_info:
        # Use the pre-calculated scores from metadata
        total_score = total_score_info.get('total_raw_score', 0)
        weighted_total_score = total_score_info.get('total_weighted_score', 0)
        levels_data = {}

        # Convert the new format to match expected structure
        for level_name, level_info in total_score_info.get('level_breakdown', {}).items():
            # Collect outperforming tests for this level
            outperform_tests = []
            for key, value in results.items():
                if isinstance(value, dict) and value.get('level') == level_name:
                    score = value.get('score', 0)
                    if score >= 220:  # Threshold for outperforming
                        outperform_tests.append((key, score))

            outperform_tests.sort(key=lambda x: x[1], reverse=True)

            levels_data[level_name] = {
                'tests': [(k, v['score']) for k, v in results.items()
                         if isinstance(v, dict) and v.get('level') == level_name],
                'total': level_info.get('count', 0),
                'compiled': level_info.get('compiled', 0),
                'correct': level_info.get('correct', 0),
                'outperform': len(outperform_tests),
                'outperform_tests': outperform_tests,
                'total_score': level_info.get('raw_score', 0),
                'weighted_score': level_info.get('weighted_score', 0)
            }

        # Add level4 if it doesn't exist
        if 'level4' not in levels_data:
            levels_data['level4'] = {
                'tests': [], 'total': 0, 'compiled': 0, 'correct': 0,
                'outperform': 0, 'outperform_tests': [], 'total_score': 0, 'weighted_score': 0
            }

        total_models = total_score_info.get('total_models', len(results))
        successful_models = total_score_info.get('successful_models', 0)

    else:
        # Fall back to calculating scores manually (old format)
        total_score = 0
        weighted_total_score = 0
        levels_data = {
            'level1': {'tests': [], 'total': 0, 'compiled': 0, 'correct': 0, 'outperform': 0,
                       'outperform_tests': [], 'total_score': 0, 'weighted_score': 0},
            'level2': {'tests': [], 'total': 0, 'compiled': 0, 'correct': 0, 'outperform': 0,
                       'outperform_tests': [], 'total_score': 0, 'weighted_score': 0},
            'level3': {'tests': [], 'total': 0, 'compiled': 0, 'correct': 0, 'outperform': 0,
                       'outperform_tests': [], 'total_score': 0, 'weighted_score': 0},
            'level4': {'tests': [], 'total': 0, 'compiled': 0, 'correct': 0, 'outperform': 0,
                       'outperform_tests': [], 'total_score': 0, 'weighted_score': 0}
        }

        level_weights = {'level1': 1, 'level2': 10, 'level3': 100, 'level4': 1000}

        for key, value in results.items():
            if isinstance(value, dict) and 'score' in value:
                score = value['score']
                total_score += score

                level = value.get('level', 'unknown')

                # Process level data
                if level in levels_data:
                    level_data = levels_data[level]
                    level_data['tests'].append((key, score))
                    level_data['total'] += 1
                    level_data['total_score'] += score
                    level_data['weighted_score'] = level_data['total_score'] * level_weights.get(level, 1)

                    if score >= 20:
                        level_data['compiled'] += 1
                    if score >= 120:
                        level_data['correct'] += 1
                    if score >= 220:
                        level_data['outperform'] += 1
                        level_data['outperform_tests'].append((key, score))

        # Calculate weighted total
        for level_name, level_data in levels_data.items():
            weighted_total_score += level_data['weighted_score']

        total_models = len(results)
        successful_models = sum(1 for v in results.values()
                              if isinstance(v, dict) and v.get('status') == 'success')

    # Process successful and failed tests
    successful_tests = []
    failed_tests = []
    scores = []

    for key, value in results.items():
        if isinstance(value, dict) and 'score' in value:
            score = value['score']
            scores.append(score)
            status = value.get('status', 'unknown')

            if status == 'success':
                successful_tests.append((key, score))
            elif status in ['failed', 'generation_failed']:
                failed_tests.append((key, score, value.get('error', 'Unknown error')))

    # Sort tests
    successful_tests.sort(key=lambda x: x[1], reverse=True)
    failed_tests.sort(key=lambda x: x[0])

    # Sort outperform tests for each level
    for level_data in levels_data.values():
        level_data['outperform_tests'].sort(key=lambda x: x[1], reverse=True)

    return {
        'total_score': total_score,
        'weighted_total_score': weighted_total_score,
        'total_entries': total_models,
        'success_count': successful_models,
        'failed_count': len(failed_tests),
        'scores': scores,
        'successful_tests': successful_tests,
        'failed_tests': failed_tests,
        'metadata': metadata,
        'levels_data': levels_data,
        'has_new_format': total_score_info is not None
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

    # Add format info
    if analysis.get('has_new_format'):
        lines.append("Format: New (with metadata.total_score)")
    else:
        lines.append("Format: Legacy (calculated from results)")

    # Metadata section
    if analysis['metadata']:
        lines.append("\n" + "-" * 70)
        lines.append("METADATA")
        lines.append("-" * 70)

        # Show various metadata fields
        if 'status' in analysis['metadata']:
            lines.append(f"  Status: {analysis['metadata']['status']}")
        if 'completed' in analysis['metadata']:
            lines.append(f"  Completed: {analysis['metadata']['completed']}")
        if 'total' in analysis['metadata']:
            lines.append(f"  Total Expected: {analysis['metadata']['total']}")

        # If we have the new total_score metadata, show summary
        if 'total_score' in analysis['metadata']:
            ts = analysis['metadata']['total_score']
            lines.append("\n  Score Summary from Metadata:")
            lines.append(f"    Total Models: {ts.get('total_models', 'N/A')}")
            lines.append(f"    Successful Models: {ts.get('successful_models', 'N/A')}")
            lines.append(f"    Compiled Models: {ts.get('compiled_models', 'N/A')}")
            lines.append(f"    Correct Models: {ts.get('correct_models', 'N/A')}")

    levels_data = analysis['levels_data']
    level_weights = {'level1': 1, 'level2': 10, 'level3': 100, 'level4': 1000}

    # Weighted final score section
    lines.append("\n" + "=" * 70)
    lines.append("WEIGHTED FINAL SCORE")
    lines.append("=" * 70)

    # Show individual level scores
    lines.append("\n  Level Scores:")
    for level_name in ['level1', 'level2', 'level3', 'level4']:
        if level_name in levels_data:
            level = levels_data[level_name]
            weight = level_weights[level_name]
            raw_score = level['total_score']
            weighted_score = level.get('weighted_score', raw_score * weight)

            if raw_score > 0 or level['total'] > 0:
                lines.append(f"    {level_name.title()}: {raw_score:,} × {weight} = {weighted_score:,}")

    lines.append(f"\n  Total Raw Score: {analysis['total_score']:,}")
    lines.append(f"  Weighted Total Score: {analysis['weighted_total_score']:,}")
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
            lines.append(f"  Weighted Score: {level.get('weighted_score', level['total_score'] * weight):,}")

            compiled_pct = (level['compiled'] / level['total'] * 100) if level['total'] > 0 else 0
            lines.append(f"  Models Passed Compilation (score >= 20): {level['compiled']} ({compiled_pct:.1f}%)")

            correct_pct = (level['correct'] / level['total'] * 100) if level['total'] > 0 else 0
            lines.append(f"  Models Passed Correctness (score >= 120): {level['correct']} ({correct_pct:.1f}%)")

            outperform_pct = (level['outperform'] / level['total'] * 100) if level['total'] > 0 else 0
            lines.append(f"  Models Outperform Original (score >= 220): {level['outperform']} ({outperform_pct:.1f}%)")

            if level['outperform_tests']:
                lines.append(f"\n  Outperforming Models:")
                for test_name, score in level['outperform_tests'][:10]:  # Limit to top 10
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

        # Limit to first 20 failed tests to keep report readable
        for test_name, score, error in analysis['failed_tests'][:20]:
            formatted_name = format_test_name(test_name)
            lines.append(f"\n  • {formatted_name}")
            lines.append(f"    Test ID: {test_name}")
            lines.append(f"    Score: {score}")
            if error:
                # Truncate long error messages
                error_msg = error if len(error) <= 100 else error[:97] + "..."
                lines.append(f"    Error: {error_msg}")

        if len(analysis['failed_tests']) > 20:
            lines.append(f"\n  ... and {len(analysis['failed_tests']) - 20} more failed tests")

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
        print("  python analyze_benchmark.py logs/benchmark_results.json")
        print("\nSupports both legacy format and new format with metadata.total_score")
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