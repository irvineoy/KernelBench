"""
Utility functions for kernelgen module.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime


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

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    return logger, log_file


def save_results(results: Dict[str, Any], output_file: Path, logger: logging.Logger) -> None:
    """
    Save benchmark results to a JSON file.

    Args:
        results: Dictionary of benchmark results
        output_file: Path to save the results
        logger: Logger instance
    """
    # Ensure parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_file}")


def print_summary(results: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Print a summary of benchmark results grouped by level.

    Args:
        results: Dictionary of benchmark results
        logger: Logger instance
    """
    # Separate results by level
    by_level = {"level1": [], "level2": [], "level3": []}

    for model_name, result in results.items():
        # Skip metadata keys
        if model_name.startswith('_'):
            continue

        if isinstance(result, dict):
            level = result.get("level", "unknown")
            if level in by_level:
                by_level[level].append((model_name, result))

    logger.info("\n" + "="*100)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*100)

    # Calculate and display score statistics
    score_stats = calculate_total_score(results)

    logger.info(f"\n📊 OVERALL STATISTICS:")
    logger.info(f"  Total Models: {score_stats['total_models']}")
    logger.info(f"  Successful: {score_stats['successful_models']}")
    logger.info(f"  Total Raw Score: {score_stats['total_raw_score']:,}")
    logger.info(f"  Total Weighted Score: {score_stats['total_weighted_score']:,}")

    # Print level breakdown
    logger.info(f"\n📈 LEVEL BREAKDOWN:")
    for level_name, stats in score_stats['level_breakdown'].items():
        if stats['count'] > 0:
            logger.info(f"\n  {level_name.upper()} (Multiplier: {LEVEL_MULTIPLIERS[level_name]}x):")
            logger.info(f"    Models: {stats['count']}")
            logger.info(f"    Successful: {stats['successful']}")
            logger.info(f"    Raw Score: {stats['raw_score']:,}")
            logger.info(f"    Weighted Score: {stats['weighted_score']:,}")
            if stats['count'] > 0:
                avg_raw = stats['raw_score'] / stats['count']
                logger.info(f"    Average Raw Score: {avg_raw:.1f}")

    # Print header for detailed results
    logger.info(f"\n📋 DETAILED RESULTS:")
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

        if by_level[level]:
            logger.info(f"{'-'*100}")
            logger.info(f"{'Level ' + level + ' Total:':<50} {'':<8} {'':<10} {level_total:<12}")
            total_weighted_score += level_total

    logger.info(f"\n{'='*100}")
    logger.info(f"{'GRAND TOTAL:':<50} {'':<8} {'':<10} {total_weighted_score:<12} ({total_models} models)")
    logger.info(f"{'='*100}\n")


def calculate_total_score(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate total score and statistics from benchmark results.

    Scoring system:
    - Compilation success: 20 points
    - Correctness pass: 100 points
    - Performance: 100 * speedup (e.g., 2x = 200, 3x = 300)

    Level multipliers:
    - Level 1: 1x
    - Level 2: 10x
    - Level 3: 100x

    Returns:
        Dictionary with total_score, weighted_score, and breakdown by level
    """
    total_raw_score = 0
    total_weighted_score = 0
    level_stats = {
        "level1": {"count": 0, "raw_score": 0, "weighted_score": 0, "successful": 0, "compiled": 0, "correct": 0},
        "level2": {"count": 0, "raw_score": 0, "weighted_score": 0, "successful": 0, "compiled": 0, "correct": 0},
        "level3": {"count": 0, "raw_score": 0, "weighted_score": 0, "successful": 0, "compiled": 0, "correct": 0}
    }

    for model_name, result in results.items():
        # Skip metadata keys
        if model_name.startswith('_'):
            continue

        if isinstance(result, dict) and 'score' in result:
            score = result.get('score', 0)
            level = result.get('level', 'unknown')
            status = result.get('status', 'failed')

            if level in level_stats:
                multiplier = LEVEL_MULTIPLIERS.get(level, 1)
                weighted = score * multiplier

                level_stats[level]['count'] += 1
                level_stats[level]['raw_score'] += score
                level_stats[level]['weighted_score'] += weighted

                # Track different success levels
                if score >= 20:  # At least compiled
                    level_stats[level]['compiled'] += 1
                if score >= 120:  # Compiled + correct
                    level_stats[level]['correct'] += 1
                if status == 'success' and score > 120:  # Has performance gain
                    level_stats[level]['successful'] += 1

                total_raw_score += score
                total_weighted_score += weighted

    return {
        "total_raw_score": total_raw_score,
        "total_weighted_score": total_weighted_score,
        "level_breakdown": level_stats,
        "total_models": sum(stats['count'] for stats in level_stats.values()),
        "successful_models": sum(stats['successful'] for stats in level_stats.values()),
        "compiled_models": sum(stats['compiled'] for stats in level_stats.values()),
        "correct_models": sum(stats['correct'] for stats in level_stats.values())
    }


def load_existing_results(results_file: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load existing results from a JSON file if it exists.

    Args:
        results_file: Path to the results file
        logger: Logger instance

    Returns:
        Dictionary of existing results, or empty dict if file doesn't exist
    """
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                # Extract just the results, not metadata
                if 'results' in data:
                    results = data['results']
                else:
                    results = data
                logger.info(f"Loaded {len(results)} existing results from {results_file}")
                return results
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")

    return {}