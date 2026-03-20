#!/usr/bin/env python3
"""CLI entry point for running ablation studies.

Usage:
    python run_ablation.py --config configs/ablation_augmentation.yaml
    python run_ablation.py --config configs/ablation_models.yaml
"""

import argparse

from src.evaluation.ablation import AblationRunner
from src.utils.logger import get_logger


logger = get_logger("run_ablation")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ablation study for segmentation models",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to ablation configuration file",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda, cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Starting ablation study with config: {args.config}")

    runner = AblationRunner(
        ablation_config_path=args.config,
        device=args.device,
    )

    results = runner.run()

    num_experiments = len(results.get("experiments", []))
    logger.info(f"\nAblation study complete! {num_experiments} experiments run.")
    logger.info("Check the output directory for results, tables, and plots.")


if __name__ == "__main__":
    main()
