#!/usr/bin/env python3
"""CLI entry point for evaluating trained segmentation models.

Usage:
    python evaluate.py --config configs/default.yaml --checkpoint experiments/checkpoints/best_model.pth
"""

import argparse
import json
import os

import torch

from src.models import get_model
from src.datasets.segmentation_dataset import create_dataloaders
from src.augmentation.transforms import get_validation_transform
from src.evaluation.evaluator import Evaluator
from src.postprocessing.postprocess import PostProcessor
from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger("evaluate")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained segmentation model",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--postprocess", action="store_true",
        help="Apply post-processing to predictions",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for inference (cuda, cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Device
    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    # Create model
    model_cfg = config.get("model", {})
    model = get_model(
        name=model_cfg.get("name", "unet"),
        num_classes=model_cfg.get("num_classes", 21),
        pretrained=False,  # Will load from checkpoint
    )

    # Load checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.warning("No checkpoint provided - using random weights")

    # Setup evaluation config
    config["evaluation"] = config.get("evaluation", {})
    config["evaluation"]["visualization"] = config["evaluation"].get("visualization", {})
    config["evaluation"]["visualization"]["save_dir"] = os.path.join(
        args.output_dir, "visualizations"
    )

    # Create data loader
    image_size = config.get("dataset", {}).get("image_size", 256)
    val_transform = get_validation_transform(image_size)
    _, val_loader = create_dataloaders(config, val_transform, val_transform)

    # Evaluate
    evaluator = Evaluator(model, config, device)
    results = evaluator.evaluate(val_loader)

    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"  Loss:            {results['loss']:.4f}")
    logger.info(f"  Mean IoU:        {results['mean_iou']:.4f}")
    logger.info(f"  Pixel Accuracy:  {results['pixel_accuracy']:.4f}")
    logger.info(f"  Dice Coefficient:{results['dice_coefficient']:.4f}")
    logger.info(f"  Samples:         {results['num_samples']}")

    if results.get("per_class_iou"):
        logger.info("\nPer-class IoU:")
        for cls, iou in sorted(results["per_class_iou"].items()):
            logger.info(f"  Class {cls}: {iou:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_save = {
        k: v for k, v in results.items()
        if k != "confusion_matrix"
    }
    results_save["per_class_iou"] = {
        str(k): v for k, v in results.get("per_class_iou", {}).items()
    }
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results_save, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Optional post-processing demo
    if args.postprocess:
        logger.info("\nApplying post-processing...")
        post_processor = PostProcessor(config)
        logger.info("Post-processing pipeline ready")
        logger.info("  Operations: morphological filtering + small region removal")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
