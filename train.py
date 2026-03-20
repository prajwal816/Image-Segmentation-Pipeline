#!/usr/bin/env python3
"""CLI entry point for training semantic segmentation models.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --model deeplabv3
"""

import argparse
import sys
import os

import torch

from src.models import get_model
from src.datasets.segmentation_dataset import create_dataloaders
from src.augmentation.transforms import get_augmentation_pipeline, get_validation_transform, Mixup
from src.training.trainer import Trainer
from src.training.transfer import setup_transfer_learning
from src.utils.config import load_config, merge_configs
from src.utils.logger import get_logger
from src.evaluation.evaluator import plot_loss_curves, plot_metrics_curves


logger = get_logger("train")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a semantic segmentation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model name (unet, deeplabv3)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--augmentation", type=str, default=None,
        help="Override augmentation pipeline (none, basic, full, mixup)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to train on (cuda, cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.model:
        config.setdefault("model", {})["name"] = args.model
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.augmentation:
        config.setdefault("augmentation", {})["pipeline"] = args.augmentation
    if args.resume:
        config.setdefault("checkpoint", {})["resume"] = args.resume

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
        pretrained=model_cfg.get("pretrained", True),
    )
    logger.info(f"Model: {model_cfg.get('name', 'unet')}")

    # Setup transfer learning
    unfreezer = setup_transfer_learning(model, config)

    # Create data loaders
    aug_cfg = config.get("augmentation", {})
    aug_name = aug_cfg.get("pipeline", "full")
    image_size = config.get("dataset", {}).get("image_size", 256)
    transform = get_augmentation_pipeline(aug_name, aug_cfg, image_size)
    val_transform = get_validation_transform(image_size)

    train_loader, val_loader = create_dataloaders(
        config, transform, val_transform
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Setup Mixup if configured
    mixup_fn = None
    if aug_name == "mixup" and aug_cfg.get("mixup", {}).get("enabled", False):
        alpha = aug_cfg.get("mixup", {}).get("alpha", 0.4)
        mixup_fn = Mixup(alpha=alpha)
        logger.info(f"Mixup enabled (alpha={alpha})")

    # Train
    trainer = Trainer(model, config, device)
    history = trainer.train(train_loader, val_loader, mixup_fn=mixup_fn)

    # Plot curves
    output_dir = config.get("checkpoint", {}).get("save_dir", "experiments/checkpoints")
    plot_loss_curves(
        history["train_losses"], history["val_losses"],
        os.path.join(output_dir, "loss_curves.png"),
    )
    plot_metrics_curves(
        history["train_metrics"], history["val_metrics"],
        os.path.join(output_dir, "metrics_curves.png"),
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
