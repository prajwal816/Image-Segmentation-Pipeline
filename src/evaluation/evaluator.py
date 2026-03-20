"""Model evaluator for semantic segmentation.

Runs a model on a validation/test set, computes all metrics, and
generates prediction visualizations.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import (
    mean_iou,
    pixel_accuracy,
    dice_coefficient,
    per_class_iou,
    confusion_matrix,
)
from src.utils.logger import get_logger


logger = get_logger("evaluator", file=False)

# Colormap for segmentation visualization
COLORMAP = np.array([
    [0, 0, 0],       [128, 0, 0],     [0, 128, 0],     [128, 128, 0],
    [0, 0, 128],     [128, 0, 128],   [0, 128, 128],   [128, 128, 128],
    [64, 0, 0],      [192, 0, 0],     [64, 128, 0],    [192, 128, 0],
    [64, 0, 128],    [192, 0, 128],   [64, 128, 128],  [192, 128, 128],
    [0, 64, 0],      [128, 64, 0],    [0, 192, 0],     [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)


class Evaluator:
    """Evaluator for semantic segmentation models.

    Args:
        model: Trained segmentation model.
        config: Configuration dictionary.
        device: Torch device for inference.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        eval_cfg = config.get("evaluation", {})
        self.num_classes = config.get("model", {}).get("num_classes", 21)
        vis_cfg = eval_cfg.get("visualization", {})
        self.vis_enabled = vis_cfg.get("enabled", True)
        self.vis_dir = vis_cfg.get("save_dir", "experiments/visualizations")
        self.num_vis_samples = vis_cfg.get("num_samples", 10)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Run full evaluation on a dataset.

        Args:
            dataloader: Evaluation data loader.

        Returns:
            Dictionary with all computed metrics.
        """
        logger.info(f"Evaluating on {len(dataloader)} batches...")

        all_preds = []
        all_targets = []
        all_images = []
        total_loss = 0.0
        num_batches = 0
        criterion = nn.CrossEntropyLoss(ignore_index=255)

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            outputs = self.model(images)
            loss = criterion(outputs, masks)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())
            all_images.append(images.cpu())
            total_loss += loss.item()
            num_batches += 1

        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_images = torch.cat(all_images, dim=0)

        # Compute metrics
        miou = mean_iou(all_preds, all_targets, self.num_classes)
        acc = pixel_accuracy(all_preds, all_targets)
        dice = dice_coefficient(all_preds, all_targets, self.num_classes)
        class_ious = per_class_iou(all_preds, all_targets, self.num_classes)
        cm = confusion_matrix(all_preds, all_targets, self.num_classes)
        avg_loss = total_loss / max(num_batches, 1)

        results = {
            "loss": avg_loss,
            "mean_iou": miou,
            "pixel_accuracy": acc,
            "dice_coefficient": dice,
            "per_class_iou": class_ious,
            "confusion_matrix": cm,
            "num_samples": len(all_preds),
        }

        logger.info(f"Results: loss={avg_loss:.4f}, mIoU={miou:.4f}, "
                     f"accuracy={acc:.4f}, dice={dice:.4f}")

        # Generate visualizations
        if self.vis_enabled:
            self._visualize(all_images, all_preds, all_targets)

        return results

    def _visualize(
        self,
        images: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Generate and save segmentation visualizations."""
        os.makedirs(self.vis_dir, exist_ok=True)
        num_samples = min(self.num_vis_samples, len(images))

        for i in range(num_samples):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            img = images[i].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[0].imshow(img)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            # Ground truth mask
            gt_colored = self._colorize_mask(targets[i].numpy())
            axes[1].imshow(gt_colored)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            # Predicted mask
            pred_colored = self._colorize_mask(preds[i].numpy())
            axes[2].imshow(pred_colored)
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            plt.tight_layout()
            save_path = os.path.join(self.vis_dir, f"sample_{i+1}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        logger.info(f"Saved {num_samples} visualizations to {self.vis_dir}")

    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert class mask to RGB colored visualization."""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for cls in range(min(self.num_classes, len(COLORMAP))):
            colored[mask == cls] = COLORMAP[cls]
        return colored


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str = "experiments/loss_curves.png",
) -> None:
    """Plot and save training/validation loss curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        save_path: Path to save the plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Loss curves saved to {save_path}")


def plot_metrics_curves(
    train_metrics: List[Dict[str, float]],
    val_metrics: List[Dict[str, float]],
    save_path: str = "experiments/metrics_curves.png",
) -> None:
    """Plot and save training/validation metric curves.

    Args:
        train_metrics: List of metric dicts per epoch.
        val_metrics: List of metric dicts per epoch.
        save_path: Path to save the plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_metrics) + 1)

    # mIoU curves
    train_miou = [m["miou"] for m in train_metrics]
    val_miou = [m["miou"] for m in val_metrics]
    axes[0].plot(epochs, train_miou, "b-", label="Train mIoU", linewidth=2)
    axes[0].plot(epochs, val_miou, "r-", label="Val mIoU", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("mIoU")
    axes[0].set_title("Mean IoU Over Training")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    train_acc = [m["accuracy"] for m in train_metrics]
    val_acc = [m["accuracy"] for m in val_metrics]
    axes[1].plot(epochs, train_acc, "b-", label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, val_acc, "r-", label="Val Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Pixel Accuracy Over Training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Metrics curves saved to {save_path}")
