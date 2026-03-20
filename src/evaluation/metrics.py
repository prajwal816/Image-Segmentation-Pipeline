"""Segmentation evaluation metrics.

Provides standard metrics for semantic segmentation evaluation:
- Mean Intersection over Union (mIoU)
- Pixel Accuracy
- Dice Coefficient (F1 Score)
"""

from typing import Optional

import torch
import numpy as np


def mean_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """Compute Mean Intersection over Union (mIoU).

    Args:
        pred: Predicted class labels (B, H, W) or (H, W).
        target: Ground truth class labels (B, H, W) or (H, W).
        num_classes: Total number of classes.
        ignore_index: Class index to ignore in computation.

    Returns:
        Mean IoU score across all present classes.
    """
    pred = pred.flatten()
    target = target.flatten()

    # Create valid mask
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    if len(pred) == 0:
        return 0.0

    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        if union == 0:
            # Class not present in either prediction or target
            continue

        ious.append(intersection / union)

    return float(np.mean(ious)) if ious else 0.0


def pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> float:
    """Compute pixel-wise accuracy.

    Args:
        pred: Predicted class labels (B, H, W) or (H, W).
        target: Ground truth class labels (B, H, W) or (H, W).
        ignore_index: Class index to ignore.

    Returns:
        Pixel accuracy score.
    """
    pred = pred.flatten()
    target = target.flatten()

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    if len(pred) == 0:
        return 0.0

    correct = (pred == target).sum().item()
    total = len(pred)

    return correct / total


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: int = 255,
) -> float:
    """Compute mean Dice coefficient (F1 score) across classes.

    Args:
        pred: Predicted class labels (B, H, W) or (H, W).
        target: Ground truth class labels (B, H, W) or (H, W).
        num_classes: Total number of classes.
        smooth: Smoothing factor to prevent division by zero.
        ignore_index: Class index to ignore.

    Returns:
        Mean Dice coefficient across present classes.
    """
    pred = pred.flatten()
    target = target.flatten()

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    if len(pred) == 0:
        return 0.0

    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum().item()
        denominator = pred_cls.sum().item() + target_cls.sum().item()

        if denominator == 0:
            continue

        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        dice_scores.append(dice)

    return float(np.mean(dice_scores)) if dice_scores else 0.0


def confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        pred: Predicted class labels.
        target: Ground truth class labels.
        num_classes: Number of classes.
        ignore_index: Class index to ignore.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    pred = pred.flatten().cpu().numpy()
    target = target.flatten().cpu().numpy()

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(pred, target):
        if 0 <= p < num_classes and 0 <= t < num_classes:
            cm[t, p] += 1

    return cm


def per_class_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> dict:
    """Compute per-class IoU scores.

    Args:
        pred: Predicted class labels.
        target: Ground truth class labels.
        num_classes: Number of classes.
        ignore_index: Class index to ignore.

    Returns:
        Dictionary mapping class index to IoU score.
    """
    pred = pred.flatten()
    target = target.flatten()

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    class_ious = {}
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        if union > 0:
            class_ious[cls] = intersection / union

    return class_ious
