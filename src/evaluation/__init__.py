"""Evaluation package."""

from src.evaluation.metrics import (
    mean_iou,
    pixel_accuracy,
    dice_coefficient,
    confusion_matrix,
    per_class_iou,
)

__all__ = [
    "mean_iou",
    "pixel_accuracy",
    "dice_coefficient",
    "confusion_matrix",
    "per_class_iou",
]


def get_evaluator():
    """Lazy import of Evaluator to avoid circular imports."""
    from src.evaluation.evaluator import Evaluator
    return Evaluator


def get_ablation_runner():
    """Lazy import of AblationRunner to avoid circular imports."""
    from src.evaluation.ablation import AblationRunner
    return AblationRunner
