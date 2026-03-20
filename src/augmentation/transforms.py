"""Data augmentation pipelines using albumentations.

Provides named augmentation pipelines for training semantic segmentation
models, including random crop, color jitter, flips, and Mixup.
"""

from typing import Any, Dict, Optional, Tuple

import albumentations as A
import numpy as np
import torch


def get_augmentation_pipeline(
    name: str = "full",
    config: Optional[Dict] = None,
    image_size: int = 256,
) -> Optional[A.Compose]:
    """Get a named augmentation pipeline.

    Args:
        name: Pipeline name - "none", "basic", "full", or "mixup".
        config: Optional augmentation config dict.
        image_size: Target image size for crop operations.

    Returns:
        Albumentations Compose pipeline, or None for "none".
    """
    if config is None:
        config = {}

    if name == "none":
        return None

    if name == "basic":
        return _build_basic_pipeline(config, image_size)
    elif name == "full":
        return _build_full_pipeline(config, image_size)
    elif name == "mixup":
        return _build_full_pipeline(config, image_size)
    else:
        raise ValueError(
            f"Unknown augmentation pipeline '{name}'. "
            f"Available: none, basic, full, mixup"
        )


def _build_basic_pipeline(config: Dict, image_size: int) -> A.Compose:
    """Build basic augmentation pipeline with flips only."""
    transforms = [
        A.HorizontalFlip(p=config.get("horizontal_flip", {}).get("p", 0.5)),
        A.VerticalFlip(p=config.get("vertical_flip", {}).get("p", 0.3)),
    ]
    return A.Compose(transforms)


def _build_full_pipeline(config: Dict, image_size: int) -> A.Compose:
    """Build full augmentation pipeline with crop, jitter, and flips."""
    crop_cfg = config.get("random_crop", {})
    jitter_cfg = config.get("color_jitter", {})
    crop_size = crop_cfg.get("size", min(224, image_size))

    transforms = []

    # Random crop
    if crop_cfg.get("enabled", True):
        transforms.append(
            A.RandomCrop(
                height=crop_size, width=crop_size, p=0.8
            )
        )
        transforms.append(
            A.Resize(height=image_size, width=image_size)
        )

    # Color jitter
    if jitter_cfg.get("enabled", True):
        transforms.append(
            A.ColorJitter(
                brightness=jitter_cfg.get("brightness", 0.3),
                contrast=jitter_cfg.get("contrast", 0.3),
                saturation=jitter_cfg.get("saturation", 0.3),
                hue=jitter_cfg.get("hue", 0.1),
                p=0.8,
            )
        )

    # Flips
    if config.get("horizontal_flip", {}).get("enabled", True):
        transforms.append(
            A.HorizontalFlip(
                p=config.get("horizontal_flip", {}).get("p", 0.5)
            )
        )
    if config.get("vertical_flip", {}).get("enabled", True):
        transforms.append(
            A.VerticalFlip(
                p=config.get("vertical_flip", {}).get("p", 0.3)
            )
        )

    # Additional augmentations for robustness
    transforms.extend([
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])

    return A.Compose(transforms)


class Mixup:
    """Mixup augmentation for batch-level data augmentation.

    Mixes pairs of images and their corresponding masks with a
    random interpolation factor drawn from a Beta distribution.

    Args:
        alpha: Beta distribution parameter. Higher values produce
               more uniform mixing ratios.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Mixup to a batch.

        Args:
            images: Batch of images (B, C, H, W).
            masks: Batch of masks (B, H, W).

        Returns:
            Tuple of (mixed_images, masks_a, masks_b) and lambda value.
        """
        batch_size = images.size(0)
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Random permutation for pairing
        indices = torch.randperm(batch_size)

        mixed_images = lam * images + (1 - lam) * images[indices]

        return mixed_images, masks, masks[indices], lam


def get_validation_transform(image_size: int = 256) -> Optional[A.Compose]:
    """Get validation transform (resize only, no augmentation).

    Args:
        image_size: Target image size.

    Returns:
        Albumentations pipeline for validation.
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
    ])
