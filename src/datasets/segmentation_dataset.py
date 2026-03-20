"""Segmentation dataset with synthetic data generation capability.

Provides a PyTorch Dataset for loading image/mask pairs and a synthetic
data generator for demo and testing purposes.
"""

import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation tasks.

    Loads image/mask pairs from directories. Supports on-the-fly
    augmentation via albumentations transforms.

    Args:
        image_dir: Path to directory containing images.
        mask_dir: Path to directory containing masks.
        transform: Albumentations transform pipeline.
        image_size: Target size for resizing images.
        num_classes: Number of segmentation classes.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Any] = None,
        image_size: int = 256,
        num_classes: int = 21,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        self.num_classes = num_classes

        # Collect matching image-mask pairs
        self.samples: List[Tuple[str, str]] = []
        if os.path.exists(image_dir) and os.path.exists(mask_dir):
            image_files = sorted([
                f for f in os.listdir(image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ])
            for img_file in image_files:
                mask_file = os.path.splitext(img_file)[0] + ".png"
                mask_path = os.path.join(mask_dir, mask_file)
                if os.path.exists(mask_path):
                    self.samples.append((
                        os.path.join(image_dir, img_file),
                        mask_path,
                    ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(
            mask, (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        # Clamp mask values to valid class range
        mask = np.clip(mask, 0, self.num_classes - 1)

        # Apply augmentation
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask.astype(np.int64)) if isinstance(mask, np.ndarray) else mask.long()

        return {"image": image, "mask": mask}


class SyntheticSegmentationDataset(Dataset):
    """Synthetic dataset that generates random images with geometric shapes.

    Useful for testing and demo purposes without real data.

    Args:
        num_samples: Number of synthetic samples to generate.
        image_size: Size of generated images (square).
        num_classes: Number of segmentation classes.
        transform: Optional albumentations transform pipeline.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 500,
        image_size: int = 256,
        num_classes: int = 21,
        transform: Optional[Any] = None,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = min(num_classes, 10)  # Cap for synthetic
        self.transform = transform
        self.rng = np.random.RandomState(seed)

        # Pre-generate data
        self.images, self.masks = self._generate_data()

    def _generate_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate synthetic images and masks with geometric shapes."""
        images = []
        masks = []

        # Color palette for shapes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]

        for _ in range(self.num_samples):
            # Random background
            bg_color = self.rng.randint(100, 200, size=3).tolist()
            image = np.full(
                (self.image_size, self.image_size, 3),
                bg_color, dtype=np.uint8
            )
            mask = np.zeros(
                (self.image_size, self.image_size), dtype=np.uint8
            )

            # Add random number of shapes
            num_shapes = self.rng.randint(3, 8)
            for _ in range(num_shapes):
                class_id = self.rng.randint(1, self.num_classes)
                color = colors[(class_id - 1) % len(colors)]
                shape_type = self.rng.choice(["circle", "rectangle", "triangle"])

                if shape_type == "circle":
                    cx = self.rng.randint(30, self.image_size - 30)
                    cy = self.rng.randint(30, self.image_size - 30)
                    r = self.rng.randint(15, 50)
                    cv2.circle(image, (cx, cy), r, color, -1)
                    cv2.circle(mask, (cx, cy), r, int(class_id), -1)

                elif shape_type == "rectangle":
                    x1 = self.rng.randint(10, self.image_size - 60)
                    y1 = self.rng.randint(10, self.image_size - 60)
                    x2 = x1 + self.rng.randint(30, 80)
                    y2 = y1 + self.rng.randint(30, 80)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), int(class_id), -1)

                elif shape_type == "triangle":
                    pts = np.array([
                        [self.rng.randint(10, self.image_size - 10),
                         self.rng.randint(10, self.image_size - 10)]
                        for _ in range(3)
                    ])
                    cv2.fillPoly(image, [pts], color)
                    cv2.fillPoly(mask, [pts], int(class_id))

            # Add noise
            noise = self.rng.normal(0, 10, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            images.append(image)
            masks.append(mask)

        return images, masks

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()

        # Apply augmentation
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask.astype(np.int64)) if isinstance(mask, np.ndarray) else mask.long()

        return {"image": image, "mask": mask}


def create_dataloaders(
    config: Dict,
    transform: Optional[Any] = None,
    val_transform: Optional[Any] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders.

    Args:
        config: Configuration dictionary.
        transform: Training augmentation pipeline.
        val_transform: Validation augmentation pipeline.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})

    root = dataset_cfg.get("root", "data")
    image_dir = os.path.join(root, dataset_cfg.get("image_dir", "images"))
    mask_dir = os.path.join(root, dataset_cfg.get("mask_dir", "masks"))
    image_size = dataset_cfg.get("image_size", 256)
    num_classes = config.get("model", {}).get("num_classes", 21)
    batch_size = training_cfg.get("batch_size", 8)
    num_workers = dataset_cfg.get("num_workers", 4)

    # Use synthetic data if enabled or no real data available
    synthetic_cfg = dataset_cfg.get("synthetic", {})
    use_synthetic = synthetic_cfg.get("enabled", False) or not (
        os.path.exists(image_dir) and len(os.listdir(image_dir)) > 0
    )

    if use_synthetic:
        num_samples = synthetic_cfg.get("num_samples", 500)
        full_dataset = SyntheticSegmentationDataset(
            num_samples=num_samples,
            image_size=image_size,
            num_classes=num_classes,
            transform=None,  # Applied per-split below
        )
        # Split indices
        indices = list(range(len(full_dataset)))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42
        )
        train_dataset = _TransformSubset(full_dataset, train_idx, transform)
        val_dataset = _TransformSubset(full_dataset, val_idx, val_transform)
    else:
        train_dataset = SegmentationDataset(
            image_dir, mask_dir, transform=transform,
            image_size=image_size, num_classes=num_classes,
        )
        val_dataset = SegmentationDataset(
            image_dir, mask_dir, transform=val_transform,
            image_size=image_size, num_classes=num_classes,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


class _TransformSubset(Dataset):
    """Subset wrapper that applies a specific transform."""

    def __init__(self, dataset: Dataset, indices: List[int], transform: Optional[Any]):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = sample["image"]
            mask = sample["mask"]
            # Convert back to numpy for albumentations
            if isinstance(image, torch.Tensor):
                image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy().astype(np.uint8)
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask.astype(np.int64))
            sample = {"image": image, "mask": mask}
        return sample
