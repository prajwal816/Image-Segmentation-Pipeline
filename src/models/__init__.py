"""Model factory for segmentation architectures."""

from typing import Optional

import torch.nn as nn

from src.models.unet import UNet
from src.models.deeplabv3 import DeepLabV3


_MODEL_REGISTRY = {
    "unet": UNet,
    "deeplabv3": DeepLabV3,
}


def get_model(
    name: str,
    num_classes: int = 21,
    pretrained: bool = True,
) -> nn.Module:
    """Factory function to create a segmentation model.

    Args:
        name: Model architecture name ("unet" or "deeplabv3").
        num_classes: Number of output segmentation classes.
        pretrained: Whether to use ImageNet pretrained backbone.

    Returns:
        Initialized segmentation model.

    Raises:
        ValueError: If model name is not recognized.
    """
    name_lower = name.lower().strip()
    if name_lower not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available}"
        )

    model_class = _MODEL_REGISTRY[name_lower]
    return model_class(num_classes=num_classes, pretrained=pretrained)


def list_models():
    """Return list of available model names."""
    return list(_MODEL_REGISTRY.keys())


__all__ = ["get_model", "list_models", "UNet", "DeepLabV3"]
