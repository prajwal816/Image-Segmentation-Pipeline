"""Transfer learning utilities for segmentation models.

Provides functions to freeze/unfreeze backbone layers and implement
staged unfreezing schedules for fine-tuning pretrained models.
"""

from typing import List, Optional

import torch.nn as nn

from src.utils.logger import get_logger


logger = get_logger("transfer", file=False)


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone/encoder parameters.

    Args:
        model: Segmentation model with get_backbone_params() method.
    """
    if hasattr(model, "get_backbone_params"):
        count = 0
        for param in model.get_backbone_params():
            param.requires_grad = False
            count += 1
        logger.info(f"Froze {count} backbone parameters")
    else:
        logger.warning("Model does not have get_backbone_params() method")


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all backbone/encoder parameters.

    Args:
        model: Segmentation model with get_backbone_params() method.
    """
    if hasattr(model, "get_backbone_params"):
        count = 0
        for param in model.get_backbone_params():
            param.requires_grad = True
            count += 1
        logger.info(f"Unfroze {count} backbone parameters")
    else:
        logger.warning("Model does not have get_backbone_params() method")


def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """Freeze specific named layers.

    Args:
        model: PyTorch model.
        layer_names: List of layer name prefixes to freeze.
    """
    count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(ln) for ln in layer_names):
            param.requires_grad = False
            count += 1
    logger.info(f"Froze {count} parameters matching {layer_names}")


def unfreeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """Unfreeze specific named layers.

    Args:
        model: PyTorch model.
        layer_names: List of layer name prefixes to unfreeze.
    """
    count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(ln) for ln in layer_names):
            param.requires_grad = True
            count += 1
    logger.info(f"Unfroze {count} parameters matching {layer_names}")


class StagedUnfreezer:
    """Gradually unfreeze backbone layers during training.

    Implements a schedule where deeper layers are unfrozen first
    (closest to the output), and shallower layers are unfrozen
    later as training progresses.

    Args:
        model: Segmentation model.
        unfreeze_epoch: Epoch at which to start unfreezing.
        total_stages: Number of unfreezing stages.
    """

    def __init__(
        self,
        model: nn.Module,
        unfreeze_epoch: int = 5,
        total_stages: int = 4,
    ):
        self.model = model
        self.unfreeze_epoch = unfreeze_epoch
        self.total_stages = total_stages
        self._unfrozen_stages = 0

        # Determine layer groups for staged unfreezing
        self.layer_groups = self._get_layer_groups()

    def _get_layer_groups(self) -> List[List[str]]:
        """Get layer groups ordered from deepest to shallowest."""
        from src.models.unet import UNet
        from src.models.deeplabv3 import DeepLabV3

        if isinstance(self.model, UNet):
            return [
                ["encoder4"],
                ["encoder3"],
                ["encoder2"],
                ["encoder1", "encoder0"],
            ]
        elif isinstance(self.model, DeepLabV3):
            return [
                ["model.backbone.layer4"],
                ["model.backbone.layer3"],
                ["model.backbone.layer2"],
                ["model.backbone.layer1", "model.backbone.conv1"],
            ]
        else:
            # Generic fallback: group parameters evenly
            all_names = [n for n, _ in self.model.named_parameters()]
            chunk_size = max(1, len(all_names) // self.total_stages)
            return [
                all_names[i:i + chunk_size]
                for i in range(0, len(all_names), chunk_size)
            ]

    def step(self, epoch: int) -> bool:
        """Check and potentially unfreeze the next stage.

        Args:
            epoch: Current training epoch.

        Returns:
            True if a new stage was unfrozen.
        """
        if self._unfrozen_stages >= len(self.layer_groups):
            return False

        # Calculate stage interval
        stage_interval = max(
            1,
            (self.unfreeze_epoch + self.total_stages - 1) // self.total_stages,
        )

        target_stage = min(
            max(0, (epoch - self.unfreeze_epoch) // stage_interval + 1),
            len(self.layer_groups),
        )

        unfrozen = False
        while self._unfrozen_stages < target_stage:
            group = self.layer_groups[self._unfrozen_stages]
            unfreeze_layers(self.model, group)
            self._unfrozen_stages += 1
            unfrozen = True
            logger.info(
                f"Stage {self._unfrozen_stages}/{len(self.layer_groups)}: "
                f"Unfroze {group}"
            )

        return unfrozen


def setup_transfer_learning(model: nn.Module, config: dict) -> Optional[StagedUnfreezer]:
    """Set up transfer learning based on configuration.

    Args:
        model: Segmentation model.
        config: Configuration dictionary.

    Returns:
        StagedUnfreezer instance if staged unfreezing is enabled, else None.
    """
    tl_cfg = config.get("transfer_learning", {})

    if tl_cfg.get("freeze_backbone", True):
        freeze_backbone(model)
        logger.info("Transfer learning: backbone frozen")

        if tl_cfg.get("staged_unfreeze", True):
            unfreezer = StagedUnfreezer(
                model,
                unfreeze_epoch=tl_cfg.get("unfreeze_epoch", 5),
            )
            logger.info("Transfer learning: staged unfreezing enabled")
            return unfreezer

    return None
