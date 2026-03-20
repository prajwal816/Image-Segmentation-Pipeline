"""DeepLabV3 semantic segmentation model wrapper.

Wraps torchvision's DeepLabV3 with ResNet101 backbone, providing a
unified forward pass that returns dense logits (B, num_classes, H, W).
Includes a patch for the ASPPPooling module compatibility issue in
torchvision >= 0.20.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as seg_models


def _patch_aspp_pooling(model: nn.Module) -> None:
    """Patch ASPPPooling forward method to fix interpolation bug.

    In some torchvision versions, the ASPPPooling branch fails because
    F.interpolate receives a (1,1) tensor with incompatible arguments.
    This patch replaces the forward with a working version.
    """
    aspp = model.classifier[0]  # The ASPP module
    pool_branch = aspp.convs[-1]  # The ASPPPooling branch

    # Store original sequential layers
    original_forward = pool_branch.forward

    def patched_forward(x):
        size = x.shape[-2:]
        for mod in pool_branch:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    pool_branch.forward = patched_forward


class DeepLabV3(nn.Module):
    """DeepLabV3 with ResNet101 backbone.

    Architecture:
        - Backbone: ResNet101 with atrous convolutions (pretrained on ImageNet)
        - ASPP (Atrous Spatial Pyramid Pooling) module
        - Custom classifier head for target num_classes

    Args:
        num_classes: Number of output segmentation classes.
        pretrained: Whether to use ImageNet pretrained backbone weights.
    """

    def __init__(self, num_classes: int = 21, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        if pretrained:
            # Use full pretrained DeepLabV3 (COCO weights) and replace head
            weights = seg_models.DeepLabV3_ResNet101_Weights.DEFAULT
            self.model = seg_models.deeplabv3_resnet101(weights=weights)
        else:
            # Build from scratch with proper num_classes
            self.model = seg_models.deeplabv3_resnet101(
                weights=None,
                weights_backbone=None,
                num_classes=num_classes,
            )

        # Replace classifier's final conv for custom num_classes
        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(
            in_channels, num_classes, kernel_size=1
        )

        # Replace auxiliary classifier if present
        if self.model.aux_classifier is not None:
            aux_in_channels = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(
                aux_in_channels, num_classes, kernel_size=1
            )

        # Patch ASPP pooling for torchvision compatibility
        _patch_aspp_pooling(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
        """
        input_size = x.shape[2:]
        output = self.model(x)

        # DeepLabV3 returns OrderedDict; 'out' is the main output
        logits = output["out"]

        # Ensure output matches input spatial dimensions
        if logits.shape[2:] != input_size:
            logits = F.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=False
            )

        return logits

    def get_backbone_params(self):
        """Return parameters of the backbone for transfer learning."""
        yield from self.model.backbone.parameters()

    def get_head_params(self):
        """Return parameters of the classifier head."""
        yield from self.model.classifier.parameters()
        if self.model.aux_classifier is not None:
            yield from self.model.aux_classifier.parameters()
