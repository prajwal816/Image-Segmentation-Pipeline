"""DeepLabV3 semantic segmentation model wrapper.

Wraps torchvision's DeepLabV3 with ResNet101 backbone, providing a
unified forward pass that returns dense logits (B, num_classes, H, W).
Handles pretrained and scratch initialization with proper head setup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as seg_models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models import resnet101, ResNet101_Weights


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
            # Replace classifier head for custom number of classes
            self.model.classifier = DeepLabHead(2048, num_classes)
            # Replace auxiliary classifier if present
            if self.model.aux_classifier is not None:
                self.model.aux_classifier = nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Conv2d(256, num_classes, kernel_size=1),
                )
        else:
            # Build from scratch with proper num_classes
            self.model = seg_models.deeplabv3_resnet101(
                weights=None,
                weights_backbone=None,
                num_classes=num_classes,
            )

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
