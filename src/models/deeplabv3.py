"""DeepLabV3 semantic segmentation model wrapper.

Wraps torchvision's DeepLabV3 with ResNet101 backbone, replacing the
classifier head for custom number of classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as seg_models


class DeepLabV3(nn.Module):
    """DeepLabV3 with ResNet101 backbone.

    Architecture:
        - Backbone: ResNet101 with atrous convolutions (pretrained on ImageNet)
        - ASPP (Atrous Spatial Pyramid Pooling) module
        - Custom classifier head for target num_classes

    Args:
        num_classes: Number of output segmentation classes.
        pretrained: Whether to use ImageNet pretrained backbone.
    """

    def __init__(self, num_classes: int = 21, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained DeepLabV3 with ResNet101
        if pretrained:
            weights = seg_models.DeepLabV3_ResNet101_Weights.DEFAULT
            self.model = seg_models.deeplabv3_resnet101(weights=weights)
        else:
            self.model = seg_models.deeplabv3_resnet101(weights=None)

        # Replace classifier head for custom number of classes
        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(
            in_channels, num_classes, kernel_size=1
        )

        # Also replace auxiliary classifier if present
        if self.model.aux_classifier is not None:
            aux_in_channels = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(
                aux_in_channels, num_classes, kernel_size=1
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
