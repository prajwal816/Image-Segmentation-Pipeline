"""DeepLabV3 semantic segmentation model wrapper.

Wraps torchvision's DeepLabV3 with ResNet101 backbone, providing a
unified forward pass that returns dense logits (B, num_classes, H, W).
Includes a fix for the ASPPPooling BatchNorm issue when the global
average pooling branch produces (B, C, 1, 1) tensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as seg_models


def _fix_aspp_pooling_bn(model: nn.Module) -> None:
    """Fix BatchNorm in ASPPPooling branch to handle (B, C, 1, 1) tensors.

    In training mode, BatchNorm requires >1 value per channel spatially.
    The ASPPPooling branch produces (B, C, 1, 1) after global avg pool,
    which causes BatchNorm to fail. This fix replaces the BN with an
    eval-mode wrapper that always runs in eval mode.
    """
    aspp = model.classifier[0]  # ASPP module
    pool_branch = aspp.convs[-1]  # ASPPPooling

    # Find and wrap the BatchNorm inside ASPPPooling
    for i, mod in enumerate(pool_branch):
        if isinstance(mod, nn.BatchNorm2d):
            pool_branch[i] = _EvalBatchNorm(mod)
            break


class _EvalBatchNorm(nn.Module):
    """BatchNorm wrapper that always runs in eval mode.

    Used for the ASPPPooling branch where spatial dims are (1,1)
    after global average pooling, making training-mode BN impossible.
    """

    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        self.bn = bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always run BN in eval mode for (1,1) spatial dims
        self.bn.eval()
        with torch.no_grad():
            # Compute running stats update manually if training
            pass
        result = self.bn(x)
        if self.training:
            self.bn.train()
        return result


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
            weights = seg_models.DeepLabV3_ResNet101_Weights.DEFAULT
            self.model = seg_models.deeplabv3_resnet101(weights=weights)
        else:
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

        # Fix ASPPPooling BatchNorm for (1,1) spatial dims
        _fix_aspp_pooling_bn(self.model)

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
