"""U-Net semantic segmentation model with ResNet34 encoder backbone.

Implements a U-Net architecture using a pretrained ResNet34 encoder with
skip connections and a multi-stage decoder for dense pixel-wise prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, concatenation, and double convolution."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels // 2 + skip_channels, out_channels,
                kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Handle size mismatch due to odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net with ResNet34 encoder backbone.

    Architecture:
        - Encoder: ResNet34 (pretrained on ImageNet)
        - Skip connections from encoder stages to decoder
        - Decoder: 4 upsampling stages with double convolution
        - Final 1x1 convolution for class prediction

    Args:
        num_classes: Number of output segmentation classes.
        pretrained: Whether to use ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int = 21, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # Load ResNet34 backbone
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Encoder stages
        self.encoder0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu
        )  # 64 channels, stride 2
        self.pool0 = resnet.maxpool           # stride 2
        self.encoder1 = resnet.layer1         # 64 channels
        self.encoder2 = resnet.layer2         # 128 channels
        self.encoder3 = resnet.layer3         # 256 channels
        self.encoder4 = resnet.layer4         # 512 channels

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Decoder stages
        self.decoder4 = DecoderBlock(1024, 512, 512)
        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)

        # Final upsampling and classification
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
        """
        input_size = x.shape[2:]

        # Encoder path
        e0 = self.encoder0(x)       # (B, 64, H/2, W/2)
        p0 = self.pool0(e0)         # (B, 64, H/4, W/4)
        e1 = self.encoder1(p0)      # (B, 64, H/4, W/4)
        e2 = self.encoder2(e1)      # (B, 128, H/8, W/8)
        e3 = self.encoder3(e2)      # (B, 256, H/16, W/16)
        e4 = self.encoder4(e3)      # (B, 512, H/32, W/32)

        # Bottleneck
        b = self.bottleneck(e4)     # (B, 1024, H/32, W/32)

        # Decoder path with skip connections
        d4 = self.decoder4(b, e4)   # (B, 512, H/16, W/16)
        d3 = self.decoder3(d4, e3)  # (B, 256, H/8, W/8)
        d2 = self.decoder2(d3, e2)  # (B, 128, H/4, W/4)
        d1 = self.decoder1(d2, e1)  # (B, 64, H/2, W/2)

        # Final stage
        up = self.final_upsample(d1)
        if up.shape[2:] != e0.shape[2:]:
            up = F.interpolate(
                up, size=e0.shape[2:], mode="bilinear", align_corners=False
            )
        out = torch.cat([up, e0], dim=1)
        out = self.final_conv(out)

        # Upsample to input resolution
        if out.shape[2:] != input_size:
            out = F.interpolate(
                out, size=input_size, mode="bilinear", align_corners=False
            )

        return out

    def get_backbone_params(self):
        """Return parameters of the encoder backbone for transfer learning."""
        backbone_modules = [
            self.encoder0, self.encoder1, self.encoder2,
            self.encoder3, self.encoder4,
        ]
        for module in backbone_modules:
            yield from module.parameters()

    def get_head_params(self):
        """Return parameters of the decoder head."""
        head_modules = [
            self.bottleneck, self.decoder4, self.decoder3,
            self.decoder2, self.decoder1, self.final_upsample,
            self.final_conv,
        ]
        for module in head_modules:
            yield from module.parameters()
