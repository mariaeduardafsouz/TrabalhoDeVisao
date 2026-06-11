from __future__ import annotations

import math

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 valid convolutions followed by ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Original U-Net layout with valid convolutions.

    Faithful to Ronneberger et al. (2015):
    - Gaussian weight init with std = sqrt(2 / fan_in)
    - Dropout (p=0.5) at the two deepest levels (512-ch and 1024-ch)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2) -> None:
        super().__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout2d(p=0.5)

        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        """Gaussian init with std = sqrt(2 / N) as in the U-Net paper (§2)."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                fan_in = (
                    module.weight.shape[1]
                    * module.weight.shape[2]
                    * module.weight.shape[3]
                )
                std = math.sqrt(2.0 / fan_in)
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        p1 = self.pool1(x1)

        x2 = self.down2(p1)
        p2 = self.pool2(x2)

        x3 = self.down3(p2)
        p3 = self.pool3(x3)

        x4 = self.down4(p3)
        x4 = self.dropout(x4)
        p4 = self.pool4(x4)

        bottleneck = self.bottleneck(p4)
        bottleneck = self.dropout(bottleneck)

        up4 = self.upconv4(bottleneck)
        d4 = self.up4(self._crop_and_concat(x4, up4))

        up3 = self.upconv3(d4)
        d3 = self.up3(self._crop_and_concat(x3, up3))

        up2 = self.upconv2(d3)
        d2 = self.up2(self._crop_and_concat(x2, up2))

        up1 = self.upconv1(d2)
        d1 = self.up1(self._crop_and_concat(x1, up1))

        return self.out_conv(d1)

    @staticmethod
    def _crop_and_concat(x_large: torch.Tensor, x_small: torch.Tensor) -> torch.Tensor:
        _, _, h_large, w_large = x_large.shape
        _, _, h_small, w_small = x_small.shape

        if h_small > h_large or w_small > w_large:
            raise ValueError(
                f"Cannot crop skip tensor from {(h_large, w_large)} to {(h_small, w_small)}"
            )

        h_crop = (h_large - h_small) // 2
        w_crop = (w_large - w_small) // 2
        x_large_cropped = x_large[
            :,
            :,
            h_crop : h_crop + h_small,
            w_crop : w_crop + w_small,
        ]
        return torch.cat([x_large_cropped, x_small], dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
