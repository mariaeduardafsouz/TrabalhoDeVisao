"""Improved U-Net with Batch Normalization and configurable padding.

This module provides ``UNetV2``, an enhanced version of the original
U-Net (``model.py``) that incorporates modern best practices while
retaining the same encoder–decoder architecture.

Key differences from the original:

* **BatchNorm2d** after every convolution (stabilises training, acts
  as a regulariser, and allows higher learning rates).
* ``bias=False`` in convolutions (redundant when followed by BatchNorm).
* **Kaiming He** weight initialisation (``kaiming_normal_``), the
  standard for ReLU networks.
* **Configurable padding** (``padding=0`` reproduces the original
  valid-convolution behaviour; ``padding=1`` preserves spatial
  dimensions, the modern default).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3×3 convolutions each followed by BatchNorm and ReLU.

    When ``padding=1`` the spatial dimensions are preserved.  With
    ``padding=0`` each convolution shrinks H and W by 2 (valid mode).
    """

    def __init__(self, in_channels: int, out_channels: int, padding: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetV2(nn.Module):
    """Improved U-Net with BatchNorm and configurable padding.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (e.g. 1 for greyscale).
    out_channels : int
        Number of output classes (e.g. 2 for binary segmentation).
    padding : int
        Convolution padding.  ``0`` = valid (original paper), ``1`` = same
        (modern default — output has the same spatial size as the input).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.padding = padding

        # ---- Encoder (contracting path) ----
        self.down1 = DoubleConv(in_channels, 64, padding)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2 = DoubleConv(64, 128, padding)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down3 = DoubleConv(128, 256, padding)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down4 = DoubleConv(256, 512, padding)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout2d(p=0.5)

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(512, 1024, padding)

        # ---- Decoder (expansive path) ----
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up4 = DoubleConv(1024, 512, padding)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = DoubleConv(512, 256, padding)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = DoubleConv(256, 128, padding)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = DoubleConv(128, 64, padding)

        # ---- Output ----
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self._init_weights()

    # ------------------------------------------------------------------
    # Kaiming He initialisation (standard for ReLU + BatchNorm)
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        d4 = self.up4(self._match_and_concat(x4, up4))

        up3 = self.upconv3(d4)
        d3 = self.up3(self._match_and_concat(x3, up3))

        up2 = self.upconv2(d3)
        d2 = self.up2(self._match_and_concat(x2, up2))

        up1 = self.upconv1(d2)
        d1 = self.up1(self._match_and_concat(x1, up1))

        return self.out_conv(d1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_and_concat(
        x_enc: torch.Tensor,
        x_dec: torch.Tensor,
    ) -> torch.Tensor:
        """Crop encoder features to decoder size (if needed) and concatenate.

        With ``padding=1`` the sizes already match, so no crop is performed.
        With ``padding=0`` the encoder features are larger and are
        centre-cropped to match the decoder.
        """
        _, _, h_enc, w_enc = x_enc.shape
        _, _, h_dec, w_dec = x_dec.shape

        if h_enc == h_dec and w_enc == w_dec:
            return torch.cat([x_enc, x_dec], dim=1)

        # Centre-crop encoder features to match decoder
        h_crop = (h_enc - h_dec) // 2
        w_crop = (w_enc - w_dec) // 2
        x_enc_cropped = x_enc[
            :, :,
            h_crop : h_crop + h_dec,
            w_crop : w_crop + w_dec,
        ]
        return torch.cat([x_enc_cropped, x_dec], dim=1)
