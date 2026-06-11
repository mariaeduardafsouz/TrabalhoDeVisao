import math

import torch
import torch.nn as nn

from unet_bccd.model import UNet


def test_unet_original_shape():
    model = UNet(in_channels=1, out_channels=2)
    x = torch.randn(1, 1, 572, 572)

    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (1, 2, 388, 388)


def test_unet_has_dropout():
    model = UNet(in_channels=1, out_channels=2)
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout2d)]
    assert len(dropout_layers) >= 1, "UNet should contain Dropout2d (paper Figure 1)"
    assert dropout_layers[0].p == 0.5


def test_unet_gaussian_init_std():
    """Verify that conv weights follow N(0, sqrt(2/fan_in))."""
    model = UNet(in_channels=1, out_channels=2)
    # Check the first conv layer (in_channels=1, out=64, kernel=3x3 → fan_in=9)
    first_conv = model.down1.conv[0]
    fan_in = (
        first_conv.weight.shape[1]
        * first_conv.weight.shape[2]
        * first_conv.weight.shape[3]
    )
    expected_std = math.sqrt(2.0 / fan_in)
    actual_std = first_conv.weight.data.std().item()
    # Allow generous tolerance since this is a single sample from a distribution
    assert abs(actual_std - expected_std) < 0.3 * expected_std

