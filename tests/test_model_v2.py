import torch
import torch.nn as nn

from unet_bccd.model_v2 import UNetV2


def test_unetv2_same_padding_preserves_shape():
    """With padding=1 the output spatial size must equal the input."""
    model = UNetV2(in_channels=1, out_channels=2, padding=1)
    x = torch.randn(1, 1, 256, 256)

    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (1, 2, 256, 256)


def test_unetv2_valid_padding_shrinks_shape():
    """With padding=0 the output should be smaller than the input (like v1)."""
    model = UNetV2(in_channels=1, out_channels=2, padding=0)
    x = torch.randn(1, 1, 572, 572)

    with torch.no_grad():
        y = model(x)

    # Same as the original U-Net: 572 → 388
    assert tuple(y.shape) == (1, 2, 388, 388)


def test_unetv2_has_batchnorm():
    model = UNetV2(in_channels=1, out_channels=2)
    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    # 5 encoder levels × 2 convs + 4 decoder levels × 2 convs = 18 BN layers
    assert len(bn_layers) >= 18, f"Expected >= 18 BatchNorm layers, got {len(bn_layers)}"


def test_unetv2_has_dropout():
    model = UNetV2(in_channels=1, out_channels=2)
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout2d)]
    assert len(dropout_layers) >= 1
    assert dropout_layers[0].p == 0.5


def test_unetv2_no_conv_bias_with_batchnorm():
    """Conv layers followed by BatchNorm should have bias=False."""
    model = UNetV2(in_channels=1, out_channels=2)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name != "out_conv":
            assert module.bias is None, (
                f"{name} has bias=True but should be False (BatchNorm handles bias)"
            )
