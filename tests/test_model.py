import torch

from unet_bccd.model import UNet


def test_unet_original_shape():
    model = UNet(in_channels=1, out_channels=2)
    x = torch.randn(1, 1, 572, 572)

    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (1, 2, 388, 388)
