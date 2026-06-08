"""Utilities for training a valid-convolution U-Net on BCCD masks."""

__all__ = ["UNet", "compute_unet_weight_map"]


def __getattr__(name: str):
    if name == "UNet":
        from .model import UNet

        return UNet
    if name == "compute_unet_weight_map":
        from .weights import compute_unet_weight_map

        return compute_unet_weight_map
    raise AttributeError(f"module 'unet_bccd' has no attribute {name!r}")
