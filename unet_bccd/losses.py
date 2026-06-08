from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import center_crop_last_dims


def weighted_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted CE after center-cropping target/weights to logits size."""
    target_h, target_w = logits.shape[-2:]
    target_crop = center_crop_last_dims(target, target_h, target_w)
    weights_crop = center_crop_last_dims(weights, target_h, target_w)

    pixel_loss = F.cross_entropy(logits, target_crop, reduction="none")
    loss = (pixel_loss * weights_crop).mean()
    return loss, target_crop
