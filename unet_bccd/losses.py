from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import center_crop_last_dims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_crop(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Centre-crop *target* (and optionally *weights*) to match *logits*.

    When the model uses ``padding=1`` the spatial sizes already match and
    no crop is performed.  With ``padding=0`` (valid convolutions) the
    target is centre-cropped to the smaller logits size.
    """
    target_h, target_w = logits.shape[-2:]
    if target.shape[-2] != target_h or target.shape[-1] != target_w:
        target = center_crop_last_dims(target, target_h, target_w)
        if weights is not None:
            weights = center_crop_last_dims(weights, target_h, target_w)
    return target, weights


# ---------------------------------------------------------------------------
# Weighted Cross-Entropy (original)
# ---------------------------------------------------------------------------

def weighted_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted CE after auto-cropping target/weights to logits size."""
    target_crop, weights_crop = _auto_crop(logits, target, weights)

    pixel_loss = F.cross_entropy(logits, target_crop, reduction="none")
    weights_crop = torch.clamp(weights_crop, max=50.0)
    loss = (pixel_loss * weights_crop).mean()
    return loss, target_crop


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Differentiable Dice loss (1 − Dice coefficient) for the foreground.

    Operates on the softmax foreground probability (class index 1).
    The *target* must already be cropped to match *logits* spatial size.
    """
    probs = F.softmax(logits, dim=1)[:, 1]  # foreground probability
    target_float = target.float()
    intersection = (probs * target_float).sum()
    return 1.0 - (2.0 * intersection + smooth) / (
        probs.sum() + target_float.sum() + smooth
    )


# ---------------------------------------------------------------------------
# Combined Loss (CE + Dice)
# ---------------------------------------------------------------------------

def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    dice_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted CE + Dice loss.

    ``dice_weight`` controls the balance::

        loss = dice_weight × Dice  +  (1 − dice_weight) × weighted_CE

    Returns ``(loss, target_crop)`` to match the ``weighted_cross_entropy``
    return signature.
    """
    target_crop, weights_crop = _auto_crop(logits, target, weights)

    # Weighted CE component
    pixel_loss = F.cross_entropy(logits, target_crop, reduction="none")
    weights_crop = torch.clamp(weights_crop, max=50.0)
    ce = (pixel_loss * weights_crop).mean()

    # Dice component
    dl = dice_loss(logits, target_crop)

    loss = dice_weight * dl + (1.0 - dice_weight) * ce
    return loss, target_crop
