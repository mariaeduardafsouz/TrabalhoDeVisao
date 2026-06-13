"""Data augmentation transforms for U-Net segmentation training.

The original paper (Ronneberger et al., 2015, §2) describes:
- Smooth elastic deformations using random displacement vectors on a
  coarse 3×3 grid with standard deviation of 10 pixels, interpolated
  with bicubic interpolation.
- Random rotations and flips for further invariance.

This module extends those augmentations with additional **intensity**
transforms (brightness/contrast, Gaussian noise) that can be
selectively enabled via the ``strategies`` parameter.

All spatial transforms are applied **jointly** to image, mask and weight
map so that pixel correspondence is preserved.  Intensity transforms
are applied to the **image only**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Augmentation strategies used in the original U-Net paper.
DEFAULT_STRATEGIES: list[str] = ["elastic", "rotate90", "flip"]

#: Every strategy supported by :func:`apply_augmentations`.
AVAILABLE_STRATEGIES: list[str] = [
    "elastic",
    "rotate90",
    "flip",
    "brightness_contrast",
    "gaussian_noise",
]


@dataclass
class AugmentationParams:
    """Tuneable parameters for each augmentation strategy."""

    # Elastic deformation
    elastic_grid_size: int = 3
    elastic_sigma: float = 10.0
    # Brightness / contrast
    brightness_range: float = 0.2
    contrast_range: float = 0.2
    brightness_contrast_prob: float = 0.5
    # Gaussian noise
    noise_sigma_max: float = 0.03
    noise_prob: float = 0.3


# ---------------------------------------------------------------------------
# Elastic deformation (paper §2)
# ---------------------------------------------------------------------------

def elastic_deformation(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    grid_size: int = 3,
    sigma: float = 10.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply smooth elastic deformation as described in the U-Net paper.

    Random displacement vectors are sampled on a coarse *grid_size × grid_size*
    grid from a Gaussian with standard deviation *sigma* (in pixels).
    Per-pixel displacements are obtained via bicubic (order-3) interpolation.
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape[:2]

    # Coarse displacement field
    dx_coarse = rng.normal(0, sigma, size=(grid_size, grid_size)).astype(np.float32)
    dy_coarse = rng.normal(0, sigma, size=(grid_size, grid_size)).astype(np.float32)

    # Upscale to full resolution via bicubic (order=3) interpolation
    coarse_y = np.linspace(0, grid_size - 1, h)
    coarse_x = np.linspace(0, grid_size - 1, w)
    grid_y, grid_x = np.meshgrid(coarse_y, coarse_x, indexing="ij")

    dx = map_coordinates(dx_coarse, [grid_y, grid_x], order=3, mode="reflect")
    dy = map_coordinates(dy_coarse, [grid_y, grid_x], order=3, mode="reflect")

    # Original pixel coordinates + displacement
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    map_y = (y_coords + dy).astype(np.float32)
    map_x = (x_coords + dx).astype(np.float32)

    coords = [map_y, map_x]

    # Apply to image (bicubic), mask (nearest-neighbour), weights (bilinear)
    image_out = map_coordinates(image, coords, order=3, mode="reflect").astype(
        image.dtype
    )
    mask_out = map_coordinates(mask, coords, order=0, mode="constant", cval=0).astype(
        mask.dtype
    )
    weights_out = map_coordinates(weights, coords, order=1, mode="reflect").astype(
        weights.dtype
    )

    return image_out, mask_out, weights_out


# ---------------------------------------------------------------------------
# Random rotation (multiples of 90°)
# ---------------------------------------------------------------------------

def random_rotate90(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly rotate all arrays by 0, 90, 180 or 270 degrees."""
    if rng is None:
        rng = np.random.default_rng()
    k = rng.integers(0, 4)  # 0..3
    if k == 0:
        return image, mask, weights
    return (
        np.rot90(image, k).copy(),
        np.rot90(mask, k).copy(),
        np.rot90(weights, k).copy(),
    )


# ---------------------------------------------------------------------------
# Random flip (horizontal and/or vertical)
# ---------------------------------------------------------------------------

def random_flip(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly flip horizontally and/or vertically."""
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() > 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
        weights = np.fliplr(weights).copy()

    if rng.random() > 0.5:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
        weights = np.flipud(weights).copy()

    return image, mask, weights


# ---------------------------------------------------------------------------
# Intensity augmentations (image only — no effect on mask or weights)
# ---------------------------------------------------------------------------

def random_brightness_contrast(
    image: np.ndarray,
    brightness_range: float = 0.2,
    contrast_range: float = 0.2,
    prob: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly adjust brightness and contrast of *image*.

    Contrast is applied as a multiplicative factor and brightness as an
    additive offset.  The result is clipped to [0, 1].

    Applied with probability *prob*; otherwise returns *image* unchanged.
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() > prob:
        return image

    contrast_factor = 1.0 + rng.uniform(-contrast_range, contrast_range)
    brightness_offset = rng.uniform(-brightness_range, brightness_range)
    image = image * contrast_factor + brightness_offset

    return np.clip(image, 0.0, 1.0).astype(np.float32)


def random_gaussian_noise(
    image: np.ndarray,
    sigma_max: float = 0.03,
    prob: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian noise with a random standard deviation in [0, *sigma_max*].

    Applied with probability *prob*; otherwise returns *image* unchanged.
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() > prob:
        return image

    sigma = rng.uniform(0, sigma_max)
    noise = rng.normal(0, sigma, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Composite augmentation pipeline
# ---------------------------------------------------------------------------

def apply_augmentations(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    strategies: list[str] | None = None,
    params: AugmentationParams | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a configurable augmentation pipeline.

    Parameters
    ----------
    strategies : list[str] or None
        Ordered list of strategy names to apply.  Each name must be one
        of :data:`AVAILABLE_STRATEGIES`.  When *None*, the original
        paper defaults (``["elastic", "rotate90", "flip"]``) are used.
    params : AugmentationParams or None
        Tuneable parameters for each strategy.  Defaults are used when
        *None*.
    rng : numpy Generator
        Random number generator for reproducibility.

    Returns
    -------
    image, mask, weights : np.ndarray
        Augmented arrays with the same shapes as the inputs.
    """
    if rng is None:
        rng = np.random.default_rng()
    if strategies is None:
        strategies = DEFAULT_STRATEGIES
    if params is None:
        params = AugmentationParams()

    for strategy in strategies:
        if strategy == "elastic":
            image, mask, weights = elastic_deformation(
                image, mask, weights,
                grid_size=params.elastic_grid_size,
                sigma=params.elastic_sigma,
                rng=rng,
            )
        elif strategy == "rotate90":
            image, mask, weights = random_rotate90(
                image, mask, weights, rng=rng,
            )
        elif strategy == "flip":
            image, mask, weights = random_flip(
                image, mask, weights, rng=rng,
            )
        elif strategy == "brightness_contrast":
            image = random_brightness_contrast(
                image,
                brightness_range=params.brightness_range,
                contrast_range=params.contrast_range,
                prob=params.brightness_contrast_prob,
                rng=rng,
            )
        elif strategy == "gaussian_noise":
            image = random_gaussian_noise(
                image,
                sigma_max=params.noise_sigma_max,
                prob=params.noise_prob,
                rng=rng,
            )
        else:
            raise ValueError(
                f"Unknown augmentation strategy: {strategy!r}. "
                f"Available: {AVAILABLE_STRATEGIES}"
            )

    return image, mask, weights
