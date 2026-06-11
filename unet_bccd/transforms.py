"""Data augmentation transforms faithful to the original U-Net paper.

The paper (Ronneberger et al., 2015, §2) describes:
- Smooth elastic deformations using random displacement vectors on a
  coarse 3×3 grid with standard deviation of 10 pixels, interpolated
  with bicubic interpolation.
- Random rotations and flips for further invariance.

All spatial transforms are applied **jointly** to image, mask and weight
map so that pixel correspondence is preserved.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


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
# Composite augmentation pipeline
# ---------------------------------------------------------------------------

def apply_augmentations(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the full set of U-Net paper augmentations.

    Pipeline: elastic deformation → random 90° rotation → random flip.
    """
    if rng is None:
        rng = np.random.default_rng()

    image, mask, weights = elastic_deformation(
        image, mask, weights, grid_size=3, sigma=10.0, rng=rng,
    )
    image, mask, weights = random_rotate90(image, mask, weights, rng=rng)
    image, mask, weights = random_flip(image, mask, weights, rng=rng)

    return image, mask, weights
