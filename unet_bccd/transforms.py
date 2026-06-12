"""Data augmentation transforms for U-Net training.

The spatial transforms are applied jointly to image, mask and weight map so that
pixel correspondence is preserved. Intensity-only transforms are applied only to
image, because masks and weight maps are labels/weights rather than visual data.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from .augmentation import random_local_rotation

AVAILABLE_AUGMENTATION_STRATEGIES = (
    "paper",
    "intensity",
    "acquisition_noise",
    "local_rotation",
)


def _get_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def normalize_strategies(strategies: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if strategies is None:
        return ()
    if isinstance(strategies, str):
        parts = strategies.replace(",", " ").split()
    else:
        parts = []
        for strategy in strategies:
            parts.extend(str(strategy).replace(",", " ").split())

    normalized = tuple(part.strip().lower() for part in parts if part.strip())
    if "none" in normalized:
        return ()

    unknown = sorted(set(normalized) - set(AVAILABLE_AUGMENTATION_STRATEGIES))
    if unknown:
        allowed = ", ".join(AVAILABLE_AUGMENTATION_STRATEGIES)
        raise ValueError(f"Unknown augmentation strategy {unknown}. Allowed: {allowed}")
    return normalized


# ---------------------------------------------------------------------------
# Paper U-Net spatial augmentation
# ---------------------------------------------------------------------------

def elastic_deformation(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    grid_size: int = 3,
    sigma: float = 10.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply smooth elastic deformation as described in the U-Net paper."""
    rng = _get_rng(rng)
    height, width = image.shape[:2]

    dx_coarse = rng.normal(0, sigma, size=(grid_size, grid_size)).astype(np.float32)
    dy_coarse = rng.normal(0, sigma, size=(grid_size, grid_size)).astype(np.float32)

    coarse_y = np.linspace(0, grid_size - 1, height)
    coarse_x = np.linspace(0, grid_size - 1, width)
    grid_y, grid_x = np.meshgrid(coarse_y, coarse_x, indexing="ij")

    dx = map_coordinates(dx_coarse, [grid_y, grid_x], order=3, mode="reflect")
    dy = map_coordinates(dy_coarse, [grid_y, grid_x], order=3, mode="reflect")

    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    coords = [(y_coords + dy).astype(np.float32), (x_coords + dx).astype(np.float32)]

    image_out = map_coordinates(image, coords, order=3, mode="reflect").astype(np.float32)
    mask_out = map_coordinates(mask, coords, order=0, mode="constant", cval=0).astype(mask.dtype)
    weights_out = map_coordinates(weights, coords, order=1, mode="reflect").astype(np.float32)
    return image_out, mask_out, weights_out


def random_rotate90(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly rotate all arrays by 0, 90, 180 or 270 degrees."""
    rng = _get_rng(rng)
    k = int(rng.integers(0, 4))
    if k == 0:
        return image, mask, weights
    return np.rot90(image, k).copy(), np.rot90(mask, k).copy(), np.rot90(weights, k).copy()


def random_flip(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly flip horizontally and/or vertically."""
    rng = _get_rng(rng)
    if rng.random() > 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
        weights = np.fliplr(weights).copy()
    if rng.random() > 0.5:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
        weights = np.flipud(weights).copy()
    return image, mask, weights


def apply_paper_augmentations(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper strategy: elastic deformation, random 90-degree rotation and flip."""
    rng = _get_rng(rng)
    image, mask, weights = elastic_deformation(image, mask, weights, grid_size=3, sigma=10.0, rng=rng)
    image, mask, weights = random_rotate90(image, mask, weights, rng=rng)
    image, mask, weights = random_flip(image, mask, weights, rng=rng)
    return image, mask, weights


# Backwards-compatible name used by the original tests and Dataset implementation.
def apply_augmentations(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return apply_paper_augmentations(image, mask, weights, rng=rng)


# ---------------------------------------------------------------------------
# Additional strategies for the medical microscopy setting
# ---------------------------------------------------------------------------

def random_brightness_contrast(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    brightness_range: float = 0.10,
    contrast_range: float = 0.10,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RandomBrightnessContrast-style transform applied only to the image."""
    rng = _get_rng(rng)
    contrast = float(rng.uniform(1.0 - contrast_range, 1.0 + contrast_range))
    brightness = float(rng.uniform(-brightness_range, brightness_range))
    image_out = image.astype(np.float32) * contrast + brightness
    return np.clip(image_out, 0.0, 1.0).astype(np.float32), mask, weights


def clahe_enhancement(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CLAHE contrast enhancement applied only to the image."""
    tile_grid_size = max(1, int(tile_grid_size))
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    image_out = clahe.apply(image_uint8).astype(np.float32) / 255.0
    return image_out, mask, weights


def apply_intensity_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intensity strategy: brightness/contrast plus CLAHE."""
    image, mask, weights = random_brightness_contrast(image, mask, weights, rng=rng)
    image, mask, weights = clahe_enhancement(image, mask, weights)
    return image, mask, weights


def gaussian_blur_noise(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    blur_probability: float = 0.5,
    noise_probability: float = 0.5,
    noise_std: float = 0.03,
    kernel_size: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply light blur/noise to simulate focus and acquisition variation."""
    rng = _get_rng(rng)
    image_out = image.astype(np.float32, copy=True)

    if rng.random() < blur_probability:
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        image_out = cv2.GaussianBlur(image_out, (kernel_size, kernel_size), sigmaX=0)

    if rng.random() < noise_probability:
        noise = rng.normal(0.0, noise_std, size=image_out.shape).astype(np.float32)
        image_out = image_out + noise

    return np.clip(image_out, 0.0, 1.0).astype(np.float32), mask, weights


def apply_acquisition_noise_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Acquisition-noise strategy: light Gaussian blur and Gaussian noise."""
    return gaussian_blur_noise(image, mask, weights, rng=rng)


def apply_local_rotation_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local-rotation strategy using Random Local Rotation (RLR)."""
    return random_local_rotation(image, mask, weights, rng=rng)


STRATEGY_FUNCTIONS = {
    "paper": apply_paper_augmentations,
    "intensity": apply_intensity_augmentation,
    "acquisition_noise": apply_acquisition_noise_augmentation,
    "local_rotation": apply_local_rotation_augmentation,
}


def apply_augmentation_strategies(
    image: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    strategies: str | list[str] | tuple[str, ...] | None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply selected augmentation strategies in order."""
    rng = _get_rng(rng)
    selected = normalize_strategies(strategies)

    image = image.astype(np.float32, copy=False)
    weights = weights.astype(np.float32, copy=False)

    for strategy in selected:
        image, mask, weights = STRATEGY_FUNCTIONS[strategy](image, mask, weights, rng=rng)

    return (
        np.ascontiguousarray(np.clip(image, 0.0, 1.0), dtype=np.float32),
        np.ascontiguousarray(mask > 0, dtype=mask.dtype),
        np.ascontiguousarray(weights, dtype=np.float32),
    )
