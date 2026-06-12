from __future__ import annotations

import cv2
import numpy as np


def random_local_rotation(
    image: np.ndarray,
    mask: np.ndarray,
    weight: np.ndarray | None = None,
    min_radius_ratio: float = 0.10,
    max_radius_ratio: float = 0.35,
    max_angle_degrees: float = 360.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random Local Rotation (RLR): rotate a random circular region.

    The same geometric transform is applied to image, mask and weight map so
    their spatial correspondence is preserved. The mask uses nearest-neighbor
    interpolation to keep labels binary; image and weights use bilinear
    interpolation.
    """
    if rng is None:
        rng = np.random.default_rng()

    if min_radius_ratio >= max_radius_ratio:
        raise ValueError(
            f"min_radius_ratio ({min_radius_ratio}) must be less than "
            f"max_radius_ratio ({max_radius_ratio})."
        )

    height, width = image.shape
    cx = int(rng.integers(0, width))
    cy = int(rng.integers(0, height))
    min_radius = max(1, int(min(height, width) * min_radius_ratio))
    max_radius = max(min_radius + 1, int(min(height, width) * max_radius_ratio))
    radius = int(rng.integers(min_radius, max_radius + 1))
    angle = float(rng.uniform(-max_angle_degrees, max_angle_degrees))

    matrix = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, 1.0)

    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    rotated_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(mask.dtype)

    yy, xx = np.ogrid[:height, :width]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2

    aug_image = np.where(circle, rotated_image, image).astype(np.float32)
    aug_mask = np.where(circle, rotated_mask, mask).astype(mask.dtype)

    if weight is None:
        aug_weight = np.ones((height, width), dtype=np.float32)
    else:
        rotated_weight = cv2.warpAffine(
            weight,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        aug_weight = np.where(circle, rotated_weight, weight).astype(np.float32)

    return (
        np.ascontiguousarray(aug_image, dtype=np.float32),
        np.ascontiguousarray(aug_mask),
        np.ascontiguousarray(aug_weight, dtype=np.float32),
    )
