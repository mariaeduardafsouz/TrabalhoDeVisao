from __future__ import annotations

import numpy as np
import cv2


def random_local_rotation(
    image: np.ndarray,
    mask: np.ndarray,
    weight: np.ndarray | None = None,
    min_radius_ratio: float = 0.10,
    max_radius_ratio: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random Local Rotation (RLR): rotates a random circular region by a random angle.

    The same geometric transform is applied to image, mask, and weight map so that
    spatial correspondence is preserved.  The mask uses nearest-neighbour
    interpolation to keep labels binary; image and weights use bilinear.

    Args:
        image:            H×W float32 array, values in [0, 1].
        mask:             H×W int64 array, values in {0, 1}.
        weight:           H×W float32 weight map, or None (returns uniform weights).
        min_radius_ratio: minimum circle radius as a fraction of min(H, W).
        max_radius_ratio: maximum circle radius as a fraction of min(H, W).

    Returns:
        Tuple (aug_image, aug_mask, aug_weight), same dtypes and shapes as inputs.
    """
    if min_radius_ratio >= max_radius_ratio:
        raise ValueError(
            f"min_radius_ratio ({min_radius_ratio}) must be less than "
            f"max_radius_ratio ({max_radius_ratio})."
        )

    H, W = image.shape

    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)
    min_r = max(1, int(min(H, W) * min_radius_ratio))
    max_r = max(min_r + 1, int(min(H, W) * max_radius_ratio))
    r = np.random.randint(min_r, max_r + 1)
    theta_deg = float(np.random.uniform(0, 360))

    M = cv2.getRotationMatrix2D((float(cx), float(cy)), theta_deg, 1.0)

    rotated_image = cv2.warpAffine(
        image, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    rotated_mask = cv2.warpAffine(
        mask.astype(np.float32), M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(mask.dtype)

    yy, xx = np.ogrid[:H, :W]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

    aug_image = np.where(circle, rotated_image, image)
    aug_mask  = np.where(circle, rotated_mask,  mask)

    if weight is None:
        aug_weight = np.ones((H, W), dtype=np.float32)
    else:
        rotated_weight = cv2.warpAffine(
            weight, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        aug_weight = np.where(circle, rotated_weight, weight).astype(np.float32)

    return aug_image, aug_mask, aug_weight
