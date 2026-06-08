from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, label


@dataclass(frozen=True)
class ClassWeights:
    background: float = 1.0
    cell: float = 1.0


def compute_binary_class_weights(mask_binary: np.ndarray) -> ClassWeights:
    """Return inverse-frequency class weights for background/cell pixels."""
    mask = mask_binary.astype(bool)
    total = mask.size
    cell_count = int(mask.sum())
    background_count = total - cell_count

    if cell_count == 0 or background_count == 0:
        return ClassWeights()

    background = total / (2.0 * background_count)
    cell = total / (2.0 * cell_count)
    return ClassWeights(background=float(background), cell=float(cell))


def compute_unet_weight_map(
    mask_binary: np.ndarray,
    w0: float = 10.0,
    sigma: float = 5.0,
    class_weights: ClassWeights | None = None,
) -> np.ndarray:
    """
    Compute the U-Net border weight map.

    This implements:
        wc(x) + w0 * exp(-((d1(x) + d2(x)) ** 2) / (2 * sigma ** 2))

    The border term is applied to background pixels only. Instance labels are
    inferred from connected components in the binary mask, so touching cells are
    treated as one instance unless the input mask already separates them.
    """
    mask = mask_binary.astype(bool)
    if class_weights is None:
        class_weights = ClassWeights()

    weight_map = np.where(mask, class_weights.cell, class_weights.background).astype(
        np.float32
    )

    instance_mask, num_instances = label(mask)
    if num_instances < 2:
        return weight_map

    distances = np.zeros((num_instances, *mask.shape), dtype=np.float32)
    for instance_index in range(1, num_instances + 1):
        current_instance = (instance_mask == instance_index).astype(np.uint8)
        distances[instance_index - 1] = distance_transform_edt(1 - current_instance)

    ordered_distances = np.sort(distances, axis=0)
    d1 = ordered_distances[0]
    d2 = ordered_distances[1]

    background = ~mask
    border_weight = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma**2))
    weight_map[background] += border_weight[background]
    return weight_map.astype(np.float32)
