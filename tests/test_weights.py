import numpy as np

from unet_bccd.weights import ClassWeights, compute_unet_weight_map


def test_unet_weight_map_has_border_boost_between_instances():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:14, 6:10] = 1
    mask[10:14, 22:26] = 1

    weights = compute_unet_weight_map(mask, w0=10, sigma=5)

    assert weights.shape == mask.shape
    assert weights.dtype == np.float32
    assert weights[11, 16] > weights[0, 0]
    assert weights[11, 7] == 1.0


def test_unet_weight_map_applies_class_weights():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    weights = compute_unet_weight_map(
        mask,
        class_weights=ClassWeights(background=0.5, cell=2.0),
    )

    assert weights[0, 0] == 0.5
    assert weights[3, 3] == 2.0
