import numpy as np

from unet_bccd.augmentation import random_local_rotation
from unet_bccd.transforms import (
    apply_augmentation_strategies,
    apply_augmentations,
    clahe_enhancement,
    elastic_deformation,
    gaussian_blur_noise,
    normalize_strategies,
    random_brightness_contrast,
    random_flip,
    random_rotate90,
)


def _make_sample(h: int = 64, w: int = 64):
    rng = np.random.default_rng(0)
    image = rng.random((h, w)).astype(np.float32)
    mask = (image > 0.5).astype(np.float32)
    weights = np.ones_like(image)
    return image, mask, weights


def test_elastic_deformation_preserves_shape():
    image, mask, weights = _make_sample()
    out_img, out_mask, out_weights = elastic_deformation(
        image, mask, weights, rng=np.random.default_rng(42),
    )
    assert out_img.shape == image.shape
    assert out_mask.shape == mask.shape
    assert out_weights.shape == weights.shape


def test_random_rotate90_preserves_shape():
    image, mask, weights = _make_sample()
    out_img, out_mask, out_weights = random_rotate90(
        image, mask, weights, rng=np.random.default_rng(42),
    )
    assert out_img.shape == image.shape
    assert out_mask.shape == mask.shape
    assert out_weights.shape == weights.shape


def test_random_flip_preserves_shape():
    image, mask, weights = _make_sample()
    out_img, out_mask, out_weights = random_flip(
        image, mask, weights, rng=np.random.default_rng(42),
    )
    assert out_img.shape == image.shape
    assert out_mask.shape == mask.shape
    assert out_weights.shape == weights.shape


def test_apply_augmentations_preserves_shape():
    image, mask, weights = _make_sample()
    out_img, out_mask, out_weights = apply_augmentations(
        image, mask, weights, rng=np.random.default_rng(42),
    )
    assert out_img.shape == image.shape
    assert out_mask.shape == mask.shape
    assert out_weights.shape == weights.shape


def test_apply_augmentations_deterministic_with_same_seed():
    image, mask, weights = _make_sample()
    r1 = apply_augmentations(image, mask, weights, rng=np.random.default_rng(99))
    r2 = apply_augmentations(image, mask, weights, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(r1[0], r2[0])
    np.testing.assert_array_equal(r1[1], r2[1])


def test_intensity_transforms_keep_mask_and_weights_aligned():
    image, mask, weights = _make_sample()
    image2, mask2, weights2 = random_brightness_contrast(
        image, mask, weights, rng=np.random.default_rng(7),
    )
    image3, mask3, weights3 = clahe_enhancement(image2, mask2, weights2)

    assert image3.shape == image.shape
    np.testing.assert_array_equal(mask3, mask)
    np.testing.assert_array_equal(weights3, weights)


def test_acquisition_noise_keeps_mask_and_weights_aligned():
    image, mask, weights = _make_sample()
    out_img, out_mask, out_weights = gaussian_blur_noise(
        image,
        mask,
        weights,
        blur_probability=1.0,
        noise_probability=1.0,
        rng=np.random.default_rng(8),
    )
    assert out_img.shape == image.shape
    np.testing.assert_array_equal(out_mask, mask)
    np.testing.assert_array_equal(out_weights, weights)


def test_random_local_rotation_preserves_shape_and_binary_mask():
    image, mask, weights = _make_sample()
    out_img, out_mask, out_weights = random_local_rotation(
        image,
        mask,
        weights,
        rng=np.random.default_rng(9),
    )
    assert out_img.shape == image.shape
    assert out_mask.shape == mask.shape
    assert out_weights.shape == weights.shape
    assert set(np.unique(out_mask)).issubset({0.0, 1.0})


def test_apply_selected_strategies_preserves_shape():
    image, mask, weights = _make_sample()
    out_img, out_mask, out_weights = apply_augmentation_strategies(
        image,
        mask,
        weights,
        ["paper", "local_rotation"],
        rng=np.random.default_rng(10),
    )
    assert out_img.shape == image.shape
    assert out_mask.shape == mask.shape
    assert out_weights.shape == weights.shape


def test_normalize_strategies_accepts_two_strategy_selection():
    assert normalize_strategies(["paper", "local_rotation"]) == (
        "paper",
        "local_rotation",
    )
