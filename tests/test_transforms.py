import numpy as np

from unet_bccd.transforms import (
    apply_augmentations,
    elastic_deformation,
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


def test_random_flip_preserves_shape():
    image, mask, weights = _make_sample()
    out_img, out_mask, out_weights = random_flip(
        image, mask, weights, rng=np.random.default_rng(42),
    )
    assert out_img.shape == image.shape
    assert out_mask.shape == mask.shape


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
