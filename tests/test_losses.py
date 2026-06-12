import torch

from unet_bccd.losses import combined_loss, dice_loss, weighted_cross_entropy


def _make_tensors(batch: int = 2, classes: int = 2, h: int = 32, w: int = 32):
    """Create synthetic logits, targets, and weights."""
    logits = torch.randn(batch, classes, h, w)
    target = torch.randint(0, classes, (batch, h, w))
    weights = torch.ones(batch, h, w)
    return logits, target, weights


def test_weighted_cross_entropy_returns_scalar():
    logits, target, weights = _make_tensors()
    loss, target_crop = weighted_cross_entropy(logits, target, weights)
    assert loss.dim() == 0
    assert target_crop.shape == target.shape


def test_weighted_cross_entropy_auto_crops():
    """When target is larger than logits, it should be centre-cropped."""
    logits = torch.randn(1, 2, 28, 28)
    target = torch.randint(0, 2, (1, 32, 32))
    weights = torch.ones(1, 32, 32)
    loss, target_crop = weighted_cross_entropy(logits, target, weights)
    assert target_crop.shape[-2:] == (28, 28)
    assert loss.dim() == 0


def test_dice_loss_range():
    logits, target, _ = _make_tensors()
    loss = dice_loss(logits, target)
    assert 0.0 <= loss.item() <= 1.0


def test_dice_loss_perfect_prediction():
    """When prediction perfectly matches target, Dice loss should be near 0."""
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 10:20, 10:20] = 1
    # Create logits that strongly predict the correct class
    logits = torch.zeros(1, 2, 32, 32)
    logits[:, 0] = 10.0  # high background confidence
    logits[:, 1] = -10.0
    logits[:, 0, 10:20, 10:20] = -10.0  # high foreground where target=1
    logits[:, 1, 10:20, 10:20] = 10.0
    loss = dice_loss(logits, target)
    assert loss.item() < 0.05


def test_combined_loss_returns_scalar():
    logits, target, weights = _make_tensors()
    loss, target_crop = combined_loss(logits, target, weights, dice_weight=0.5)
    assert loss.dim() == 0
    assert target_crop.shape == target.shape


def test_combined_loss_dice_weight_zero_equals_ce():
    """With dice_weight=0 the combined loss should equal pure weighted CE."""
    logits, target, weights = _make_tensors()
    torch.manual_seed(42)
    loss_ce, _ = weighted_cross_entropy(logits, target, weights)
    torch.manual_seed(42)
    loss_combined, _ = combined_loss(logits, target, weights, dice_weight=0.0)
    assert torch.allclose(loss_ce, loss_combined, atol=1e-6)
