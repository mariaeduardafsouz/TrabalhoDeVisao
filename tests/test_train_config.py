from pathlib import Path

import pytest

from unet_bccd.train import (
    build_train_config,
    epochs_since_best,
    trim_history,
    validate_class_weights,
)


def test_build_train_config_accepts_resume_checkpoint():
    config = build_train_config(
        None,
        {"resume_checkpoint": "runs/unet/checkpoints/unet_epoch_15.pth"},
    )

    assert config.resume_checkpoint == Path("runs/unet/checkpoints/unet_epoch_15.pth")


def test_build_train_config_accepts_class_weights():
    config = build_train_config(None, {"class_weights": [1, 4]})

    assert config.class_weights == [1.0, 4.0]


def test_validate_class_weights_rejects_invalid_lengths():
    with pytest.raises(ValueError):
        validate_class_weights([1.0])


def test_epochs_since_best_counts_epochs_without_improvement():
    assert epochs_since_best([0.50, 0.42, 0.45, 0.46]) == 2


def test_trim_history_removes_epochs_after_checkpoint():
    history = {
        "train_loss": [0.5, 0.4, 0.3],
        "val_loss": [0.6, 0.5, 0.4],
    }

    assert trim_history(history, 2) == {
        "train_loss": [0.5, 0.4],
        "val_loss": [0.6, 0.5],
    }
