from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .utils import ensure_dir


def plot_history(history: dict[str, list[float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.get("train_loss", []), linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss")
    axes[0].grid(True, alpha=0.3)

    learning_rates = history.get("learning_rates", [])
    if learning_rates:
        axes[1].plot(learning_rates, linewidth=2, color="orange")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning rate")
        axes[1].set_title("Learning rate schedule")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_prediction_grid(samples: list[dict], output_path: str | Path) -> None:
    if not samples:
        return

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows), squeeze=False)
    fig.suptitle("U-Net inference results", fontsize=16)

    for row, sample in enumerate(samples):
        axes[row, 0].imshow(sample["image"], cmap="gray")
        axes[row, 0].set_title(f"Sample {row + 1} - image")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(sample["target"], cmap="jet", alpha=0.8)
        axes[row, 1].set_title(f"Sample {row + 1} - ground truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(sample["prediction"], cmap="jet", alpha=0.8)
        axes[row, 2].set_title(f"Sample {row + 1} - prediction")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
