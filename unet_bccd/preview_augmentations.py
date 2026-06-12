from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dataset import SegmentationTilesDataset
from .transforms import AVAILABLE_AUGMENTATION_STRATEGIES, apply_augmentation_strategies
from .utils import ensure_dir


DEFAULT_STRATEGIES = list(AVAILABLE_AUGMENTATION_STRATEGIES)


def load_sample(
    data_root: Path,
    split: str,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = SegmentationTilesDataset(
        data_root / split / "original_tiles",
        data_root / split / "mask_tiles",
        data_root / split / "pesos" if split == "train" else None,
    )
    image, mask, weights = dataset[sample_index]
    return image.squeeze(0).numpy(), mask.numpy(), weights.numpy()


def save_comparison(
    output_path: Path,
    title: str,
    original_image: np.ndarray,
    original_mask: np.ndarray,
    augmented_image: np.ndarray,
    augmented_mask: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(title.replace("_", " "), fontsize=14)

    axes[0, 0].imshow(original_image, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(original_mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Original mask")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(augmented_image, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Augmented image")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(augmented_mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Augmented mask")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close(fig)


def generate_previews(
    data_root: Path,
    output_dir: Path,
    strategies: list[str],
    split: str = "train",
    sample_index: int = 0,
    seed: int = 42,
) -> None:
    output_dir = ensure_dir(output_dir)
    image, mask, weights = load_sample(data_root, split, sample_index)

    for name in strategies:
        rng = np.random.default_rng(seed)
        image_aug, mask_aug, _ = apply_augmentation_strategies(
            image,
            mask,
            weights,
            [name],
            rng=rng,
        )
        save_comparison(
            output_dir / f"{name}.png",
            name,
            image,
            mask,
            image_aug,
            mask_aug,
        )

    print(f"Saved augmentation previews to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate original/augmented image-mask comparison grids."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/BCCD_processado"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/augmentation_examples"))
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=DEFAULT_STRATEGIES,
        choices=AVAILABLE_AUGMENTATION_STRATEGIES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_previews(
        data_root=args.data_root,
        output_dir=args.output_dir,
        strategies=args.strategies,
        split=args.split,
        sample_index=args.sample_index,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
