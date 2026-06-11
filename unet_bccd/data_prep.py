from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .utils import ensure_dir, find_file_by_stem, list_image_files
from .weights import compute_binary_class_weights, compute_unet_weight_map

BCCD_KAGGLE_SLUG = "jeetblahiri/bccd-dataset-with-mask"
DEFAULT_DATASET_NAME = "BCCD Dataset with mask"


@dataclass(frozen=True)
class ProcessedDatasetPaths:
    output_base: Path
    train_images: Path
    train_masks: Path
    train_weights: Path
    val_images: Path
    val_masks: Path
    test_images: Path
    test_masks: Path


def download_bccd_dataset(
    destination: str | Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
    kaggle_slug: str = BCCD_KAGGLE_SLUG,
) -> Path:
    """Download the Kaggle dataset and copy it into destination/dataset_name."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError("Install kagglehub to download the dataset.") from exc

    destination = ensure_dir(destination)
    downloaded_path = Path(kagglehub.dataset_download(kaggle_slug))
    dataset_root = _resolve_downloaded_root(downloaded_path, dataset_name)
    target_root = destination / dataset_name

    shutil.copytree(dataset_root, target_root, dirs_exist_ok=True)
    return target_root


def process_dataset(
    base_dir: str | Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
    output_dir: str | Path | None = None,
    tile_size: int = 572,
    w0: float = 10.0,
    sigma: float = 5.0,
    val_ratio: float = 0.2,
    seed: int = 42,
    overwrite: bool = False,
) -> ProcessedDatasetPaths:
    """Create fixed-size tiles and train weights from the raw BCCD dataset."""
    base_dir = Path(base_dir)
    raw_root = resolve_raw_dataset_root(base_dir, dataset_name)
    output_base = Path(output_dir) if output_dir else base_dir / "BCCD_processado"
    if output_base.exists():
        if overwrite:
            shutil.rmtree(output_base)
        else:
            print(
                f"Output directory already exists: {output_base}. "
                "Existing files may be overwritten, but stale files are kept. "
                "Use --overwrite to rebuild it from scratch."
            )

    train_original_dir = raw_root / "train" / "original"
    train_mask_dir = raw_root / "train" / "mask"
    test_original_dir = raw_root / "test" / "original"
    test_mask_dir = raw_root / "test" / "mask"

    train_pairs = collect_valid_pairs(train_original_dir, train_mask_dir)
    test_pairs = collect_valid_pairs(test_original_dir, test_mask_dir)
    train_pairs, val_pairs = split_pairs(train_pairs, val_ratio=val_ratio, seed=seed)

    paths = create_processed_dirs(output_base)
    total = len(train_pairs) + len(val_pairs) + len(test_pairs)

    print(f"Raw dataset: {raw_root}")
    print(f"Output: {output_base}")
    print(f"Train images: {len(train_pairs)}")
    print(f"Val images: {len(val_pairs)}")
    print(f"Test images: {len(test_pairs)}")

    with tqdm(total=total, desc="Processing images", unit="img") as progress:
        _process_split(
            train_pairs,
            paths.train_images,
            paths.train_masks,
            paths.train_weights,
            tile_size=tile_size,
            w0=w0,
            sigma=sigma,
            progress=progress,
        )
        _process_split(
            val_pairs,
            paths.val_images,
            paths.val_masks,
            None,
            tile_size=tile_size,
            w0=w0,
            sigma=sigma,
            progress=progress,
        )
        _process_split(
            test_pairs,
            paths.test_images,
            paths.test_masks,
            None,
            tile_size=tile_size,
            w0=w0,
            sigma=sigma,
            progress=progress,
        )

    print_summary(paths)
    return paths


def resolve_raw_dataset_root(base_dir: str | Path, dataset_name: str) -> Path:
    base_dir = Path(base_dir)
    candidates = [base_dir / dataset_name, base_dir]
    for candidate in candidates:
        if (candidate / "train" / "original").exists() and (
            candidate / "train" / "mask"
        ).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find raw BCCD folders under {base_dir}. "
        "Expected train/original and train/mask."
    )


def collect_valid_pairs(
    image_dir: str | Path,
    mask_dir: str | Path,
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    skipped: list[str] = []

    for image_path in list_image_files(image_dir):
        mask_path = find_file_by_stem(mask_dir, image_path.stem)
        if mask_path is None:
            skipped.append(f"{image_path.name}: missing mask")
            continue

        if validate_image_mask_pair(image_path, mask_path):
            pairs.append((image_path, mask_path))
        else:
            skipped.append(f"{image_path.name}: invalid pair")

    if skipped:
        print(f"Skipped {len(skipped)} invalid/missing pairs.")
        for item in skipped[:5]:
            print(f"  - {item}")

    if not pairs:
        raise ValueError(f"No valid image/mask pairs found in {image_dir}")

    return pairs


def validate_image_mask_pair(image_path: Path, mask_path: Path) -> bool:
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        return False
    if image.shape[:2] != mask.shape[:2]:
        return False
    if np.sum(mask > 127) == 0:
        return False
    return True


def split_pairs(
    pairs: list[tuple[Path, Path]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if val_ratio <= 0:
        return pairs, []
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)
    val_size = max(1, int(round(len(pairs) * val_ratio)))
    val_indices = set(indices[:val_size].tolist())

    train_pairs = [pair for idx, pair in enumerate(pairs) if idx not in val_indices]
    val_pairs = [pair for idx, pair in enumerate(pairs) if idx in val_indices]
    return train_pairs, val_pairs


def create_processed_dirs(output_base: Path) -> ProcessedDatasetPaths:
    paths = ProcessedDatasetPaths(
        output_base=output_base,
        train_images=output_base / "train" / "original_tiles",
        train_masks=output_base / "train" / "mask_tiles",
        train_weights=output_base / "train" / "pesos",
        val_images=output_base / "val" / "original_tiles",
        val_masks=output_base / "val" / "mask_tiles",
        test_images=output_base / "test" / "original_tiles",
        test_masks=output_base / "test" / "mask_tiles",
    )

    for directory in (
        paths.train_images,
        paths.train_masks,
        paths.train_weights,
        paths.val_images,
        paths.val_masks,
        paths.test_images,
        paths.test_masks,
    ):
        ensure_dir(directory)

    return paths


def compute_padding(height: int, width: int, tile_size: int) -> tuple[int, int]:
    pad_h = (tile_size - (height % tile_size)) % tile_size
    pad_w = (tile_size - (width % tile_size)) % tile_size
    return pad_h, pad_w


def _process_split(
    pairs: list[tuple[Path, Path]],
    output_image_dir: Path,
    output_mask_dir: Path,
    output_weight_dir: Path | None,
    tile_size: int,
    w0: float,
    sigma: float,
    progress: tqdm,
) -> None:
    for image_path, mask_path in pairs:
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            progress.update(1)
            continue

        height, width = image.shape[:2]
        pad_h, pad_w = compute_padding(height, width, tile_size)

        image_padded = cv2.copyMakeBorder(
            image,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_REFLECT,
        )
        mask_padded = cv2.copyMakeBorder(
            mask,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        padded_h, padded_w = image_padded.shape[:2]
        for y in range(0, padded_h, tile_size):
            for x in range(0, padded_w, tile_size):
                tile_image = image_padded[y : y + tile_size, x : x + tile_size]
                tile_mask = mask_padded[y : y + tile_size, x : x + tile_size]
                tile_mask_binary = (tile_mask > 127).astype(np.uint8)
                tile_id = f"{image_path.stem}_tile_{y}_{x}"

                cv2.imwrite(str(output_image_dir / f"{tile_id}.png"), tile_image)
                cv2.imwrite(str(output_mask_dir / f"{tile_id}.png"), tile_mask_binary * 255)

                if output_weight_dir is not None:
                    # Always use class-frequency weights (w_c) as per the
                    # original U-Net paper (Eq. 2)
                    class_weights = compute_binary_class_weights(
                        tile_mask_binary
                    )
                    weights = compute_unet_weight_map(
                        tile_mask_binary,
                        w0=w0,
                        sigma=sigma,
                        class_weights=class_weights,
                    )
                    np.save(output_weight_dir / f"{tile_id}.npy", weights)

        progress.update(1)


def _resolve_downloaded_root(downloaded_path: Path, dataset_name: str) -> Path:
    if (downloaded_path / "train" / "original").exists():
        return downloaded_path
    named = downloaded_path / dataset_name
    if (named / "train" / "original").exists():
        return named

    for child in downloaded_path.iterdir():
        if child.is_dir() and (child / "train" / "original").exists():
            return child

    raise FileNotFoundError(
        f"Could not find BCCD train/original inside downloaded path {downloaded_path}"
    )


def print_summary(paths: ProcessedDatasetPaths) -> None:
    print("\nProcessing complete.")
    print(f"Processed dataset: {paths.output_base}")
    print(f"  train images:  {paths.train_images}")
    print(f"  train masks:   {paths.train_masks}")
    print(f"  train weights: {paths.train_weights}")
    print(f"  val images:    {paths.val_images}")
    print(f"  val masks:     {paths.val_masks}")
    print(f"  test images:   {paths.test_images}")
    print(f"  test masks:    {paths.test_masks}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and process BCCD for U-Net.")
    parser.add_argument("--destination", type=Path, default=Path("data"))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--tile-size", type=int, default=572)
    parser.add_argument("--w0", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_download:
        raw_base = args.destination
    else:
        download_bccd_dataset(args.destination, dataset_name=args.dataset_name)
        raw_base = args.destination

    process_dataset(
        base_dir=raw_base,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        w0=args.w0,
        sigma=args.sigma,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
