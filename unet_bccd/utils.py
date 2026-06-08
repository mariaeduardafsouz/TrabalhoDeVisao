from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_image_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_file_by_stem(
    directory: str | Path,
    stem: str,
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> Path | None:
    directory = Path(directory)
    for extension in extensions:
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def match_image_mask_pairs(
    image_dir: str | Path,
    mask_dir: str | Path,
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []

    for image_path in list_image_files(image_dir):
        mask_path = find_file_by_stem(mask_dir, image_path.stem)
        if mask_path is None:
            missing.append(image_path.name)
            continue
        pairs.append((image_path, mask_path))

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} masks in {mask_dir}. Examples: {preview}"
        )

    return pairs


def center_crop_last_dims(tensor, target_h: int, target_w: int):
    source_h, source_w = tensor.shape[-2:]
    if target_h > source_h or target_w > source_w:
        raise ValueError(
            f"Cannot crop tensor from {(source_h, source_w)} to {(target_h, target_w)}"
        )

    h_crop = (source_h - target_h) // 2
    w_crop = (source_w - target_w) // 2
    return tensor[..., h_crop : h_crop + target_h, w_crop : w_crop + target_w]


def get_device(device_name: str | None = None):
    import torch

    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
