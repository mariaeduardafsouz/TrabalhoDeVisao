from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import find_file_by_stem, list_image_files


class SegmentationTilesDataset(Dataset):
    """Dataset for image/mask tiles and optional per-pixel weights."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        weight_dir: str | Path | None = None,
        mask_threshold: int = 128,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.weight_dir = Path(weight_dir) if weight_dir is not None else None
        self.mask_threshold = mask_threshold
        self.samples = self._collect_samples()

        if not self.samples:
            raise ValueError(f"No samples found in {self.image_dir}")

    def _collect_samples(self) -> list[tuple[Path, Path, Path | None]]:
        samples: list[tuple[Path, Path, Path | None]] = []
        missing: list[str] = []

        for image_path in list_image_files(self.image_dir):
            mask_path = find_file_by_stem(self.mask_dir, image_path.stem)
            if mask_path is None:
                missing.append(f"mask:{image_path.name}")
                continue

            weight_path = None
            if self.weight_dir is not None:
                weight_path = self.weight_dir / f"{image_path.stem}.npy"
                if not weight_path.exists():
                    missing.append(f"weight:{image_path.stem}.npy")
                    continue

            samples.append((image_path, mask_path, weight_path))

        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Missing {len(missing)} files while building dataset. Examples: {preview}"
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, mask_path, weight_path = self.samples[idx]

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image_np = np.array(image, dtype=np.float32) / 255.0
        mask_np = (np.array(mask, dtype=np.uint8) > self.mask_threshold).astype(np.int64)

        if image_np.shape != mask_np.shape:
            raise ValueError(
                f"Image/mask size mismatch for {image_path.name}: "
                f"{image_np.shape} vs {mask_np.shape}"
            )

        if weight_path is None:
            weight_np = np.ones(mask_np.shape, dtype=np.float32)
        else:
            weight_np = np.load(weight_path).astype(np.float32)
            if weight_np.shape != mask_np.shape:
                raise ValueError(
                    f"Weight/mask size mismatch for {image_path.name}: "
                    f"{weight_np.shape} vs {mask_np.shape}"
                )

        image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_np).long()
        weight_tensor = torch.from_numpy(weight_np).float()
        return image_tensor, mask_tensor, weight_tensor
