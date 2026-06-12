from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import SegmentationTilesDataset
from .metrics import MetricTotals, binary_segmentation_metrics
from .model import UNet
from .utils import center_crop_last_dims, ensure_dir, get_device
from .visualization import plot_prediction_grid


@dataclass
class EvalConfig:
    data_root: Path = Path("data/BCCD_processado")
    model_path: Path = Path("runs/unet/unet_final.pth")
    output_dir: Path = Path("runs/unet/eval")
    batch_size: int = 8
    num_workers: int = 0
    device: str | None = None
    num_visualizations: int = 2
    model_version: str = "v1"
    use_padding: bool = True


def _build_model(config: EvalConfig) -> torch.nn.Module:
    """Instantiate the correct model variant."""
    if config.model_version == "v2":
        from .model_v2 import UNetV2

        padding = 1 if config.use_padding else 0
        return UNetV2(in_channels=1, out_channels=2, padding=padding)
    return UNet(in_channels=1, out_channels=2)


@torch.no_grad()
def run_evaluation(config: EvalConfig) -> dict[str, float]:
    device = get_device(config.device)
    output_dir = ensure_dir(config.output_dir)

    dataset = SegmentationTilesDataset(
        config.data_root / "test" / "original_tiles",
        config.data_root / "test" / "mask_tiles",
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    model = _build_model(config).to(device)
    state_dict = torch.load(config.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    totals = MetricTotals()
    samples: list[dict] = []

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        predictions = torch.argmax(logits, dim=1)

        # Auto-crop only when output is smaller than input (valid convolutions)
        if masks.shape[-2:] != logits.shape[-2:]:
            target_crop = center_crop_last_dims(
                masks, logits.shape[-2], logits.shape[-1],
            )
            image_crop = center_crop_last_dims(
                images, logits.shape[-2], logits.shape[-1],
            )
        else:
            target_crop = masks
            image_crop = images

        for image, prediction, target in zip(image_crop, predictions, target_crop):
            totals.update(binary_segmentation_metrics(prediction, target))
            if len(samples) < config.num_visualizations:
                samples.append(
                    {
                        "image": image.cpu().squeeze(0).numpy(),
                        "target": target.cpu().numpy(),
                        "prediction": prediction.cpu().numpy(),
                    }
                )

    metrics = totals.averages()
    metrics["samples"] = float(totals.count)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if samples:
        plot_prediction_grid(samples, output_dir / "predictions.png")

    print(f"Evaluated samples: {totals.count}")
    print(f"Pixel accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean IoU: {metrics['iou']:.4f}")
    print(f"Mean Dice: {metrics['dice']:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    return metrics


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate U-Net on processed BCCD tiles.")
    parser.add_argument("--data-root", type=Path, default=EvalConfig.data_root)
    parser.add_argument("--model-path", type=Path, default=EvalConfig.model_path)
    parser.add_argument("--output-dir", type=Path, default=EvalConfig.output_dir)
    parser.add_argument("--batch-size", type=int, default=EvalConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=EvalConfig.num_workers)
    parser.add_argument("--device", default=EvalConfig.device)
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=EvalConfig.num_visualizations,
    )
    parser.add_argument(
        "--model-version", default=EvalConfig.model_version,
        choices=["v1", "v2"],
    )
    parser.add_argument("--use-padding", action="store_true", default=None)
    parser.add_argument("--no-padding", dest="use_padding", action="store_false")
    args = parser.parse_args()
    args_dict = vars(args)
    # Handle None for use_padding (keep default)
    if args_dict["use_padding"] is None:
        args_dict["use_padding"] = EvalConfig.use_padding
    return EvalConfig(**args_dict)


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
