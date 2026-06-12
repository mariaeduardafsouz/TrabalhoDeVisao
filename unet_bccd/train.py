from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dc_fields
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SegmentationTilesDataset
from .losses import combined_loss, weighted_cross_entropy
from .metrics import MetricTotals, binary_segmentation_metrics
from .model import UNet, count_parameters
from .transforms import AugmentationParams
from .utils import ensure_dir, get_device, set_seed
from .visualization import plot_history


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class TrainConfig:
    # ---- Data ----
    data_root: Path = Path("data/BCCD_processado")

    # ---- Model ----
    model_version: str = "v1"       # "v1" (original) or "v2" (BatchNorm)
    use_padding: bool = True        # padding=1 for v2 (ignored by v1)

    # ---- Training ----
    epochs: int = 50
    batch_size: int = 1
    num_workers: int = 0
    lr: float = 0.01
    momentum: float = 0.99
    weight_decay: float = 0.0
    augment: bool = True
    early_stopping_patience: int = 0
    early_stopping_metric: str = "loss"   # "loss" or "iou"
    seed: int = 42
    device: str | None = None

    # ---- Optimizer ----
    optimizer_type: str = "sgd"     # "sgd" or "adamw"

    # ---- Scheduler ----
    use_scheduler: bool = False     # backward compat (enables StepLR)
    scheduler_type: str = "none"    # "none", "step", or "onecycle"
    step_size: int = 10
    gamma: float = 0.5

    # ---- Loss ----
    loss_type: str = "ce"           # "ce" or "combined"
    dice_weight: float = 0.5

    # ---- Augmentation ----
    augment_strategies: list[str] = field(
        default_factory=lambda: ["elastic", "rotate90", "flip"],
    )
    elastic_sigma: float = 10.0
    elastic_grid_size: int = 3
    brightness_range: float = 0.2
    contrast_range: float = 0.2
    brightness_contrast_prob: float = 0.5
    noise_sigma_max: float = 0.03
    noise_prob: float = 0.3

    # ---- Output ----
    checkpoint_every: int = 5
    output_dir: Path = Path("runs/unet")
    final_model_name: str = "unet_final.pth"


CONFIG_SECTIONS = {
    "data": {"data_root"},
    "model": {"model_version", "use_padding"},
    "training": {
        "epochs", "batch_size", "num_workers", "seed", "device",
        "augment", "early_stopping_patience", "early_stopping_metric",
    },
    "optimizer": {"lr", "momentum", "weight_decay", "optimizer_type"},
    "scheduler": {"step_size", "gamma", "use_scheduler", "scheduler_type"},
    "loss": {"loss_type", "dice_weight"},
    "augmentation": {
        "augment_strategies", "elastic_sigma", "elastic_grid_size",
        "brightness_range", "contrast_range", "brightness_contrast_prob",
        "noise_sigma_max", "noise_prob",
    },
    "output": {"output_dir", "checkpoint_every", "final_model_name"},
}


# ======================================================================
# Factory helpers
# ======================================================================

def build_model(config: TrainConfig) -> nn.Module:
    """Instantiate the model according to *config.model_version*."""
    if config.model_version == "v2":
        from .model_v2 import UNetV2

        padding = 1 if config.use_padding else 0
        return UNetV2(in_channels=1, out_channels=2, padding=padding)
    return UNet(in_channels=1, out_channels=2)


def build_optimizer(model: nn.Module, config: TrainConfig) -> optim.Optimizer:
    """Create the optimiser specified by *config.optimizer_type*."""
    if config.optimizer_type == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    # Default: SGD
    return optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


LossFn = type(weighted_cross_entropy)  # callable alias for type hints


def build_loss_fn(config: TrainConfig):
    """Return a loss callable with signature ``(logits, target, weights) -> (loss, target_crop)``."""
    if config.loss_type == "combined":
        dw = config.dice_weight

        def _combined(logits, target, weights):
            return combined_loss(logits, target, weights, dice_weight=dw)

        return _combined
    return weighted_cross_entropy


def build_scheduler(
    optimizer: optim.Optimizer,
    config: TrainConfig,
    steps_per_epoch: int,
) -> tuple[optim.lr_scheduler.LRScheduler | None, str]:
    """Create a learning-rate scheduler.

    Returns ``(scheduler, mode)`` where *mode* is ``"batch"``, ``"epoch"``
    or ``"none"``.  The caller must invoke ``scheduler.step()`` at the
    appropriate granularity.
    """
    # Determine effective scheduler type
    stype = config.scheduler_type
    if stype == "none" and config.use_scheduler:
        stype = "step"  # backward compatibility

    if stype == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
        return scheduler, "epoch"

    if stype == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
        )
        return scheduler, "batch"

    return None, "none"


def _build_aug_params(config: TrainConfig) -> AugmentationParams:
    return AugmentationParams(
        elastic_grid_size=config.elastic_grid_size,
        elastic_sigma=config.elastic_sigma,
        brightness_range=config.brightness_range,
        contrast_range=config.contrast_range,
        brightness_contrast_prob=config.brightness_contrast_prob,
        noise_sigma_max=config.noise_sigma_max,
        noise_prob=config.noise_prob,
    )


# ======================================================================
# Data
# ======================================================================

def build_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader | None]:
    aug_params = _build_aug_params(config) if config.augment else None
    aug_strategies = config.augment_strategies if config.augment else None

    train_dataset = SegmentationTilesDataset(
        config.data_root / "train" / "original_tiles",
        config.data_root / "train" / "mask_tiles",
        config.data_root / "train" / "pesos",
        augment=config.augment,
        augment_strategies=aug_strategies,
        augment_params=aug_params,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_image_dir = config.data_root / "val" / "original_tiles"
    val_mask_dir = config.data_root / "val" / "mask_tiles"
    val_loader = None
    if val_image_dir.exists() and any(val_image_dir.iterdir()):
        val_dataset = SegmentationTilesDataset(
            val_image_dir,
            val_mask_dir,
            augment=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


# ======================================================================
# Training loop
# ======================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn,
    device: torch.device,
    epoch: int,
    epochs: int,
    batch_scheduler=None,
) -> float:
    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
    for images, masks, weights in progress:
        images = images.to(device)
        masks = masks.to(device)
        weights = weights.to(device)

        logits = model(images)
        loss, _ = loss_fn(logits, masks, weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if batch_scheduler is not None:
            batch_scheduler.step()

        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    totals = MetricTotals()

    for images, masks, weights in loader:
        images = images.to(device)
        masks = masks.to(device)
        weights = weights.to(device)

        logits = model(images)
        loss, target_crop = loss_fn(logits, masks, weights)
        predictions = torch.argmax(logits, dim=1)
        total_loss += loss.item()

        for prediction, target in zip(predictions, target_crop):
            totals.update(binary_segmentation_metrics(prediction, target))

    averages = totals.averages()
    averages["loss"] = total_loss / max(len(loader), 1)
    return averages


# ======================================================================
# Main training driver
# ======================================================================

def run_training(config: TrainConfig) -> dict[str, list[float]]:
    set_seed(config.seed)
    device = get_device(config.device)
    output_dir = ensure_dir(config.output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")

    train_loader, val_loader = build_dataloaders(config)

    # ---- Build model, optimizer, scheduler, loss ----
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler, scheduler_mode = build_scheduler(
        optimizer, config, steps_per_epoch=len(train_loader),
    )
    loss_fn = build_loss_fn(config)

    batch_scheduler = scheduler if scheduler_mode == "batch" else None
    epoch_scheduler = scheduler if scheduler_mode == "epoch" else None

    # ---- Early stopping setup ----
    use_early_stopping = config.early_stopping_patience > 0 and val_loader is not None
    higher_is_better = config.early_stopping_metric == "iou"
    best_metric = float("-inf") if higher_is_better else float("inf")
    epochs_without_improvement = 0

    # ---- Print summary ----
    print(f"Device: {device}")
    print(f"Model: {config.model_version}"
          + (f" (padding={'same' if config.use_padding else 'valid'})"
             if config.model_version == "v2" else ""))
    print(f"Optimizer: {config.optimizer_type} | Loss: {config.loss_type}"
          + (f" (dice_weight={config.dice_weight})"
             if config.loss_type == "combined" else ""))
    if scheduler is not None:
        print(f"Scheduler: {scheduler_mode} ({type(scheduler).__name__})")
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Parameters: {count_parameters(model):,}")
    if config.augment:
        print(f"Augmentation: {config.augment_strategies}")
    if use_early_stopping:
        print(f"Early stopping: patience={config.early_stopping_patience} "
              f"metric={config.early_stopping_metric}")
    elif config.early_stopping_patience > 0:
        print("Early stopping: disabled (no validation set)")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "learning_rates": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
        "val_accuracy": [],
    }

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epoch,
            config.epochs,
            batch_scheduler=batch_scheduler,
        )
        if epoch_scheduler is not None:
            epoch_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["learning_rates"].append(current_lr)

        message = f"Epoch {epoch} - train_loss={train_loss:.4f} - lr={current_lr:.6f}"
        if val_loader is not None:
            val_metrics = validate(model, val_loader, loss_fn, device)
            history["val_loss"].append(val_metrics["loss"])
            history["val_iou"].append(val_metrics["iou"])
            history["val_dice"].append(val_metrics["dice"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            message += (
                f" - val_loss={val_metrics['loss']:.4f}"
                f" - val_iou={val_metrics['iou']:.4f}"
                f" - val_dice={val_metrics['dice']:.4f}"
            )
        print(message)

        # Early stopping check
        if use_early_stopping:
            current_val = val_metrics[config.early_stopping_metric]
            is_better = (
                current_val > best_metric if higher_is_better
                else current_val < best_metric
            )
            if is_better:
                best_metric = current_val
                epochs_without_improvement = 0
                # Save best model
                best_model_path = output_dir / "unet_best.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved "
                      f"({config.early_stopping_metric}={best_metric:.4f})")
            else:
                epochs_without_improvement += 1
                print(
                    f"No improvement for {epochs_without_improvement}/"
                    f"{config.early_stopping_patience} epochs"
                )
                if epochs_without_improvement >= config.early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best {config.early_stopping_metric}={best_metric:.4f})"
                    )
                    break

        if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_dir / f"unet_epoch_{epoch}.pth",
                model,
                optimizer,
                epoch,
                train_loss,
                config,
            )

    final_model_path = output_dir / config.final_model_name
    torch.save(model.state_dict(), final_model_path)

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(
        json.dumps(_json_ready(asdict(config)), indent=2),
        encoding="utf-8",
    )
    plot_history(history, output_dir / "training_history.png")

    print(f"Final model saved to: {final_model_path}")
    return history


# ======================================================================
# Checkpoint I/O
# ======================================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    config: TrainConfig,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": _json_ready(asdict(config)),
        },
        path,
    )
    print(f"Checkpoint saved to: {path}")


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


# ======================================================================
# Config loading
# ======================================================================

def load_train_config(config_path: Path) -> dict:
    raw_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    valid_fields = {f.name for f in dc_fields(TrainConfig)}
    values = {}

    for key, value in raw_config.items():
        if key in CONFIG_SECTIONS:
            if not isinstance(value, dict):
                raise ValueError(f"Config section [{key}] must be a table.")
            allowed_keys = CONFIG_SECTIONS[key]
            unknown = set(value) - allowed_keys
            if unknown:
                raise ValueError(
                    f"Unknown keys in [{key}]: {', '.join(sorted(unknown))}"
                )
            values.update(value)
            continue

        if key not in valid_fields:
            raise ValueError(f"Unknown train config key: {key}")
        values[key] = value

    return values


def build_train_config(config_path: Path | None, overrides: dict) -> TrainConfig:
    values = asdict(TrainConfig())
    if config_path is not None:
        values.update(load_train_config(config_path))

    values.update({key: value for key, value in overrides.items() if value is not None})
    values["data_root"] = Path(values["data_root"])
    values["output_dir"] = Path(values["output_dir"])
    return TrainConfig(**values)


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train U-Net on processed BCCD tiles.")
    parser.add_argument("--config", type=Path, default=None)

    # Data
    parser.add_argument("--data-root", type=Path, default=None)

    # Model
    parser.add_argument("--model-version", default=None, choices=["v1", "v2"])
    parser.add_argument("--use-padding", action="store_true", default=None)
    parser.add_argument("--no-padding", dest="use_padding", action="store_false")

    # Training
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--augment", action="store_true", default=None)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-metric", default=None, choices=["loss", "iou"])

    # Optimizer
    parser.add_argument("--optimizer-type", default=None, choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)

    # Scheduler
    parser.add_argument("--use-scheduler", action="store_true", default=None)
    parser.add_argument("--scheduler-type", default=None, choices=["none", "step", "onecycle"])
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)

    # Loss
    parser.add_argument("--loss-type", default=None, choices=["ce", "combined"])
    parser.add_argument("--dice-weight", type=float, default=None)

    # Output
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--final-model-name", default=None)

    args = parser.parse_args()
    args_dict = vars(args)
    config_path = args_dict.pop("config")
    return build_train_config(config_path, args_dict)


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
