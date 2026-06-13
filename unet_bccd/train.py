from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from dataclasses import fields
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SegmentationTilesDataset
from .losses import weighted_cross_entropy
from .metrics import MetricTotals, binary_segmentation_metrics
from .model import UNet, count_parameters
from .transforms import AVAILABLE_AUGMENTATION_STRATEGIES, normalize_strategies
from .utils import ensure_dir, get_device, set_seed
from .visualization import plot_history


@dataclass
class TrainConfig:
    data_root: Path = Path("data/BCCD_processado")
    epochs: int = 50
    batch_size: int = 1
    num_workers: int = 0
    lr: float = 0.01
    momentum: float = 0.99
    weight_decay: float = 0.0
    class_weights: list[float] | None = None
    step_size: int = 10
    gamma: float = 0.5
    use_scheduler: bool = False
    augment: bool = True
    augmentation_strategies: list[str] = field(default_factory=lambda: ["paper"])
    early_stopping_patience: int = 0
    checkpoint_every: int = 5
    output_dir: Path = Path("runs/unet")
    final_model_name: str = "unet_final.pth"
    resume_checkpoint: Path | None = None
    seed: int = 42
    device: str | None = None


CONFIG_SECTIONS = {
    "data": {"data_root"},
    "training": {"epochs", "batch_size", "num_workers", "seed", "device", "augment", "early_stopping_patience", "resume_checkpoint"},
    "augmentation": {"strategies"},
    "optimizer": {"lr", "momentum", "weight_decay"},
    "loss": {"class_weights"},
    "scheduler": {"step_size", "gamma", "use_scheduler"},
    "output": {"output_dir", "checkpoint_every", "final_model_name"},
}


def build_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader | None]:
    strategies = normalize_strategies(config.augmentation_strategies)
    train_dataset = SegmentationTilesDataset(
        config.data_root / "train" / "original_tiles",
        config.data_root / "train" / "mask_tiles",
        config.data_root / "train" / "pesos",
        augment=config.augment,
        augmentation_strategies=strategies,
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


def train_one_epoch(
    model: UNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    class_weights: torch.Tensor | None = None,
) -> float:
    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
    for images, masks, weights in progress:
        images = images.to(device)
        masks = masks.to(device)
        weights = weights.to(device)

        logits = model(images)
        loss, _ = weighted_cross_entropy(
            logits,
            masks,
            weights,
            class_weights=class_weights,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    totals = MetricTotals()
    pred_cell_pixels = 0
    target_cell_pixels = 0
    total_pixels = 0

    for images, masks, weights in loader:
        images = images.to(device)
        masks = masks.to(device)
        weights = weights.to(device)

        logits = model(images)
        loss, target_crop = weighted_cross_entropy(
            logits,
            masks,
            weights,
            class_weights=class_weights,
        )
        predictions = torch.argmax(logits, dim=1)
        total_loss += loss.item()
        pred_cell_pixels += int((predictions == 1).sum().item())
        target_cell_pixels += int((target_crop == 1).sum().item())
        total_pixels += int(predictions.numel())

        for prediction, target in zip(predictions, target_crop):
            totals.update(binary_segmentation_metrics(prediction, target))

    averages = totals.averages()
    averages["loss"] = total_loss / max(len(loader), 1)
    averages["pred_cell_ratio"] = pred_cell_pixels / max(total_pixels, 1)
    averages["target_cell_ratio"] = target_cell_pixels / max(total_pixels, 1)
    return averages


def run_training(config: TrainConfig) -> dict[str, list[float]]:
    set_seed(config.seed)
    device = get_device(config.device)
    output_dir = ensure_dir(config.output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    strategies = normalize_strategies(config.augmentation_strategies)

    train_loader, val_loader = build_dataloaders(config)
    model = UNet(in_channels=1, out_channels=2).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    class_weight_tensor = build_class_weight_tensor(config.class_weights, device)
    scheduler = None
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )

    use_early_stopping = config.early_stopping_patience > 0 and val_loader is not None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1

    history: dict[str, list[float]] = {
        "train_loss": [],
        "learning_rates": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
        "val_accuracy": [],
        "val_pred_cell_ratio": [],
        "val_target_cell_ratio": [],
    }

    if config.resume_checkpoint is not None:
        checkpoint = load_checkpoint(
            config.resume_checkpoint,
            model,
            optimizer,
            scheduler,
            device,
        )
        resumed_epoch = int(checkpoint["epoch"])
        start_epoch = resumed_epoch + 1
        history = load_history_from_checkpoint(checkpoint, history)
        history = load_existing_history(output_dir, history)
        history = trim_history(history, resumed_epoch)
        if history["val_loss"]:
            best_val_loss = min(history["val_loss"])
            epochs_without_improvement = epochs_since_best(history["val_loss"])

    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Augment: {config.augment}")
    print(f"Augmentation strategies: {', '.join(strategies) if strategies else 'none'}")
    if class_weight_tensor is not None:
        print(f"Class weights: {config.class_weights}")
    print(f"Parameters: {count_parameters(model):,}")
    if use_early_stopping:
        print(f"Early stopping: patience={config.early_stopping_patience} epochs")
    elif config.early_stopping_patience > 0:
        print("Early stopping: disabled (no validation set)")
    if config.resume_checkpoint is not None:
        print(f"Resumed from checkpoint: {config.resume_checkpoint}")
        print(f"Starting at epoch: {start_epoch}/{config.epochs}")

    for epoch in range(start_epoch, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            config.epochs,
            class_weights=class_weight_tensor,
        )
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["learning_rates"].append(current_lr)

        message = f"Epoch {epoch} - train_loss={train_loss:.4f} - lr={current_lr:.6f}"
        if val_loader is not None:
            val_metrics = validate(
                model,
                val_loader,
                device,
                class_weights=class_weight_tensor,
            )
            history["val_loss"].append(val_metrics["loss"])
            history["val_iou"].append(val_metrics["iou"])
            history["val_dice"].append(val_metrics["dice"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_pred_cell_ratio"].append(val_metrics["pred_cell_ratio"])
            history["val_target_cell_ratio"].append(val_metrics["target_cell_ratio"])
            message += (
                f" - val_loss={val_metrics['loss']:.4f}"
                f" - val_iou={val_metrics['iou']:.4f}"
                f" - val_dice={val_metrics['dice']:.4f}"
                f" - val_pred_cell={val_metrics['pred_cell_ratio']:.4f}"
                f" - val_target_cell={val_metrics['target_cell_ratio']:.4f}"
            )
        print(message)
        write_history(history, output_dir / "history.json")

        # Early stopping check
        if use_early_stopping:
            current_val_loss = val_metrics["loss"]
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epochs_without_improvement = 0
                # Save best model
                best_model_path = output_dir / "unet_best.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved (val_loss={best_val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                print(
                    f"No improvement for {epochs_without_improvement}/"
                    f"{config.early_stopping_patience} epochs"
                )
                if epochs_without_improvement >= config.early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best val_loss={best_val_loss:.4f})"
                    )
                    break

        if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_dir / f"unet_epoch_{epoch}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                train_loss,
                config,
                history,
            )

    final_model_path = output_dir / config.final_model_name
    torch.save(model.state_dict(), final_model_path)

    history_path = output_dir / "history.json"
    write_history(history, history_path)
    (output_dir / "config.json").write_text(
        json.dumps(_json_ready(asdict(config)), indent=2),
        encoding="utf-8",
    )
    plot_history(history, output_dir / "training_history.png")

    print(f"Final model saved to: {final_model_path}")
    return history


def save_checkpoint(
    path: Path,
    model: UNet,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    loss: float,
    config: TrainConfig,
    history: dict[str, list[float]],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": _json_ready(asdict(config)),
        "history": history,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(
        checkpoint,
        path,
    )
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(
    path: Path,
    model: UNet,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" not in checkpoint or "optimizer_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint {path} must contain model_state_dict and optimizer_state_dict."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "epoch" not in checkpoint:
        raise ValueError(f"Checkpoint {path} must contain epoch.")
    return checkpoint


def load_history_from_checkpoint(
    checkpoint: dict,
    default_history: dict[str, list[float]],
) -> dict[str, list[float]]:
    loaded = checkpoint.get("history")
    if not isinstance(loaded, dict):
        return default_history

    for key in default_history:
        values = loaded.get(key, [])
        default_history[key] = list(values) if isinstance(values, list) else []
    return default_history


def load_existing_history(
    output_dir: Path,
    default_history: dict[str, list[float]],
) -> dict[str, list[float]]:
    history_path = output_dir / "history.json"
    if not history_path.exists():
        return default_history

    loaded = json.loads(history_path.read_text(encoding="utf-8"))
    for key in default_history:
        values = loaded.get(key, [])
        default_history[key] = list(values) if isinstance(values, list) else []
    return default_history


def write_history(history: dict[str, list[float]], path: Path) -> None:
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def trim_history(
    history: dict[str, list[float]],
    max_epochs: int,
) -> dict[str, list[float]]:
    return {
        key: values[:max_epochs] if isinstance(values, list) else []
        for key, values in history.items()
    }


def epochs_since_best(values: list[float]) -> int:
    if not values:
        return 0
    best_index = min(range(len(values)), key=values.__getitem__)
    return len(values) - best_index - 1


def validate_class_weights(class_weights: list[float] | None) -> list[float] | None:
    if class_weights is None:
        return None
    if len(class_weights) != 2:
        raise ValueError("class_weights must contain two values: [background, cell].")
    validated = [float(value) for value in class_weights]
    if any(value <= 0 for value in validated):
        raise ValueError("class_weights values must be positive.")
    return validated


def build_class_weight_tensor(
    class_weights: list[float] | None,
    device: torch.device,
) -> torch.Tensor | None:
    class_weights = validate_class_weights(class_weights)
    if class_weights is None:
        return None
    return torch.tensor(class_weights, dtype=torch.float32, device=device)


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def load_train_config(config_path: Path) -> dict:
    raw_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    valid_fields = {field.name for field in fields(TrainConfig)}
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
            if key == "augmentation":
                values["augmentation_strategies"] = value.get("strategies", [])
            else:
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
    if values["resume_checkpoint"] is not None:
        values["resume_checkpoint"] = Path(values["resume_checkpoint"])
    values["class_weights"] = validate_class_weights(values.get("class_weights"))
    values["augmentation_strategies"] = list(
        normalize_strategies(values.get("augmentation_strategies"))
    )
    if not values["augmentation_strategies"]:
        values["augment"] = False
    return TrainConfig(**values)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train U-Net on processed BCCD tiles.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--class-weights", nargs=2, type=float, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--use-scheduler", action="store_true", default=None)
    parser.add_argument("--augment", action="store_true", default=None)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument(
        "--augmentation-strategies",
        nargs="*",
        default=None,
        choices=AVAILABLE_AUGMENTATION_STRATEGIES,
    )
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--final-model-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    args_dict = vars(args)
    config_path = args_dict.pop("config")
    return build_train_config(config_path, args_dict)


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
