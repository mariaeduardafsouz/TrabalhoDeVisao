from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MetricTotals:
    iou: float = 0.0
    dice: float = 0.0
    accuracy: float = 0.0
    count: int = 0

    def update(self, metrics: dict[str, float]) -> None:
        self.iou += metrics["iou"]
        self.dice += metrics["dice"]
        self.accuracy += metrics["accuracy"]
        self.count += 1

    def averages(self) -> dict[str, float]:
        if self.count == 0:
            return {"iou": 0.0, "dice": 0.0, "accuracy": 0.0}
        return {
            "iou": self.iou / self.count,
            "dice": self.dice / self.count,
            "accuracy": self.accuracy / self.count,
        }


def binary_segmentation_metrics(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    smooth: float = 1e-6,
) -> dict[str, float]:
    pred = pred_mask.flatten().float()
    target = true_mask.flatten().float()

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()

    iou = (tp + smooth) / (tp + fp + fn + smooth)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    accuracy = (tp + tn) / (tp + fp + fn + tn + smooth)

    return {
        "iou": float(iou.item()),
        "dice": float(dice.item()),
        "accuracy": float(accuracy.item()),
    }
