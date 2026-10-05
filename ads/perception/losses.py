"""Losses for the multi-task model: Ultralytics' YOLO detection loss plus BCE + Dice per segmentation head."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from ultralytics.utils.loss import v8DetectionLoss

# Segmentation gains relative to the detection loss. Lanes are thin and rare, so they weigh more.
DRIVABLE_GAIN = 2.0
LANE_GAIN = 4.0


def bce_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Mean BCE plus soft Dice loss for a binary mask; target is 0/1 with the logits' shape."""
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = logits.sigmoid()
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + eps) / (denom + eps)
    return bce + dice.mean()


class MultiTaskLoss:
    def __init__(self, model) -> None:
        self.detection = v8DetectionLoss(model)

    def __call__(self, outputs: tuple, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        det_out, da_logits, ll_logits = outputs
        bs = da_logits.shape[0]
        det, det_items = self.detection(det_out, batch)  # already scaled by batch size
        da = bce_dice(da_logits, batch["drivable"].unsqueeze(1))
        ll = bce_dice(ll_logits, batch["lane"].unsqueeze(1))
        total = det.sum() + bs * (DRIVABLE_GAIN * da + LANE_GAIN * ll)
        items = {k: float(v) for k, v in det_items.items()}
        items.update(drivable_loss=float(da), lane_loss=float(ll))
        return total, items
