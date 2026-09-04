"""Validation loop for sparse edge segmentation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from losses.dynamic_loss import compute_dynamic_loss
from metrics import compute_sparse_edge_metrics


def _router_loss(keep_logits: Tensor, patch_targets: Tensor) -> Tensor:
    """Class-balanced Keep/Drop cross entropy for patch routing."""
    targets = patch_targets.long()
    positive = targets.sum()
    negative = targets.numel() - positive
    class_weights = keep_logits.new_tensor([1.0, negative / positive.clamp_min(1)])
    return F.cross_entropy(keep_logits.reshape(-1, 2), targets.reshape(-1), weight=class_weights)


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    segmentation_loss: nn.Module,
    device: torch.device,
    keep_ratio: float,
    ratio_weight: float = 2.0,
    router_weight: float = 1.0,
) -> dict[str, float]:
    """Evaluate loss and foreground-focused segmentation metrics."""
    model.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    sample_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        outputs = model(images, labels=masks)
        pixel_loss = segmentation_loss(outputs["mask_logits"], masks)
        route_loss = _router_loss(outputs["keep_logits"], outputs["patch_targets"])
        total_loss = compute_dynamic_loss(
            pixel_loss + router_weight * route_loss,
            outputs["keep_probs"],
            target_ratio=keep_ratio,
            lambda_ratio=ratio_weight,
        )
        metrics = compute_sparse_edge_metrics(outputs["mask_logits"], masks)
        totals["loss"] += total_loss.item()
        totals["pixel_loss"] += pixel_loss.item()
        totals["router_loss"] += route_loss.item()
        for name, value in metrics.items():
            totals[name] += value.item()
        sample_count += 1

    if sample_count == 0:
        raise ValueError("validation loader is empty")
    return {name: value / sample_count for name, value in totals.items()}
