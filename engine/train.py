"""One-epoch training loop for sparse edge segmentation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from losses.dynamic_loss import compute_dynamic_loss


def router_loss(keep_logits: Tensor, patch_targets: Tensor) -> Tensor:
    """Class-balanced Keep/Drop cross entropy for sparse positive patches."""
    targets = patch_targets.long()
    positive = targets.sum()
    negative = targets.numel() - positive
    class_weights = keep_logits.new_tensor([1.0, negative / positive.clamp_min(1)])
    return F.cross_entropy(keep_logits.reshape(-1, 2), targets.reshape(-1), weight=class_weights)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    segmentation_loss: nn.Module,
    device: torch.device,
    keep_ratio: float,
    scaler: torch.amp.GradScaler | None = None,
    ratio_weight: float = 2.0,
    router_weight: float = 1.0,
) -> dict[str, float]:
    """Optimize one epoch and return average scalar losses and keep ratio."""
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    sample_count = 0
    amp_enabled = scaler is not None and device.type == "cuda"

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images, labels=masks)
            pixel_loss = segmentation_loss(outputs["mask_logits"], masks)
            route_loss = router_loss(outputs["keep_logits"], outputs["patch_targets"])
            total_loss = compute_dynamic_loss(
                pixel_loss + router_weight * route_loss,
                outputs["keep_probs"],
                target_ratio=keep_ratio,
                lambda_ratio=ratio_weight,
            )

        if scaler is None:
            total_loss.backward()
            optimizer.step()
        else:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        totals["loss"] += total_loss.detach().item()
        totals["pixel_loss"] += pixel_loss.detach().item()
        totals["router_loss"] += route_loss.detach().item()
        totals["mean_keep_probability"] += outputs["keep_probs"].detach().mean().item()
        sample_count += 1

    if sample_count == 0:
        raise ValueError("training loader is empty")
    return {name: value / sample_count for name, value in totals.items()}
