"""Loss utilities for DynamicViT token routing."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def compute_dynamic_loss(
    task_loss: Tensor,
    pred_keep_probs: Tensor,
    target_ratio: float = 0.3,
    lambda_ratio: float = 2.0,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Add a keep-ratio MSE penalty to a task loss.

    Args:
        task_loss: Scalar task loss tensor.
        pred_keep_probs: Keep probabilities with shape ``[B, N]``.
        target_ratio: Desired mean token retention rate.
        lambda_ratio: Weight of the retention-rate penalty.
        valid_mask: Optional valid token mask with shape ``[B, N]``.
    """
    if valid_mask is None:
        mean_keep_ratio = pred_keep_probs.mean()
    else:
        mask = valid_mask.to(dtype=pred_keep_probs.dtype)
        mean_keep_ratio = (pred_keep_probs * mask).sum() / mask.sum().clamp_min(1.0)
    ratio_loss = F.mse_loss(mean_keep_ratio, pred_keep_probs.new_tensor(target_ratio))
    return task_loss + lambda_ratio * ratio_loss
