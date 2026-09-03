"""Losses for sparse binary edge segmentation."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


class SparseEdgeLoss(nn.Module):
    """Combine foreground-weighted BCE and Dice loss for sparse masks.

    Args:
        bce_weight: Weight assigned to the BCE term.
        dice_weight: Weight assigned to the Dice term.
        positive_weight: Optional fixed foreground weight. If omitted, the
            foreground weight is computed per batch as ``negative / positive``.
        eps: Numerical-stability constant.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        positive_weight: float | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.positive_weight = positive_weight
        self.eps = eps

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Return weighted BCE plus Dice loss for ``[B, 1, H, W]`` tensors."""
        if self.positive_weight is None:
            positive = targets.sum()
            negative = targets.numel() - positive
            pos_weight = (negative / positive.clamp_min(self.eps)).detach()
        else:
            pos_weight = logits.new_tensor(self.positive_weight)

        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        probabilities = logits.sigmoid()
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * intersection + self.eps) / (denominator + self.eps)).mean()
        return self.bce_weight * bce + self.dice_weight * dice
