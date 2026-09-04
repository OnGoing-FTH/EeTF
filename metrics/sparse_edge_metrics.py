"""Foreground-focused metrics for sparse edge segmentation."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_sparse_edge_metrics(
    logits: Tensor,
    targets: Tensor,
    probability_threshold: float = 0.5,
    foreground_threshold: float = 0.0,
    background_weight: float = 0.05,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Compute metrics that prevent true-negative background dominance.

    Args:
        logits: Predicted segmentation logits with shape ``[B, 1, H, W]``.
        targets: Grayscale line targets in ``[0, 1]`` with the same shape.
        probability_threshold: Threshold applied after sigmoid for predictions.
        foreground_threshold: Target threshold identifying line support.
        background_weight: Relative contribution of true-negative pixels to
            ``weighted_accuracy``. Use 0 to exclude background completely.

    Returns:
        Scalar tensors for foreground precision, recall, F1, IoU, Dice,
        balanced accuracy, background-deweighted accuracy, and raw accuracy.
        ``foreground_f1`` is the recommended main validation metric.
    """
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    if not 0.0 <= background_weight <= 1.0:
        raise ValueError("background_weight must be in [0, 1]")

    predicted = logits.sigmoid() >= probability_threshold
    target = targets > foreground_threshold
    predicted, target = predicted.reshape(-1), target.reshape(-1)

    true_positive = (predicted & target).sum().to(torch.float32)
    false_positive = (predicted & ~target).sum().to(torch.float32)
    false_negative = (~predicted & target).sum().to(torch.float32)
    true_negative = (~predicted & ~target).sum().to(torch.float32)

    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive / (true_positive + false_negative + eps)
    foreground_f1 = 2.0 * precision * recall / (precision + recall + eps)
    foreground_iou = true_positive / (true_positive + false_positive + false_negative + eps)
    dice = 2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative + eps)
    specificity = true_negative / (true_negative + false_positive + eps)
    balanced_accuracy = 0.5 * (recall + specificity)
    weighted_accuracy = (true_positive + background_weight * true_negative) / (
        true_positive + false_negative + false_positive + background_weight * true_negative + eps
    )
    raw_accuracy = (true_positive + true_negative) / (
        true_positive + false_positive + false_negative + true_negative + eps
    )

    return {
        "foreground_precision": precision,
        "foreground_recall": recall,
        "foreground_f1": foreground_f1,
        "foreground_iou": foreground_iou,
        "dice": dice,
        "balanced_accuracy": balanced_accuracy,
        "weighted_accuracy": weighted_accuracy,
        "raw_accuracy": raw_accuracy,
    }
