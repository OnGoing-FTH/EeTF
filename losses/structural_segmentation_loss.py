"""Differentiable structure losses for sparse line segmentation."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _soft_erode(image: Tensor) -> Tensor:
    horizontal = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    vertical = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    return torch.minimum(horizontal, vertical)


def _soft_dilate(image: Tensor) -> Tensor:
    return F.max_pool2d(image, 3, stride=1, padding=1)


def _soft_open(image: Tensor) -> Tensor:
    return _soft_dilate(_soft_erode(image))


def soft_skeletonize(image: Tensor, iterations: int = 10) -> Tensor:
    """Approximate a morphological skeleton while retaining gradients."""
    skeleton = F.relu(image - _soft_open(image))
    current = image
    for _ in range(iterations):
        current = _soft_erode(current)
        delta = F.relu(current - _soft_open(current))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0, 1)


def soft_cldice_loss(probabilities: Tensor, targets: Tensor, iterations: int = 10,
                     eps: float = 1e-6) -> Tensor:
    """Topology loss using binary target support and a soft prediction skeleton."""
    support = (targets > 0).to(probabilities.dtype)
    pred_skeleton = soft_skeletonize(probabilities, iterations)
    target_skeleton = soft_skeletonize(support, iterations)
    foreground = support.sum(dim=(1, 2, 3)) > 0
    topology_precision = ((pred_skeleton * support).sum(dim=(1, 2, 3)) + eps) / (
        pred_skeleton.sum(dim=(1, 2, 3)) + eps)
    topology_sensitivity = ((target_skeleton * probabilities).sum(dim=(1, 2, 3)) + eps) / (
        target_skeleton.sum(dim=(1, 2, 3)) + eps)
    cldice = (2 * topology_precision * topology_sensitivity + eps) / (
        topology_precision + topology_sensitivity + eps)
    loss = 1 - cldice
    # An empty target has no skeleton; suppress predicted foreground instead.
    loss = torch.where(foreground, loss, probabilities.mean(dim=(1, 2, 3)))
    return loss.mean()



def structural_segmentation_losses(logits: Tensor, targets: Tensor,
                                   cldice_iterations: int = 10) -> dict[str, Tensor]:
    probabilities = logits.float().sigmoid()
    return {'cldice': soft_cldice_loss(probabilities, targets.float(), cldice_iterations)}
