"""Grayscale edge-mask loading and patch-level target generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F


def load_grayscale_mask(path: str | Path) -> Tensor:
    """Load an 8-bit single-channel mask as ``[1, H, W]`` in ``[0, 1]``.

    The source mask may contain anti-aliased line values such as 35..255;
    those values are intentionally preserved as a soft pixel target.
    """
    with Image.open(path) as image:
        mask = np.array(image.convert("L"), dtype=np.float32, copy=True)
    return torch.from_numpy(mask).div(255.0).unsqueeze(0)


def load_binary_png_label(path: str | Path, threshold: float = 0.0) -> Tensor:
    """Load the same PNG as a hard foreground mask in ``[1, H, W]``.

    ``threshold`` is expressed in normalized ``[0, 1]`` units. The default
    treats every non-zero line pixel as foreground for patch routing.
    """
    return (load_grayscale_mask(path) > threshold).to(torch.float32)


def pool_patch_targets(
    labels: Tensor,
    patch_grid: tuple[int, int],
    patch_size: tuple[int, int] = (64, 32),
    threshold: float = 0.0,
) -> Tensor:
    """Max-pool a mask and return hard patch occupancy targets ``[B, N]``.

    This is intended for Drop/Keep routing: a patch is positive when it has
    any pixel above ``threshold``. The input can be ``[B, 1, H, W]`` or
    ``[B, H, W]``; values may be normalized grayscale soft targets.
    """
    if labels.ndim == 3:
        labels = labels.unsqueeze(1)
    if labels.ndim != 4 or labels.shape[1] != 1:
        raise ValueError("labels must have shape [B, 1, H, W] or [B, H, W]")
    pooled = F.max_pool2d(labels, kernel_size=patch_size, stride=patch_size)
    if pooled.shape[-2:] != patch_grid:
        raise ValueError("label size and patch_grid do not match")
    return (pooled > threshold).flatten(start_dim=1).to(labels.dtype)


def pool_patch_soft_targets(
    labels: Tensor,
    patch_grid: tuple[int, int],
    patch_size: tuple[int, int] = (64, 32),
) -> Tensor:
    """Max-pool grayscale line intensity, returning soft targets ``[B, N]``."""
    if labels.ndim == 3:
        labels = labels.unsqueeze(1)
    pooled = F.max_pool2d(labels, kernel_size=patch_size, stride=patch_size)
    if pooled.shape[-2:] != patch_grid:
        raise ValueError("label size and patch_grid do not match")
    return pooled.flatten(start_dim=1)
