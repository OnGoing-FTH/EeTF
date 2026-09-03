"""Binary PNG label loading and patch-level target generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F


def load_binary_png_label(path: str | Path) -> Tensor:
    """Load palette/RGB/grayscale PNG and convert it to a [1, H, W] binary label.

    Palette indices or non-black colormap pixels are mapped to foreground 1.
    """
    with Image.open(path) as image:
        if image.mode == "P":
            label = torch.from_numpy(np.array(image, copy=True))
            return (label > 0).to(torch.float32).unsqueeze(0)
        rgb = torch.from_numpy(np.array(image.convert("RGB"), copy=True))
    return rgb.any(dim=-1).to(torch.float32).unsqueeze(0)


def pool_patch_targets(
    labels: Tensor,
    patch_grid: tuple[int, int],
    patch_size: tuple[int, int] = (64, 32),
) -> Tensor:
    """Use max pooling to form binary patch targets of shape ``[B, N]``.

    A patch target is 1 when its label patch contains at least one foreground
    pixel. ``labels`` has shape ``[B, 1, H, W]``.
    """
    h_patches, w_patches = patch_grid
    pooled = F.max_pool2d(labels, kernel_size=patch_size, stride=patch_size)
    if pooled.shape[-2:] != (h_patches, w_patches):
        raise ValueError("label size and patch_grid do not match")
    return pooled.flatten(start_dim=1)
