"""Aspect-ratio-preserving resize and padding utilities."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

TARGET_SIZES: Final[tuple[tuple[int, int], ...]] = (
    (768, 768),
    (512, 1024),
    (1024, 512),
)


def select_target_size(height: int, width: int, target_sizes: tuple[tuple[int, int], ...] = TARGET_SIZES) -> tuple[int, int]:
    """Choose the supported resolution with the closest aspect ratio."""
    source_ratio = width / height
    return min(target_sizes, key=lambda size: abs((size[1] / size[0]) - source_ratio))


def letterbox(
    image: Tensor,
    target_size: tuple[int, int],
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, dict[str, int | float]]:
    """Resize proportionally and zero-pad to ``target_size``.

    Image interpolation is bilinear. A supplied grayscale mask uses nearest
    interpolation. Metadata is sufficient to remove padding during inference.
    """
    target_height, target_width = target_size
    height, width = image.shape[-2:]
    scale = min(target_height / height, target_width / width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    image = TF.resize(
        image,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    if mask is not None:
        if mask.shape[-2:] != (height, width):
            raise ValueError("image and mask must have the same spatial size")
        mask = TF.resize(mask, [resized_height, resized_width], interpolation=InterpolationMode.NEAREST)

    pad_height, pad_width = target_height - resized_height, target_width - resized_width
    left, right = pad_width // 2, pad_width - pad_width // 2
    top, bottom = pad_height // 2, pad_height - pad_height // 2
    padding = [left, top, right, bottom]
    image = TF.pad(image, padding, fill=0.0)
    if mask is not None:
        mask = TF.pad(mask, padding, fill=0.0)
    metadata: dict[str, int | float] = {
        "original_height": height,
        "original_width": width,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "pad_left": left,
        "pad_top": top,
        "scale": scale,
    }
    return image, mask, metadata


def unletterbox(mask: Tensor, metadata: dict[str, int | float]) -> Tensor:
    """Remove padding and resize a ``[1, H, W]`` output to original resolution."""
    top, left = int(metadata["pad_top"]), int(metadata["pad_left"])
    resized_height, resized_width = int(metadata["resized_height"]), int(metadata["resized_width"])
    original_height, original_width = int(metadata["original_height"]), int(metadata["original_width"])
    mask = mask[..., top : top + resized_height, left : left + resized_width]
    return TF.resize(mask, [original_height, original_width], interpolation=InterpolationMode.BILINEAR, antialias=True)
