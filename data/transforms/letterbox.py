"""Aspect-ratio-preserving resize, padding, and 256-aligned tile utilities."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

TILE_SIZE: Final[int] = 256
TARGET_SIZES: Final[tuple[tuple[int, int], ...]] = ((256, 256),)


def select_target_size(height: int, width: int, target_sizes: tuple[tuple[int, int], ...] = TARGET_SIZES) -> tuple[int, int]:
    """Choose the supported resolution with the closest aspect ratio."""
    source_ratio = width / height
    return min(target_sizes, key=lambda size: abs((size[1] / size[0]) - source_ratio))


def letterbox(
    image: Tensor,
    target_size: tuple[int, int],
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, dict[str, int | float]]:
    """Resize proportionally and zero-pad to ``target_size``."""
    target_height, target_width = target_size
    height, width = image.shape[-2:]
    if target_height < 1 or target_width < 1:
        raise ValueError("target_size must be positive")
    scale = min(target_height / height, target_width / width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    image = TF.resize(image, [resized_height, resized_width], interpolation=InterpolationMode.BILINEAR, antialias=True)
    if mask is not None:
        if mask.shape[-2:] != (height, width):
            raise ValueError("image and mask must have the same spatial size")
        mask = TF.resize(mask, [resized_height, resized_width], interpolation=InterpolationMode.NEAREST)
    pad_height, pad_width = target_height - resized_height, target_width - resized_width
    if pad_height < 0 or pad_width < 0:
        raise ValueError("target_size is smaller than the resized image")
    left, right = pad_width // 2, pad_width - pad_width // 2
    top, bottom = pad_height // 2, pad_height - pad_height // 2
    padding = [left, top, right, bottom]
    image = TF.pad(image, padding, fill=0.0)
    if mask is not None:
        mask = TF.pad(mask, padding, fill=0.0)
    metadata: dict[str, int | float] = {
        "original_height": height, "original_width": width,
        "resized_height": resized_height, "resized_width": resized_width,
        "pad_left": left, "pad_top": top, "pad_right": right, "pad_bottom": bottom,
        "scale": scale, "target_height": target_height, "target_width": target_width,
    }
    return image, mask, metadata


def letterbox_to_multiple(
    image: Tensor,
    multiple: int = TILE_SIZE,
    mask: Tensor | None = None,
    max_size: int | None = None,
) -> tuple[Tensor, Tensor | None, dict[str, int | float]]:
    """Letterbox while preserving aspect ratio and align both sides to a multiple.

    ``max_size`` optionally limits the resized long side before alignment. With
    no limit, the image keeps its native scale and only receives alignment pad.
    """
    if multiple < 1:
        raise ValueError("multiple must be positive")
    height, width = image.shape[-2:]
    if max_size is None:
        resized_height, resized_width = height, width
    else:
        scale = min(1.0, max_size / max(height, width))
        resized_height = max(1, round(height * scale))
        resized_width = max(1, round(width * scale))
    target = ((resized_height + multiple - 1) // multiple * multiple,
              (resized_width + multiple - 1) // multiple * multiple)
    if max_size is None and target == (height, width):
        return image, mask, {
            "original_height": height, "original_width": width,
            "resized_height": height, "resized_width": width,
            "pad_left": 0, "pad_top": 0, "pad_right": 0, "pad_bottom": 0,
            "scale": 1.0, "target_height": height, "target_width": width,
        }
    return letterbox(image, target, mask)


def split_tiles(
    image: Tensor,
    tile_size: int = TILE_SIZE,
) -> tuple[Tensor, list[dict[str, int]]]:
    """Split a padded CHW/BCHW tensor into non-overlapping tiles and coordinates."""
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim != 4:
        raise ValueError("image must have shape [C,H,W] or [B,C,H,W]")
    batch, channels, height, width = image.shape
    if height % tile_size or width % tile_size:
        raise ValueError("padded image dimensions must be divisible by tile_size")
    tiles = image.unfold(2, tile_size, tile_size).unfold(3, tile_size, tile_size)
    rows, cols = tiles.shape[2:4]
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(batch * rows * cols, channels, tile_size, tile_size)
    metadata = [{"batch_index": b, "tile_index": r * cols + c, "row": r, "col": c,
                 "top": r * tile_size, "left": c * tile_size,
                 "bottom": (r + 1) * tile_size, "right": (c + 1) * tile_size}
                for b in range(batch) for r in range(rows) for c in range(cols)]
    # Always preserve the flattened tile dimension, including the one-tile
    # case: [1,C,tile_size,tile_size]. Dataset and inference callers index N.
    return tiles, metadata


def merge_tiles(tiles: Tensor, metadata: list[dict[str, int]], canvas_size: tuple[int, int], tile_size: int = TILE_SIZE) -> Tensor:
    """Merge non-overlapping tile predictions into a BCHW canvas."""
    if tiles.ndim == 3:
        tiles = tiles.unsqueeze(1)
    if tiles.ndim != 4 or tiles.shape[-2:] != (tile_size, tile_size):
        raise ValueError("tiles must have shape [N,C,tile_size,tile_size]")
    if tiles.shape[0] != len(metadata):
        raise ValueError("tile count and metadata count differ")
    height, width = canvas_size
    canvas = tiles.new_zeros(1, tiles.shape[1], height, width)
    for tile, item in zip(tiles, metadata):
        canvas[0, :, item["top"]:item["bottom"], item["left"]:item["right"]] = tile
    return canvas


def unletterbox(mask: Tensor, metadata: dict[str, int | float]) -> Tensor:
    """Remove padding and resize a ``[1,H,W]`` or ``[B,1,H,W]`` output."""
    top, left = int(metadata["pad_top"]), int(metadata["pad_left"])
    resized_height, resized_width = int(metadata["resized_height"]), int(metadata["resized_width"])
    original_height, original_width = int(metadata["original_height"]), int(metadata["original_width"])
    mask = mask[..., top : top + resized_height, left : left + resized_width]
    return TF.resize(mask, [original_height, original_width], interpolation=InterpolationMode.BILINEAR, antialias=True)
