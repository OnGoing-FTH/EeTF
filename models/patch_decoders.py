"""Pixel-level decoders for hierarchical sparse image patches."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SubPatchDecoder(nn.Module):
    """Predict a mask for every retained 32x32 sub-patch.

    The decoder is intentionally independent per sub-patch: the first-level
    router supplies sparsity, while all 64 sub-patches of a retained 256x256
    macro patch are decoded and reconstructed.
    """

    def __init__(self, context_dim: int = 256, output_channels: int = 64,
                 patch_size: int = 32) -> None:
        super().__init__()
        if patch_size < 2 or patch_size % 2:
            raise ValueError("patch_size must be a positive even number")
        self.context_dim = context_dim
        self.patch_size = patch_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels), nn.GELU(),
        )
        self.context_projection = nn.Linear(context_dim, output_channels)
        self.block_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(output_channels, 2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(output_channels, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, patches: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        if patches.ndim != 4 or patches.shape[1:] != (3, self.patch_size, self.patch_size):
            raise ValueError(f"patches must have shape [M,3,{self.patch_size},{self.patch_size}]")
        if context.ndim != 2 or context.shape != (patches.shape[0], self.context_dim):
            raise ValueError(f"context must have shape [M,{self.context_dim}]")
        features = self.encoder(patches)
        features = features + self.context_projection(context).unsqueeze(-1).unsqueeze(-1)
        block_logits = self.block_head(features)
        pixel_logits = self.decoder(F.interpolate(features, size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False))
        return pixel_logits, block_logits


class RemainingPatchDecoder(nn.Module):
    """Compatibility decoder retained for external imports."""

    def __init__(self, input_dim: int = 256, patch_height: int = 64, patch_width: int = 64) -> None:
        super().__init__()
        self.patch_height, self.patch_width = patch_height, patch_width
        self.to_low_resolution = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, patch_height * patch_width // 4), nn.GELU()
        )
        self.super_resolution = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.GELU(), nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim == 3:
            b, n, d = features.shape
            flat = features.reshape(b * n, d)
        elif features.ndim == 2:
            flat, b, n = features, features.shape[0], 1
        else:
            raise ValueError("features must be [B,N,D] or [M,D]")
        low = self.to_low_resolution(flat).reshape(-1, 1, self.patch_height // 2, self.patch_width // 2)
        masks = self.super_resolution(low)
        return masks if features.ndim == 2 else masks.reshape(b, n, 1, self.patch_height, self.patch_width)


def merge_patch_logits(selected_logits: Tensor, selected_indices: Tensor, remaining_logits: Tensor,
                       remaining_indices: Tensor, patch_grid: tuple[int, int],
                       patch_size: tuple[int, int] = (64, 64)) -> Tensor:
    """Compatibility merge helper for the former one-level model."""
    hp, wp = patch_grid
    ph, pw = patch_size
    full = selected_logits.new_zeros(1, hp * wp, 1, ph, pw)
    if selected_indices.numel():
        full[:, selected_indices] = selected_logits.unsqueeze(0)
    if remaining_indices.numel():
        full[:, remaining_indices] = remaining_logits.unsqueeze(0)
    return full.reshape(1, hp, wp, 1, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(1, 1, hp * ph, wp * pw)
