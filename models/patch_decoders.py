"""Pixel-level decoders for selected and unselected image patches."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SelectedPatchDecoder(nn.Module):
    """Produce detailed masks from selected raw image patches."""

    def __init__(self, embed_dim: int = 128, nheads: int = 8,
                 patch_height: int = 64, patch_width: int = 64) -> None:
        super().__init__()
        if patch_height < 2 or patch_width < 2 or patch_height % 2 or patch_width % 2:
            raise ValueError("patch dimensions must be even and at least 2")
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nheads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        self.context_projection = nn.Linear(768, embed_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, patches: Tensor, context: Tensor) -> Tensor:
        """Decode packed raw patches conditioned on fused context ``[M, 768]``."""
        if context.ndim not in (2, 3) or context.shape[-1] != 768:
            raise ValueError("context must be [M,768] or [B,N1,768]")
        if context.ndim == 3:
            context = context.reshape(-1, 768)
        if context.shape[0] != patches.shape[0]:
            raise ValueError("context and patches must have matching flattened counts")
        if patches.shape[-2:] != (self.patch_height, self.patch_width):
            raise ValueError("patches do not match decoder patch dimensions")
        features = self.encoder(patches)
        conditioning = self.context_projection(context).reshape(-1, features.shape[1], 1, 1)
        features = features + conditioning
        batch_size, channels, height, width = features.shape
        tokens = features.flatten(start_dim=2).transpose(1, 2)
        tokens = self.transformer(tokens)
        features = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        return self.decoder(features)


class RemainingPatchDecoder(nn.Module):
    """Recover coarse masks from the 768-dimensional unselected patch features."""

    def __init__(self, input_dim: int = 768, patch_height: int = 64, patch_width: int = 64) -> None:
        super().__init__()
        if patch_height < 2 or patch_width < 2 or patch_height % 2 or patch_width % 2:
            raise ValueError("patch dimensions must be even and at least 2")
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.to_low_resolution = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, patch_height * patch_width // 4),
            nn.GELU(),
        )
        self.super_resolution = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Return coarse pixel logits with shape ``[B, N2, 1, Hpatch, Wpatch]``."""
        if features.ndim == 3:
            batch_size, patch_count, _ = features.shape
            flat = features.reshape(batch_size * patch_count, -1)
            shape = (batch_size, patch_count)
        elif features.ndim == 2:
            flat = features
            shape = (features.shape[0], 1)
        else:
            raise ValueError("features must be [B,N,768] or [M,768]")
        low_resolution = self.to_low_resolution(flat).reshape(
            flat.shape[0], 1, self.patch_height // 2, self.patch_width // 2
        )
        masks = self.super_resolution(low_resolution)
        if features.ndim == 2:
            return masks
        return masks.reshape(shape[0], shape[1], 1, self.patch_height, self.patch_width)


def merge_patch_logits(
    selected_logits: Tensor,
    selected_indices: Tensor,
    remaining_logits: Tensor,
    remaining_indices: Tensor,
    patch_grid: tuple[int, int],
    patch_size: tuple[int, int] = (64, 64),
) -> Tensor:
    """Restore full-resolution mask logits for a single sample.

    Multi-batch packed routing is handled by ``EdgeDynamicViT``.
    """
    h_patches, w_patches = patch_grid
    patch_height, patch_width = patch_size
    patch_count = h_patches * w_patches
    full_patches = selected_logits.new_zeros(1, patch_count, 1, patch_height, patch_width)
    full_patches[:, selected_indices[0]] = selected_logits.unsqueeze(0)
    full_patches[:, remaining_indices[0]] = remaining_logits
    rows = full_patches.reshape(1, h_patches, w_patches, 1, patch_height, patch_width)
    rows = rows.permute(0, 3, 1, 4, 2, 5)
    return rows.reshape(1, 1, h_patches * patch_height, w_patches * patch_width)
