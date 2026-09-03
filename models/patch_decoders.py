"""Pixel-level decoders for selected and unselected image patches."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SelectedPatchDecoder(nn.Module):
    """Produce detailed masks from selected raw image patches.

    Input: ``[N1, 3, 64, 32]``.
    Output: ``[N1, 1, 64, 32]`` logits.
    """

    def __init__(self, embed_dim: int = 128, nheads: int = 8) -> None:
        super().__init__()
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
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, patches: Tensor) -> Tensor:
        """Return detailed pixel logits with shape ``[N1, 1, 64, 32]``."""
        features = self.encoder(patches)  # [N1, 128, 32, 16]
        batch_size, channels, height, width = features.shape
        tokens = features.flatten(start_dim=2).transpose(1, 2)
        tokens = self.transformer(tokens)
        features = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        return self.decoder(features)


class RemainingPatchDecoder(nn.Module):
    """Recover coarse masks from the 768-dimensional unselected patch features.

    Input: ``[B, N2, 768]``.
    Output: ``[B, N2, 1, 64, 32]`` logits.
    """

    def __init__(self, input_dim: int = 768) -> None:
        super().__init__()
        self.to_low_resolution = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
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
        """Return coarse pixel logits with shape ``[B, N2, 1, 64, 32]``."""
        batch_size, patch_count, _ = features.shape
        low_resolution = self.to_low_resolution(features).reshape(
            batch_size * patch_count, 1, 32, 16
        )
        masks = self.super_resolution(low_resolution)
        return masks.reshape(batch_size, patch_count, 1, 64, 32)


def merge_patch_logits(
    selected_logits: Tensor,
    selected_indices: Tensor,
    remaining_logits: Tensor,
    remaining_indices: Tensor,
    patch_grid: tuple[int, int],
) -> Tensor:
    """Restore full-resolution mask logits from selected and remaining patches.

    Only batch size 1 is supported. Returns ``[1, 1, H, W]``.
    """
    h_patches, w_patches = patch_grid
    patch_count = h_patches * w_patches
    full_patches = selected_logits.new_zeros(1, patch_count, 1, 64, 32)
    full_patches[:, selected_indices[0]] = selected_logits.unsqueeze(0)
    full_patches[:, remaining_indices[0]] = remaining_logits
    rows = full_patches.reshape(1, h_patches, w_patches, 1, 64, 32)
    rows = rows.permute(0, 3, 1, 4, 2, 5)
    return rows.reshape(1, 1, h_patches * 64, w_patches * 32)
