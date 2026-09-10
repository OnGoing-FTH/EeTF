"""Patch-level CNN features with lightweight cross-block interaction."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class PatchLevelCrossBlockFeatureInteraction(nn.Module):
    """Fuse neighboring Patch descriptors on their 2D Patch grid.

    Input and output use ``[B, N, C, H, W]``. The descriptor path performs
    spatial mixing only on ``[B, C, Hp, Wp]``; it never treats Patch count as
    channels. A channel gate controls how much cross-Patch context is added.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.descriptor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.patch_mixer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor, patch_grid: tuple[int, int]) -> Tensor:
        if features.ndim != 5:
            raise ValueError("features must have shape [B,N,C,H,W]")
        batch_size, patch_count, channels, height, width = features.shape
        if patch_grid[0] * patch_grid[1] != patch_count:
            raise ValueError(
                f"patch_grid {patch_grid} does not match N={patch_count}"
            )
        flat = features.reshape(batch_size * patch_count, channels, height, width)
        descriptors = self.descriptor(flat).reshape(batch_size, patch_count, channels)
        grid = descriptors.reshape(batch_size, patch_grid[0], patch_grid[1], channels)
        grid = grid.permute(0, 3, 1, 2).contiguous()
        mixed = self.patch_mixer(grid)
        cross = mixed.permute(0, 2, 3, 1).reshape(batch_size, patch_count, channels)
        cross = cross[..., None, None].expand(-1, -1, -1, height, width)
        gate = self.gate(flat).reshape(batch_size, patch_count, channels, height, width)
        return features + gate * cross


class CNNBase(nn.Module):
    """Extract full-resolution per-Patch CNN features and mix Patch neighbors.

    Args:
        output_channels: Number of channels in the returned Patch feature map.

    Input:
        Flattened patches ``[B*N, 3, H1, W1]`` and ``patch_grid=(Hp,Wp)``.

    Output:
        ``[B, N, output_channels, H1, W1]``.
    """

    def __init__(self, output_channels: int = 64) -> None:
        super().__init__()
        if output_channels < 1:
            raise ValueError("output_channels must be positive")
        hidden = max(16, output_channels // 2)
        self.output_channels = output_channels
        self.intra_patch = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )
        self.cross_block = PatchLevelCrossBlockFeatureInteraction(output_channels)

    def forward(self, patches: Tensor, patch_grid: tuple[int, int]) -> Tensor:
        if patches.ndim != 4 or patches.shape[1] != 3:
            raise ValueError("patches must have shape [B*N,3,H1,W1]")
        patch_count = patch_grid[0] * patch_grid[1]
        if patch_count < 1 or patches.shape[0] % patch_count != 0:
            raise ValueError(
                f"patch batch dimension {patches.shape[0]} is not divisible by N={patch_count}"
            )
        batch_size = patches.shape[0] // patch_count
        features = self.intra_patch(patches)
        _, channels, height, width = features.shape
        features = features.reshape(batch_size, patch_count, channels, height, width)
        return self.cross_block(features, patch_grid)


if __name__ == "__main__":
    model = CNNBase(output_channels=64)
    patches = torch.randn(2 * 16, 3, 64, 64)
    output = model(patches, (4, 4))
    print(tuple(output.shape))
