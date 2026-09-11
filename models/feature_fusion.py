"""Fusion of CNN patch features and statistical MLP features."""

from __future__ import annotations

from torch import Tensor, nn


class FeatureFusion(nn.Module):
    """Project CNN features and fuse them with MLP features.

    Args:
        cnn_dim: CNN channel dimension after Patch-level spatial fusion.
        feature_dim: Shared fusion dimension, default ``256``.

    Inputs:
        cnn_features: ``(B, N, cnn_dim)`` or ``(B,N,cnn_dim,H,W)``.
        mlp_features: ``(B, N, feature_dim)``.

    Output:
        ``(B, N, feature_dim)``.
    """

    def __init__(self, cnn_dim: int = 64, feature_dim: int = 256) -> None:
        super().__init__()
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )

    def forward(self, cnn_features: Tensor, mlp_features: Tensor) -> Tensor:
        """Return fused features with shape ``(B, N, feature_dim)``."""
        if cnn_features.ndim == 5:
            cnn_features = cnn_features.mean(dim=(-2, -1))
        if cnn_features.ndim != 3 or mlp_features.ndim != 3:
            raise ValueError("cnn_features and mlp_features must be [B,N,C]")
        if cnn_features.shape[:2] != mlp_features.shape[:2]:
            raise ValueError("CNN and MLP features must have matching B,N")
        projected_cnn = self.cnn_projection(cnn_features)
        fused = projected_cnn + mlp_features
        return self.output_projection(fused)


if __name__ == "__main__":
    import torch

    batch_size, patch_count = 2, 8
    cnn_features = torch.randn(batch_size, patch_count, 64, 64, 64)
    mlp_features = torch.randn(batch_size, patch_count, 256)
    model = FeatureFusion(cnn_dim=64)
    output = model(cnn_features, mlp_features)

    print(f"CNN input shape:  {tuple(cnn_features.shape)}")
    print(f"MLP input shape:  {tuple(mlp_features.shape)}")
    print(f"output shape:     {tuple(output.shape)}")
