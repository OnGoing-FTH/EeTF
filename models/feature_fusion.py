"""Fusion of CNN patch features and statistical MLP features."""

from __future__ import annotations

from torch import Tensor, nn


class FeatureFusion(nn.Module):
    """Project CNN features and fuse them with MLP features.

    Args:
        cnn_dim: Flattened CNN feature dimension, default ``64 * 16 * 8``.
        feature_dim: Shared fusion dimension, default ``768``.

    Inputs:
        cnn_features: ``(B, N, cnn_dim)``.
        mlp_features: ``(B, N, feature_dim)``.

    Output:
        ``(B, N, feature_dim)``.
    """

    def __init__(self, cnn_dim: int = 64 * 16 * 8, feature_dim: int = 768) -> None:
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
        projected_cnn = self.cnn_projection(cnn_features)
        fused = projected_cnn + mlp_features
        return self.output_projection(fused)


if __name__ == "__main__":
    import torch

    batch_size, patch_count = 2, 8
    cnn_features = torch.randn(batch_size, patch_count, 64 * 16 * 8)
    mlp_features = torch.randn(batch_size, patch_count, 768)
    model = FeatureFusion()
    output = model(cnn_features, mlp_features)

    print(f"CNN input shape:  {tuple(cnn_features.shape)}")
    print(f"MLP input shape:  {tuple(mlp_features.shape)}")
    print(f"output shape:     {tuple(output.shape)}")
