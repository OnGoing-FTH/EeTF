"""MLP projection for block statistical features."""

from __future__ import annotations

from torch import Tensor, nn


class MLPBase(nn.Module):
    """Project per-patch statistics to one 768-dimensional feature.

    Input shape:
        ``(B * N, 28)``.

    Output shape:
        ``(B, N, 768)``, where ``N = h_patches * w_patches``.
    """

    def __init__(self, input_dim: int = 28) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 768

        self.projection = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, self.output_dim),
        )

    def forward(
        self,
        features: Tensor,
        patch_grid: tuple[int, int],
    ) -> Tensor:
        """Return block features with shape ``(B, N, 768)``."""
        h_patches, w_patches = patch_grid
        patch_count = h_patches * w_patches
        batch_patch_count = features.shape[0]
        if batch_patch_count % patch_count != 0:
            raise ValueError(
                "the feature batch dimension must be divisible by the patch count: "
                f"got {batch_patch_count} and N={patch_count}"
            )

        projected = self.projection(features)
        batch_size = batch_patch_count // patch_count
        return projected.reshape(batch_size, patch_count, self.output_dim)


if __name__ == "__main__":
    import torch

    block_features = torch.randn(24, 28)
    model = MLPBase(input_dim=28)
    output = model(block_features, patch_grid=(2, 4))

    print(f"input shape:  {tuple(block_features.shape)}")
    print(f"output shape: {tuple(output.shape)}")
