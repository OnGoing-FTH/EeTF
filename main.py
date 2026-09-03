"""Main model that combines patch processing, feature fusion, and DynamicViT."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utils.label_utils import pool_patch_targets
from models.cnn_base import CNNBase
from models.dynamic_vit import RoPEAttention, TokenSelector
from models.feature_fusion import FeatureFusion
from models.mlp_base import MLPBase
from patching.patching import ImagePatchingRect
from utils.block_feature_extractor import BlockFeatureExtractor


class EdgeDynamicViT(nn.Module):
    """Patch-based edge model with DynamicViT token pruning.

    Args:
        keep_ratio: Fraction of patches retained by the token selector.
        stats_dim: Input dimension expected by the statistical MLP branch.

    Inputs:
        images: RGB images with shape ``[B, 3, H, W]``.
        labels: Optional binary labels with shape ``[B, 1, H, W]``.
        block_features: Optional precomputed statistics with shape ``[B*N, 28]``.

    Returns a dictionary with selected features and DynamicViT routing values.
    """

    def __init__(self, keep_ratio: float = 0.3, stats_dim: int = 28) -> None:
        super().__init__()
        if stats_dim < 4:
            raise ValueError("stats_dim must be at least 4")
        self.stats_dim = stats_dim
        self.patching = ImagePatchingRect(patch_height=64, patch_width=32)
        self.block_extractor = BlockFeatureExtractor()
        self.cnn_base = CNNBase()
        self.mlp_base = MLPBase(input_dim=stats_dim)
        self.feature_fusion = FeatureFusion()
        self.selector = TokenSelector(keep_ratio=keep_ratio)
        self.pre_attention_norm = nn.LayerNorm(768)
        self.attention = RoPEAttention(d_model=768, nheads=12)
        self.post_attention_norm = nn.LayerNorm(768)

    @staticmethod
    def _make_patch_coords(
        batch_size: int,
        patch_grid: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Generate normalized patch-center coordinates with shape ``[B, N, 2]``."""
        h_patches, w_patches = patch_grid
        y = (torch.arange(h_patches, device=device, dtype=dtype) + 0.5) / h_patches
        x = (torch.arange(w_patches, device=device, dtype=dtype) + 0.5) / w_patches
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((grid_x, grid_y), dim=-1).reshape(1, -1, 2)
        return coords.expand(batch_size, -1, -1)

    def _prepare_block_features(self, patches: Tensor, block_features: Optional[Tensor]) -> Tensor:
        """Use supplied 28D statistics or pad the current four statistics to stats_dim."""
        if block_features is not None:
            if block_features.shape != (patches.shape[0], self.stats_dim):
                raise ValueError(
                    "block_features must have shape "
                    f"({patches.shape[0]}, {self.stats_dim}), got {tuple(block_features.shape)}"
                )
            return block_features
        basic_features = self.block_extractor(patches)
        return F.pad(basic_features, (0, self.stats_dim - basic_features.shape[-1]))

    def forward(
        self,
        images: Tensor,
        labels: Optional[Tensor] = None,
        block_features: Optional[Tensor] = None,
    ) -> dict[str, Tensor | tuple[int, int]]:
        """Run the full image-to-selected-token processing flow."""
        patches, patch_grid = self.patching(images)
        batch_size = images.shape[0]
        coords = self._make_patch_coords(batch_size, patch_grid, images.device, images.dtype)

        cnn_features = self.cnn_base(patches, patch_grid)
        statistics = self._prepare_block_features(patches, block_features)
        mlp_features = self.mlp_base(statistics, patch_grid)
        fused_features = self.feature_fusion(cnn_features, mlp_features)

        selected_x, selected_coords, keep_logits, keep_probs, selected_indices = self.selector(
            fused_features, coords
        )
        selected_x = selected_x + self.attention(self.pre_attention_norm(selected_x), selected_coords)
        selected_x = self.post_attention_norm(selected_x)

        output: dict[str, Tensor | tuple[int, int]] = {
            "features": selected_x,
            "coords": selected_coords,
            "keep_logits": keep_logits,
            "keep_probs": keep_probs,
            "selected_indices": selected_indices,
            "patch_grid": patch_grid,
        }
        if labels is not None:
            patch_targets = pool_patch_targets(labels, patch_grid)
            selected_targets = patch_targets.gather(1, selected_indices)
            output["patch_targets"] = patch_targets
            output["selected_targets"] = selected_targets
        return output


if __name__ == "__main__":
    images = torch.randn(1, 3, 512, 256)
    labels = (torch.rand(1, 1, 512, 256) > 0.98).to(torch.float32)
    model = EdgeDynamicViT(keep_ratio=0.5)
    outputs = model(images, labels=labels)

    print(f"input shape:              {tuple(images.shape)}")
    print(f"patch grid:               {outputs['patch_grid']}")
    print(f"selected feature shape:   {tuple(outputs['features'].shape)}")
    print(f"selected coordinate shape: {tuple(outputs['coords'].shape)}")
    print({outputs['coords']})
    print(f"keep probability shape:   {tuple(outputs['keep_probs'].shape)}")
    print(f"patch target shape:       {tuple(outputs['patch_targets'].shape)}")
