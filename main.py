"""Main sparse-patch segmentation model."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.cnn_base import CNNBase
from models.dynamic_vit import RoPEAttention, TokenSelector
from models.feature_fusion import FeatureFusion
from models.mlp_base import MLPBase
from models.patch_decoders import RemainingPatchDecoder, SelectedPatchDecoder
from patching.patching import ImagePatchingRect
from utils.block_feature_extractor import BlockFeatureExtractor
from utils.label_utils import pool_patch_targets


class EdgeDynamicViT(nn.Module):
    """B=1 sparse patch segmentation model with selected and remaining branches."""

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
        self.selected_decoder = SelectedPatchDecoder()
        self.remaining_decoder = RemainingPatchDecoder()

    @staticmethod
    def _make_patch_coords(
        batch_size: int,
        patch_grid: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return patch-center coordinates ``[B, N, 2]`` in (x, y) order."""
        h_patches, w_patches = patch_grid
        y = (torch.arange(h_patches, device=device, dtype=dtype) + 0.5) / h_patches
        x = (torch.arange(w_patches, device=device, dtype=dtype) + 0.5) / w_patches
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((grid_x, grid_y), dim=-1).reshape(1, -1, 2)
        return coords.expand(batch_size, -1, -1)

    def _prepare_block_features(self, patches: Tensor, block_features: Optional[Tensor]) -> Tensor:
        """Use supplied 28D statistics or pad the four current statistics."""
        if block_features is not None:
            if block_features.shape != (patches.shape[0], self.stats_dim):
                raise ValueError(
                    "block_features must have shape "
                    f"({patches.shape[0]}, {self.stats_dim}), got {tuple(block_features.shape)}"
                )
            return block_features
        basic_features = self.block_extractor(patches)
        return F.pad(basic_features, (0, self.stats_dim - basic_features.shape[-1]))

    @staticmethod
    def _gather_patches(patches: Tensor, indices: Tensor, batch_size: int, patch_count: int) -> Tensor:
        """Gather flattened patches using [B, K] indices and return [B*K, C, H, W]."""
        channels, height, width = patches.shape[1:]
        patches = patches.reshape(batch_size, patch_count, channels, height, width)
        selected = patches.gather(
            1, indices[..., None, None, None].expand(-1, -1, channels, height, width)
        )
        return selected.reshape(-1, channels, height, width)

    @staticmethod
    def _merge_patch_masks(
        selected_masks: Tensor,
        selected_indices: Tensor,
        remaining_masks: Tensor,
        remaining_indices: Tensor,
        patch_grid: tuple[int, int],
    ) -> Tensor:
        """Restore branch masks to ``[B, 1, H, W]`` using original patch indices."""
        batch_size = selected_indices.shape[0]
        h_patches, w_patches = patch_grid
        patch_count = h_patches * w_patches
        full = selected_masks.new_zeros(batch_size, patch_count, 1, 64, 32)
        selected_masks = selected_masks.reshape(batch_size, -1, 1, 64, 32)
        full.scatter_(1, selected_indices[..., None, None, None].expand_as(selected_masks), selected_masks)
        if remaining_masks.shape[1] > 0:
            full.scatter_(
                1,
                remaining_indices[..., None, None, None].expand_as(remaining_masks),
                remaining_masks,
            )
        full = full.reshape(batch_size, h_patches, w_patches, 1, 64, 32)
        full = full.permute(0, 3, 1, 4, 2, 5).contiguous()
        return full.reshape(batch_size, 1, h_patches * 64, w_patches * 32)

    def forward(
        self,
        images: Tensor,
        labels: Optional[Tensor] = None,
        block_features: Optional[Tensor] = None,
    ) -> dict[str, Tensor | tuple[int, int]]:
        """Run both patch branches and return the full-resolution mask logits."""
        if images.shape[0] != 1:
            raise ValueError("EdgeDynamicViT currently requires batch size B=1")

        patches, patch_grid = self.patching(images)
        batch_size, patch_count = images.shape[0], patch_grid[0] * patch_grid[1]
        coords = self._make_patch_coords(batch_size, patch_grid, images.device, images.dtype)

        cnn_features = self.cnn_base(patches, patch_grid)
        statistics = self._prepare_block_features(patches, block_features)
        mlp_features = self.mlp_base(statistics, patch_grid)
        fused_features = self.feature_fusion(cnn_features, mlp_features)

        selected_features, selected_coords, keep_logits, keep_probs, selected_indices, remaining_indices = self.selector(
            fused_features, coords
        )
        selected_features = selected_features + self.attention(
            self.pre_attention_norm(selected_features), selected_coords
        )
        selected_features = self.post_attention_norm(selected_features)

        selected_patches = self._gather_patches(
            patches, selected_indices, batch_size, patch_count
        )
        selected_masks = self.selected_decoder(selected_patches)

        remaining_features = fused_features.gather(
            1, remaining_indices[..., None].expand(-1, -1, fused_features.shape[-1])
        )
        remaining_masks = self.remaining_decoder(remaining_features)
        full_mask_logits = self._merge_patch_masks(
            selected_masks, selected_indices, remaining_masks, remaining_indices, patch_grid
        )

        output: dict[str, Tensor | tuple[int, int]] = {
            "mask_logits": full_mask_logits,
            "selected_features": selected_features,
            "selected_coords": selected_coords,
            "remaining_features": remaining_features,
            "keep_logits": keep_logits,
            "keep_probs": keep_probs,
            "selected_indices": selected_indices,
            "remaining_indices": remaining_indices,
            "patch_grid": patch_grid,
        }
        if labels is not None:
            patch_targets = pool_patch_targets(labels, patch_grid)
            output["patch_targets"] = patch_targets
            output["selected_targets"] = patch_targets.gather(1, selected_indices)
            output["remaining_targets"] = patch_targets.gather(1, remaining_indices)
        return output


if __name__ == "__main__":
    test_resolutions = ((768, 768), (512, 1024), (1024, 512))
    model = EdgeDynamicViT(keep_ratio=0.3).eval()

    with torch.inference_mode():
        for height, width in test_resolutions:
            images = torch.randn(1, 3, height, width)
            labels = (torch.rand(1, 1, height, width) > 0.99).float()
            outputs = model(images, labels=labels)
            print(f"input shape:              {tuple(images.shape)}")
            print(f"patch grid:               {outputs['patch_grid']}")
            print(f"mask logits shape:        {tuple(outputs['mask_logits'].shape)}")
            print(f"selected feature shape:   {tuple(outputs['selected_features'].shape)}")
            print(f"remaining feature shape:  {tuple(outputs['remaining_features'].shape)}")
            print(f"keep probability shape:   {tuple(outputs['keep_probs'].shape)}")
            print(f"selected index shape:     {tuple(outputs['selected_indices'].shape)}")
            print(f"remaining index shape:    {tuple(outputs['remaining_indices'].shape)}")
            print(f"patch target shape:       {tuple(outputs['patch_targets'].shape)}")