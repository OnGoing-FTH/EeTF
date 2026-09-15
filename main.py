"""Hierarchical sparse segmentation with 256x256 macro patches and 32x32 sub-patches."""
from __future__ import annotations

from typing import Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.cnn_base import CNNBase
from models.dynamic_vit import TokenSelector
from models.feature_fusion import FeatureFusion
from models.mlp_base import MLPBase
from models.patch_decoders import SubPatchDecoder
from patching.patching import ImagePatchingRect
from utils.block_feature_extractor import BlockFeatureExtractor
from utils.label_utils import pool_patch_targets


class EdgeDynamicViT(nn.Module):
    """Route 256x256 macro patches, then decode all 32x32 sub-patches."""

    def __init__(self, selection_threshold: float = 0.5, stats_dim: int = 28,
                 macro_size: int = 256, sub_size: int = 32,
                 feature_dim: int = 256) -> None:
        super().__init__()
        if stats_dim != 28:
            raise ValueError("stats_dim must be 28")
        if macro_size != 8 * sub_size:
            raise ValueError("macro_size must equal 8 * sub_size")
        self.stats_dim, self.macro_size, self.sub_size = stats_dim, macro_size, sub_size
        self.sub_grid = (macro_size // sub_size, macro_size // sub_size)
        self.sub_count = self.sub_grid[0] * self.sub_grid[1]
        self.feature_dim = feature_dim
        self.macro_patching = ImagePatchingRect(macro_size, macro_size)
        self.sub_patching = ImagePatchingRect(sub_size, sub_size)
        self.block_extractor = BlockFeatureExtractor()
        self.cnn_base = CNNBase(output_channels=64, spatial_stride=4)
        self.mlp_base = MLPBase(input_dim=stats_dim, feature_dim=feature_dim)
        self.feature_fusion = FeatureFusion(cnn_dim=64, feature_dim=feature_dim)
        self.selector = TokenSelector(d_model=feature_dim, selection_threshold=selection_threshold)
        self.sub_decoder = SubPatchDecoder(context_dim=feature_dim, patch_size=sub_size)
        self.training_stage = "finetune"

    def routing_modules(self):
        return (self.block_extractor, self.cnn_base, self.mlp_base,
                self.feature_fusion, self.selector)

    def set_stage(self, stage: str) -> None:
        if stage not in {"routing", "segmentation", "finetune"}:
            raise ValueError(f"unknown stage: {stage}")
        self.training_stage = stage
        self.requires_grad_(False)
        if stage == "routing":
            for module in self.routing_modules():
                module.requires_grad_(True)
            self.sub_decoder.requires_grad_(True)  # explicit 32x32 block supervision
        elif stage == "segmentation":
            self.sub_decoder.requires_grad_(True)
        else:
            self.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.training_stage == "segmentation":
            for module in self.routing_modules():
                module.eval()
        return self

    @staticmethod
    def _pack_by_indices(features: Tensor, indices: list[Tensor]) -> Tensor:
        parts = [features[b, idx] for b, idx in enumerate(indices) if idx.numel()]
        return torch.cat(parts, dim=0) if parts else features.new_empty((0, features.shape[-1]))

    @staticmethod
    def _gather_patches(patches: Tensor, indices: list[Tensor], batch_size: int, count: int) -> Tensor:
        c, h, w = patches.shape[1:]
        patches = patches.reshape(batch_size, count, c, h, w)
        parts = [patches[b, idx] for b, idx in enumerate(indices) if idx.numel()]
        return torch.cat(parts, dim=0) if parts else patches.new_empty((0, c, h, w))

    @staticmethod
    def _merge_sub_masks(sub_masks: Tensor, selected_indices: list[Tensor],
                         macro_grid: tuple[int, int], sub_size: int, batch_size: int) -> Tensor:
        hp, wp = macro_grid
        sub_grid = 256 // sub_size
        full = sub_masks.new_zeros(batch_size, hp * wp, 1, sub_grid * sub_size, sub_grid * sub_size)
        offset = 0
        for b, indices in enumerate(selected_indices):
            for macro_index in indices.tolist():
                tiles = sub_masks[offset:offset + sub_grid * sub_grid]
                offset += sub_grid * sub_grid
                tiles = tiles.reshape(sub_grid, sub_grid, 1, sub_size, sub_size)
                full[b, macro_index] = tiles.permute(2, 0, 3, 1, 4).reshape(1, sub_grid * sub_size, sub_grid * sub_size)
        full = full.reshape(batch_size, hp, wp, 1, sub_grid * sub_size, sub_grid * sub_size)
        return full.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, 1, hp * sub_grid * sub_size, wp * sub_grid * sub_size)

    def _sub_targets(self, labels: Tensor, selected_indices: list[Tensor], macro_grid: tuple[int, int]) -> Tensor:
        macro_patches, _ = self.macro_patching(labels)
        selected = self._gather_patches(macro_patches, selected_indices, labels.shape[0], macro_grid[0] * macro_grid[1])
        if not selected.shape[0]:
            return labels.new_empty((0, 64))
        sub, _ = self.sub_patching(selected)
        return (F.max_pool2d(sub, self.sub_size, self.sub_size).flatten(1) > 0).reshape(-1, self.sub_count).to(labels.dtype)

    def forward(self, images: Tensor, labels: Optional[Tensor] = None,
                block_features: Optional[Tensor] = None, routing_only: bool = False):
        if images.ndim != 4:
            raise ValueError("images must have shape [B,3,H,W]")
        batch_size = images.shape[0]
        macro_patches, macro_grid = self.macro_patching(images)
        macro_count = macro_grid[0] * macro_grid[1]
        cnn_maps = self.cnn_base(macro_patches, macro_grid)
        if block_features is None:
            block_features = self.block_extractor(macro_patches, batch_size, macro_grid)
        mlp_features = self.mlp_base(block_features, macro_grid)
        fused = self.feature_fusion(cnn_maps, mlp_features)
        selected_features, macro_logits, macro_probs, selected_indices, remaining_indices = self.selector(fused)
        output = {'keep_logits': macro_logits, 'keep_probs': macro_probs,
                  'macro_logits': macro_logits, 'macro_probs': macro_probs,
                  'selected_features': selected_features, 'selected_indices': selected_indices,
                  'remaining_indices': remaining_indices, 'patch_grid': macro_grid,
                  'macro_grid': macro_grid}
        if labels is not None:
            macro_targets = pool_patch_targets(labels, macro_grid, (self.macro_size, self.macro_size))
            output['patch_targets'] = macro_targets
            output['macro_targets'] = macro_targets
            output['sub_targets'] = self._sub_targets(labels, selected_indices, macro_grid)
        if not selected_features.shape[0]:
            output['sub_block_logits'] = fused.new_empty((0, self.sub_count, 2))
            output['sub_block_probs'] = fused.new_empty((0, self.sub_count))
            output['sub_mask_logits'] = images.new_empty((0, 1, self.sub_size, self.sub_size))
            if not routing_only:
                output['mask_logits'] = images.new_zeros(batch_size, 1, images.shape[-2], images.shape[-1])
            return output
        selected_macro = self._gather_patches(macro_patches, selected_indices, batch_size, macro_count)
        sub_patches, _ = self.sub_patching(selected_macro)
        sub_context = selected_features[:, None, :].expand(-1, self.sub_count, -1).reshape(-1, self.feature_dim)
        sub_mask_logits, sub_block_logits = self.sub_decoder(sub_patches, sub_context)
        output['sub_block_logits'] = sub_block_logits.reshape(-1, self.sub_count, 2)
        output['sub_block_probs'] = sub_block_logits.softmax(-1)[..., 1].reshape(-1, self.sub_count)
        output['sub_mask_logits'] = sub_mask_logits
        if not routing_only:
            output['mask_logits'] = self._merge_sub_masks(
                sub_mask_logits, selected_indices, macro_grid, self.sub_size, batch_size)
        return output
