"""Sparse Patch segmentation model with Patch-grid CNN interaction."""
from __future__ import annotations

from typing import Optional
import torch
from torch import Tensor, nn

from models.cnn_base import CNNBase
from models.dynamic_vit import TokenSelector
from models.feature_fusion import FeatureFusion
from models.mlp_base import MLPBase
from models.patch_decoders import RemainingPatchDecoder, SelectedPatchDecoder
from patching.patching import ImagePatchingRect
from utils.block_feature_extractor import BlockFeatureExtractor
from utils.label_utils import pool_patch_targets


class EdgeDynamicViT(nn.Module):
    """Dynamic Patch segmentation model supporting variable selected counts per batch."""

    def __init__(self, selection_threshold: float = 0.5, stats_dim: int = 28,
                 patch_height: int = 64, patch_width: int = 64) -> None:
        super().__init__()
        if stats_dim != 28:
            raise ValueError("stats_dim must be 28 (4 base + 24 neighborhood features)")
        self.stats_dim = stats_dim
        self.patch_size = (patch_height, patch_width)
        self.patching = ImagePatchingRect(patch_height, patch_width)
        self.block_extractor = BlockFeatureExtractor()
        self.cnn_base = CNNBase(output_channels=64)
        self.mlp_base = MLPBase(input_dim=stats_dim)
        self.feature_fusion = FeatureFusion(cnn_dim=64)
        self.selector = TokenSelector(selection_threshold=selection_threshold)
        self.training_stage = "finetune"
        self.selected_decoder = SelectedPatchDecoder(patch_height=patch_height, patch_width=patch_width)
        self.remaining_decoder = RemainingPatchDecoder(patch_height=patch_height, patch_width=patch_width)

    def routing_modules(self):
        return (self.block_extractor, self.cnn_base, self.mlp_base, self.feature_fusion, self.selector)

    def set_stage(self, stage: str) -> None:
        if stage not in {"routing", "segmentation", "finetune"}:
            raise ValueError(f"unknown stage: {stage}")
        self.training_stage = stage
        # Explicitly define trainable ownership for every stage.
        self.requires_grad_(False)
        if stage == "routing":
            for module in self.routing_modules():
                module.requires_grad_(True)
        elif stage == "segmentation":
            routing_ids = {id(module) for module in self.routing_modules()}
            for module in self.children():
                if id(module) not in routing_ids:
                    module.requires_grad_(True)
        else:
            self.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.training_stage == "segmentation":
            for module in self.routing_modules():
                module.eval()
        return self

    def _prepare_block_features(self, patches, block_features, batch_size, patch_grid):
        if block_features is not None:
            if block_features.shape != (patches.shape[0], self.stats_dim):
                raise ValueError(f"block_features must have shape ({patches.shape[0]}, {self.stats_dim})")
            return block_features
        return self.block_extractor(patches, batch_size=batch_size, grid_size=patch_grid)

    @staticmethod
    def _pack_by_indices(features, indices):
        parts = [features[b, idx] for b, idx in enumerate(indices) if idx.numel()]
        return torch.cat(parts, dim=0) if parts else features.new_empty((0, features.shape[-1]))

    @staticmethod
    def _gather_patches(patches, indices, batch_size, patch_count):
        c, h, w = patches.shape[1:]
        patches = patches.reshape(batch_size, patch_count, c, h, w)
        parts = [patches[b, idx] for b, idx in enumerate(indices) if idx.numel()]
        return torch.cat(parts, dim=0) if parts else patches.new_empty((0, c, h, w))

    @staticmethod
    def _merge_packed(selected_masks, selected_indices, remaining_masks, remaining_indices,
                      patch_grid, patch_size, batch_size):
        hp, wp = patch_grid
        ph, pw = patch_size
        n = hp * wp
        full = selected_masks.new_zeros(batch_size, n, 1, ph, pw)
        so = ro = 0
        for b in range(batch_size):
            ns, nr = selected_indices[b].numel(), remaining_indices[b].numel()
            if ns:
                full[b, selected_indices[b]] = selected_masks[so:so + ns]
                so += ns
            if nr:
                full[b, remaining_indices[b]] = remaining_masks[ro:ro + nr]
                ro += nr
        full = full.reshape(batch_size, hp, wp, 1, ph, pw).permute(0, 3, 1, 4, 2, 5)
        return full.reshape(batch_size, 1, hp * ph, wp * pw)

    def forward(self, images: Tensor, labels: Optional[Tensor] = None,
                block_features: Optional[Tensor] = None, routing_only: bool = False):
        if images.ndim != 4:
            raise ValueError("images must have shape [B,3,H,W]")
        batch_size = images.shape[0]
        patches, patch_grid = self.patching(images)
        patch_count = patch_grid[0] * patch_grid[1]
        cnn_maps = self.cnn_base(patches, patch_grid)
        statistics = self._prepare_block_features(patches, block_features, batch_size, patch_grid)
        mlp_features = self.mlp_base(statistics, patch_grid)
        fused = self.feature_fusion(cnn_maps, mlp_features)
        logits = self.selector.router(fused)
        probs = logits.softmax(-1)[..., 1]
        if routing_only:
            output = {'keep_logits': logits, 'keep_probs': probs, 'patch_grid': patch_grid}
            if labels is not None:
                output['patch_targets'] = pool_patch_targets(labels, patch_grid, self.patch_size)
            return output
        selected_features, _, _, selected_indices, remaining_indices = self.selector(fused)
        selected_patches = self._gather_patches(patches, selected_indices, batch_size, patch_count)
        remaining_features = self._pack_by_indices(fused, remaining_indices)
        if selected_patches.shape[0]:
            selected_masks = self.selected_decoder(selected_patches, selected_features)
        else:
            selected_masks = patches.new_empty(0, 1, *self.patch_size)
        if remaining_features.shape[0]:
            remaining_masks = self.remaining_decoder(remaining_features)
        else:
            remaining_masks = fused.new_empty(0, 1, *self.patch_size)
        full_mask = self._merge_packed(selected_masks, selected_indices, remaining_masks,
                                       remaining_indices, patch_grid, self.patch_size, batch_size)
        output = {'mask_logits': full_mask, 'selected_features': selected_features,
                  'remaining_features': remaining_features,
                  'keep_logits': logits, 'keep_probs': probs,
                  'selected_indices': selected_indices, 'remaining_indices': remaining_indices,
                  'patch_grid': patch_grid}
        if labels is not None:
            targets = pool_patch_targets(labels, patch_grid, self.patch_size)
            output['patch_targets'] = targets
            output['selected_targets'] = [targets[b, idx] for b, idx in enumerate(selected_indices)]
            output['remaining_targets'] = [targets[b, idx] for b, idx in enumerate(remaining_indices)]
        return output
