"""Dynamic Patch token routing."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class TokenSelector(nn.Module):
    """Predict Keep/Drop logits and partition every sample by probability."""

    def __init__(self, d_model: int = 768, selection_threshold: float = 0.5, hidden_dim: int = 192) -> None:
        super().__init__()
        if not 0.0 <= selection_threshold <= 1.0:
            raise ValueError("selection_threshold must be in [0, 1]")
        self.selection_threshold = selection_threshold
        self.router = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))

    def forward(
        self,
        x: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor], list[Tensor]]:
        """Return selected tokens and the complementary remaining indices."""
        logits = self.router(x)
        keep_probs = logits.softmax(dim=-1)[..., 1]
        selected_mask = keep_probs >= self.selection_threshold
        if valid_mask is not None:
            selected_mask = selected_mask & valid_mask.bool()
        # Packed outputs support different selected counts for each sample.
        selected_parts = []
        selected_indices = []
        remaining_indices = []
        for batch_index in range(x.shape[0]):
            selected = selected_mask[batch_index].nonzero(as_tuple=True)[0]
            remaining = (~selected_mask[batch_index]).nonzero(as_tuple=True)[0]
            selected_indices.append(selected)
            remaining_indices.append(remaining)
            selected_parts.append(x[batch_index, selected])
        selected_x = torch.cat(selected_parts, dim=0) if selected_parts else x.new_empty(0, x.shape[-1])
        return selected_x, logits, keep_probs, selected_indices, remaining_indices
