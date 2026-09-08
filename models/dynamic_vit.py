"""2D-RoPE attention and DynamicViT token selection."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate adjacent feature pairs: (x0, x1) -> (-x1, x0)."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack((-x[..., 1], x[..., 0]), dim=-1)
    return rotated.flatten(start_dim=-2)


def _rope_angles(coords: Tensor, dim: int, base: float) -> Tensor:
    """Build angles with shape [B, N, dim] for one coordinate axis."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=coords.device, dtype=coords.dtype) / dim)
    )
    return coords.unsqueeze(-1) * inv_freq


def apply_2d_rope(
    q: Tensor,
    k: Tensor,
    coords: Tensor,
    base: float = 10000.0,
) -> tuple[Tensor, Tensor]:
    """Apply continuous 2D rotary embeddings to Q and K.

    Args:
        q, k: ``[B, heads, N, head_dim]`` tensors, with ``head_dim=64``.
        coords: ``[B, N, 2]`` normalized coordinates in ``[0, 1]``.

    The first 32 dimensions rotate with x and the last 32 with y.
    """
    head_dim = q.shape[-1]
    if head_dim != 64 or k.shape != q.shape or coords.shape[:2] != (q.shape[0], q.shape[2]):
        raise ValueError("expected q/k=[B, heads, N, 64], coords=[B, N, 2]")

    coord_x = coords[..., 0]
    coord_y = coords[..., 1]
    angles_x = _rope_angles(coord_x, 32, base)
    angles_y = _rope_angles(coord_y, 32, base)
    angles = torch.cat((angles_x, angles_y), dim=-1).unsqueeze(1)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class RoPEAttention(nn.Module):
    """Multi-head self-attention with 2D rotary position encoding."""

    def __init__(self, d_model: int = 768, nheads: int = 12, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model != 768 or nheads != 12:
            raise ValueError("this implementation requires d_model=768 and nheads=12")
        self.d_model, self.nheads = d_model, nheads
        self.head_dim = d_model // nheads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        batch_size, token_count, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, token_count, 3, self.nheads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_2d_rope(q, k, coords)
        attention = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        output = (attention @ v).transpose(1, 2).reshape(batch_size, token_count, self.d_model)
        return self.out(output)


class TokenSelector(nn.Module):
    """Predict Keep/Drop logits and partition B=1 tokens by probability."""

    def __init__(self, d_model: int = 768, selection_threshold: float = 0.5, hidden_dim: int = 192) -> None:
        super().__init__()
        if not 0.0 <= selection_threshold <= 1.0:
            raise ValueError("selection_threshold must be in [0, 1]")
        self.selection_threshold = selection_threshold
        self.router = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))

    def forward(
        self,
        x: Tensor,
        coords: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return selected tokens and the complementary remaining indices."""
        logits = self.router(x)
        keep_probs = logits.softmax(dim=-1)[..., 1]
        if x.shape[0] != 1:
            raise ValueError("threshold selection requires B=1")
        selected_mask = keep_probs >= self.selection_threshold
        if valid_mask is not None:
            selected_mask = selected_mask & valid_mask.bool()
        indices = selected_mask[0].nonzero(as_tuple=True)[0].unsqueeze(0)
        remaining_indices = (~selected_mask[0]).nonzero(as_tuple=True)[0].unsqueeze(0)
        selected_x = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        selected_coords = coords.gather(1, indices.unsqueeze(-1).expand(-1, -1, 2))
        return selected_x, selected_coords, logits, keep_probs, indices, remaining_indices
