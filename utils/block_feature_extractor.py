"""Block-level statistical and neighborhood feature extraction.

The extractor accepts RGB patches produced by ``patching.ImagePatching``.
When patch-grid metadata is supplied, it also models the relationships between
spatially adjacent patches from the same source image.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BlockFeatureExtractor(nn.Module):
    """Extract per-patch statistics and optional four-neighbor features.

    Base features, always returned, are:

    1. structure tensor eigenvalue difference;
    2. within-patch grayscale standard deviation;
    3. GLCM inverse difference moment (homogeneity);
    4. grayscale Shannon entropy (base 2).

    Supplying ``batch_size`` and ``grid_size=(grid_height, grid_width)`` adds
    24 topology-aware features. The complete column order then is:

    - columns 0--3: base features above;
    - columns 4--19: absolute base-feature differences to the
      ``(up, down, left, right)`` neighbors, four values per direction;
    - columns 20--27: boundary intensity discontinuity and boundary normal-
      gradient discontinuity for ``(up, down, left, right)``, two values per
      direction.

    Missing neighbors at image-grid borders are represented by zero values.
    Patches from different source images are never treated as neighbors.

    Args:
        num_levels: Number of quantized grayscale levels for GLCM and entropy.
        offset: Neighbor offset ``(vertical, horizontal)`` used by the GLCM.
        eps: Numerical-stability constant.
    """

    _DIRECTIONS = ("up", "down", "left", "right")

    def __init__(
        self,
        num_levels: int = 16,
        offset: tuple[int, int] = (0, 1),
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("num_levels must be at least 2")
        if offset == (0, 0):
            raise ValueError("offset cannot be (0, 0)")

        self.num_levels = num_levels
        self.offset = offset
        self.eps = eps
        self.register_buffer(
            "rgb_weights", torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "sobel_x", torch.tensor(
                [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
            ).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y", torch.tensor(
                [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
            ).unsqueeze(0),
            persistent=False,
        )

    def _to_grayscale(self, patches: Tensor) -> Tensor:
        return (patches * self.rgb_weights.to(dtype=patches.dtype)).sum(dim=1, keepdim=True)

    def _normalize_per_patch(self, grayscale: Tensor) -> Tensor:
        minimum = grayscale.amin(dim=(-2, -1), keepdim=True)
        maximum = grayscale.amax(dim=(-2, -1), keepdim=True)
        return (grayscale - minimum) / (maximum - minimum + self.eps)

    def _structure_tensor_difference(self, grayscale: Tensor) -> Tensor:
        grad_x = F.conv2d(grayscale, self.sobel_x.to(dtype=grayscale.dtype), padding=1)
        grad_y = F.conv2d(grayscale, self.sobel_y.to(dtype=grayscale.dtype), padding=1)
        j_xx = grad_x.square().mean(dim=(-2, -1))
        j_yy = grad_y.square().mean(dim=(-2, -1))
        j_xy = (grad_x * grad_y).mean(dim=(-2, -1))
        return torch.sqrt((j_xx - j_yy).square() + 4.0 * j_xy.square() + self.eps).squeeze(1)

    def _quantize(self, normalized: Tensor) -> Tensor:
        return normalized.mul(self.num_levels - 1).round().long().squeeze(1)

    def _gray_level_histogram(self, levels: Tensor) -> Tensor:
        one_hot = F.one_hot(levels.flatten(start_dim=1), num_classes=self.num_levels)
        histogram = one_hot.sum(dim=1).to(dtype=torch.float32)
        return histogram / histogram.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def _inverse_difference_moment(self, levels: Tensor) -> Tensor:
        delta_h, delta_w = self.offset
        height, width = levels.shape[-2:]
        source_h_start, source_h_end = max(0, -delta_h), min(height, height - delta_h)
        source_w_start, source_w_end = max(0, -delta_w), min(width, width - delta_w)
        target_h_start, target_h_end = max(0, delta_h), min(height, height + delta_h)
        target_w_start, target_w_end = max(0, delta_w), min(width, width + delta_w)

        source = levels[:, source_h_start:source_h_end, source_w_start:source_w_end]
        target = levels[:, target_h_start:target_h_end, target_w_start:target_w_end]
        source_hot = F.one_hot(source.flatten(start_dim=1), self.num_levels).float()
        target_hot = F.one_hot(target.flatten(start_dim=1), self.num_levels).float()
        glcm = source_hot.transpose(1, 2).bmm(target_hot)
        glcm = glcm / glcm.sum(dim=(1, 2), keepdim=True).clamp_min(self.eps)

        indices = torch.arange(self.num_levels, device=levels.device, dtype=glcm.dtype)
        weights = 1.0 / (1.0 + (indices[:, None] - indices[None, :]).square())
        return (glcm * weights).sum(dim=(1, 2))

    def _base_features(self, patches: Tensor) -> tuple[Tensor, Tensor]:
        grayscale = self._to_grayscale(patches)
        normalized = self._normalize_per_patch(grayscale)
        levels = self._quantize(normalized)
        histogram = self._gray_level_histogram(levels)
        entropy = -(histogram * histogram.clamp_min(self.eps).log2()).sum(dim=1)
        features = torch.stack(
            (
                self._structure_tensor_difference(grayscale),
                grayscale.std(dim=(-2, -1), unbiased=False).squeeze(1),
                self._inverse_difference_moment(levels),
                entropy.to(dtype=patches.dtype),
            ),
            dim=1,
        )
        return features, grayscale.squeeze(1)

    @staticmethod
    def _neighbor_grid(grid: Tensor, direction: str) -> tuple[Tensor, Tensor]:
        """Return neighbor values and a boolean mask for one grid direction."""
        neighbor = torch.zeros_like(grid)
        valid = torch.zeros(grid.shape[:3], dtype=torch.bool, device=grid.device)
        if direction == "up":
            neighbor[:, 1:] = grid[:, :-1]
            valid[:, 1:] = True
        elif direction == "down":
            neighbor[:, :-1] = grid[:, 1:]
            valid[:, :-1] = True
        elif direction == "left":
            neighbor[:, :, 1:] = grid[:, :, :-1]
            valid[:, :, 1:] = True
        elif direction == "right":
            neighbor[:, :, :-1] = grid[:, :, 1:]
            valid[:, :, :-1] = True
        else:
            raise ValueError(f"unsupported direction: {direction}")
        return neighbor, valid

    @staticmethod
    def _boundary_pair(gray_grid: Tensor, direction: str) -> tuple[Tensor, Tensor]:
        """Return per-patch intensity and normal-gradient boundary differences."""
        batch, grid_h, grid_w, height, width = gray_grid.shape
        intensity = gray_grid.new_zeros((batch, grid_h, grid_w))
        gradient = gray_grid.new_zeros((batch, grid_h, grid_w))

        if direction == "up":
            intensity[:, 1:] = (gray_grid[:, 1:, :, 0] - gray_grid[:, :-1, :, -1]).abs().mean(dim=-1)
            gradient[:, 1:] = (
                (gray_grid[:, 1:, :, 1] - gray_grid[:, 1:, :, 0])
                - (gray_grid[:, :-1, :, -1] - gray_grid[:, :-1, :, -2])
            ).abs().mean(dim=-1)
        elif direction == "down":
            intensity[:, :-1] = (gray_grid[:, :-1, :, -1] - gray_grid[:, 1:, :, 0]).abs().mean(dim=-1)
            gradient[:, :-1] = (
                (gray_grid[:, :-1, :, -1] - gray_grid[:, :-1, :, -2])
                - (gray_grid[:, 1:, :, 1] - gray_grid[:, 1:, :, 0])
            ).abs().mean(dim=-1)
        elif direction == "left":
            intensity[:, :, 1:] = (gray_grid[:, :, 1:, :, 0] - gray_grid[:, :, :-1, :, -1]).abs().mean(dim=-1)
            gradient[:, :, 1:] = (
                (gray_grid[:, :, 1:, :, 1] - gray_grid[:, :, 1:, :, 0])
                - (gray_grid[:, :, :-1, :, -1] - gray_grid[:, :, :-1, :, -2])
            ).abs().mean(dim=-1)
        elif direction == "right":
            intensity[:, :, :-1] = (gray_grid[:, :, :-1, :, -1] - gray_grid[:, :, 1:, :, 0]).abs().mean(dim=-1)
            gradient[:, :, :-1] = (
                (gray_grid[:, :, :-1, :, -1] - gray_grid[:, :, :-1, :, -2])
                - (gray_grid[:, :, 1:, :, 1] - gray_grid[:, :, 1:, :, 0])
            ).abs().mean(dim=-1)
        else:
            raise ValueError(f"unsupported direction: {direction}")
        return intensity, gradient

    def forward(
        self,
        patches: Tensor,
        batch_size: int | None = None,
        grid_size: tuple[int, int] | None = None,
    ) -> Tensor:
        """Return per-patch base or topology-aware statistical features.

        Args:
            patches: RGB patches with shape ``(batch_size * N, 3, patch_h, patch_w)``.
            batch_size: Original image batch size. Required with ``grid_size``.
            grid_size: Patch grid ``(grid_height, grid_width)``. Required with
                ``batch_size``. Its product must equal ``N``.
        """
        base_features, grayscale = self._base_features(patches)
        if batch_size is None and grid_size is None:
            return base_features
        if batch_size is None or grid_size is None:
            raise ValueError("batch_size and grid_size must be provided together")

        grid_h, grid_w = grid_size
        patch_count = grid_h * grid_w
        if patches.shape[0] != batch_size * patch_count:
            raise ValueError(
                "patch count does not match batch_size * grid_size: "
                f"got {patches.shape[0]}, expected {batch_size} * {grid_h} * {grid_w}"
            )

        feature_grid = base_features.reshape(batch_size, grid_h, grid_w, 4)
        gray_grid = grayscale.reshape(batch_size, grid_h, grid_w, *grayscale.shape[-2:])
        relation_features: list[Tensor] = []
        boundary_features: list[Tensor] = []

        for direction in self._DIRECTIONS:
            neighbor, valid = self._neighbor_grid(feature_grid, direction)
            relation_features.append((feature_grid - neighbor).abs() * valid.unsqueeze(-1))
            intensity, gradient = self._boundary_pair(gray_grid, direction)
            boundary_features.extend((intensity.unsqueeze(-1), gradient.unsqueeze(-1)))

        relations = torch.cat(relation_features, dim=-1)
        boundaries = torch.cat(boundary_features, dim=-1)
        return torch.cat((feature_grid, relations, boundaries), dim=-1).reshape(-1, 28)


if __name__ == "__main__":
    patches = torch.randn(120, 3, 64, 32)
    extractor = BlockFeatureExtractor()

    base = extractor(patches)
    contextual = extractor(patches, batch_size=2, grid_size=(6, 10))
    print(f"input patch shape:       {tuple(patches.shape)}")
    print(f"base feature shape:      {tuple(base.shape)}")
    print(f"contextual feature shape:{tuple(contextual.shape)}")
    print(f"contains NaN/Inf:        {not torch.isfinite(contextual).all().item()}")
