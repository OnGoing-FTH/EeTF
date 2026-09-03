"""Rectangular image patch extraction based on torch unfold."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ImagePatchingRect(nn.Module):
    """Split a BCHW image tensor into non-overlapping rectangular patches.

    Each input image of shape ``(C, H, W)`` is split into a grid of
    ``(H // patch_height) * (W // patch_width)`` patches. The batch and patch
    dimensions are flattened in the output.
    """

    def __init__(self, patch_height: int = 64, patch_width: int = 32) -> None:
        super().__init__()
        if patch_height <= 0 or patch_width <= 0:
            raise ValueError("patch_height and patch_width must be positive")

        self.patch_height = patch_height
        self.patch_width = patch_width

    def forward(self, images: Tensor) -> Tensor:
        """Return patches with shape ``(B * N, C, patch_height, patch_width)``."""
        batch_size, channels, height, width = images.shape
        if height % self.patch_height != 0 or width % self.patch_width != 0:
            raise ValueError(
                "image height and width must be divisible by the patch size: "
                f"got image {(height, width)} and patch "
                f"{(self.patch_height, self.patch_width)}"
            )


        h_patches = height // self.patch_height
        w_patches = width // self.patch_width
        # F.unfold returns (B, C * patch_height * patch_width, N).
        patches = F.unfold(
            images,
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
        )
        patch_count = patches.shape[-1]
        patches = patches.transpose(1, 2).reshape(
            batch_size * patch_count,
            channels,
            self.patch_height,
            self.patch_width,
        )
        return patches,(h_patches,w_patches)


if __name__ == "__main__":
    batch_size, channels, height, width = 2, 3, 128, 128
    images = torch.randn(batch_size, channels, height, width)
    patching = ImagePatchingRect(patch_height=64, patch_width=32)
    patches, (h_patches, w_patches) = patching(images) # (B*N, 3, 64, 32)


    print(f"input shape:  {tuple(images.shape)}")
    print(f"output shape: {tuple(patches.shape)}")
    print(f"patch count: {h_patches} x {w_patches}")
