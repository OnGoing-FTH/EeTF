"""CNN backbone for rectangular image patches."""

from __future__ import annotations

from torch import Tensor, nn


class CNNBase(nn.Module):
    """Extract token features from image patches.

    Input shape:
        ``(B * N, 3, 64, 32)``.

    Output shape:
        ``(B, N, 8192)``, where ``N = h_patches * w_patches`` and
        ``8192 = 64 * 16 * 8``.
    """

    def __init__(self) -> None:
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        patches: Tensor,
        patch_grid: tuple[int, int],
    ) -> Tensor:
        """Return CNN features reshaped to ``(B, N, 8192)``."""
        h_patches, w_patches = patch_grid
        patch_count = h_patches * w_patches
        if patches.shape[0] % patch_count != 0:
            raise ValueError(
                "the patch batch dimension must be divisible by the patch count: "
                f"got {patches.shape[0]} and N={patch_count}"
            )

        features = self.stage1(patches)   # (B*N, 16, 64, 32)
        features = self.stage2(features)  # (B*N, 64, 32, 16)
        features = self.stage3(features)  # (B*N, 128, 32, 16)
        features = self.stage4(features)  # (B*N, 128, 16, 8)
        features = self.stage5(features)  # (B*N, 64, 16, 8)

        features = features.flatten(start_dim=1)  # (B*N, 8192)
        batch_size = patches.shape[0] // patch_count
        return features.reshape(batch_size, patch_count, -1)


if __name__ == "__main__":
    import torch

    patches = torch.randn(16, 3, 64, 32)
    model = CNNBase()

    x = model.stage1(patches)
    print(f"input shape:  {tuple(patches.shape)}")
    print(f"stage 1:      {tuple(x.shape)}")
    x = model.stage2(x)
    print(f"stage 2:      {tuple(x.shape)}")
    x = model.stage3(x)
    print(f"stage 3:      {tuple(x.shape)}")
    x = model.stage4(x)
    print(f"stage 4:      {tuple(x.shape)}")
    x = model.stage5(x)
    print(f"stage 5:      {tuple(x.shape)}")
    output = x.flatten(start_dim=1).reshape(16 // (2 * 4), 2 * 4, -1)
    print(f"output shape: {tuple(output.shape)}")
