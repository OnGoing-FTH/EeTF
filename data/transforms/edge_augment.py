"""Synchronized image/mask augmentation for sparse edge segmentation."""

from __future__ import annotations

import random
from typing import Final

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .letterbox import TARGET_SIZES, letterbox


class EdgeAugment:
    """Apply topology-preserving geometric and image-only photometric transforms.

    Inputs are one image ``[C, H, W]`` and one grayscale soft mask ``[1, H, W]``.
    The same flip, affine, and resize parameters are applied to both tensors.
    Image geometry uses bilinear interpolation; mask geometry always uses nearest
    interpolation so thin line topology and original 8-bit intensities survive.
    After augmentation, proportional letterbox resizing pads both tensors to one
    of the configured target sizes.
    """

    def __init__(
        self,
        target_sizes: tuple[tuple[int, int], ...] = TARGET_SIZES,
        rotation_degrees: float = 15.0,
        shear_degrees: float = 5.0,
        scale_range: tuple[float, float] = (0.8, 1.2),
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.5,
        clahe_prob: float = 0.4,
        sharpen_prob: float = 0.3,
        blur_prob: float = 0.2,
        noise_prob: float = 0.25,
        shadow_prob: float = 0.3,
    ) -> None:
        self.target_sizes = target_sizes
        self.rotation_degrees = rotation_degrees
        self.shear_degrees = shear_degrees
        self.scale_range = scale_range
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.clahe_prob = clahe_prob
        self.sharpen_prob = sharpen_prob
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.shadow_prob = shadow_prob

    @staticmethod
    def _clahe(image: Tensor) -> Tensor:
        """Apply CLAHE independently to image channels in OpenCV space."""
        device, dtype = image.device, image.dtype
        array = image.detach().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        output = np.stack([clahe.apply(channel) for channel in array], axis=0)
        return torch.from_numpy(output).to(device=device, dtype=dtype).div(255.0)

    @staticmethod
    def _motion_or_gaussian_blur(image: Tensor) -> Tensor:
        if random.random() < 0.5:
            return TF.gaussian_blur(image, kernel_size=[3, 3], sigma=[0.2, 0.8])
        kernel = image.new_zeros(1, 1, 3, 3)
        if random.random() < 0.5:
            kernel[0, 0, 1, :] = 1.0 / 3.0
        else:
            kernel[0, 0, :, 1] = 1.0 / 3.0
        return torch.nn.functional.conv2d(
            image.unsqueeze(0), kernel.expand(image.shape[0], -1, -1, -1),
            padding=1, groups=image.shape[0]
        ).squeeze(0)

    @staticmethod
    def _shadow(image: Tensor) -> Tensor:
        """Multiply the image by a smooth randomly-oriented illumination ramp."""
        _, height, width = image.shape
        y = torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype)
        x = torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        angle = random.uniform(0.0, 2.0 * np.pi)
        ramp = grid_x * np.cos(angle) + grid_y * np.sin(angle)
        strength = random.uniform(0.2, 0.5)
        illumination = 1.0 - strength * (ramp - ramp.min()) / (ramp.max() - ramp.min() + 1e-6)
        return image * illumination.unsqueeze(0)

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return synchronized augmented image/mask at one allowed resolution."""
        if image.ndim != 3 or mask.ndim != 3 or mask.shape[0] != 1:
            raise ValueError("image must be [C, H, W] and mask must be [1, H, W]")
        if image.shape[-2:] != mask.shape[-2:]:
            raise ValueError("image and mask must have the same spatial size")

        if random.random() < self.horizontal_flip_prob:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if random.random() < self.vertical_flip_prob:
            image, mask = TF.vflip(image), TF.vflip(mask)

        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        shear = random.uniform(-self.shear_degrees, self.shear_degrees)
        scale = random.uniform(*self.scale_range)
        image = TF.affine(image, angle, [0, 0], scale, [shear, 0.0], InterpolationMode.BILINEAR)
        mask = TF.affine(mask, angle, [0, 0], scale, [shear, 0.0], InterpolationMode.NEAREST)

        if random.random() < self.clahe_prob:
            image = self._clahe(image)
        if random.random() < self.sharpen_prob:
            image = TF.adjust_sharpness(image, sharpness_factor=random.uniform(1.2, 1.8))
        if random.random() < self.blur_prob:
            image = self._motion_or_gaussian_blur(image)
        if random.random() < self.noise_prob:
            image = image + torch.randn_like(image) * random.uniform(0.005, 0.02)
        if random.random() < self.shadow_prob:
            image = self._shadow(image)

        target_size = random.choice(self.target_sizes)
        image, mask, _ = letterbox(image, target_size, mask)
        return image.clamp(0, 1), mask.clamp(0, 1)
