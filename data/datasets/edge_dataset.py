"""Paired image and grayscale edge-mask dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from data.transforms.edge_augment import EdgeAugment
from data.transforms.letterbox import TARGET_SIZES, letterbox, select_target_size
from utils.label_utils import load_grayscale_mask

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class EdgeDataset(Dataset[dict[str, Tensor | str]]):
    """Read image/mask pairs with identical filename stems.

    Training samples receive randomized synchronized augmentation. Validation
    samples only receive aspect-ratio-aware letterboxing, preserving the exact
    image/mask relationship. The model currently requires DataLoader batch size 1.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        training: bool = False,
        target_sizes: tuple[tuple[int, int], ...] = TARGET_SIZES,
        samples: list[tuple[Path, Path]] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError("image_dir and mask_dir must both exist")

        if samples is None:
            image_paths = sorted(path for path in self.image_dir.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)
            mask_paths = {path.stem: path for path in self.mask_dir.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES}
            samples = []
            for image_path in image_paths:
                mask_path = mask_paths.pop(image_path.stem, None)
                if mask_path is None:
                    raise FileNotFoundError(f"no mask found for image: {image_path.name}")
                samples.append((image_path, mask_path))
            if mask_paths:
                names = ", ".join(sorted(path.name for path in mask_paths.values())[:5])
                raise FileNotFoundError(f"masks without matching images: {names}")
        if not samples:
            raise FileNotFoundError(f"no supported images found in {self.image_dir}")
        self.samples = list(samples)

        self.training = training
        self.target_sizes = target_sizes
        self.augment = EdgeAugment(target_sizes=target_sizes) if training else None

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_image(path: Path) -> Tensor:
        with Image.open(path) as image:
            array = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).div(255.0)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        image_path, mask_path = self.samples[index]
        image = self._load_image(image_path)
        mask = load_grayscale_mask(mask_path)
        if image.shape[-2:] != mask.shape[-2:]:
            raise ValueError(f"image/mask resolution mismatch: {image_path.name} and {mask_path.name}")

        if self.augment is not None:
            image, mask = self.augment(image, mask)
        else:
            target_size = select_target_size(*image.shape[-2:], self.target_sizes)
            image, mask, _ = letterbox(image, target_size, mask)

        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
