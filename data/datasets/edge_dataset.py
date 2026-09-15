"""Paired image and grayscale edge-mask dataset expanded into 256x256 tiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.transforms.edge_augment import EdgeAugment
from data.transforms.letterbox import TILE_SIZE, letterbox_to_multiple, split_tiles
from utils.label_utils import load_grayscale_mask

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class EdgeDataset(Dataset):
    """Split each original image into fixed-size tiles after image-level split."""

    def __init__(self, image_dir, mask_dir, training=False, samples=None,
                 tile_size=TILE_SIZE, max_size=None, target_sizes=None):
        self.image_dir, self.mask_dir = Path(image_dir), Path(mask_dir)
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError("image_dir and mask_dir must both exist")
        if samples is None:
            image_paths = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
            mask_paths = {p.stem: p for p in self.mask_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES}
            samples = []
            for image_path in image_paths:
                mask_path = mask_paths.pop(image_path.stem, None)
                if mask_path is None:
                    raise FileNotFoundError(f"no mask found for image: {image_path.name}")
                samples.append((image_path, mask_path))
            if mask_paths:
                raise FileNotFoundError("masks without matching images: " + ", ".join(p.name for p in mask_paths.values()))
        if not samples:
            raise FileNotFoundError(f"no supported images found in {self.image_dir}")
        self.samples = list(samples)
        self.training, self.tile_size, self.max_size = training, tile_size, max_size
        self.augment = EdgeAugment(tile_size=tile_size, max_size=max_size) if training else None
        self._tiles = self._index_tiles()

    @staticmethod
    def _load_image(path):
        with Image.open(path) as image:
            array = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).div(255.0)

    def _index_tiles(self):
        records = []
        for sample_index, (image_path, mask_path) in enumerate(self.samples):
            image = self._load_image(image_path)
            padded, _, meta = letterbox_to_multiple(image, self.tile_size, max_size=self.max_size)
            _, tile_meta = split_tiles(padded, self.tile_size)
            for item in tile_meta:
                records.append((sample_index, item, meta))
        return records

    def __len__(self):
        return len(self._tiles)

    def __getitem__(self, index):
        sample_index, tile_meta, letterbox_meta = self._tiles[index]
        image_path, mask_path = self.samples[sample_index]
        image = self._load_image(image_path)
        mask = load_grayscale_mask(mask_path)
        if image.shape[-2:] != mask.shape[-2:]:
            raise ValueError(f"image/mask resolution mismatch: {image_path.name} and {mask_path.name}")
        padded_image, padded_mask, _ = letterbox_to_multiple(image, self.tile_size, mask, max_size=self.max_size)
        image_tiles, _ = split_tiles(padded_image, self.tile_size)
        mask_tiles, _ = split_tiles(padded_mask, self.tile_size)
        tile_index = tile_meta["tile_index"]
        image, mask, item = image_tiles[tile_index], mask_tiles[tile_index], tile_meta
        if self.augment is not None:
            image, mask = self.augment(image, mask)
        return {"image": image, "mask": mask, "image_path": str(image_path), "mask_path": str(mask_path),
                "image_id": image_path.stem, "tile_index": item["tile_index"], "tile_row": item["row"],
                "tile_col": item["col"], "tile_top": item["top"], "tile_left": item["left"],
                "original_size": (int(letterbox_meta["original_height"]), int(letterbox_meta["original_width"])),
                "padded_size": (int(letterbox_meta["target_height"]), int(letterbox_meta["target_width"]))}
