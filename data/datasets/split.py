"""Deterministic train/validation split for paired edge samples."""

from __future__ import annotations

import random
from pathlib import Path

from .edge_dataset import EdgeDataset


def split_edge_dataset(
    image_dir: str | Path,
    mask_dir: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    target_sizes: tuple[tuple[int, int], ...] | None = None,
) -> tuple[EdgeDataset, EdgeDataset]:
    """Pair first, shuffle with ``seed``, and return train/validation datasets."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    common = {"image_dir": image_dir, "mask_dir": mask_dir}
    if target_sizes is not None:
        common["target_sizes"] = target_sizes
    full = EdgeDataset(training=False, **common)
    samples = list(full.samples)
    random.Random(seed).shuffle(samples)
    val_count = max(1, round(len(samples) * val_ratio))
    if val_count >= len(samples):
        raise ValueError("dataset must contain at least two samples")
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]
    return (
        EdgeDataset(training=True, samples=train_samples, **common),
        EdgeDataset(training=False, samples=val_samples, **common),
    )
