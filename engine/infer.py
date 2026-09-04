"""Inference helpers for sparse edge segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn

from data.transforms.letterbox import TARGET_SIZES, letterbox, select_target_size, unletterbox

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_image(path: str | Path) -> Tensor:
    """Read an RGB image as a normalized ``[3, H, W]`` tensor."""
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).div(255.0)


def predict_image(
    model: nn.Module,
    image: Tensor,
    device: torch.device,
    target_sizes: tuple[tuple[int, int], ...] = TARGET_SIZES,
) -> Tensor:
    """Return unletterboxed probability map with shape ``[1, H, W]``."""
    if image.ndim != 3:
        raise ValueError("image must have shape [3, H, W]")
    target_size = select_target_size(*image.shape[-2:], target_sizes)
    input_image, _, metadata = letterbox(image, target_size)
    model.eval()
    with torch.inference_mode():
        logits = model(input_image.unsqueeze(0).to(device))["mask_logits"]
        probability = logits.sigmoid().cpu().squeeze(0)
    return unletterbox(probability, metadata).clamp(0, 1)


def make_overlay(image: Tensor, probability: Tensor, threshold: float = 0.5, alpha: float = 0.55) -> Image.Image:
    """Return an RGB image with predicted foreground highlighted in red."""
    if image.ndim != 3 or image.shape[0] != 3 or probability.shape[0] != 1:
        raise ValueError("image must be [3,H,W] and probability must be [1,H,W]")
    if image.shape[-2:] != probability.shape[-2:]:
        raise ValueError("image and probability must have identical spatial sizes")
    base = Image.fromarray(image.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy(), mode="RGB")
    mask = (probability.squeeze(0) >= threshold).cpu().numpy()
    color = np.zeros((*mask.shape, 4), dtype=np.uint8)
    color[..., 0] = 255
    color[..., 3] = (mask * round(255 * alpha)).astype(np.uint8)
    return Image.alpha_composite(base.convert("RGBA"), Image.fromarray(color, mode="RGBA")).convert("RGB")


def save_prediction(
    probability: Tensor,
    output_dir: str | Path,
    stem: str,
    threshold: float = 0.5,
    image: Tensor | None = None,
) -> tuple[Path, Path, Path | None]:
    """Write probability, binary mask, and optional original-image overlay PNGs."""
    if probability.shape[0] != 1:
        raise ValueError("probability must have shape [1, H, W]")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    probability_image = probability.squeeze(0).mul(255).round().to(torch.uint8).numpy()
    binary_image = (probability.squeeze(0) >= threshold).to(torch.uint8).mul(255).numpy()
    probability_path = output_path / f"{stem}_prob.png"
    binary_path = output_path / f"{stem}_mask.png"
    Image.fromarray(probability_image, mode="L").save(probability_path)
    Image.fromarray(binary_image, mode="L").save(binary_path)
    overlay_path = None
    if image is not None:
        overlay_path = output_path / f"{stem}_overlay.png"
        make_overlay(image, probability, threshold).save(overlay_path)
    return probability_path, binary_path, overlay_path


def save_contact_sheet(
    records: list[tuple[str, Image.Image, Image.Image]],
    output_path: str | Path,
    tile_size: tuple[int, int] = (320, 240),
) -> Path:
    """Save a grid containing original and overlay images for each sample."""
    if not records:
        raise ValueError("records must not be empty")
    tile_width, tile_height = tile_size
    columns = 2
    label_height = 24
    sheet = Image.new("RGB", (columns * tile_width, len(records) * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (name, original, overlay) in enumerate(records):
        y = row * (tile_height + label_height)
        for column, (title, picture) in enumerate(((f"{name} original", original), (f"{name} overlay", overlay))):
            picture = picture.copy()
            picture.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
            x = column * tile_width + (tile_width - picture.width) // 2
            image_y = y + label_height + (tile_height - picture.height) // 2
            sheet.paste(picture, (x, image_y))
            draw.text((column * tile_width + 6, y + 5), title, fill="black")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def iter_images(path: str | Path) -> list[Path]:
    """Return one supported image or all supported images in a directory."""
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.suffix.lower() in _IMAGE_SUFFIXES)
    raise FileNotFoundError(path)
