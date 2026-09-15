"""Inference helpers for tiled sparse edge segmentation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn

from data.transforms.letterbox import TILE_SIZE, letterbox_to_multiple, split_tiles, merge_tiles, unletterbox

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_image(path: str | Path) -> Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).div(255.0)


def predict_image(model: nn.Module, image: Tensor, device: torch.device,
                  tile_batch_size: int = 16) -> Tensor:
    """Predict a full image by tiled inference and return [1,H,W]."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3,H,W]")
    padded, _, metadata = letterbox_to_multiple(image, TILE_SIZE)
    tiles, tile_meta = split_tiles(padded, TILE_SIZE)
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, tiles.shape[0], tile_batch_size):
            batch = tiles[start:start + tile_batch_size].to(device, non_blocking=True)
            if hasattr(model, "forward_deploy"):
                logits = model.forward_deploy(batch)["mask_logits"]
            else:
                logits = model(batch)["mask_logits"]
            outputs.append(logits.sigmoid().cpu())
    probability = merge_tiles(torch.cat(outputs, dim=0), tile_meta, padded.shape[-2:], TILE_SIZE).squeeze(0)
    return unletterbox(probability, metadata).clamp(0, 1)


def make_overlay(image: Tensor, probability: Tensor, threshold: float = 0.5, alpha: float = 0.55) -> Image.Image:
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


def save_prediction(probability: Tensor, output_dir: str | Path, stem: str,
                    threshold: float = 0.5, image: Tensor | None = None):
    if probability.shape[0] != 1:
        raise ValueError("probability must have shape [1,H,W]")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    probability_image = probability.squeeze(0).mul(255).round().to(torch.uint8).numpy()
    binary_image = (probability.squeeze(0) >= threshold).to(torch.uint8).mul(255).numpy()
    probability_path, binary_path = output_path / f"{stem}_prob.png", output_path / f"{stem}_mask.png"
    Image.fromarray(probability_image, mode="L").save(probability_path)
    Image.fromarray(binary_image, mode="L").save(binary_path)
    overlay_path = None
    if image is not None:
        overlay_path = output_path / f"{stem}_overlay.png"
        make_overlay(image, probability, threshold).save(overlay_path)
    return probability_path, binary_path, overlay_path


def save_contact_sheet(records, output_path: str | Path, tile_size=(320, 240)) -> Path:
    if not records:
        raise ValueError("records must not be empty")
    tile_width, tile_height = tile_size
    label_height, columns = 24, 2
    sheet = Image.new("RGB", (columns * tile_width, len(records) * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (name, original, overlay) in enumerate(records):
        y = row * (tile_height + label_height)
        for column, (title, picture) in enumerate(((f"{name} original", original), (f"{name} overlay", overlay))):
            picture = picture.copy(); picture.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
            x = column * tile_width + (tile_width - picture.width) // 2
            image_y = y + label_height + (tile_height - picture.height) // 2
            sheet.paste(picture, (x, image_y)); draw.text((column * tile_width + 6, y + 5), title, fill="black")
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True); sheet.save(output_path)
    return output_path


def iter_images(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file(): return [path]
    if path.is_dir(): return sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    raise FileNotFoundError(path)
