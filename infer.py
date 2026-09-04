"""Command-line inference entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from engine.infer import (
    iter_images,
    load_image,
    make_overlay,
    predict_image,
    save_contact_sheet,
    save_prediction,
)
from main import EdgeDynamicViT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="one image or an image directory")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=240)
    parser.add_argument("--no-contact-sheet", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeDynamicViT().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    records = []
    image_paths = iter_images(args.input)
    for image_path in image_paths:
        image = load_image(image_path)
        probability = predict_image(model, image, device)
        overlay = make_overlay(image, probability, args.threshold)
        paths = save_prediction(probability, args.output_dir, image_path.stem, args.threshold, image=image)
        original_pil = Image.fromarray(
            image.mul(255).byte().permute(1, 2, 0).cpu().numpy(), mode="RGB"
        )
        records.append((image_path.stem, original_pil, overlay))
        print(f"{image_path} -> {paths[0]}, {paths[1]}, {paths[2]}")
    if len(records) > 1 and not args.no_contact_sheet:
        sheet_path = save_contact_sheet(
            records,
            Path(args.output_dir) / "contact_sheet.png",
            tile_size=(args.tile_width, args.tile_height),
        )
        print(f"contact sheet -> {sheet_path}")


if __name__ == "__main__":
    main()
