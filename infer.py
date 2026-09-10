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
from engine.benchmark import benchmark_forward
from data.transforms.letterbox import letterbox, select_target_size
from utils.run_logging import create_run, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="one image or an image directory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=240)
    parser.add_argument("--no-contact-sheet", action="store_true")
    parser.add_argument('--benchmark', action='store_true', help='measure forward-only FPS; skip mask saving')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50, help='timed forwards per image')
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error('warmup must be >= 0 and iterations must be >= 1')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get('stage') == 'routing':
        raise ValueError('routing-only checkpoint has no trained segmentation decoder')
    if 'model' in checkpoint and checkpoint.get('format_version') != 3:
        raise ValueError('use a format_version=3 1024x1024 multi-batch checkpoint for inference')
    model = EdgeDynamicViT(selection_threshold=checkpoint.get('args', {}).get('selection_threshold', 0.5)).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    records = []
    image_paths = iter_images(args.input)
    if not image_paths:
        raise ValueError('no input images found')
    output_dir = create_run(args.output_dir)
    save_json(output_dir / 'args.json', vars(args))
    print(f'output directory: {output_dir}')
    if args.benchmark:
        results = []
        for image_path in image_paths:
            image = load_image(image_path)
            resized, _, _ = letterbox(image, select_target_size(*image.shape[-2:]))
            inputs = resized.unsqueeze(0).to(device)
            result = benchmark_forward(model, inputs, args.warmup, args.iterations)
            result['image'] = str(image_path)
            results.append(result)
            print(f'{image_path.name}: {result["mean_latency_ms"]:.3f} ms/batch, {result["image_fps"]:.2f} image FPS', flush=True)
        frames = sum(row['iterations'] for row in results)
        elapsed = sum(row['total_seconds'] for row in results)
        summary = {'scope': 'model forward only, synchronized wall time; FP32',
                   'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                   'torch_version': str(torch.__version__), 'cpu_threads': torch.get_num_threads(),
                   'selection_threshold': model.selector.selection_threshold,
                   'batches': frames, 'images_per_batch': 1, 'total_seconds': elapsed,
                   'batch_fps': frames / elapsed, 'image_fps': frames / elapsed,
                   'fps': frames / elapsed, 'mean_latency_ms': elapsed * 1000 / frames, 'images': results}
        save_json(output_dir / 'benchmark.json', summary)
        print(f'Overall: {summary["image_fps"]:.2f} image FPS, {summary["mean_latency_ms"]:.3f} ms/batch')
        return
    for image_path in image_paths:
        image = load_image(image_path)
        probability = predict_image(model, image, device)
        overlay = make_overlay(image, probability, args.threshold)
        paths = save_prediction(probability, output_dir, image_path.stem, args.threshold, image=image)
        original_pil = Image.fromarray(
            image.mul(255).byte().permute(1, 2, 0).cpu().numpy(), mode="RGB"
        )
        records.append((image_path.stem, original_pil, overlay))
        print(f"{image_path} -> {paths[0]}, {paths[1]}, {paths[2]}")
    if len(records) > 1 and not args.no_contact_sheet:
        sheet_path = save_contact_sheet(
            records,
            output_dir / "contact_sheet.png",
            tile_size=(args.tile_width, args.tile_height),
        )
        print(f"contact sheet -> {sheet_path}")


if __name__ == "__main__":
    main()
