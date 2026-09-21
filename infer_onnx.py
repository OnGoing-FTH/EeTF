#!/usr/bin/env python3
"""ONNX Runtime inference for large images with automatic tiling.

This script handles arbitrary-sized images by:
1. Letterboxing to 256-pixel multiples without distortion
2. Splitting into 256×256 tiles
3. Batched ONNX Runtime inference
4. Merging tiles back to letterbox canvas
5. Removing padding and resizing to original dimensions
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from data.transforms.letterbox import (
    letterbox_to_multiple,
    split_tiles,
    merge_tiles,
    unletterbox,
)
from engine.infer import (
    make_overlay,
    save_prediction,
    save_contact_sheet,
    iter_images,
)


def create_onnx_session(
    onnx_path: Path,
    provider: str,
    device_id: int = 0,
    trt_fp16: bool = False,
    trt_engine_cache_dir: Path | None = None,
) -> ort.InferenceSession:
    """Create ONNX Runtime session with specified execution provider."""
    
    providers_map = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }
    
    available_providers = ort.get_available_providers()
    
    if provider == "auto":
        # Priority: TensorRT > CUDA > CPU
        if "TensorrtExecutionProvider" in available_providers:
            provider = "tensorrt"
        elif "CUDAExecutionProvider" in available_providers:
            provider = "cuda"
        else:
            provider = "cpu"
        print(f"Auto-selected provider: {provider}")
    
    if provider not in providers_map:
        raise ValueError(f"Unknown provider: {provider}. Choose from: auto, cpu, cuda, tensorrt")
    
    provider_name = providers_map[provider]
    
    if provider_name not in available_providers:
        print(f"Error: {provider_name} not available in this ONNX Runtime build.", file=sys.stderr)
        print(f"Available providers: {available_providers}", file=sys.stderr)
        sys.exit(1)
    
    # Build provider options
    provider_options = []
    
    if provider == "cuda":
        cuda_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "DEFAULT",
        }
        provider_options = [(provider_name, cuda_options)]
    
    elif provider == "tensorrt":
        trt_options = {
            "device_id": device_id,
            "trt_fp16_enable": trt_fp16,
        }
        if trt_engine_cache_dir is not None:
            trt_engine_cache_dir.mkdir(parents=True, exist_ok=True)
            trt_options["trt_engine_cache_enable"] = True
            trt_options["trt_engine_cache_path"] = str(trt_engine_cache_dir)
        provider_options = [(provider_name, trt_options)]
    
    else:  # CPU
        provider_options = [provider_name]
    
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=provider_options,
    )
    
    # Print active providers
    active_providers = session.get_providers()
    print(f"Active providers: {active_providers}")
    
    # Validate expected provider is active
    if provider != "auto" and provider_name not in active_providers:
        print(f"Warning: Requested {provider_name} but it's not active. Falling back to {active_providers[0]}", file=sys.stderr)
    
    return session


def predict_image_onnx(
    session: ort.InferenceSession,
    image_bgr: np.ndarray,
    tile_size: int = 256,
    tile_batch_size: int = 16,
    threshold: float = 0.5,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run ONNX inference on a large image with automatic tiling.
    
    Returns:
        probability_map: [H, W] float32 in [0, 1], same size as input image
        metadata: dict with timing and routing statistics
    """
    import torch
    
    original_height, original_width = image_bgr.shape[:2]
    
    if verbose:
        print(f"  Original size: {original_width}×{original_height}")
    
    # Convert to RGB tensor [3, H, W]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    
    # Letterbox
    letterbox_tensor, _, letterbox_meta = letterbox_to_multiple(image_tensor, tile_size)
    
    if verbose:
        print(f"  Letterbox size: {letterbox_meta['target_width']}×{letterbox_meta['target_height']}")
    
    # Split into tiles
    tile_tensor, tile_meta = split_tiles(letterbox_tensor, tile_size)
    num_tiles = tile_tensor.shape[0]
    
    # Extract grid dimensions from tile_meta list
    grid_rows = max(m['row'] for m in tile_meta) + 1
    grid_cols = max(m['col'] for m in tile_meta) + 1
    
    if verbose:
        print(f"  Tiles: {num_tiles} ({grid_cols}×{grid_rows})")
    
    # Batch inference
    all_mask_logits = []
    all_keep_probs = []
    
    inference_start = time.perf_counter()
    
    for batch_start in range(0, num_tiles, tile_batch_size):
        batch_end = min(batch_start + tile_batch_size, num_tiles)
        batch_tiles = tile_tensor[batch_start:batch_end].numpy()  # [B, 3, 256, 256]
        
        # ONNX Runtime inference
        ort_inputs = {"images": batch_tiles}
        mask_logits, keep_logits, keep_probs = session.run(None, ort_inputs)
        
        all_mask_logits.append(mask_logits)
        all_keep_probs.append(keep_probs)
    
    inference_time = time.perf_counter() - inference_start
    
    # Concatenate batches
    all_mask_logits = np.concatenate(all_mask_logits, axis=0)  # [N, 1, 256, 256]
    all_keep_probs = np.concatenate(all_keep_probs, axis=0)    # [N, 256]
    
    # Convert logits to probabilities
    mask_probs = 1.0 / (1.0 + np.exp(-all_mask_logits))  # sigmoid
    
    # Convert back to torch for merge operations
    mask_probs_tensor = torch.from_numpy(mask_probs)
    
    # Merge tiles
    canvas_size = (letterbox_meta['target_height'], letterbox_meta['target_width'])
    letterbox_prob = merge_tiles(
        mask_probs_tensor,
        tile_meta,
        canvas_size,
        tile_size,
    )  # [1, 1, letterbox_h, letterbox_w]
    
    # Remove padding
    unpadded_prob = unletterbox(
        letterbox_prob.squeeze(0),  # [1, letterbox_h, letterbox_w]
        letterbox_meta,
    )  # [1, valid_h, valid_w]
    
    # Resize to original
    prob_np = unpadded_prob.numpy()  # Keep [1, valid_h, valid_w]
    
    if prob_np.shape[1:] != (original_height, original_width):
        # Resize the [1,H,W] tensor
        prob_squeezed = prob_np.squeeze(0)  # [H,W]
        prob_resized = cv2.resize(
            prob_squeezed,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )
        prob_np = prob_resized[None, ...]  # [1,H,W]
    
    # Compute routing statistics
    selected_ratio = (all_keep_probs >= threshold).mean()
    
    # Extract grid dimensions from tile_meta
    grid_rows = max(m['row'] for m in tile_meta) + 1
    grid_cols = max(m['col'] for m in tile_meta) + 1
    
    metadata = {
        "original_size": (original_width, original_height),
        "letterbox_size": (letterbox_meta['target_width'], letterbox_meta['target_height']),
        "num_tiles": num_tiles,
        "grid_size": (grid_cols, grid_rows),
        "inference_time": inference_time,
        "selected_patch_ratio": float(selected_ratio),
    }
    
    return prob_np, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ONNX Runtime inference for edge detection on large images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Single image with CUDA
  python infer_onnx.py --input data/test.png --onnx deployment/eetf_tile_256.onnx --output-dir outputs_onnx --provider cuda

  # Directory with TensorRT FP16
  python infer_onnx.py --input data/test_images --onnx deployment/eetf_tile_256.onnx --output-dir outputs_trt \\
    --provider tensorrt --trt-fp16 --trt-engine-cache-dir deployment/trt_cache

  # CPU inference
  python infer_onnx.py --input data/test.png --onnx deployment/eetf_tile_256.onnx --output-dir outputs_cpu --provider cpu
        """,
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image or directory")
    parser.add_argument("--onnx", type=Path, required=True, help="ONNX model path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda", "tensorrt"], default="auto",
                        help="ONNX Runtime execution provider (default: auto)")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA/TensorRT device ID (default: 0)")
    parser.add_argument("--tile-batch-size", type=int, default=16, help="Tile batch size (default: 16)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary mask threshold (default: 0.5)")
    parser.add_argument("--trt-fp16", action="store_true", help="Enable TensorRT FP16 mode")
    parser.add_argument("--trt-engine-cache-dir", type=Path, help="TensorRT engine cache directory")
    parser.add_argument("--no-contact-sheet", action="store_true", help="Skip contact sheet generation for directories")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-image information")
    args = parser.parse_args()
    
    if not args.onnx.is_file():
        parser.error(f"ONNX model not found: {args.onnx}")
    
    if not args.input.exists():
        parser.error(f"Input not found: {args.input}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"ONNX model: {args.onnx}")
    print(f"Output directory: {args.output_dir}")
    
    # Create ONNX Runtime session
    session = create_onnx_session(
        args.onnx,
        args.provider,
        args.device_id,
        args.trt_fp16,
        args.trt_engine_cache_dir,
    )
    
    # Get input/output info
    input_info = session.get_inputs()[0]
    print(f"\nModel input: {input_info.name} {input_info.shape} {input_info.type}")
    
    # Collect images
    if args.input.is_file():
        image_paths = [args.input]
    else:
        image_paths = list(iter_images(args.input))
        if not image_paths:
            print(f"No images found in {args.input}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(image_paths)} images")
    
    # Process images
    results = []
    total_start = time.perf_counter()
    
    for i, image_path in enumerate(image_paths, start=1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {image_path.name}")
        
        try:
            # Load image as BGR numpy array
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Run inference
            prob_map, metadata = predict_image_onnx(
                session,
                image_bgr,
                tile_size=256,
                tile_batch_size=args.tile_batch_size,
                threshold=args.threshold,
                verbose=args.verbose,
            )
            
            # Generate binary mask
            prob_map_2d = prob_map.squeeze(0)  # [H, W]
            binary_mask = (prob_map_2d >= args.threshold).astype(np.uint8) * 255
            
            # Create overlay - convert to torch tensors
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0  # [3,H,W]
            prob_tensor = torch.from_numpy(prob_map)  # Already [1,H,W]
            overlay_pil = make_overlay(image_tensor, prob_tensor, threshold=args.threshold)
            overlay = cv2.cvtColor(np.array(overlay_pil), cv2.COLOR_RGB2BGR)
            
            # Save outputs
            output_name = image_path.stem
            prob_path = args.output_dir / f"{output_name}_prob.png"
            mask_path = args.output_dir / f"{output_name}_mask.png"
            overlay_path = args.output_dir / f"{output_name}_overlay.png"
            
            cv2.imwrite(str(prob_path), (prob_map_2d * 255).astype(np.uint8))
            cv2.imwrite(str(mask_path), binary_mask)
            cv2.imwrite(str(overlay_path), overlay)
            
            print(f"  Inference: {metadata['inference_time']:.3f}s")
            print(f"  Selected patches: {metadata['selected_patch_ratio']:.1%}")
            print(f"  Saved: {mask_path.name}")
            
            results.append({
                "image_path": image_path,
                "prob_map": prob_map,
                "binary_mask": binary_mask,
                "overlay": overlay,
                "metadata": metadata,
            })
        
        except Exception as e:
            import traceback
            print(f"  Error: {e}", file=sys.stderr)
            traceback.print_exc()
            continue
    
    total_time = time.perf_counter() - total_start
    
    # Save contact sheet for directories
    if len(image_paths) > 1 and not args.no_contact_sheet:
        if results:
            print("\nGenerating contact sheet...")
            contact_path = args.output_dir / "contact_sheet.png"
            contact_records = []
            for result in results[:50]:
                original_bgr = cv2.imread(str(result["image_path"]), cv2.IMREAD_COLOR)
                if original_bgr is None:
                    continue
                original = Image.fromarray(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
                overlay = Image.fromarray(cv2.cvtColor(result["overlay"], cv2.COLOR_BGR2RGB))
                contact_records.append((result["image_path"].stem, original, overlay))
            if contact_records:
                save_contact_sheet(contact_records, contact_path)
                print(f"Contact sheet: {contact_path}")
        else:
            print("\nSkipping contact sheet: no successful inference results.")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Completed {len(results)}/{len(image_paths)} images in {total_time:.2f}s")
    if results:
        avg_time = sum(r["metadata"]["inference_time"] for r in results) / len(results)
        print(f"Average inference time: {avg_time:.3f}s per image")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
