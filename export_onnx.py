#!/usr/bin/env python3
"""Export the fixed 256x256 tile inference graph for ONNX/TensorRT.

This script exports forward_deploy(), which uses fixed-shape dual-branch
computation and torch.where fusion instead of dynamic routing indices.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from main import EdgeDynamicViT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export EdgeDynamicViT deployment graph to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Basic export with default settings
  python export_onnx.py --checkpoint runs/train/best.pt --output deployment/eetf_tile_256.onnx

  # Export with ONNX simplification and opset 18
  python export_onnx.py --checkpoint runs/train/best.pt --output deployment/eetf_tile_256.onnx --simplify --opset 18

  # Export with dynamic batch size=4 for testing
  python export_onnx.py --checkpoint runs/train/best.pt --output deployment/eetf_tile_256.onnx --batch-size 4
        """,
    )
    parser.add_argument('--checkpoint', type=Path, required=True, help='PyTorch checkpoint (.pt) with format_version=6')
    parser.add_argument('--output', type=Path, required=True, help='Output ONNX model path')
    parser.add_argument('--opset', type=int, default=18, help='ONNX opset version (default: 18)')
    parser.add_argument('--batch-size', type=int, default=1, help='Sample batch size for shape inference (default: 1)')
    parser.add_argument('--simplify', action='store_true', help='Simplify ONNX graph using onnxsim (requires onnxsim package)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed export information')
    args = parser.parse_args()
    
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Validate checkpoint format
    format_version = checkpoint.get('format_version')
    if format_version != 6:
        print(f"Error: ONNX export requires format_version=6, got {format_version}", file=sys.stderr)
        print("This checkpoint was trained with an incompatible data/model contract.", file=sys.stderr)
        sys.exit(1)
    
    tile_size = checkpoint.get('tile_size')
    patch_size = checkpoint.get('patch_size')
    if tile_size != 256 or patch_size != 16:
        print(f"Error: Expected tile_size=256, patch_size=16, got {tile_size}/{patch_size}", file=sys.stderr)
        sys.exit(1)
    
    # Check training stage
    stage = checkpoint.get('stage', 'unknown')
    epoch = checkpoint.get('epoch', 0)
    if stage == 'routing':
        print(f"Warning: Checkpoint is from routing stage (epoch {epoch}).", file=sys.stderr)
        print("Routing-stage checkpoints have not finalized Selected/Remaining decoders.", file=sys.stderr)
        response = input("Continue export anyway? [y/N]: ")
        if response.lower() not in ('y', 'yes'):
            print("Export cancelled.")
            sys.exit(0)
    
    ckpt_args = checkpoint.get('args', {})
    threshold = ckpt_args.get('selection_threshold', 0.5)
    
    if args.verbose:
        print(f"Checkpoint metadata:")
        print(f"  format_version: {format_version}")
        print(f"  stage: {stage}")
        print(f"  epoch: {epoch}")
        print(f"  tile_size: {tile_size}")
        print(f"  patch_size: {patch_size}")
        print(f"  selection_threshold: {threshold}")
    model = EdgeDynamicViT(selection_threshold=threshold, patch_height=16, patch_width=16)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    class ExportWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, images):
            output = self.module.forward_deploy(images)
            return output['mask_logits'], output['keep_logits'], output['keep_probs']

    wrapper = ExportWrapper(model)
    sample = torch.zeros(args.batch_size, 3, 256, 256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting to ONNX (opset {args.opset})...")
    torch.onnx.export(
        wrapper,
        sample,
        str(args.output),
        opset_version=args.opset,
        input_names=['images'],
        output_names=['mask_logits', 'keep_logits', 'keep_probs'],
        dynamic_axes={
            'images': {0: 'batch'},
            'mask_logits': {0: 'batch'},
            'keep_logits': {0: 'batch'},
            'keep_probs': {0: 'batch'},
        },
        do_constant_folding=False,
        verbose=args.verbose,
    )
    
    print(f"✓ ONNX exported: {args.output}")
    
    # Optional simplification
    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify
            
            print("Simplifying ONNX graph...")
            model_onnx = onnx.load(str(args.output))
            model_simplified, check = simplify(model_onnx)
            if not check:
                print("Warning: ONNX simplification validation failed.", file=sys.stderr)
            else:
                onnx.save(model_simplified, str(args.output))
                print("✓ ONNX graph simplified")
        except ImportError:
            print("Warning: onnxsim not installed. Install with: pip install onnxsim", file=sys.stderr)
            print("Skipping simplification.")
        except Exception as e:
            print(f"Warning: ONNX simplification failed: {e}", file=sys.stderr)
    
    # Print output info
    print("\nONNX Model Summary:")
    print(f"  Input:  images       [batch, 3, 256, 256]  float32")
    print(f"  Output: mask_logits  [batch, 1, 256, 256]  float32")
    print(f"  Output: keep_logits  [batch, 256, 2]      float32")
    print(f"  Output: keep_probs   [batch, 256]         float32")
    print(f"  Router threshold: {threshold}")
    print(f"  Opset: {args.opset}")
    print(f"\nNext steps:")
    print(f"  1. Verify with ONNX Runtime: python infer_onnx.py --input <image> --onnx {args.output}")
    print(f"  2. Build TensorRT engine: trtexec --onnx={args.output} --saveEngine=deployment/eetf.trt --fp16")


if __name__ == '__main__':
    main()
