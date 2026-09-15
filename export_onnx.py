#!/usr/bin/env python3
"""Export the fixed 256x256 tile inference graph for ONNX/TensorRT."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from main import EdgeDynamicViT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('format_version') != 6:
        raise ValueError('ONNX export requires format_version=6 tile checkpoint')
    if checkpoint.get('tile_size') != 256 or checkpoint.get('patch_size') != 16:
        raise ValueError('checkpoint does not match the 256/16 tile contract')
    threshold = checkpoint.get('args', {}).get('selection_threshold', 0.5)
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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, sample, str(output), opset_version=args.opset,
        input_names=['images'], output_names=['mask_logits', 'keep_logits', 'keep_probs'],
        dynamic_axes={'images': {0: 'batch'}, 'mask_logits': {0: 'batch'},
                      'keep_logits': {0: 'batch'}, 'keep_probs': {0: 'batch'}},
        do_constant_folding=True,
    )
    print(f'exported: {output}')


if __name__ == '__main__':
    main()
