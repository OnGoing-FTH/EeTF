"""Timestamped runs and headless epoch visualizations."""
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METRICS = ('loss', 'router_loss', 'pixel_loss', 'route_f1', 'foreground_f1', 'selected_fraction')
FIELDS = ['epoch', 'stage'] + [f'{split}_{key}' for key in METRICS for split in ('train', 'val')]


def create_run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        path = root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        try:
            path.mkdir()
            return path.resolve()
        except FileExistsError:
            continue


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def metric_row(epoch, stage, training, validation):
    row = dict(epoch=epoch, stage=stage)
    for split, values in (('train', training), ('val', validation)):
        for key in METRICS:
            inactive = (stage == 'routing' and key in ('pixel_loss', 'foreground_f1')) or (stage == 'segmentation' and key == 'router_loss')
            row[f'{split}_{key}'] = None if inactive else values.get(key)
    return row


def save_history(run_dir, history, routing_epochs, frozen_epochs, epochs):
    run_dir = Path(run_dir)
    with (run_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(history)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    boundaries = (0.5, routing_epochs + 0.5, routing_epochs + frozen_epochs + 0.5, epochs + 0.5)
    for ax, key in zip(axes.flat, METRICS):
        for index, (name, color) in enumerate(zip(('Routing', 'Frozen', 'Fine-tune'), ('#dceadf', '#f8e8ce', '#eadff0'))):
            left, right = boundaries[index:index + 2]
            ax.axvspan(left, right, color=color, alpha=.5)
            ax.text((left + right) / 2, .98, name, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8)
        for boundary in boundaries[1:3]:
            ax.axvline(boundary, color='gray', linestyle='--', linewidth=1)
        for split, color in (('train', '#226b42'), ('val', '#ba3b46')):
            # Total loss changes meaning at stage boundaries; do not connect them.
            stages = ('routing', 'segmentation', 'finetune') if key == 'loss' else (None,)
            for index, stage in enumerate(stages):
                points = [row for row in history if stage is None or row['stage'] == stage]
                ax.plot([row['epoch'] for row in points],
                        [np.nan if row.get(f'{split}_{key}') is None else row[f'{split}_{key}'] for row in points],
                        color=color, marker='.', label=split if index == 0 else None)
        ax.set(title=key, xlabel='Epoch', xlim=(.5, epochs + .5))
        if key.endswith('f1') or key == 'selected_fraction':
            ax.set_ylim(0, 1.12)
        ax.grid(alpha=.2)
        ax.legend(loc='best')
    fig.savefig(run_dir / 'results.png', dpi=140)
    plt.close(fig)


def save_validation(directory, metrics, stage):
    directory = Path(directory)
    save_json(directory / 'metrics.json', metrics)
    if stage != 'routing':
        return
    counts = np.asarray(metrics['route_confusion'], dtype=float)
    normalized = np.divide(counts, counts.sum(axis=1, keepdims=True),
                           out=np.zeros_like(counts), where=counts.sum(axis=1, keepdims=True) != 0)
    for matrix, suffix in ((counts, ''), (normalized, '_normalized')):
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        image = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1 if suffix else max(counts.max(), 1))
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Drop', 'Keep'], yticklabels=['Drop', 'Keep'],
               xlabel='Predicted', ylabel='True', title='Patch confusion' + (' (row normalized)' if suffix else ''))
        for row in range(2):
            for col in range(2):
                text = f'{matrix[row, col]:.3f}' if suffix else str(int(matrix[row, col]))
                ax.text(col, row, text, ha='center', va='center', color='white' if matrix[row, col] > image.norm.vmax / 2 else 'black')
        fig.colorbar(image, ax=ax)
        fig.savefig(directory / f'confusion_matrix{suffix}.png', dpi=140)
        plt.close(fig)
