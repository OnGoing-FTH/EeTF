"""Timestamped runs and headless epoch visualizations."""
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METRICS = ('loss', 'router_loss', 'router_ce_loss', 'router_pairwise_loss',
           'router_boundary_loss', 'router_morphology_loss', 'segmentation_loss',
           'pixel_loss', 'seg_cldice_loss', 'route_f1', 'foreground_f1',
           'selected_fraction')
CONFUSION_FIELDS = ('val_tn_rate', 'val_fp_rate', 'val_fn_rate', 'val_tp_rate')
FIELDS = (['epoch', 'stage']
          + [f'{split}_{key}' for key in METRICS for split in ('train', 'val')]
          + list(CONFUSION_FIELDS))


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
            segmentation_key = key in ('segmentation_loss', 'pixel_loss',
                                       'seg_cldice_loss', 'foreground_f1')
            inactive = (stage == 'routing' and segmentation_key) or (stage == 'segmentation' and key.startswith('router_'))
            row[f'{split}_{key}'] = None if inactive else values.get(key)
    for key in CONFUSION_FIELDS:
        row[key] = None
    if stage == 'routing' and validation.get('route_confusion') is not None:
        (tn, fp), (fn, tp) = validation['route_confusion']
        negative = max(tn + fp, 1)
        positive = max(fn + tp, 1)
        row.update(val_tn_rate=tn / negative, val_fp_rate=fp / negative,
                   val_fn_rate=fn / positive, val_tp_rate=tp / positive)
    return row


def save_history(run_dir, history, routing_epochs, frozen_epochs, epochs):
    run_dir = Path(run_dir)
    with (run_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(history)
    columns = 3
    rows = (len(METRICS) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4.2 * rows), constrained_layout=True)
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
    for ax in axes.flat[len(METRICS):]:
        ax.set_visible(False)
    fig.savefig(run_dir / 'results.png', dpi=140)
    plt.close(fig)
    _save_route_confusion_history(run_dir / 'validation', history)


def _save_route_confusion_history(directory, history):
    points = [row for row in history if row['stage'] == 'routing' and row.get('val_tn_rate') is not None]
    if not points:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    styles = (('val_tn_rate', 'TN rate', '#24733f'), ('val_fp_rate', 'FP rate', '#d1495b'),
              ('val_fn_rate', 'FN rate', '#d9822b'), ('val_tp_rate', 'TP rate', '#2864a8'))
    for key, label, color in styles:
        ax.plot([row['epoch'] for row in points], [row[key] for row in points],
                marker='o', linewidth=2, color=color, label=label)
    ax.set_yscale('symlog', linthresh=0.01, linscale=1.0, base=10)
    ticks = [0, .01, .02, .05, .1, .2, .5, .9, 1.0]
    ax.set_yticks(ticks, [f'{value:g}' for value in ticks])
    ax.set_ylim(0, 1.05)
    ax.set(xlabel='Routing Epoch', ylabel='Row-normalized rate (symlog)',
           title='Patch routing confusion over time')
    ax.grid(True, which='both', alpha=.25)
    ax.legend(ncol=4, loc='best')
    fig.savefig(directory / 'route_confusion_history.png', dpi=150)
    plt.close(fig)


def save_validation(directory, metrics, stage):
    save_json(Path(directory) / 'metrics.json', metrics)
