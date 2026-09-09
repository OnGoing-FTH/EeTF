"""Routing pretraining, frozen segmentation, and joint fine-tuning."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import split_edge_dataset
from engine.train import train_one_epoch
from engine.validate import validate
from losses.sparse_segmentation_loss import SparseEdgeLoss
from main import EdgeDynamicViT
from utils.run_logging import create_run, save_json, metric_row, save_history, save_validation


def stage_for_epoch(epoch, routing_epochs, frozen_epochs):
    if epoch < routing_epochs:
        return 'routing'
    if epoch < routing_epochs + frozen_epochs:
        return 'segmentation'
    return 'finetune'


def make_optimizer(model, learning_rate, weight_decay):
    routing = [p for module in model.routing_modules() for p in module.parameters()]
    routing_ids = {id(p) for p in routing}
    decoding = [p for p in model.parameters() if id(p) not in routing_ids]
    return torch.optim.AdamW([
        {'params': routing, 'name': 'routing'},
        {'params': decoding, 'name': 'decoding'},
    ], lr=learning_rate, weight_decay=weight_decay)


def configure_stage(model, optimizer, stage, learning_rate, finetune_multiplier):
    model.set_stage(stage)
    for group in optimizer.param_groups:
        factor = finetune_multiplier if stage == 'finetune' and group['name'] == 'routing' else 1.0
        group['lr'] = learning_rate * factor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--image-dir', default='images')
    parser.add_argument('--mask-dir', default='edge_maps')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100, help='total epochs across all three stages')
    parser.add_argument('--routing-epochs', type=int, default=20)
    parser.add_argument('--frozen-epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--finetune-lr-multiplier', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--selection-threshold', type=float, default=0.5)
    parser.add_argument('--router-weight', type=float, default=1.0)
    parser.add_argument('--router-boundary-weight', type=float, default=0.2)
    parser.add_argument('--router-pairwise-weight', type=float, default=0.1)
    parser.add_argument('--router-morphology-weight', type=float, default=0.1)
    parser.add_argument('--router-boundary-margin', type=float, default=0.1)
    parser.add_argument('--router-pair-margin', type=float, default=0.2)
    parser.add_argument('--seg-cldice-weight', type=float, default=0.5)
    parser.add_argument('--seg-cldice-iterations', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--run-dir', '--checkpoint-dir', dest='run_dir', default='runs/train', help='parent directory for timestamped training runs')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    if args.routing_epochs < 1 or args.frozen_epochs < 1 or args.epochs <= args.routing_epochs + args.frozen_epochs:
        parser.error('require routing-epochs >= 1, frozen-epochs >= 1 and epochs > their sum')
    if not 0 < args.finetune_lr_multiplier <= 1 or args.learning_rate <= 0 or args.router_weight <= 0:
        parser.error('require positive learning rate/router weight and finetune multiplier in (0,1]')
    weights = (args.router_boundary_weight, args.router_pairwise_weight,
               args.router_morphology_weight, args.seg_cldice_weight)
    margins = (args.router_boundary_margin, args.router_pair_margin)
    if any(value < 0 for value in weights + margins):
        parser.error('structure loss weights and margins must be non-negative')
    if args.seg_cldice_iterations < 1:
        parser.error('clDice iterations must be >= 1')
    return args


def main():
    args = parse_args()
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint.get('format_version') != 2:
            raise ValueError('resume requires a three-stage checkpoint (format_version=2)')
        # Resume the saved schedule and split, rather than silently changing stages.
        for key, value in checkpoint['args'].items():
            if key not in {'resume', 'epochs', 'checkpoint_dir', 'run_dir', 'num_workers'}:
                setattr(args, key, value)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, val_data = split_edge_dataset(Path(args.data_root) / args.image_dir,
                                             Path(args.data_root) / args.mask_dir,
                                             val_ratio=args.val_ratio, seed=args.seed)
    split = {'train': [str(p[0].resolve()) for p in train_data.samples],
             'val': [str(p[0].resolve()) for p in val_data.samples]}
    if checkpoint and checkpoint['split'] != split:
        raise ValueError('dataset split changed since checkpoint; refusing resume')
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, generator=generator,
                              num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_data, batch_size=1, num_workers=args.num_workers)
    model = EdgeDynamicViT(selection_threshold=args.selection_threshold).to(device)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    loss_fn = SparseEdgeLoss()
    start_epoch = 0
    best = {'route_f1': float('-inf'), 'foreground_f1': float('-inf')}
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if device.type == 'cuda' and checkpoint['scaler']:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best = checkpoint['best']
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['numpy_state'])
        torch.set_rng_state(checkpoint['torch_state'])
        generator.set_state(checkpoint['loader_state'])
        if device.type == 'cuda' and checkpoint['cuda_states'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_states'])
    if args.epochs <= start_epoch or args.epochs <= args.routing_epochs + args.frozen_epochs:
        raise ValueError('epochs must extend past the restored epoch and frozen stage')
    if checkpoint and 'history' in checkpoint:
        run_dir = Path(args.resume).resolve().parent.parent
        # An older best checkpoint must not overwrite later epochs in this run.
        if Path(args.resume).name != 'latest.pt' and (run_dir / 'checkpoints/latest.pt').exists():
            latest = torch.load(run_dir / 'checkpoints/latest.pt', map_location='cpu', weights_only=False)
            if latest['epoch'] > checkpoint['epoch']:
                raise ValueError('resume latest.pt to continue this run; older best checkpoint would overwrite history')
    else:
        run_dir = create_run(args.run_dir)
    destination = run_dir / 'checkpoints'
    destination.mkdir(parents=True, exist_ok=True)
    history = [row for row in checkpoint.get('history', []) if row['epoch'] <= start_epoch] if checkpoint else []
    save_json(run_dir / 'args.json', vars(args))
    print(f'run directory: {run_dir}', flush=True)
    for epoch in range(start_epoch, args.epochs):
        stage = stage_for_epoch(epoch, args.routing_epochs, args.frozen_epochs)
        configure_stage(model, optimizer, stage, args.learning_rate, args.finetune_lr_multiplier)
        loss_args = dict(router_weight=args.router_weight,
                          boundary_weight=args.router_boundary_weight,
                          pairwise_weight=args.router_pairwise_weight,
                          morphology_weight=args.router_morphology_weight,
                          boundary_margin=args.router_boundary_margin,
                          pair_margin=args.router_pair_margin,
                          cldice_weight=args.seg_cldice_weight,
                          cldice_iterations=args.seg_cldice_iterations)
        training = train_one_epoch(model, train_loader, optimizer, loss_fn, device,
                                   stage=stage, scaler=scaler, **loss_args)
        validation = validate(model, val_loader, loss_fn, device, stage=stage, **loss_args)
        metric = 'route_f1' if stage == 'routing' else 'foreground_f1'
        improved = validation[metric] > best[metric]
        best[metric] = max(best[metric], validation[metric])
        history.append(metric_row(epoch + 1, stage, training, validation))
        save_history(run_dir, history, args.routing_epochs, args.frozen_epochs, args.epochs)
        validation_metrics = dict(validation)
        if stage == 'routing':
            validation_metrics['pixel_loss'] = None
        if stage == 'segmentation':
            validation_metrics['router_loss'] = None
        save_validation(run_dir / 'validation' / f'epoch_{epoch + 1:04d}_{stage}', validation_metrics, stage)
        state = dict(history=history, format_version=2, epoch=epoch, stage=stage, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), best=best,
                     args=vars(args), split=split, random_state=random.getstate(), numpy_state=np.random.get_state(),
                     torch_state=torch.get_rng_state(), loader_state=generator.get_state(),
                     cuda_states=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        torch.save(state, destination / 'latest.pt')
        if improved:
            torch.save(state, destination / ('best_router.pt' if stage == 'routing' else 'best.pt'))
        print(f'epoch={epoch + 1}/{args.epochs} stage={stage} train_loss={training["loss"]:.4f} '
              f'ce={training.get("router_ce_loss", 0.0):.4f} pair={training.get("router_pairwise_loss", 0.0):.4f} '
              f'boundary={training.get("router_boundary_loss", 0.0):.4f} morphology={training.get("router_morphology_loss", 0.0):.4f} '
              f'cldice={training.get("seg_cldice_loss", 0.0):.4f} val_loss={validation["loss"]:.4f} {metric}={validation[metric]:.4f} '
              f'selected_fraction={validation["selected_fraction"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
