"""Stage-specific optimization for routing and segmentation."""
from collections import defaultdict

import torch
from torch.nn import functional as F
from metrics import compute_sparse_edge_metrics


def router_loss(keep_logits, patch_targets):
    targets = patch_targets.long().reshape(-1)
    positive = targets.sum().float()
    negative = targets.numel() - positive
    # Both classes retain nonzero weights for all-positive/all-negative batches.
    weights = torch.stack((positive.clamp_min(1), negative.clamp_min(1)))
    weights = weights / weights.max()
    return F.cross_entropy(keep_logits.float().reshape(-1, 2), targets, weight=weights)


def stage_loss(outputs, masks, segmentation_loss, stage, router_weight=1.0):
    if stage not in {"routing", "segmentation", "finetune"}:
        raise ValueError(f"unknown stage: {stage}")
    zero = masks.new_zeros(())
    route = router_loss(outputs['keep_logits'], outputs['patch_targets']) if stage != 'segmentation' else zero
    pixel = segmentation_loss(outputs['mask_logits'], masks) if stage != 'routing' else zero
    total = route if stage == 'routing' else pixel + router_weight * route
    return total, pixel, route


def train_one_epoch(model, loader, optimizer, segmentation_loss, device,
                    stage='finetune', scaler=None, router_weight=1.0):
    model.set_stage(stage)
    model.train()
    totals = defaultdict(float)
    count = 0
    tp = fp = fn = 0
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None and device.type == 'cuda'):
            outputs = model(images, labels=masks, routing_only=stage == 'routing')
            loss, pixel, route = stage_loss(outputs, masks, segmentation_loss, stage, router_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite {stage} loss')
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        for name, value in (('loss', loss), ('pixel_loss', pixel), ('router_loss', route)):
            totals[name] += value.detach().item()
        with torch.no_grad():
            predicted = outputs['keep_probs'] >= model.selector.selection_threshold
            target = outputs['patch_targets'].bool()
            tp += (predicted & target).sum().item()
            fp += (predicted & ~target).sum().item()
            fn += (~predicted & target).sum().item()
            totals['selected_fraction'] += predicted.float().mean().item()
            if stage != 'routing':
                totals['foreground_f1'] += compute_sparse_edge_metrics(outputs['mask_logits'].detach(), masks)['foreground_f1'].item()
        count += 1
    if not count:
        raise ValueError('training loader is empty')
    result = {key: value / count for key, value in totals.items()}
    result['route_f1'] = 2 * tp / max(2 * tp + fp + fn, 1)
    return result
