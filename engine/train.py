"""Stage-specific optimization for routing and segmentation."""
from collections import defaultdict

import torch
from torch.nn import functional as F
from metrics import compute_sparse_edge_metrics


def _grid_edges(values):
    """Return horizontal and vertical neighboring values from [B,H,W]."""
    horizontal = (values[:, :, :-1], values[:, :, 1:])
    vertical = (values[:, :-1, :], values[:, 1:, :])
    return horizontal, vertical


def routing_loss_components(keep_logits, patch_targets, patch_grid, selection_threshold=0.5,
                           boundary_margin=0.1, pair_margin=0.2):
    """Return CE, neighbor, threshold-boundary and mismatch-isolation losses."""
    targets = patch_targets.long().reshape(-1)
    positive = targets.sum().float()
    negative = targets.numel() - positive
    weights = torch.stack((positive.clamp_min(1), negative.clamp_min(1))
                          ).to(keep_logits.device)
    weights = weights / weights.max()
    ce = F.cross_entropy(keep_logits.float().reshape(-1, 2), targets, weight=weights)

    probs = keep_logits.float().softmax(-1)[..., 1]
    batch_size = probs.shape[0]
    height, width = patch_grid
    if probs.shape[1] != height * width:
        raise ValueError('patch_grid does not match router token count')
    p = probs.reshape(batch_size, height, width)
    y = patch_targets.float().reshape(batch_size, height, width)

    pair_terms = []
    for (p0, p1), (y0, y1) in zip(_grid_edges(p), _grid_edges(y)):
        same = (y0 == y1)
        same_loss = (p0 - p1).square()
        different_loss = F.relu(pair_margin - (p0 - p1).abs()).square()
        term = torch.where(same, same_loss, different_loss).reshape(-1)
        if term.numel():
            pair_terms.append(term)
    pairwise = torch.cat(pair_terms).mean() if pair_terms else p.new_zeros(())

    threshold = p.new_tensor(selection_threshold)
    positive_boundary = F.relu(threshold + boundary_margin - p).square()
    negative_boundary = F.relu(p - (threshold - boundary_margin)).square()
    boundary = torch.where(y > 0.5, positive_boundary, negative_boundary).mean()

    predicted = (p >= threshold).float()
    mismatch = predicted.ne(y)
    neighbor_count = F.conv2d(predicted.unsqueeze(1), p.new_tensor([[ [ [0., 1., 0.], [1., 0., 1.], [0., 1., 0.] ] ]]), padding=1).squeeze(1)
    isolated_mismatch = mismatch & (neighbor_count == 0)
    morphology = torch.where(predicted > 0.5, p.square(), (1.0 - p).square())
    morphology = morphology.masked_select(isolated_mismatch).mean() if isolated_mismatch.any() else p.new_zeros(())
    return {'ce': ce, 'pairwise': pairwise, 'boundary': boundary, 'morphology': morphology}


def router_loss(keep_logits, patch_targets, patch_grid=None, selection_threshold=0.5,
                boundary_weight=0.2, pairwise_weight=0.1, morphology_weight=0.1,
                boundary_margin=0.1, pair_margin=0.2):
    """Weighted CE plus label-aware spatial supervision for the Patch Router."""
    if patch_grid is None:
        # Backward-compatible CE-only call for external users without grid metadata.
        targets = patch_targets.long().reshape(-1)
        positive = targets.sum().float()
        negative = targets.numel() - positive
        weights = torch.stack((positive.clamp_min(1), negative.clamp_min(1))).to(keep_logits.device)
        weights = weights / weights.max()
        return F.cross_entropy(keep_logits.float().reshape(-1, 2), targets, weight=weights)
    terms = routing_loss_components(keep_logits, patch_targets, patch_grid, selection_threshold,
                                    boundary_margin, pair_margin)
    return terms['ce'] + pairwise_weight * terms['pairwise'] + boundary_weight * terms['boundary'] + morphology_weight * terms['morphology']


def stage_loss(outputs, masks, segmentation_loss, stage, router_weight=1.0,
               boundary_weight=0.2, pairwise_weight=0.1, morphology_weight=0.1,
               boundary_margin=0.1, pair_margin=0.2):
    if stage not in {"routing", "segmentation", "finetune"}:
        raise ValueError(f"unknown stage: {stage}")
    zero = masks.new_zeros(())
    terms = ({key: zero for key in ('ce', 'pairwise', 'boundary', 'morphology')}
             if stage == 'segmentation' else routing_loss_components(
                 outputs['keep_logits'], outputs['patch_targets'], outputs['patch_grid'],
                 outputs.get('selection_threshold', 0.5), boundary_margin, pair_margin))
    route = terms['ce'] + pairwise_weight * terms['pairwise'] + boundary_weight * terms['boundary'] + morphology_weight * terms['morphology']
    pixel = segmentation_loss(outputs['mask_logits'], masks) if stage != 'routing' else zero
    total = route if stage == 'routing' else pixel + router_weight * route
    return total, pixel, route, terms


def train_one_epoch(model, loader, optimizer, segmentation_loss, device,
                    stage='finetune', scaler=None, router_weight=1.0,
                    boundary_weight=0.2, pairwise_weight=0.1, morphology_weight=0.1,
                    boundary_margin=0.1, pair_margin=0.2):
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
            outputs['selection_threshold'] = model.selector.selection_threshold
            loss, pixel, route, terms = stage_loss(outputs, masks, segmentation_loss, stage, router_weight,
                                                    boundary_weight, pairwise_weight, morphology_weight,
                                                    boundary_margin, pair_margin)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite {stage} loss')
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        for name, value in (('loss', loss), ('pixel_loss', pixel), ('router_loss', route),
                            *[(f'router_{key}_loss', value) for key, value in terms.items()]):
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
