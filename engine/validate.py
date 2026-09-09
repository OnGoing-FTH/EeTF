"""Stage-aware validation without pixel decoding during routing pretraining."""
from collections import defaultdict

import torch

from engine.train import stage_loss
from metrics import compute_sparse_edge_metrics


@torch.inference_mode()
def validate(model, loader, segmentation_loss, device, stage='finetune', router_weight=1.0,
             boundary_weight=0.2, pairwise_weight=0.1, morphology_weight=0.1,
             boundary_margin=0.1, pair_margin=0.2, cldice_weight=0.2,
             cldice_iterations=10):
    model.eval()
    totals = defaultdict(float)
    count = 0
    tp = fp = fn = tn = 0
    for batch in loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        outputs = model(images, labels=masks, routing_only=stage == 'routing')
        outputs['selection_threshold'] = model.selector.selection_threshold
        loss, pixel, route, segmentation, route_terms, structure_terms = stage_loss(
            outputs, masks, segmentation_loss, stage, router_weight,
            boundary_weight, pairwise_weight, morphology_weight,
            boundary_margin, pair_margin, cldice_weight, cldice_iterations)
        for name, value in (
                ('loss', loss), ('pixel_loss', pixel), ('segmentation_loss', segmentation),
                ('router_loss', route),
                *[(f'router_{key}_loss', value) for key, value in route_terms.items()],
                *[(f'seg_{key}_loss', value) for key, value in structure_terms.items()]):
            totals[name] += value.item()
        predicted = outputs['keep_probs'] >= model.selector.selection_threshold
        target = outputs['patch_targets'].bool()
        tn += (~predicted & ~target).sum().item()
        tp += (predicted & target).sum().item()
        fp += (predicted & ~target).sum().item()
        fn += (~predicted & target).sum().item()
        totals['selected_fraction'] += predicted.float().mean().item()
        if stage != 'routing':
            for name, value in compute_sparse_edge_metrics(outputs['mask_logits'], masks).items():
                totals[name] += value.item()
        count += 1
    if not count:
        raise ValueError('validation loader is empty')
    result = {key: value / count for key, value in totals.items()}
    result.update(route_precision=tp / max(tp + fp, 1), route_recall=tp / max(tp + fn, 1),
                  route_f1=2 * tp / max(2 * tp + fp + fn, 1))
    result['route_confusion'] = [[tn, fp], [fn, tp]]
    return result
