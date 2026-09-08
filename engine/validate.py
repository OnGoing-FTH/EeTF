"""Stage-aware validation without pixel decoding during routing pretraining."""
from collections import defaultdict

import torch

from engine.train import stage_loss
from metrics import compute_sparse_edge_metrics


@torch.inference_mode()
def validate(model, loader, segmentation_loss, device, stage='finetune', router_weight=1.0):
    model.eval()
    totals = defaultdict(float)
    count = 0
    tp = fp = fn = tn = 0
    for batch in loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        outputs = model(images, labels=masks, routing_only=stage == 'routing')
        loss, pixel, route = stage_loss(outputs, masks, segmentation_loss, stage, router_weight)
        for name, value in (('loss', loss), ('pixel_loss', pixel), ('router_loss', route)):
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
