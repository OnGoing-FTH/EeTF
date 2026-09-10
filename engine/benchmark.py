"""Forward-only latency measurement with inputs already on device."""
from time import perf_counter

import torch


@torch.inference_mode()
def benchmark_forward(model, inputs, warmup=10, iterations=50):
    """Time model(inputs) only; caller performs preprocessing and transfer.

    Synchronized wall time includes CPU dispatch and model-internal operations,
    including dynamic routing. No output sigmoid, transfer or postprocessing.
    """
    if warmup < 0 or iterations < 1:
        raise ValueError('warmup must be >= 0 and iterations must be >= 1')
    if inputs.ndim != 4 or inputs.shape[0] < 1:
        raise ValueError('benchmark expects inputs with a positive batch dimension')
    model.eval()
    cuda = inputs.device.type == 'cuda'
    for _ in range(warmup):
        result = model(inputs)
        del result
    if cuda:
        torch.cuda.synchronize(inputs.device)
    seconds = []
    for _ in range(iterations):
        start = perf_counter()
        result = model(inputs)
        if cuda:
            torch.cuda.synchronize(inputs.device)
        seconds.append(perf_counter() - start)
        del result
    total = sum(seconds)
    batch_size = inputs.shape[0]
    return {'shape': list(inputs.shape), 'batch_size': batch_size, 'dtype': str(inputs.dtype),
            'device': str(inputs.device), 'warmup': warmup, 'iterations': iterations,
            'total_seconds': total, 'mean_latency_ms': total * 1000 / iterations,
            'batch_fps': iterations / total, 'image_fps': iterations * batch_size / total,
            'fps': iterations * batch_size / total}
