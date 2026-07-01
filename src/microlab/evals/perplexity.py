"""Perplexity — the pretraining/LM eval metric (exp of mean per-token cross-entropy). This
is the correct eval for a raw pretraining model; the task-check harness needs a capable
(instruction-tuned) model. Lower perplexity = better language modeling."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def evaluate_perplexity(model, data, block_size: int, batch_size: int, iters: int = 200,
                        device: str = "cpu", seed: int = 0) -> float:
    """Mean per-token cross-entropy over `iters` random blocks, returned as perplexity.
    `data` may be a 1-D LongTensor OR anything with `.get_batch(block, batch, device, gen)`
    (e.g. ShardDataset)."""
    from microlab.data.reference.dataset import get_batch as tensor_batch

    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(iters):
        if hasattr(data, "get_batch"):
            x, y = data.get_batch(block_size, batch_size, device, gen)
        else:
            x, y = tensor_batch(data, block_size, batch_size, device, gen)
        _, loss = model(x, y)
        total += loss.item()
    if was_training:
        model.train()
    return math.exp(total / iters)
