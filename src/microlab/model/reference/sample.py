"""Autoregressive token sampler for the reference GPT model.

Supports greedy decoding (temperature=0), temperature scaling, and top-k filtering.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Autoregressively extend `idx` (B, T). temperature==0 -> greedy argmax."""
    model.eval()
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if temperature == 0.0:
            idx_next = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
