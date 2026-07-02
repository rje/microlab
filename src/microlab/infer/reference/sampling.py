"""Reference next-token sampling (Phase 6): temperature, top-k, and top-p (nucleus) in
the standard order — scale, filter, filter, softmax, sample."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """logits (B, V) -> next token ids (B, 1). temperature=0 is greedy argmax."""
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # drop tokens once the cumulative prob BEFORE them already reached top_p
        drop = (cum - F.softmax(sorted_logits, dim=-1)) >= top_p
        sorted_logits = sorted_logits.masked_fill(drop, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
