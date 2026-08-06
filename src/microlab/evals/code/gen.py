"""Greedy/sampled generation with stop-string early exit, on top of the reference
KVCache. generate_cached always runs max_new_tokens steps; eval completions usually hit a
stop marker ("### End", a top-level `def `, ...) long before that, so stopping early is a
several-fold throughput win across a few hundred tasks. Greedy decoding here is
step-for-step identical to microlab.infer.reference.kv_cache.generate_cached."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from microlab.infer.reference.kv_cache import build_cache


@torch.no_grad()
def generate_until(
    model,
    tok,
    prompt_ids: list[int],
    *,
    max_new: int,
    stops: list[str],
    device: str,
    temperature: float = 0.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> str:
    """Generate up to max_new tokens after prompt_ids; return the decoded completion
    truncated at the earliest stop string (the stop itself is not included). Raises when
    the prompt + budget cannot fit the model's context — size the prompt, don't clip it."""
    cfg = model.config
    if len(prompt_ids) + max_new > cfg.block_size:
        raise ValueError(
            f"prompt ({len(prompt_ids)} tokens) + max_new ({max_new}) exceeds "
            f"block_size {cfg.block_size}; shrink the prompt or the budget"
        )
    # build_cache, not a hand-rolled KVCache: hybrid (KDA/MLA) models need HybridCache's
    # conv_hist/linear state — the same crash the console's serve path hit at the 4k
    # milestone. The cache is block_size-long rather than prompt+max_new; the guard
    # above already bounds that.
    cache = build_cache(model, 1, device)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, _ = model(idx, kv_cache=cache)

    out_ids: list[int] = []
    for _ in range(max_new):
        step = logits[:, -1, :]
        if temperature == 0.0:
            nxt = step.argmax(dim=-1, keepdim=True)
        else:
            step = step / temperature
            if top_k is not None:
                v, _ = torch.topk(step, min(top_k, step.size(-1)))
                step[step < v[:, [-1]]] = -float("inf")
            probs = F.softmax(step, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
        out_ids.append(int(nxt.item()))
        text = tok.decode(out_ids)
        cut = min((text.find(s) for s in stops if s in text), default=-1)
        if cut >= 0:
            return text[:cut]
        logits, _ = model(nxt, kv_cache=cache)
    return tok.decode(out_ids)
