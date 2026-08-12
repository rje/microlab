"""Batched same-prompt generation: n samples of one prompt decoded as one batch.

The sampled-eval and best-of-k workloads generate n completions of the SAME prompt
(eval_code --n, generate_best_of --k). Sequential generation pays the full
weight-bandwidth cost per sample; decoding all n as one batch reads the weights once per
step, so throughput scales nearly with n until the GPU saturates.

Semantics vs microlab.evals.code.gen.generate_until:
- Greedy (temperature=0) rows are step-for-step identical to generate_until — locked by
  tests. (All greedy rows are identical to each other; n>1 greedy is only useful there.)
- Sampled rows draw from ONE shared generator in slot order each step, so the stream a
  row sees depends on n and on when other rows finish: batched output is deterministic
  for a fixed (seed, n, config) but is a DIFFERENT sampling scheme than sequential
  per-sample generators. Result files must therefore carry an engine marker in their
  config header — resuming a sequential file with the batched engine has to raise, not
  silently mix schemes.
- Stop handling matches generate_until exactly: each row is truncated at the earliest
  stop-string occurrence in its decoded text; the stop itself is excluded. A finished row
  keeps stepping (batch shape is fixed) but its later tokens are discarded and its
  completion is frozen at detection time.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from microlab.infer.reference.kv_cache import build_cache


@torch.no_grad()
def generate_batch(
    model,
    tok,
    prompt_ids: list[int],
    *,
    n: int,
    max_new: int,
    stops: list[str],
    device: str,
    temperature: float = 0.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> list[str]:
    """n completions of one prompt, decoded as a single batch. Returns completions in
    slot order. Raises when the prompt + budget cannot fit the model's context."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    cfg = model.config
    if len(prompt_ids) + max_new > cfg.block_size:
        raise ValueError(
            f"prompt ({len(prompt_ids)} tokens) + max_new ({max_new}) exceeds "
            f"block_size {cfg.block_size}; shrink the prompt or the budget"
        )
    dtype = next(model.parameters()).dtype
    cache = build_cache(model, n, device, dtype=dtype)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device).expand(n, -1)
    logits, _ = model(idx.contiguous(), kv_cache=cache)

    out_ids: list[list[int]] = [[] for _ in range(n)]
    done: list[str | None] = [None] * n
    for _ in range(max_new):
        step = logits[:, -1, :].float()
        if temperature == 0.0:
            nxt = step.argmax(dim=-1, keepdim=True)
        else:
            step = step / temperature
            if top_k is not None:
                v, _ = torch.topk(step, min(top_k, step.size(-1)))
                step[step < v[:, [-1]]] = -float("inf")
            probs = F.softmax(step, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
        toks = nxt.squeeze(1).tolist()
        for i in range(n):
            if done[i] is not None:
                continue
            out_ids[i].append(toks[i])
            text = tok.decode(out_ids[i])
            cut = min((text.find(s) for s in stops if s in text), default=-1)
            if cut >= 0:
                done[i] = text[:cut]
        if all(d is not None for d in done):
            break
        logits, _ = model(nxt, kv_cache=cache)
    return [done[i] if done[i] is not None else tok.decode(out_ids[i]) for i in range(n)]
