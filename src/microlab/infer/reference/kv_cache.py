"""Reference KV cache (Phase 6): the single most important inference optimization.
Without it, generating token T recomputes attention keys/values for all T-1 previous
tokens; with it, each new token costs one forward over ONE position. This is also why
inference is memory-bound — the cache is (n_layer, B, n_kv_head, T, head_dim) big — and
why GQA (Phase 3) exists: fewer KV heads, smaller cache."""

from __future__ import annotations

import torch
from torch.nn import functional as F


class KVCache:
    """Preallocated per-layer K/V buffers. append() writes new keys/values at the current
    position and returns full views; the LAST layer's append advances seq_len (all layers
    see the same positions each step)."""

    def __init__(self, n_layer: int, batch_size: int, n_kv_head: int, capacity: int,
                 head_dim: int, dtype=torch.float32, device="cpu") -> None:
        self.n_layer = n_layer
        self.capacity = capacity
        self.seq_len = 0
        shape = (batch_size, n_kv_head, capacity, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layer)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layer)]

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        t = k.size(2)
        assert self.seq_len == 0 or t == 1, "append supports full prefill or single-token steps"
        assert self.seq_len + t <= self.capacity, "KV cache overflow"
        self.k[layer][:, :, self.seq_len:self.seq_len + t] = k
        self.v[layer][:, :, self.seq_len:self.seq_len + t] = v
        k_all = self.k[layer][:, :, : self.seq_len + t]
        v_all = self.v[layer][:, :, : self.seq_len + t]
        if layer == self.n_layer - 1:
            self.seq_len += t
        return k_all, v_all


@torch.no_grad()
def generate_cached(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.0,
                    top_k: int | None = None, generator: torch.Generator | None = None):
    """Autoregressive generation with a KV cache: one full prefill, then one-token steps.
    Greedy (temperature=0) output is token-for-token identical to the uncached
    microlab.model.reference.sample.generate."""
    model.eval()
    cfg = model.config
    n_kv = cfg.n_kv_head if getattr(cfg, "n_kv_head", None) else cfg.n_head
    cache = KVCache(cfg.n_layer, idx.size(0), n_kv, cfg.block_size,
                    cfg.n_embd // cfg.n_head, device=idx.device)
    logits, _ = model(idx, kv_cache=cache)
    for _ in range(max_new_tokens):
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
        idx = torch.cat((idx, nxt), dim=1)
        if cache.seq_len >= cfg.block_size:
            break  # context full — matches uncached crop-free semantics up to block_size
        logits, _ = model(nxt, kv_cache=cache)
    return idx
