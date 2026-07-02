"""Hand-write exercise (Phase 6): inference engineering — the KV cache, the sampling zoo,
groupwise quantization, and the speculative-decoding accept rule.

Fill in the ``NotImplementedError`` bodies so ``tests/exercises/test_phase06_inference.py``
passes. Graded against ``microlab.infer.reference``. See docs/hand-write/phase6-inference.md.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F  # noqa: F401  (you'll want it)


@torch.no_grad()
def generate_cached(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.0,
                    top_k: int | None = None, generator: torch.Generator | None = None):
    """KV-cached generation: build a microlab.infer.reference.kv_cache.KVCache sized to
    the model config, prefill once with the full prompt, then feed ONE token at a time.
    Greedy output must EXACTLY match the uncached reference generate — and be faster."""
    raise NotImplementedError(
        "cache = KVCache(n_layer, B, n_kv_head or n_head, block_size, head_dim); "
        "logits, _ = model(idx, kv_cache=cache); then loop: pick next from logits[:, -1], "
        "append, model(next_token, kv_cache=cache)"
    )
