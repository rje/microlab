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

    def kv_bytes(self) -> int:
        """Bytes of K/V actually occupied at the current position (not the preallocated
        capacity). Defined here so a dense cache and a HybridCache can be measured by the
        same call — the memory comparison between them is the whole point."""
        used = 0
        for t in self.k + self.v:
            if t is not None:
                used += t.element_size() * t[:, :, : max(self.seq_len, 1)].numel()
        return used

    def state_bytes(self) -> int:
        """Recurrent-state bytes. Always 0 for pure attention: that asymmetry IS the
        architectural difference being measured."""
        return 0


class HybridCache(KVCache):
    """Cache for a GDN/KDA hybrid: K/V buffers ONLY for the global-attention layers, plus a
    fixed-size recurrent state (and a tiny conv history) for each GatedDeltaNet layer.

    This is where the architecture's claim lives. A KV buffer grows with context; the GDN
    state is (n_head, head_dim, head_dim) per layer and never grows at all. `kv_bytes()`
    and `state_bytes()` report what was actually allocated, so the memory claim can be
    measured instead of computed on paper.

    `linear_layers` is the set of layer indices that are GatedDeltaNet."""

    def __init__(self, n_layer: int, batch_size: int, n_kv_head: int, capacity: int,
                 head_dim: int, linear_layers: set[int], n_head: int,
                 dtype=torch.float32, device="cpu") -> None:
        self.n_layer = n_layer
        self.capacity = capacity
        self.seq_len = 0
        self.linear_layers = set(linear_layers)
        # seq_len is advanced by the LAST layer's append() (inherited protocol), so the
        # last layer must be a global-attention one or the position counter never moves.
        # True for every n_layer divisible by hybrid_every; assert rather than silently
        # generate garbage if that ever stops holding.
        if (n_layer - 1) in self.linear_layers:
            raise ValueError(
                f"last layer ({n_layer - 1}) is a GatedDeltaNet layer; HybridCache needs "
                "the final layer to be global attention so seq_len advances"
            )
        self._dtype = dtype
        self._device = device
        shape = (batch_size, n_kv_head, capacity, head_dim)
        # None for linear layers — the whole point is that these are never allocated.
        self.k = [None if i in self.linear_layers
                  else torch.zeros(shape, dtype=dtype, device=device)
                  for i in range(n_layer)]
        self.v = [None if i in self.linear_layers
                  else torch.zeros(shape, dtype=dtype, device=device)
                  for i in range(n_layer)]
        self._state: dict[int, torch.Tensor] = {}
        self._conv: dict[int, torch.Tensor] = {}
        self._n_head = n_head
        self._head_dim = head_dim

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        if self.k[layer] is None:
            raise AssertionError(
                f"layer {layer} is a GatedDeltaNet layer and has no K/V buffer"
            )
        t = k.size(2)
        assert self.seq_len == 0 or t == 1, "full prefill or single-token steps only"
        assert self.seq_len + t <= self.capacity, "KV cache overflow"
        self.k[layer][:, :, self.seq_len:self.seq_len + t] = k
        self.v[layer][:, :, self.seq_len:self.seq_len + t] = v
        out = (self.k[layer][:, :, : self.seq_len + t],
               self.v[layer][:, :, : self.seq_len + t])
        if layer == self.n_layer - 1:
            self.seq_len += t
        return out

    def gdn_state(self, layer: int, B: int, n_head: int, head_dim: int, dtype, device):
        if layer not in self._state:
            self._state[layer] = torch.zeros(B, n_head, head_dim, head_dim,
                                             dtype=dtype, device=device)
        return self._state[layer]

    def set_gdn_state(self, layer: int, S: torch.Tensor) -> None:
        self._state[layer] = S

    def conv_hist(self, layer: int, B: int, channels: int, width: int, dtype, device):
        """Depthwise-conv history. Initialised to `width` ZEROS, not an empty tensor, so
        the first (prefill) call is exactly equivalent to the uncached path's causal
        zero-padding — that equivalence is what makes cached and uncached generation
        produce identical tokens."""
        if layer not in self._conv:
            self._conv[layer] = torch.zeros(B, channels, width, dtype=dtype, device=device)
        return self._conv[layer]

    def set_conv_hist(self, layer: int, h: torch.Tensor) -> None:
        self._conv[layer] = h

    def state_bytes(self) -> int:
        """Recurrent state + conv history. Overrides the base class's 0. This should be
        CONSTANT in context length — that is the claim under test."""
        return sum(t.element_size() * t.numel()
                   for t in list(self._state.values()) + list(self._conv.values()))


def build_cache(model, batch_size: int, device, dtype=torch.float32):
    """KVCache for a dense model, HybridCache when the model has GatedDeltaNet layers."""
    cfg = model.config
    n_kv = cfg.n_kv_head if getattr(cfg, "n_kv_head", None) else cfg.n_head
    head_dim = cfg.n_embd // cfg.n_head
    linear = {i for i, b in enumerate(model.transformer.h) if getattr(b, "is_linear", False)}
    if not linear:
        return KVCache(cfg.n_layer, batch_size, n_kv, cfg.block_size, head_dim,
                       dtype=dtype, device=device)
    return HybridCache(cfg.n_layer, batch_size, n_kv, cfg.block_size, head_dim,
                       linear_layers=linear, n_head=cfg.n_head, dtype=dtype, device=device)


@torch.no_grad()
def generate_cached(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.0,
                    top_k: int | None = None, generator: torch.Generator | None = None):
    """Autoregressive generation with a KV cache: one full prefill, then one-token steps.
    Greedy (temperature=0) output is token-for-token identical to the uncached
    microlab.model.reference.sample.generate."""
    model.eval()
    cfg = model.config
    cache = build_cache(model, idx.size(0), idx.device)
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
