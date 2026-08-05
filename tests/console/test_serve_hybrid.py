"""The Playground must serve HYBRID models, not just dense ones.

The generate endpoint hand-rolled a plain KVCache, which predated the hybrid runs: the
first coder-1b milestone anyone prompted crashed with "'KVCache' object has no attribute
'conv_hist'" — after silently building the WRONG architecture entirely on a stale
service. Serving goes through build_cache, which picks per the model's layers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from microlab.infer.reference.kv_cache import HybridCache, build_cache  # noqa: E402
from microlab.model.reference.variants import VariantConfig, VariantGPT  # noqa: E402


def _hybrid():
    return VariantGPT(VariantConfig(
        vocab_size=64, block_size=32, n_layer=4, n_head=2, n_embd=32,
        norm="rms", pos="nope", mlp="swiglu", block_norm="peri",
        hybrid_every=4, gdn_gate="channel", global_attn="mla", mla_kv_lora=16,
        qk_norm=True, gdn_fused=False)).eval()


def test_build_cache_returns_a_hybrid_cache_for_hybrid_models():
    m = _hybrid()
    cache = build_cache(m, 1, "cpu")
    assert isinstance(cache, HybridCache), type(cache).__name__


def test_hybrid_generation_through_the_serve_cache_path():
    """The exact op sequence the Playground endpoint runs: prefill with a fresh cache,
    then single-token steps. This is what crashed."""
    m = _hybrid()
    cache = build_cache(m, 1, "cpu")
    idx = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        logits, _ = m(idx, kv_cache=cache)
        for _ in range(4):
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            logits, _ = m(nxt, kv_cache=cache)
    assert cache.seq_len == 12


def test_console_serve_uses_the_factory_not_a_hand_rolled_cache():
    src = (Path(__file__).resolve().parents[2] / "src" / "microlab" / "console"
           / "serve.py").read_text()
    assert "build_cache(" in src
    assert "KVCache(" not in src, \
        "a hand-rolled KVCache in serve.py is how hybrid serving broke"
