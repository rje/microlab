"""Reference interpretability tools (Phase 5): residual-stream capture, the logit lens,
attention-pattern extraction, and induction-head scoring — run against the from-scratch
models whose every weight we own. The oracle the owner diffs hand-written lenses against."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from microlab.model.reference.variants import VariantGPT, apply_rope


@torch.no_grad()
def collect_residual_stream(model: VariantGPT, idx: torch.Tensor) -> list[torch.Tensor]:
    """Mirror VariantGPT.forward, keeping the residual stream after embedding and after
    each block. Returns n_layer+1 tensors of shape (B, T, C)."""
    x = model.transformer.wte(idx)
    if model.config.pos == "learned":
        pos = torch.arange(idx.size(1), device=idx.device)
        x = x + model.transformer.wpe(pos)
    x = model.transformer.drop(x)
    stream = [x]
    for block in model.transformer.h:
        x = block(x)
        stream.append(x)
    return stream


@torch.no_grad()
def logit_lens(residuals, ln_f, lm_head) -> torch.Tensor:
    """Decode EVERY layer's residual state through the model's own final norm + unembed:
    what would the model predict if it had to stop here? Returns (L+1, B, T, V)."""
    return torch.stack([lm_head(ln_f(r)) for r in residuals])


@torch.no_grad()
def attention_patterns(model: VariantGPT, idx: torch.Tensor) -> torch.Tensor:
    """Recompute softmax attention probabilities per layer/head for the RoPE block (SDPA
    never materializes them). Returns (n_layer, n_head, T, T)."""
    assert model.config.pos == "rope", "attention_patterns supports the RoPE block"
    x = model.transformer.drop(model.transformer.wte(idx))
    B, T = idx.shape
    out = []
    for block in model.transformer.h:
        h = block.ln_1(x)
        a = block.attn
        if hasattr(a, "q_proj"):  # GQAAttention
            q = a.q_proj(h).view(B, T, a.n_head, a.head_dim).transpose(1, 2)
            k, _ = a.kv_proj(h).split(a.n_kv_head * a.head_dim, dim=2)
            k = k.view(B, T, a.n_kv_head, a.head_dim).transpose(1, 2)
            k = k.repeat_interleave(a.n_head // a.n_kv_head, dim=1)
        else:  # RoPECausalSelfAttention
            q, k, _ = a.c_attn(h).split(a.n_embd, dim=2)
            q = q.view(B, T, a.n_head, a.n_embd // a.n_head).transpose(1, 2)
            k = k.view(B, T, a.n_head, a.n_embd // a.n_head).transpose(1, 2)
        q = apply_rope(q, a.rope_cos.to(q.dtype), a.rope_sin.to(q.dtype))
        k = apply_rope(k, a.rope_cos.to(k.dtype), a.rope_sin.to(k.dtype))
        scores = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        out.append(F.softmax(scores, dim=-1)[0])
        x = block(x)
    return torch.stack(out)


def repeated_token_sequence(
    vocab_size: int, period: int, repeats: int, generator: torch.Generator
) -> torch.Tensor:
    """A random block of `period` tokens tiled `repeats` times — the classic induction
    probe: after the first repetition, [A B] ... [A ?] is predictable by copying."""
    block = torch.randint(0, vocab_size, (period,), generator=generator)
    return block.repeat(repeats).unsqueeze(0)


def induction_score(attn: torch.Tensor, period: int) -> torch.Tensor:
    """Mean attention mass on the induction target: from position i, the token AFTER the
    previous occurrence of the current token, i.e. offset i - period + 1. Scores near 1
    mean 'this head is an induction head'. attn: (..., T, T) -> (...)."""
    T = attn.size(-1)
    idx = torch.arange(period, T)
    return attn[..., idx, idx - period + 1].mean(-1)
