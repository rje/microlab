"""Reference speculative-decoding accept rule (Phase 6, Leviathan et al. 2022): a cheap
draft model proposes K tokens; the target model verifies them in ONE forward. Accepting
with prob min(1, p_target/p_draft) and resampling rejections from the positive residual
provably samples from the target distribution — free speedup, zero quality loss."""

from __future__ import annotations

import torch


def speculative_accept(draft_tokens: torch.Tensor, draft_probs: torch.Tensor,
                       target_probs: torch.Tensor, generator: torch.Generator):
    """draft_tokens (K,), draft/target probs (K, V). Returns (n_accepted, correction):
    correction is a token sampled from normalize(max(0, target - draft)) at the first
    rejected position, or None when all K drafts are accepted."""
    k = draft_tokens.size(0)
    for i in range(k):
        tok = draft_tokens[i]
        p_d = draft_probs[i, tok].clamp(min=1e-12)
        p_t = target_probs[i, tok]
        u = torch.rand((), generator=generator)
        if u <= torch.clamp(p_t / p_d, max=1.0):
            continue
        residual = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
        residual = residual / residual.sum().clamp(min=1e-12)
        fix = torch.multinomial(residual, 1, generator=generator)[0]
        return i, fix
    return k, None
