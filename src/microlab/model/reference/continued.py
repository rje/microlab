"""Reference continued-pretraining tools (Phase 8): measure catastrophic forgetting when a
model is trained on a new domain, and mitigate it with replay (rehearsing old-domain data).
The oracle the owner diffs hand-written forgetting/replay against."""

from __future__ import annotations

import torch

from microlab.model.reference.train import TrainConfig, estimate_loss, train


def evaluate_on_corpora(
    model: torch.nn.Module, corpora: dict[str, torch.Tensor], block_size: int,
    batch_size: int, iters: int = 20, device: str = "cpu",
) -> dict[str, float]:
    """Held-out loss on each named corpus (e.g. {'shakespeare': ..., 'sherlock': ...})."""
    return {
        name: estimate_loss(model, data, block_size, batch_size, iters=iters, device=device)
        for name, data in corpora.items()
    }


def forgetting_score(loss_before: float, loss_after: float) -> float:
    """Increase in loss on the ORIGINAL domain after continued training. Positive means the
    model forgot; ~0 means it retained; negative means it even improved there."""
    return loss_after - loss_before


def build_replay_mix(
    new_tokens: torch.Tensor, old_tokens: torch.Tensor, replay_fraction: float
) -> torch.Tensor:
    """Combine new-domain tokens with a `replay_fraction` slice of old-domain tokens so
    continued training rehearses the old domain. replay_fraction is the target share of
    OLD tokens in the result. 0.0 -> just the new tokens."""
    assert 0.0 <= replay_fraction < 1.0, "replay_fraction must be in [0, 1)"
    if replay_fraction == 0.0:
        return new_tokens
    n_new = len(new_tokens)
    n_old = min(round(replay_fraction / (1.0 - replay_fraction) * n_new), len(old_tokens))
    return torch.cat([new_tokens, old_tokens[:n_old]])


def continued_pretrain(
    model: torch.nn.Module, new_data: torch.Tensor, eval_corpora: dict[str, torch.Tensor],
    train_cfg: TrainConfig, replay_data: torch.Tensor | None = None,
    replay_fraction: float = 0.0,
) -> dict:
    """Evaluate on all corpora, continue-train on new_data (optionally mixed with replay),
    re-evaluate, and report per-corpus forgetting. eval_corpora should include the ORIGINAL
    domain (to measure forgetting) and the NEW domain (to measure learning)."""
    bs, bsz, dev = train_cfg.block_size, train_cfg.batch_size, train_cfg.device
    before = evaluate_on_corpora(model, eval_corpora, bs, bsz, device=dev)
    data = (
        build_replay_mix(new_data, replay_data, replay_fraction)
        if replay_data is not None and replay_fraction > 0.0
        else new_data
    )
    stats = train(model, data, train_cfg)
    after = evaluate_on_corpora(model, eval_corpora, bs, bsz, device=dev)
    forgetting = {name: forgetting_score(before[name], after[name]) for name in eval_corpora}
    return {"before": before, "after": after, "forgetting": forgetting, "train": stats}
