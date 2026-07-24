"""Reward model over the production VariantGPT (Phase 11 at scale): the 1B chat backbone with
its LM head swapped for a fresh scalar head, scored at the LAST NON-PAD token of a right-padded
(prompt + response) sequence, trained with the Bradley-Terry pairwise loss.

This is the scale counterpart of model/reference/reward.py (which wraps the tiny learned-pos
GPT); here the backbone is the RoPE/RMSNorm/SwiGLU VariantGPT and sequences are right-padded
batches, so the score must be gathered per-row at ``lengths - 1``, not at ``[:, -1]``. Causal
attention means pad tokens sit strictly AFTER the scored position and cannot leak into it.

Checkpoints follow the repo's servable pattern ({"cfg", "model", "step", ...}), plus a
``"kind": "reward"`` marker so an LM checkpoint can never be half-loaded as a reward model
(and vice versa)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from microlab.model.reference.variants import VariantConfig, VariantGPT

REWARD_CKPT_KIND = "reward"


class RewardModel(nn.Module):
    """A VariantGPT trunk (embedding + blocks + final norm, LM head unused) with a fresh
    scalar head. ``forward`` returns one score per sequence, read at its last non-pad token."""

    def __init__(self, backbone: VariantGPT) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.n_embd, 1, bias=False)
        # Fresh head, small init: scores start near 0 so the Bradley-Terry loss starts at
        # ~log 2 and early updates don't slam the pretrained trunk.
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Score (B,) for right-padded ``input_ids`` (B, T); ``lengths[b]`` = real (non-pad)
        token count of row b. The score is the head applied to the final-norm hidden state at
        position ``lengths[b] - 1`` — the last real token."""
        B, T = input_ids.shape
        cfg = self.backbone.config
        if T > cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {cfg.block_size}")
        if lengths.shape != (B,):
            raise ValueError(f"lengths shape {tuple(lengths.shape)} != ({B},)")
        if lengths.min().item() < 1 or lengths.max().item() > T:
            raise ValueError(f"lengths must be in [1, {T}], got "
                             f"[{lengths.min().item()}, {lengths.max().item()}]")
        # Walk the backbone trunk directly (VariantGPT.forward would also apply the LM head,
        # materializing (B, T, vocab) logits we don't need).
        t = self.backbone.transformer
        x = t.wte(input_ids)
        if cfg.pos == "learned":
            pos = torch.arange(T, device=input_ids.device)
            x = x + t.wpe(pos)
        x = t.drop(x)
        for block in t.h:
            x = block(x)
        x = t.ln_f(x)
        last = x[torch.arange(B, device=x.device), lengths.to(x.device) - 1]  # (B, n_embd)
        return self.head(last).squeeze(-1)


def bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """-log sigmoid(r_chosen - r_rejected), averaged over the batch: the probability the
    chosen response wins under the Bradley-Terry model, maximized."""
    if r_chosen.shape != r_rejected.shape:
        raise ValueError(f"shape mismatch: {tuple(r_chosen.shape)} vs {tuple(r_rejected.shape)}")
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def pairwise_accuracy(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> float:
    """Fraction of pairs where chosen STRICTLY outscores rejected (a tie is not a win)."""
    if r_chosen.shape != r_rejected.shape:
        raise ValueError(f"shape mismatch: {tuple(r_chosen.shape)} vs {tuple(r_rejected.shape)}")
    return (r_chosen > r_rejected).float().mean().item()


def collate_reward(seqs: list[list[int]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Right-pad token sequences to the batch max. Returns {"input_ids": (B, T) long,
    "lengths": (B,) long} — lengths are recorded here, at collate time, rather than inferred
    from pad_id later (pad_id could be a legitimate in-sequence token)."""
    if any(len(s) == 0 for s in seqs):
        raise ValueError("collate_reward got an empty sequence (nothing to score)")
    maxlen = max(len(s) for s in seqs)
    input_ids = [s + [pad_id] * (maxlen - len(s)) for s in seqs]
    return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
            "lengths": torch.tensor([len(s) for s in seqs], dtype=torch.long)}


def save_reward_checkpoint(path: str | Path, model: RewardModel, step: int,
                           extra: dict | None = None) -> None:
    """Servable-pattern checkpoint: backbone cfg + full RewardModel state_dict + step, marked
    kind="reward". ``extra`` merges additional provenance (e.g. base_ckpt). The optimizer state
    is deliberately NOT saved (it would triple the file; these runs are minutes-to-hours and
    restart from the base checkpoint, not resume)."""
    payload = {"model": model.state_dict(), "cfg": model.backbone.config, "step": step,
               "kind": REWARD_CKPT_KIND}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_reward_checkpoint(path: str | Path, device: str = "cpu") -> tuple[RewardModel, int]:
    """Rebuild the RewardModel from a save_reward_checkpoint file. Refuses non-reward
    checkpoints loudly. Loads to CPU first, then moves to ``device`` (same rationale as
    load_variant_from_run)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") != REWARD_CKPT_KIND:
        raise ValueError(f"{path} is not a reward-model checkpoint "
                         f"(kind={ckpt.get('kind')!r}); refusing to load it as one")
    cfg = ckpt["cfg"]
    backbone = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, norm=cfg.norm, pos=cfg.pos,
        mlp=cfg.mlp, n_kv_head=getattr(cfg, "n_kv_head", None)))
    model = RewardModel(backbone)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["step"]
