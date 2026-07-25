"""Convert an MHA VariantGPT checkpoint to GQA via the Ainslie et al. (2023) mean-pool
recipe ("GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
Checkpoints", arXiv 2305.13245): the K and V head projections are mean-pooled into
n_kv_head groups (the paper found mean-pooling beats picking the first head or random
init), Q and everything else are untouched. The pooled model is degraded by construction
and must be uptrained (~alpha=0.05 of pretraining compute in the paper) to recover.

    python scripts/convert_gqa.py runs/1b/ckpt_40000.pt out/converted.pt --n-kv-head 2 \
        --kl-data-dir data/shards/fineweb-100bt --device cuda

Output is a WEIGHTS checkpoint: {"model", "cfg", "converted_from", "conversion"} — no
optimizer state (the source run's optimizer moments have MHA shapes and the uptrain uses
Muon anyway) and no RNG state. Warm-start it with `pretrain.py --init-ckpt`.

Authority rule: the embedded cfg (source cfg + n_kv_head) is METADATA describing the
weights. At uptrain time the run config wins; the strict state-dict load in
`pretrain.py --init-ckpt` asserts the shapes agree and fails loudly when they don't.
"""

from __future__ import annotations

import argparse
import dataclasses
import re

import torch
from torch.nn import functional as F

from microlab.model.reference.variants import VariantConfig, VariantGPT

_CATTN_W = re.compile(r"^transformer\.h\.(\d+)\.attn\.c_attn\.weight$")


def pool_heads(
    t: torch.Tensor, *, n_head: int, n_kv_head: int, scale_correct: bool = False
) -> torch.Tensor:
    """Mean-pool the head blocks of a K or V projection tensor along dim 0.

    `t` is (n_head * head_dim, ...) — rows grouped head-major, exactly how the fused
    c_attn K/V slices are laid out. Heads are split into n_kv_head contiguous groups of
    n_head // n_kv_head and averaged elementwise within each group. Works on 2-D weights
    and 1-D biases alike. Deterministic: a plain reshape + mean, no RNG.

    `scale_correct` multiplies each pooled block by sqrt(group_size). Averaging g fully
    DEcorrelated head projections shrinks their norm by 1/sqrt(g) — measured on the 1B:
    within-group K/V head cosine ~ 0.000 at every depth, pooled-norm ratio 0.378 =
    1/sqrt(7) — which mutes the attention branch (~5x smaller outputs) and collapses the
    converted model to the unigram floor. The sqrt(g) rescale restores per-head RMS: on
    K it restores the attention-logit scale q.k, on V the value magnitude. Exact only
    for orthogonal heads (what we measured); for correlated heads it overshoots — e.g.
    identical heads within a group would be overscaled by sqrt(g). Applied to weight AND
    bias so the pooled head function k(x) = W x + b is rescaled as a whole.
    """
    if n_head % n_kv_head != 0:
        raise ValueError(f"n_kv_head ({n_kv_head}) must divide n_head ({n_head})")
    if t.size(0) % n_head != 0:
        raise ValueError(f"dim 0 ({t.size(0)}) is not divisible by n_head ({n_head})")
    head_dim = t.size(0) // n_head
    group = n_head // n_kv_head
    pooled = t.reshape(n_kv_head, group, head_dim, *t.shape[1:]).mean(dim=1)
    if scale_correct:
        pooled = pooled * (group ** 0.5)
    return pooled.reshape(n_kv_head * head_dim, *t.shape[1:])


def convert_state_dict(
    sd: dict[str, torch.Tensor], *, n_head: int, n_embd: int, n_kv_head: int,
    scale_correct: bool = False,
) -> dict[str, torch.Tensor]:
    """MHA state dict -> GQA state dict. Per attention layer the fused c_attn (3C rows:
    Q,K,V stacked) becomes q_proj (the Q rows, untouched) and kv_proj (mean-pooled K rows
    then mean-pooled V rows — matching GQAAttention's `.split(n_kv_head*head_dim)`
    layout). Biases pool the same way. Every other tensor is carried over unchanged."""
    if n_head % n_kv_head != 0:
        raise ValueError(f"n_kv_head ({n_kv_head}) must divide n_head ({n_head})")
    layers = [m.group(1) for k in sd if (m := _CATTN_W.match(k))]
    if not layers:
        raise ValueError("no transformer.h.*.attn.c_attn.weight keys — not an MHA state dict")
    new: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        m = re.match(r"^transformer\.h\.(\d+)\.attn\.c_attn\.(weight|bias)$", k)
        if m is None:
            new[k] = v
            continue
        prefix = f"transformer.h.{m.group(1)}.attn."
        if v.size(0) != 3 * n_embd:
            raise ValueError(f"{k} has {v.size(0)} rows, expected 3*n_embd={3 * n_embd}")
        q, kk, vv = v.split(n_embd, dim=0)
        new[prefix + f"q_proj.{m.group(2)}"] = q.clone()
        new[prefix + f"kv_proj.{m.group(2)}"] = torch.cat(
            [pool_heads(kk, n_head=n_head, n_kv_head=n_kv_head, scale_correct=scale_correct),
             pool_heads(vv, n_head=n_head, n_kv_head=n_kv_head, scale_correct=scale_correct)],
            dim=0)
    return new


@torch.no_grad()
def mean_kl(model_a, model_b, x: torch.Tensor, autocast: bool = False) -> float:
    """Mean per-token KL(P_a || P_b) in nats on batch x. Logits are computed (optionally
    under bf16 autocast, matching training numerics) then reduced in float32."""
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast
           else torch.no_grad())
    with ctx:
        la, _ = model_a(x)
        lb, _ = model_b(x)
    pa = F.log_softmax(la.float(), dim=-1)
    pb = F.log_softmax(lb.float(), dim=-1)
    return float((pa.exp() * (pa - pb)).sum(-1).mean())


@torch.no_grad()
def mean_ce(model, x: torch.Tensor, y: torch.Tensor, autocast: bool = False) -> float:
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast
           else torch.no_grad())
    with ctx:
        _, loss = model(x, y)
    return float(loss)


def _variant_cfg(cfg, n_kv_head: int | None) -> VariantConfig:
    return VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, norm=cfg.norm, pos=cfg.pos,
        mlp=cfg.mlp, n_kv_head=n_kv_head, rope_base=getattr(cfg, "rope_base", 10000.0))


def convert_checkpoint(src: str, out: str, *, n_kv_head: int, scale_correct: bool = False) -> dict:
    """Load a trainer checkpoint at `src`, mean-pool to n_kv_head, write a weights-only
    checkpoint to `out`, and return a summary (param counts before/after)."""
    ckpt = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    cfg = ckpt["cfg"]
    if getattr(cfg, "n_kv_head", None) is not None:
        raise ValueError(
            f"{src} is not an MHA checkpoint (cfg.n_kv_head={cfg.n_kv_head}); "
            "re-pooling an already-GQA model is not supported")
    sd = ckpt["model"]
    new_sd = convert_state_dict(sd, n_head=cfg.n_head, n_embd=cfg.n_embd, n_kv_head=n_kv_head,
                                scale_correct=scale_correct)
    new_cfg = dataclasses.replace(cfg, n_kv_head=n_kv_head)
    # wte/lm_head are tied in the module but appear as two identical entries in the state
    # dict; count unique storages the same way num_params() counts parameters (once).
    params_before = sum(v.numel() for k, v in sd.items() if k != "lm_head.weight")
    params_after = sum(v.numel() for k, v in new_sd.items() if k != "lm_head.weight")
    torch.save(
        {
            "model": new_sd,
            "cfg": new_cfg,
            "converted_from": {"path": src, "step": ckpt.get("step")},
            "conversion": {
                "method": ("mean-pool (Ainslie et al., 2023)"
                           + (" + sqrt(group) scale correction" if scale_correct else "")),
                "n_kv_head": n_kv_head,
                "scale_correct": scale_correct,
            },
        },
        out,
    )
    return {
        "params_before": params_before,
        "params_after": params_after,
        "n_layers_converted": cfg.n_layer,
    }


def _build_model(cfg, sd, n_kv_head, device: str) -> VariantGPT:
    model = VariantGPT(_variant_cfg(cfg, n_kv_head))
    model.load_state_dict(sd)
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="MHA trainer checkpoint (ckpt_*.pt)")
    ap.add_argument("out", help="output weights checkpoint path")
    ap.add_argument("--n-kv-head", type=int, required=True)
    ap.add_argument("--scale-correct", action="store_true",
                    help="rescale pooled K/V blocks by sqrt(group_size) to restore "
                         "per-head RMS (exact for decorrelated heads; see pool_heads)")
    ap.add_argument("--kl-data-dir", default=None,
                    help="shard dir: report KL(orig||pooled) + CE on a fixed val batch")
    ap.add_argument("--kl-batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="device for the KL evaluation")
    args = ap.parse_args()

    summary = convert_checkpoint(args.src, args.out, n_kv_head=args.n_kv_head,
                                 scale_correct=args.scale_correct)
    delta = summary["params_before"] - summary["params_after"]
    mode = "mean-pool + sqrt(group) scale correction" if args.scale_correct else "mean-pool"
    print(f"converted {summary['n_layers_converted']} layers ({mode}) -> {args.out}")
    print(f"params: {summary['params_before']:,} -> {summary['params_after']:,} "
          f"(-{delta:,}, -{100 * delta / summary['params_before']:.2f}%)")

    if args.kl_data_dir is not None:
        from microlab.data.shard_dataset import ShardDataset

        src_ckpt = torch.load(args.src, map_location="cpu", weights_only=False, mmap=True)
        out_ckpt = torch.load(args.out, map_location="cpu", weights_only=False)
        cfg = src_ckpt["cfg"]
        orig = _build_model(cfg, src_ckpt["model"], None, args.device)
        pooled = _build_model(cfg, out_ckpt["model"], args.n_kv_head, args.device)
        val = ShardDataset(args.kl_data_dir, split="val")
        x, y = val.get_batch(cfg.block_size, args.kl_batch_size, args.device,
                             torch.Generator().manual_seed(1337))
        autocast = args.device.startswith("cuda")
        kl = mean_kl(orig, pooled, x, autocast=autocast)
        ce_orig = mean_ce(orig, x, y, autocast=autocast)
        ce_pooled = mean_ce(pooled, x, y, autocast=autocast)
        print(f"val batch ({args.kl_batch_size} x {cfg.block_size} tokens, seed 1337):")
        print(f"  KL(orig||pooled) = {kl:.4f} nats/token")
        print(f"  CE orig = {ce_orig:.4f}  CE pooled = {ce_pooled:.4f} "
              f"(+{ce_pooled - ce_orig:.4f})")


if __name__ == "__main__":
    main()
