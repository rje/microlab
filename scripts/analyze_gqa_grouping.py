"""Audit tooling for the MHA->GQA conversion (scripts/convert_gqa.py): measure the K/V
head-similarity structure of an MHA checkpoint and the conversion CE of alternative
recipes. Inference-only — no training anywhere in this script.

Subcommands:

  similarity  — full n_head x n_head cosine-similarity matrices per layer for K and V, in
                three spaces: weights (flattened head projection blocks), activations
                (per-token head outputs on a fixed val batch, averaged over tokens), and
                ALIGNED activations (cosine after the best function-preserving per-head
                rotation — full orthogonal for V, RoPE-commuting plane rotations for K —
                the Procrustes view of arXiv 2412.20677, which shows raw cosine is
                gauge-confounded: heads computing similar functions in rotated bases look
                orthogonal). Exhaustively searches head partitions (bitmask DP, exact)
                for the grouping that maximizes within-group similarity vs the adjacent
                grouping the converter uses, reports the pooled/original norm ratio and
                the per-layer attention-entropy shift caused by pooling (the collapse
                mechanism). Writes <out>/similarity.pt and <out>/similarity_summary.json.

  eval        — conversion CE on the same fixed val batch (seed 1337, batch 8, matching
                convert_gqa.py's --kl-data-dir eval) for a grid of recipes: mean-pool vs
                medoid-select, adjacent vs similarity-optimal grouping, Procrustes-
                aligned vs unaligned pooling, n_kv_head in {1, 2, 7}, with and without
                norm restoration. Requires similarity.pt (fails loudly if missing) and
                appends each result to <out>/eval_variants.json as it lands.

    python scripts/analyze_gqa_grouping.py similarity runs/1b/ckpt_40000.pt \
        --data-dir data/shards/fineweb-100bt --out runs/gqa_audit --device cuda
    python scripts/analyze_gqa_grouping.py eval runs/1b/ckpt_40000.pt \
        --data-dir data/shards/fineweb-100bt --out runs/gqa_audit --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from functools import cache
from itertools import combinations
from pathlib import Path

import torch
from torch.nn import functional as F

from microlab.model.reference.variants import VariantGPT, apply_rope, build_rope_cache

_SPEC = importlib.util.spec_from_file_location(
    "convert_gqa_script", Path(__file__).resolve().parent / "convert_gqa.py")
convert_gqa = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(convert_gqa)

_ATTN_KEY = re.compile(r"^transformer\.h\.(\d+)\.attn\.(c_attn|c_proj)\.(weight|bias)$")


# ---------------------------------------------------------------------------
# pure logic: similarity + partition search
# ---------------------------------------------------------------------------

def head_cosine_matrix(t: torch.Tensor, n_head: int) -> torch.Tensor:
    """(n_head*head_dim, ...) K or V projection tensor -> (n_head, n_head) cosine
    similarity between the flattened per-head blocks (weight-space similarity)."""
    if t.size(0) % n_head != 0:
        raise ValueError(f"dim 0 ({t.size(0)}) is not divisible by n_head ({n_head})")
    blocks = F.normalize(t.reshape(n_head, -1).float(), dim=1)
    return blocks @ blocks.T


def activation_cosine_matrix(acts: torch.Tensor) -> torch.Tensor:
    """acts: (N, n_head, head_dim) per-token head outputs -> (n_head, n_head) cosine
    similarity between heads, averaged over the N tokens. RoPE-invariant for K: every
    head gets the identical per-position rotation, which preserves inner products."""
    if acts.dim() != 3:
        raise ValueError(f"expected (N, n_head, head_dim), got {tuple(acts.shape)}")
    a = F.normalize(acts.float(), dim=-1)
    return torch.einsum("nid,njd->ij", a, a) / a.size(0)


def group_score(sim, groups) -> float:
    """Mean within-group pairwise similarity (off-diagonal pairs only) over all groups.
    This is the quantity mean-pooling cares about: for unit-norm heads the pooled norm is
    sqrt((1 + (g-1)*mean_cos) / g), so score 0 -> the measured 1/sqrt(g) norm shrink."""
    total, count = 0.0, 0
    for g in groups:
        for i, a in enumerate(g):
            for b in g[i + 1:]:
                total += float(sim[a][b])
                count += 1
    if count == 0:
        raise ValueError("all groups have size 1: no within-group pairs to score")
    return total / count


def best_partition(sim, group_size: int) -> tuple[list[list[int]], float]:
    """Exact search over all partitions of range(n) into equal groups of `group_size`,
    maximizing mean within-group pairwise similarity. Bitmask DP with the lowest
    unassigned head as forced group leader, so each partition is visited once:
    C(13,6)=1716 partitions for 14 heads -> 2 groups, 13!!=135135 for pairs. Ties keep
    the first (lexicographically smallest) partition found. Returns (groups, score)."""
    n = len(sim)
    if group_size < 2:
        raise ValueError("group_size must be >= 2 (singleton groups have no pairs)")
    if n % group_size != 0:
        raise ValueError(f"group_size ({group_size}) must divide n_head ({n})")
    s = [[float(v) for v in row] for row in sim]

    @cache
    def solve(mask: int) -> tuple[float, tuple[tuple[int, ...], ...]]:
        if mask == 0:
            return 0.0, ()
        lead = (mask & -mask).bit_length() - 1
        rest = [i for i in range(lead + 1, n) if mask >> i & 1]
        best_total, best_groups = -math.inf, ()
        for comb in combinations(rest, group_size - 1):
            group = (lead, *comb)
            within = sum(s[a][b] for i, a in enumerate(group) for b in group[i + 1:])
            sub_mask = mask
            for h in group:
                sub_mask &= ~(1 << h)
            sub_total, sub_groups = solve(sub_mask)
            if within + sub_total > best_total:
                best_total, best_groups = within + sub_total, (group, *sub_groups)
        return best_total, best_groups

    total, groups = solve((1 << n) - 1)
    n_pairs = (n // group_size) * group_size * (group_size - 1) // 2
    return [list(g) for g in groups], total / n_pairs


def adjacent_groups(n_head: int, n_kv_head: int) -> list[list[int]]:
    """The converter's implicit grouping: contiguous index blocks of n_head//n_kv_head."""
    if n_head % n_kv_head != 0:
        raise ValueError(f"n_kv_head ({n_kv_head}) must divide n_head ({n_head})")
    g = n_head // n_kv_head
    return [list(range(i * g, (i + 1) * g)) for i in range(n_kv_head)]


def medoid(sim, group) -> int:
    """The group member with the highest mean similarity to the rest of the group (the
    most representative head to keep under selection-init). Ties keep the lowest index."""
    if len(group) == 1:
        return group[0]
    return max(group, key=lambda h: (sum(float(sim[h][o]) for o in group if o != h), -h))


# ---------------------------------------------------------------------------
# pure logic: function-preserving per-head rotations (Procrustes alignment)
# ---------------------------------------------------------------------------

def procrustes_rotation(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """(N, d) activations -> the orthogonal Q minimizing sum_n ||Q src_n - tgt_n||^2
    (classic orthogonal Procrustes via SVD). Used for V heads, whose basis is free: a
    per-head orthogonal rewrite v' = Q v is absorbed exactly by c_proj's columns."""
    m = src.T.float() @ tgt.float()
    u, _, vt = torch.linalg.svd(m)
    return vt.T @ u.T


def rope_plane_rotation(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """(N, d) activations -> the best rotation for K heads under RoPE: independent 2-D
    rotations in each RoPE frequency plane (coords (i, i + d/2)), the exact family of
    orthogonal maps that commute with every RoPE position rotation. Rotating W_k and W_q
    of the same head by such a Q preserves the head's function exactly."""
    d = src.size(1)
    if d % 2 != 0:
        raise ValueError(f"head_dim must be even, got {d}")
    half = d // 2
    a, b = src[:, :half].float(), src[:, half:].float()
    c, e = tgt[:, :half].float(), tgt[:, half:].float()
    alpha = (a * c + b * e).sum(0)
    beta = (a * e - b * c).sum(0)
    phi = torch.atan2(beta, alpha)  # atan2(0, 0) = 0 -> identity for degenerate planes
    rot = torch.zeros(d, d)
    idx = torch.arange(half)
    rot[idx, idx] = phi.cos()
    rot[idx + half, idx + half] = phi.cos()
    rot[idx, idx + half] = -phi.sin()
    rot[idx + half, idx] = phi.sin()
    return rot


def align_group(acts: torch.Tensor, group, constrained: bool, iters: int = 3) -> torch.Tensor:
    """Generalized Procrustes over one group: iteratively rotate every member head onto
    the running aligned mean. acts: (N, n_head, head_dim). Returns (len(group), d, d)
    rotations in group order (Q_h maps head h's activations toward the common frame)."""
    rot_fn = rope_plane_rotation if constrained else procrustes_rotation
    tgt = acts[:, group[0]].float()
    rots = [torch.eye(acts.size(2)) for _ in group]
    for _ in range(iters):
        rots = [rot_fn(acts[:, h], tgt) for h in group]
        tgt = torch.stack(
            [acts[:, h].float() @ q.T for h, q in zip(group, rots, strict=True)]).mean(0)
    return torch.stack(rots)


def aligned_similarity_matrix(acts: torch.Tensor, constrained: bool) -> torch.Tensor:
    """(N, n_head, d) -> (n_head, n_head) cosine between two heads' activations after
    the best allowed rotation of one onto the other (nuclear norm of the cross-Gram for
    full orthogonal; closed-form per-plane maximum for the RoPE-commuting family). This
    is the gauge-invariant similarity that raw cosine hides. Symmetric by construction."""
    n_head, d = acts.size(1), acts.size(2)
    a = acts.float()
    norms = torch.stack([a[:, h].norm() for h in range(n_head)])
    out = torch.eye(n_head)
    half = d // 2
    for i in range(n_head):
        for j in range(i + 1, n_head):
            src, tgt = a[:, j], a[:, i]
            if constrained:
                sa, sb = src[:, :half], src[:, half:]
                ta, tb = tgt[:, :half], tgt[:, half:]
                alpha = (sa * ta + sb * tb).sum(0)
                beta = (sa * tb - sb * ta).sum(0)
                val = torch.sqrt(alpha ** 2 + beta ** 2).sum()
            else:
                val = torch.linalg.svdvals(src.T @ tgt).sum()
            out[i, j] = out[j, i] = val / (norms[i] * norms[j])
    return out


# ---------------------------------------------------------------------------
# pure logic: generalized MHA -> GQA conversion
# ---------------------------------------------------------------------------

def rotate_heads(t: torch.Tensor | None, rots: torch.Tensor) -> torch.Tensor | None:
    """Apply per-head rotations to a head-major projection tensor: weight (n_head*d, C)
    or bias (n_head*d,). Head h's block W_h becomes Q_h @ W_h. None passes through."""
    if t is None:
        return None
    n_head, d = rots.size(0), rots.size(1)
    heads = t.reshape(n_head, d, -1)
    out = torch.einsum("hde,hec->hdc", rots.to(t.dtype), heads)
    return out.reshape(t.shape)


def absorb_output_rotation(cw: torch.Tensor, rots: torch.Tensor) -> torch.Tensor:
    """Fold per-head V rotations into c_proj: each input-column block becomes
    W_o,h @ Q_h^T, so W_o,h' (Q_h v_h) == W_o,h v_h and the rewrite is exact."""
    n_head, d = rots.size(0), rots.size(1)
    blocks = cw.reshape(cw.size(0), n_head, d)
    out = torch.einsum("chd,hed->che", blocks, rots.to(cw.dtype))
    return out.reshape(cw.shape)


def _for_layer(spec, li: int):
    """Per-layer specs are either one value for every layer, or {layer_idx: value}."""
    if spec is None or not isinstance(spec, dict):
        return spec
    return spec[li]


def _check_partition(par, n_head: int) -> None:
    sizes = {len(g) for g in par}
    if len(sizes) != 1:
        raise ValueError(f"groups must be equal-size, got sizes {sorted(sizes)}")
    if sorted(h for g in par for h in g) != list(range(n_head)):
        raise ValueError(f"groups must cover range({n_head}) exactly: {par}")


def _pool_group(w, b, par, sel, mode, scale_correct, renorm, n_head, head_dim):
    """Pool one projection's heads by group. Weight and bias are pooled together so
    norm-restoring factors stay consistent across the head function k(x) = W x + b."""
    wh = w.reshape(n_head, head_dim, -1)
    bh = b.reshape(n_head, head_dim) if b is not None else None
    w_parts, b_parts = [], []
    for gi, g in enumerate(par):
        if mode == "select":
            if sel[gi] not in g:
                raise ValueError(f"select[{gi}]={sel[gi]} is not a member of group {g}")
            w_parts.append(wh[sel[gi]].clone())
            if bh is not None:
                b_parts.append(bh[sel[gi]].clone())
            continue
        pw = wh[list(g)].mean(dim=0)
        pb = bh[list(g)].mean(dim=0) if bh is not None else None
        if scale_correct:
            factor = len(g) ** 0.5
        elif renorm:
            # restore the mean member Frobenius norm (exact whatever the correlation;
            # equals sqrt(g) only in the fully-decorrelated case scale_correct assumes)
            factor = float(wh[list(g)].reshape(len(g), -1).norm(dim=1).mean() / pw.norm())
        else:
            factor = 1.0
        w_parts.append(pw * factor)
        if pb is not None:
            b_parts.append(pb * factor)
    pooled_w = torch.cat(w_parts, dim=0).reshape(len(par) * head_dim, *w.shape[1:])
    pooled_b = torch.cat(b_parts, dim=0).reshape(-1) if b_parts else None
    return pooled_w, pooled_b


def convert_with_groups(
    sd: dict[str, torch.Tensor], *, n_head: int, n_embd: int, groups, mode: str = "mean",
    select=None, scale_correct: bool = False, renorm: bool = False, rotations=None,
) -> dict[str, torch.Tensor]:
    """Generalized MHA -> GQA conversion with an explicit head partition per layer.

    `groups` is one partition (list of equal-size groups covering range(n_head)) applied
    everywhere, or {layer_idx: partition}. Query heads (q_proj rows) and c_proj input
    columns are permuted so each group's query heads sit adjacent — function-preserving,
    since attention is equivariant to a consistent head permutation — then each group's
    K/V heads are mean-pooled (mode="mean"; `scale_correct` multiplies by sqrt(group),
    `renorm` restores the mean member norm exactly) or replaced by one representative
    (mode="select"; `select` holds one absolute member index per group, per layer).

    `rotations` optionally supplies function-preserving per-head rewrites applied BEFORE
    pooling: {layer_idx: {"k": (n_head,d,d), "v": (n_head,d,d)}}. K rotations must be
    RoPE-commuting (rope_plane_rotation) and are applied to W_q and W_k of the same head
    (q.k is invariant); V rotations are arbitrary orthogonal and are absorbed into
    c_proj's columns. With adjacent groups, mode="mean", and no rotations this
    reproduces convert_gqa.convert_state_dict exactly."""
    if mode not in ("mean", "select"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "select" and select is None:
        raise ValueError("mode='select' requires `select` (one member index per group)")
    if mode == "mean" and select is not None:
        raise ValueError("`select` is only meaningful with mode='select'")
    if mode == "select" and (scale_correct or renorm):
        raise ValueError("scale_correct/renorm apply to mean-pooling, not selection")
    if scale_correct and renorm:
        raise ValueError("scale_correct and renorm are mutually exclusive")
    head_dim = n_embd // n_head
    layer_ids = sorted({int(m.group(1)) for k in sd if (m := _ATTN_KEY.match(k))})
    if not layer_ids:
        raise ValueError("no transformer.h.*.attn.* keys — not an MHA state dict")
    new = {k: v for k, v in sd.items() if not _ATTN_KEY.match(k)}
    for li in layer_ids:
        par = _for_layer(groups, li)
        _check_partition(par, n_head)
        sel = _for_layer(select, li)
        if mode == "select" and len(sel) != len(par):
            raise ValueError(f"select has {len(sel)} entries for {len(par)} groups")
        p = f"transformer.h.{li}.attn."
        w = sd[p + "c_attn.weight"]
        if w.size(0) != 3 * n_embd:
            raise ValueError(f"{p}c_attn.weight has {w.size(0)} rows, expected {3 * n_embd}")
        b = sd.get(p + "c_attn.bias")
        cw = sd[p + "c_proj.weight"]
        q, k, v = w.split(n_embd, dim=0)
        qb, kb, vb = b.split(n_embd, dim=0) if b is not None else (None, None, None)
        rot = _for_layer(rotations, li)
        if rot is not None:
            q, qb = rotate_heads(q, rot["k"]), rotate_heads(qb, rot["k"])
            k, kb = rotate_heads(k, rot["k"]), rotate_heads(kb, rot["k"])
            v, vb = rotate_heads(v, rot["v"]), rotate_heads(vb, rot["v"])
            cw = absorb_output_rotation(cw, rot["v"])
        order = [h for g in par for h in g]
        new[p + "q_proj.weight"] = (
            q.reshape(n_head, head_dim, -1)[order].reshape(n_embd, -1).clone())
        if qb is not None:
            new[p + "q_proj.bias"] = qb.reshape(n_head, head_dim)[order].reshape(-1).clone()
        pk, pkb = _pool_group(k, kb, par, sel, mode, scale_correct, renorm, n_head, head_dim)
        pv, pvb = _pool_group(v, vb, par, sel, mode, scale_correct, renorm, n_head, head_dim)
        new[p + "kv_proj.weight"] = torch.cat([pk, pv], dim=0)
        if pkb is not None:
            new[p + "kv_proj.bias"] = torch.cat([pkb, pvb], dim=0)
        new[p + "c_proj.weight"] = (
            cw.reshape(cw.size(0), n_head, head_dim)[:, order].reshape(cw.shape).clone())
        cb = sd.get(p + "c_proj.bias")
        if cb is not None:
            new[p + "c_proj.bias"] = cb  # lives in residual space: untouched
    return new


# ---------------------------------------------------------------------------
# measurement drivers (checkpoint + val batch; sequential GPU residency)
# ---------------------------------------------------------------------------

def _load(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    cfg = ckpt["cfg"]
    if getattr(cfg, "n_kv_head", None) is not None:
        raise ValueError(f"{ckpt_path} is not an MHA checkpoint (n_kv_head={cfg.n_kv_head})")
    return ckpt["model"], cfg


def _val_batch(cfg, data_dir: str, batch_size: int, device: str):
    from microlab.data.shard_dataset import ShardDataset

    val = ShardDataset(data_dir, split="val")
    return val.get_batch(cfg.block_size, batch_size, device,
                         torch.Generator().manual_seed(1337))


def _qkv_weights(sd, li: int, n_embd: int):
    p = f"transformer.h.{li}.attn.c_attn."
    w, b = sd[p + "weight"], sd.get(p + "bias")
    parts = list(w.split(n_embd, dim=0))
    bparts = list(b.split(n_embd, dim=0)) if b is not None else [None, None, None]
    return parts, bparts


@torch.no_grad()
def _capture_ln1(sd, cfg, x: torch.Tensor, device: str) -> list[torch.Tensor]:
    """Run the MHA model once, capturing each block's ln_1 output (the tensor the K/V
    projections consume) to CPU float32. Returns one (B*T, n_embd) tensor per layer."""
    model = VariantGPT(convert_gqa._variant_cfg(cfg, None))
    model.load_state_dict(sd)
    model = model.to(device).eval()
    captured: list = [None] * cfg.n_layer
    hooks = []
    for li, block in enumerate(model.transformer.h):
        def hook(_mod, _inp, out, li=li):
            captured[li] = out.detach().float().reshape(-1, cfg.n_embd).cpu()
        hooks.append(block.ln_1.register_forward_hook(hook))
    model(x)
    for h in hooks:
        h.remove()
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if any(c is None for c in captured):
        raise RuntimeError("ln_1 capture missed a layer")
    return captured


def _head_acts(ln1_l: torch.Tensor, w, b, n_head: int, device: str) -> torch.Tensor:
    """(N, C) ln_1 activations x one projection -> (N, n_head, head_dim) head outputs
    including the bias (the full head function the conversion pools)."""
    x = ln1_l.to(device)
    acts = x @ w.to(device).float().T
    if b is not None:
        acts = acts + b.to(device).float()
    return acts.reshape(x.size(0), n_head, -1)


@torch.no_grad()
def _attn_entropy(sd, cfg, shape, ln1_l: torch.Tensor, li: int, device: str,
                  kv_variant: str, n_kv_head: int) -> float:
    """Mean causal-attention entropy (nats/query) at layer `li` on the captured ln_1
    activations, with K from the original heads ('orig'), adjacent mean-pooling ('mean'),
    or scale-corrected pooling ('mean_scale'). Isolates the per-layer attention-
    flattening effect of pooling from the downstream cascade."""
    B, T = shape
    n_head, n_embd = cfg.n_head, cfg.n_embd
    head_dim = n_embd // n_head
    (qw, kw, _), (qb, kb, _) = _qkv_weights(sd, li, n_embd)
    if kv_variant in ("mean", "mean_scale"):
        sc = kv_variant == "mean_scale"
        kw = convert_gqa.pool_heads(kw, n_head=n_head, n_kv_head=n_kv_head, scale_correct=sc)
        if kb is not None:
            kb = convert_gqa.pool_heads(kb, n_head=n_head, n_kv_head=n_kv_head,
                                        scale_correct=sc)
        n_k = n_kv_head
    elif kv_variant == "orig":
        n_k = n_head
    else:
        raise ValueError(f"unknown kv_variant {kv_variant!r}")
    x = ln1_l.to(device).reshape(B, T, n_embd)
    q = _head_acts(ln1_l, qw, qb, n_head, device).reshape(B, T, n_head, head_dim)
    q = q.transpose(1, 2)
    k = _head_acts(ln1_l, kw, kb, n_k, device).reshape(B, T, n_k, head_dim).transpose(1, 2)
    del x
    cos, sin = build_rope_cache(T, head_dim, base=getattr(cfg, "rope_base", 10000.0),
                                device=device)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if n_k != n_head:
        k = k.repeat_interleave(n_head // n_k, dim=1)
    mask = torch.full((T, T), -torch.inf, device=device).triu(1)
    ent_sum, n_q = 0.0, 0
    for i in range(B):  # one sequence at a time keeps the (n_head, T, T) logits small
        logits = q[i] @ k[i].transpose(-1, -2) / math.sqrt(head_dim) + mask
        prob = F.softmax(logits.float(), dim=-1)
        ent = -(prob * prob.clamp_min(1e-12).log()).sum(-1)
        ent_sum += float(ent.sum())
        n_q += ent.numel()
    return ent_sum / n_q


def compute_rotations(sd, cfg, ln1, groups_by_layer, device: str) -> dict:
    """Generalized-Procrustes rotations per layer for `groups_by_layer` ({li: partition}):
    RoPE-commuting plane rotations for K (shared with Q), full orthogonal for V."""
    rotations: dict[int, dict[str, torch.Tensor]] = {}
    n_head, n_embd = cfg.n_head, cfg.n_embd
    head_dim = n_embd // n_head
    for li, par in groups_by_layer.items():
        (_, kw, vw), (_, kb, vb) = _qkv_weights(sd, li, n_embd)
        out: dict[str, torch.Tensor] = {}
        for name, w, b, constrained in (("k", kw, kb, True), ("v", vw, vb, False)):
            acts = _head_acts(ln1[li], w, b, n_head, device).cpu()
            rots = torch.zeros(n_head, head_dim, head_dim)
            for g in par:
                rots[list(g)] = align_group(acts, g, constrained)
            out[name] = rots
        rotations[li] = out
    return rotations


@torch.no_grad()
def run_similarity(args) -> None:
    sd, cfg = _load(args.ckpt)
    n_head, n_embd = cfg.n_head, cfg.n_embd
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    x, _ = _val_batch(cfg, args.data_dir, args.batch_size, args.device)
    print(f"capturing ln_1 activations on {tuple(x.shape)} val tokens ...")
    ln1 = _capture_ln1(sd, cfg, x, args.device)

    names = ("k_weight", "v_weight", "k_act", "v_act", "k_act_aligned", "v_act_aligned")
    mats = {n: torch.zeros(cfg.n_layer, n_head, n_head) for n in names}
    norm_ratio = {"k": [], "v": []}
    group = n_head // 2  # the shipped conversion: n_kv_head=2, groups of 7
    for li in range(cfg.n_layer):
        (_, kw, vw), (_, kb, vb) = _qkv_weights(sd, li, n_embd)
        mats["k_weight"][li] = head_cosine_matrix(kw, n_head)
        mats["v_weight"][li] = head_cosine_matrix(vw, n_head)
        for name, w, b, constrained in (("k", kw, kb, True), ("v", vw, vb, False)):
            acts = _head_acts(ln1[li], w, b, n_head, args.device)
            mats[f"{name}_act"][li] = activation_cosine_matrix(acts).cpu()
            mats[f"{name}_act_aligned"][li] = aligned_similarity_matrix(acts.cpu(), constrained)
            heads = w.reshape(n_head, -1)
            pooled = convert_gqa.pool_heads(w, n_head=n_head, n_kv_head=2)
            norm_ratio[name].append(
                float(pooled.reshape(2, -1).norm(dim=1).mean() / heads.norm(dim=1).mean()))
        print(f"  layer {li:2d} matrices done")
    torch.save(dict(mats) | {"n_head": n_head}, out_dir / "similarity.pt")

    summary: dict = {"ckpt": args.ckpt, "n_layer": cfg.n_layer, "n_head": n_head,
                     "norm_ratio_kv2": {k: sum(v) / len(v) for k, v in norm_ratio.items()},
                     "expected_orthogonal_ratio": 1 / math.sqrt(group), "partitions": {}}
    print(f"pooled/orig norm ratio (kv2): K {summary['norm_ratio_kv2']['k']:.3f} "
          f"V {summary['norm_ratio_kv2']['v']:.3f} (fully-decorrelated predicts "
          f"{1 / math.sqrt(group):.3f})")
    combos = list(names) + ["kv_act", "kv_act_aligned"]
    for n_kv in (2, 7):
        adj = adjacent_groups(n_head, n_kv)
        for name in combos:
            adj_scores, best_scores, off_diag = [], [], []
            for li in range(cfg.n_layer):
                if name.startswith("kv_"):
                    suffix = name.removeprefix("kv_")
                    sim = (mats[f"k_{suffix}"][li] + mats[f"v_{suffix}"][li]) / 2
                else:
                    sim = mats[name][li]
                sim_l = sim.tolist()
                adj_scores.append(group_score(sim_l, adj))
                best_scores.append(best_partition(sim_l, n_head // n_kv)[1])
                off = sim - torch.diag(sim.diagonal())
                off_diag.append(float(off.abs().sum() / (n_head * (n_head - 1))))
            key = f"kv{n_kv}_{name}"
            summary["partitions"][key] = {
                "adjacent_mean": sum(adj_scores) / len(adj_scores),
                "best_mean": sum(best_scores) / len(best_scores),
                "best_max_layer": max(best_scores), "best_min_layer": min(best_scores),
                "mean_abs_offdiag": sum(off_diag) / len(off_diag)}
            p = summary["partitions"][key]
            print(f"{key:>22}: adjacent {p['adjacent_mean']:+.4f}  best {p['best_mean']:+.4f}"
                  f"  (best per-layer range [{p['best_min_layer']:+.4f}, "
                  f"{p['best_max_layer']:+.4f}], mean|offdiag| {p['mean_abs_offdiag']:.4f})")

    if not args.skip_entropy:
        print("attention entropy, kv2 pooling (nats/query; uniform over the causal prefix "
              f"~= {math.log(x.size(1)) - 1:.2f} at the mean position):")
        ent: dict[str, list[float]] = {"orig": [], "mean": [], "mean_scale": []}
        for li in range(cfg.n_layer):
            for variant in ent:
                ent[variant].append(_attn_entropy(sd, cfg, tuple(x.shape), ln1[li], li,
                                                  args.device, variant, 2))
            print(f"  layer {li:2d}: orig {ent['orig'][-1]:5.2f}  pooled {ent['mean'][-1]:5.2f}"
                  f"  pooled+scale {ent['mean_scale'][-1]:5.2f}")
        summary["attn_entropy_kv2"] = ent
    (out_dir / "similarity_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'similarity.pt'} and {out_dir / 'similarity_summary.json'}")


@torch.no_grad()
def run_eval(args) -> None:
    sd, cfg = _load(args.ckpt)
    n_head, n_embd = cfg.n_head, cfg.n_embd
    out_dir = Path(args.out)
    sim_path = out_dir / "similarity.pt"
    if not sim_path.exists():
        raise FileNotFoundError(f"{sim_path} missing — run the `similarity` subcommand first")
    mats = torch.load(sim_path, weights_only=True)
    x, y = _val_batch(cfg, args.data_dir, args.batch_size, args.device)
    autocast = args.device.startswith("cuda")
    results_path = out_dir / "eval_variants.json"
    results: list[dict] = []

    def record(name: str, ce: float, note: str = "") -> None:
        results.append({"variant": name, "ce": ce, "note": note})
        results_path.write_text(json.dumps(
            {"ckpt": args.ckpt, "batch": [args.batch_size, cfg.block_size],
             "seed": 1337, "results": results}, indent=2))
        print(f"{name:>26}: CE {ce:.4f}  {note}")

    def sim_of(li: int, kind: str) -> list[list[float]]:
        return ((mats[f"k_{kind}"][li] + mats[f"v_{kind}"][li]) / 2).tolist()

    def eval_sd(new_sd, n_kv) -> float:
        model = VariantGPT(convert_gqa._variant_cfg(cfg, n_kv))
        model.load_state_dict(new_sd)
        model = model.to(args.device).eval()
        ce = convert_gqa.mean_ce(model, x, y, autocast=autocast)
        del model
        if autocast:
            torch.cuda.empty_cache()
        return ce

    def conv(n_kv, groups, **kw) -> dict:
        return convert_with_groups(sd, n_head=n_head, n_embd=n_embd, groups=groups, **kw)

    record("orig-mha", eval_sd(sd, None), "unconverted reference")
    ln1 = _capture_ln1(sd, cfg, x, args.device)
    for n_kv in (7, 2, 1):
        adj = {li: adjacent_groups(n_head, n_kv) for li in range(cfg.n_layer)}
        ratio = f"{n_head // n_kv}:1"
        record(f"kv{n_kv}-adjacent-mean", eval_sd(conv(n_kv, adj), n_kv),
               f"{ratio} — the shipped converter recipe" if n_kv == 2 else ratio)
        record(f"kv{n_kv}-adjacent-mean-scale", eval_sd(
            conv(n_kv, adj, scale_correct=True), n_kv), "sqrt(group) rescale")
        sel = {li: [medoid(sim_of(li, "act"), g) for g in adj[li]]
               for li in range(cfg.n_layer)}
        record(f"kv{n_kv}-adjacent-select", eval_sd(
            conv(n_kv, adj, mode="select", select=sel), n_kv), "keep medoid head per group")
        aligned_rot = compute_rotations(sd, cfg, ln1, adj, args.device)
        record(f"kv{n_kv}-aligned-mean", eval_sd(
            conv(n_kv, adj, rotations=aligned_rot), n_kv), "Procrustes-align, then pool")
        record(f"kv{n_kv}-aligned-mean-renorm", eval_sd(
            conv(n_kv, adj, rotations=aligned_rot, renorm=True), n_kv),
            "align, pool, restore norms")
        if n_kv == 1:
            continue  # one group of all heads: no partition freedom to exploit
        opt = {li: best_partition(sim_of(li, "act"), n_head // n_kv)[0]
               for li in range(cfg.n_layer)}
        record(f"kv{n_kv}-optimal-mean", eval_sd(conv(n_kv, opt), n_kv),
               "best per-layer partition on raw activation cosine")
        opt_al = {li: best_partition(sim_of(li, "act_aligned"), n_head // n_kv)[0]
                  for li in range(cfg.n_layer)}
        rot_al = compute_rotations(sd, cfg, ln1, opt_al, args.device)
        record(f"kv{n_kv}-alignopt-mean", eval_sd(
            conv(n_kv, opt_al, rotations=rot_al), n_kv),
            "best partition on ALIGNED similarity + Procrustes")
        record(f"kv{n_kv}-alignopt-mean-renorm", eval_sd(
            conv(n_kv, opt_al, rotations=rot_al, renorm=True), n_kv),
            "aligned partition + pool + restore norms")
    print(f"wrote {results_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("similarity", "eval"):
        p = sub.add_parser(name)
        p.add_argument("ckpt", help="MHA trainer checkpoint (ckpt_*.pt)")
        p.add_argument("--data-dir", required=True, help="shard dir with a val split")
        p.add_argument("--out", default="runs/gqa_audit", help="output directory")
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--device", default="cuda")
        if name == "similarity":
            p.add_argument("--skip-entropy", action="store_true",
                           help="skip the per-layer attention-entropy measurement")
    args = ap.parse_args()
    if args.cmd == "similarity":
        run_similarity(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
