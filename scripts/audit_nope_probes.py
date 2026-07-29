"""NoPE-verdict audit probes (positive controls for the NoPE-vs-RoPE ablation).

    python scripts/audit_nope_probes.py --out evals/nope_audit/probes.json

Two saved-checkpoint measurements, run on BOTH arms (runs/nope-ab-nope, runs/nope-ab-rope)
so the losing arm is checked against the literature with our own instrument:

(a) Haviv et al. 2022 (arXiv 2203.16634) position probe: a 2-layer ReLU MLP predicts a
    token's ABSOLUTE position (0..1023) from the frozen residual stream, per tap layer.
    Val windows are random shard crops (no BOS alignment), so content carries ~no position
    signal: probe skill >> the shuffled-label control means the network itself encodes
    position. Haviv's finding — a NoPE LM acquires implicit absolute position within a few
    layers — is the positive control that our NoPE implementation behaves as literature
    says. The "emb" tap is the built-in negative control: with no wpe, token embeddings
    contain zero position signal, so the probe must sit at chance there.

(b) Attention entropy/distance profiles (Wang et al. 2404.12224 link NoPE's length-gen
    failure to attention-distribution distraction beyond train length): per layer and
    query position — within and beyond the 1024 trained window — the head-mean attention
    entropy (normalized by ln(visible keys)), mean attention distance, and mass on the
    last-64 keys. If NoPE rows go toward uniform beyond 1024 while RoPE rows stay sharp,
    the length-gen cliff has a mechanistic explanation and is not an eval artifact.

GPU budget: brief <8GB bursts (bf16 forwards on one 124M model at a time); probes train
in seconds. Output JSON is written progressively per section.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from microlab.data.shard_dataset import ShardDataset  # noqa: E402
from microlab.model.reference.variants import apply_rope  # noqa: E402

# scripts/ is not a package: load the passkey module (checkpoint rebuild) from the
# sibling file, same trick eval_length_gen.py uses.
_SPEC = importlib.util.spec_from_file_location(
    "eval_passkey", Path(__file__).resolve().parent / "eval_passkey.py")
ep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ep)


# ------------------------------------------------------------------ probe dataset logic

def features_to_dataset(feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, T, C) features -> (N*T, C) samples with per-position labels 0..T-1."""
    n, t, c = feats.shape
    x = feats.reshape(n * t, c)
    y = torch.arange(t).repeat(n)
    return x, y


def split_sequences(n_seqs: int, n_test: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic SEQUENCE-level train/test split (no token leakage across the split)."""
    if not 0 < n_test < n_seqs:
        raise ValueError(f"need 0 < n_test < n_seqs (got n_test={n_test}, n_seqs={n_seqs})")
    perm = torch.randperm(n_seqs, generator=torch.Generator().manual_seed(seed))
    return perm[n_test:], perm[:n_test]


def probe_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    """Top-1 accuracy + mean absolute distance (Haviv's metric) between pred and label."""
    acc = (preds == labels).float().mean().item()
    mad = (preds - labels).abs().float().mean().item()
    return {"acc": acc, "mad": mad}


def train_position_probe(train_feats: torch.Tensor, test_feats: torch.Tensor,
                         n_positions: int, hidden: int, epochs: int, batch_size: int,
                         lr: float, seed: int, device: str,
                         shuffle_labels: bool = False) -> dict:
    """Train a 2-layer ReLU MLP (Haviv et al. 2022) to classify absolute position from
    frozen features; return test acc/mad. `shuffle_labels` permutes TRAIN labels — the
    chance-level control (test labels stay true, so the score reflects real skill)."""
    x_tr, y_tr = features_to_dataset(train_feats)
    x_te, y_te = features_to_dataset(test_feats)
    gen = torch.Generator().manual_seed(seed)
    if shuffle_labels:
        y_tr = y_tr[torch.randperm(y_tr.numel(), generator=gen)]
    # standardize with TRAIN statistics only (applied identically to every arm/tap)
    mu, sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    torch.manual_seed(seed)
    probe = nn.Sequential(nn.Linear(x_tr.size(1), hidden), nn.ReLU(),
                          nn.Linear(hidden, n_positions)).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        order = torch.randperm(x_tr.size(0), generator=gen)
        for start in range(0, order.numel(), batch_size):
            idx = order[start:start + batch_size]
            xb, yb = x_tr[idx].to(device), y_tr[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(probe(xb), yb).backward()
            opt.step()
    probe.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, x_te.size(0), batch_size):
            preds.append(probe(x_te[start:start + batch_size].to(device)).argmax(-1).cpu())
    out = probe_metrics(torch.cat(preds), y_te)
    out.update({"shuffled": shuffle_labels, "n_train": int(x_tr.size(0)),
                "n_test": int(x_te.size(0))})
    return out


# ------------------------------------------------------------------- feature collection

def collect_features(model, x: torch.Tensor, taps: list) -> dict:
    """Frozen residual-stream features via forward hooks, keyed by tap: "emb" = the
    post-dropout token embedding (pre-blocks), int i = block i's output (pre-ln_f).
    Returned tensors are (B, T, C) fp32 on CPU."""
    feats: dict = {}
    handles = []

    def save(name):
        def hook(_mod, _inp, out):
            feats[name] = out.detach().float().cpu()
        return hook

    for tap in taps:
        if tap == "emb":
            mod = model.transformer.drop
        elif isinstance(tap, int) and 0 <= tap < len(model.transformer.h):
            mod = model.transformer.h[tap]
        else:
            raise ValueError(f"unknown tap {tap!r}: expected 'emb' or a block index")
        handles.append(mod.register_forward_hook(save(tap)))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return feats


# ------------------------------------------------------------------------ attention rows

def attention_rows(attn, x_norm: torch.Tensor, qpos: list[int]) -> torch.Tensor:
    """Causal-masked softmax attention rows for the given ABSOLUTE query positions:
    (B, n_head, len(qpos), T) in fp32. `x_norm` is the attention module's own input
    (i.e. ln_1(x)). RoPE modules are detected by their cos/sin buffers and get the
    rotation applied exactly as in their forward."""
    b, t, c = x_norm.shape
    if any(not 0 <= p < t for p in qpos):
        raise ValueError(f"query positions {qpos} out of range for T={t}")
    q, k, _ = attn.c_attn(x_norm).split(attn.n_embd, dim=2)
    head_dim = c // attn.n_head
    q = q.view(b, t, attn.n_head, head_dim).transpose(1, 2)
    k = k.view(b, t, attn.n_head, head_dim).transpose(1, 2)
    if hasattr(attn, "rope_cos"):
        cos = attn.rope_cos[:t].to(q.dtype)
        sin = attn.rope_sin[:t].to(q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
    q_sel = q[:, :, qpos, :].float()
    scores = q_sel @ k.float().transpose(-2, -1) / math.sqrt(head_dim)
    key_idx = torch.arange(t, device=x_norm.device)
    mask = key_idx[None, :] > torch.tensor(qpos, device=x_norm.device)[:, None]
    scores = scores.masked_fill(mask[None, None], float("-inf"))
    return scores.softmax(-1)


# ------------------------------------------------------------------- entropy summaries

def row_entropy(rows: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) over the last dim; exact 0 for one-hot rows."""
    return -torch.special.xlogy(rows, rows).sum(-1)


def attention_summary(rows: torch.Tensor, qpos: list[int], last_k: int) -> list[dict]:
    """Per query position: head-mean entropy, entropy normalized by ln(visible keys),
    mean attention distance E[p - j], mass on the last `last_k` visible keys, and the
    per-head normalized entropies (batch-and-head means; batch means for per-head)."""
    b, n_head, n_q, t = rows.shape
    key_idx = torch.arange(t, dtype=torch.float32)
    out = []
    for qi, p in enumerate(qpos):
        r = rows[:, :, qi, :]                       # (B, n_head, T)
        ent = row_entropy(r)                        # (B, n_head)
        norm = math.log(p + 1) if p > 0 else 1.0    # ln(1)=0: define norm-entropy as 0
        ent_norm = ent / norm if p > 0 else torch.zeros_like(ent)
        dist = (r * (p - key_idx)).sum(-1)          # future mass is exactly 0
        last_mass = r[..., max(0, p + 1 - last_k): p + 1].sum(-1)
        out.append({
            "qpos": p,
            "entropy": ent.mean().item(),
            "entropy_norm": ent_norm.mean().item(),
            "mean_dist": dist.mean().item(),
            "last_k": last_k,
            "last_k_mass": last_mass.mean().item(),
            "entropy_norm_per_head": ent_norm.mean(0).tolist(),
        })
    return out


# ---------------------------------------------------------------------------- sections

def _val_windows(data_dir: str, length: int, n_seqs: int, batch: int, seed: int):
    """Identical seeded val windows for every arm (same convention as eval_length_gen)."""
    ds = ShardDataset(data_dir, split="val")
    gen = torch.Generator().manual_seed(seed * 1_000_003 + length)
    batches = []
    remaining = n_seqs
    while remaining > 0:
        bs = min(batch, remaining)
        x, _ = ds.get_batch(length, bs, "cpu", gen)
        batches.append(x)
        remaining -= bs
    return torch.cat(batches)


def probe_section(run_dir: Path, args, windows: torch.Tensor) -> dict:
    model, step, cfg, _ = ep.load_for_eval(run_dir, windows.size(1), args.device)
    taps = ["emb"] + [int(t) for t in args.tap_layers.split(",")]
    feats: dict = {}
    for start in range(0, windows.size(0), args.forward_batch):
        x = windows[start:start + args.forward_batch].to(args.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) \
                if args.device.startswith("cuda") else torch.no_grad():
            got = collect_features(model, x, taps)
        for tap, v in got.items():
            feats.setdefault(tap, []).append(v)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    tr, te = split_sequences(windows.size(0), n_test=args.n_test, seed=args.seed)
    section = {"run": str(run_dir), "step": step, "pos": cfg.pos, "taps": {}}
    for tap in taps:
        f = torch.cat(feats[tap])
        entry = {}
        for shuffled in (False, True):
            r = train_position_probe(
                f[tr], f[te], n_positions=windows.size(1), hidden=args.probe_hidden,
                epochs=args.probe_epochs, batch_size=args.probe_batch, lr=args.probe_lr,
                seed=args.seed, device=args.device, shuffle_labels=shuffled)
            entry["shuffled" if shuffled else "real"] = r
        section["taps"][str(tap)] = entry
        print(f"  probe {run_dir.name} tap={tap}: acc={entry['real']['acc']:.3f} "
              f"mad={entry['real']['mad']:.1f} | shuffled acc="
              f"{entry['shuffled']['acc']:.3f} mad={entry['shuffled']['mad']:.1f}",
              flush=True)
    return section


def attention_section(run_dir: Path, args, windows: torch.Tensor) -> dict:
    model, step, cfg, _ = ep.load_for_eval(run_dir, windows.size(1), args.device)
    layers = [int(x) for x in args.attn_layers.split(",")]
    qpos = [int(x) for x in args.query_positions.split(",")]
    captured: dict = {}
    handles = []

    def save(layer):
        def hook(_mod, inp):
            captured.setdefault(layer, []).append(inp[0].detach().float().cpu())
        return hook

    for layer in layers:
        handles.append(model.transformer.h[layer].attn.register_forward_pre_hook(save(layer)))
    try:
        with torch.no_grad():
            for start in range(0, windows.size(0), args.attn_batch):
                x = windows[start:start + args.attn_batch].to(args.device)
                if args.device.startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        model(x)
                else:
                    model(x)
    finally:
        for h in handles:
            h.remove()
    section = {"run": str(run_dir), "step": step, "pos": cfg.pos,
               "trained_block_size": cfg.block_size, "length": int(windows.size(1)),
               "layers": {}}
    for layer in layers:
        x_norm = torch.cat(captured[layer])
        with torch.no_grad():
            rows = attention_rows(model.transformer.h[layer].attn.cpu(), x_norm, qpos)
        section["layers"][str(layer)] = attention_summary(rows, qpos, last_k=args.last_k)
        line = " ".join(f"p={e['qpos']}:H%={e['entropy_norm']:.2f}"
                        for e in section["layers"][str(layer)])
        print(f"  attn {run_dir.name} layer={layer}: {line}", flush=True)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return section


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nope-run", default="runs/nope-ab-nope")
    ap.add_argument("--rope-run", default="runs/nope-ab-rope")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-dir", default="data/shards/fineweb-100bt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    # position probe
    ap.add_argument("--probe-length", type=int, default=1024)
    ap.add_argument("--n-seqs", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=16)
    ap.add_argument("--forward-batch", type=int, default=8)
    ap.add_argument("--tap-layers", default="0,3,6,9,11")
    ap.add_argument("--probe-hidden", type=int, default=512)
    ap.add_argument("--probe-epochs", type=int, default=8)
    ap.add_argument("--probe-batch", type=int, default=8192)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    # attention profiles
    ap.add_argument("--attn-length", type=int, default=4096)
    ap.add_argument("--attn-seqs", type=int, default=8)
    ap.add_argument("--attn-batch", type=int, default=2)
    ap.add_argument("--attn-layers", default="0,6,11")
    ap.add_argument("--query-positions", default="64,256,512,1000,1023,1536,2047,3071,4095")
    ap.add_argument("--last-k", type=int, default=64)
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {args.device!r} requested but CUDA is unavailable")

    torch.set_float32_matmul_precision("high")
    report: dict = {"seed": args.seed, "position_probe": {}, "attention": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:  # progressive write: on-disk report is current per section
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    t0 = time.time()
    probe_windows = _val_windows(args.data_dir, args.probe_length, args.n_seqs,
                                 args.forward_batch, args.seed)
    for name, run in (("nope", Path(args.nope_run)), ("rope", Path(args.rope_run))):
        print(f"== position probe: {run} ==", flush=True)
        report["position_probe"][name] = probe_section(run, args, probe_windows)
        flush()
    attn_windows = _val_windows(args.data_dir, args.attn_length, args.attn_seqs,
                                args.attn_batch, args.seed)
    for name, run in (("nope", Path(args.nope_run)), ("rope", Path(args.rope_run))):
        print(f"== attention profiles: {run} ==", flush=True)
        report["attention"][name] = attention_section(run, args, attn_windows)
        flush()
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
