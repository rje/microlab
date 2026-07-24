"""Train a Bradley-Terry reward model on preference pairs, warm-started from the SFT chat
model (Phase 11 at 1B scale).

    python scripts/train_reward_model.py --base-ckpt runs/1b-sft-mix \
        --prefs data/corpora/rlaif_prefs_1b.jsonl --out runs/1b-rm

The backbone is the SFT VariantGPT; its LM head is replaced by a fresh scalar head scored at
the last non-pad token of each (prompt + response + "\\n### End") sequence — the sentinel is
included so the reward reads a COMPLETE, terminated response, matching how sft.py/dpo.py
define one. A seeded shuffle holds out the LAST --holdout pairs (written to holdout.jsonl so
future evals reuse the exact split; the split is over raw pairs, before any tokenize-time
skip). Prompts too long for the block are truncated from the LEFT (the response is never cut);
a pair whose response + sentinel fills the whole block is SKIPPED with a counted, printed
total — one absurd pair shouldn't kill a run, but it must never pass silently.

After each epoch the model is evaluated on the holdout (pairwise accuracy + mean margin) and,
as a cross-distribution check, on the first --uf-n pairs of --uf-prefs (UltraFeedback,
off-policy). The checkpoint kept at the end is the epoch with the BEST holdout accuracy
(earliest wins ties — less overfit); eval.json records it as "kept_epoch". Progressive
outputs: holdout.jsonl before training, train_log.jsonl appended per optimizer step and per
epoch eval, per-epoch checkpoints (pruned to the best at the end), eval.json last.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.model.reference.checkpoint import latest_checkpoint  # noqa: E402
from microlab.model.reference.variants import VariantConfig, VariantGPT  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402
from microlab.train.reward import (  # noqa: E402
    RewardModel,
    bradley_terry_loss,
    collate_reward,
    save_reward_checkpoint,
)

# Appended to every response so the score is read after a complete, terminated response —
# the same definition of "complete" that sft.py trains and dpo.py optimizes.
END_SENTINEL = "\n### End"
PAD_ID = 0  # matches collate_sft's default and the tokenizer convention here


def load_prefs(path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    """Read the preference JSONL ({prompt, chosen, rejected} per line)."""
    rows: list[dict[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def split_holdout(n: int, holdout: int, seed: int) -> tuple[list[int], list[int]]:
    """Seeded shuffle of range(n); the LAST `holdout` indices are held out, the rest train.
    Deterministic in (n, holdout, seed) so the split is reproducible forever."""
    if not 0 < holdout < n:
        raise ValueError(f"holdout must be in (0, {n}), got {holdout}")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[:-holdout], perm[-holdout:]


def build_reward_sequences(
    tok, rows: list[dict[str, str]], block_size: int
) -> tuple[list[tuple[list[int], list[int]]], int]:
    """Tokenize each pair into (prompt+chosen+sentinel, prompt+rejected+sentinel) token lists.
    Overlong prompts are truncated from the LEFT (keeping the context nearest the response);
    the response is never cut. A pair either of whose responses (+ sentinel) fills the whole
    block — leaving no prompt at all — is dropped; the count of dropped pairs is returned so
    the caller can report it. Raises if every pair was dropped."""
    pairs: list[tuple[list[int], list[int]]] = []
    skipped = 0
    for row in rows:
        prompt_ids = tok.encode(row["prompt"])
        sides: list[list[int]] = []
        for key in ("chosen", "rejected"):
            resp_ids = tok.encode(row[key] + END_SENTINEL)
            keep = block_size - len(resp_ids)
            if keep < 1:
                break  # response can't fit with any prompt context -> drop the whole pair
            sides.append(prompt_ids[-keep:] + resp_ids)
        if len(sides) == 2:
            pairs.append((sides[0], sides[1]))
        else:
            skipped += 1
    if rows and not pairs:
        raise ValueError(f"all {len(rows)} pairs were dropped (block_size {block_size} too "
                         f"small for these responses)")
    return pairs, skipped


def load_backbone(base_ckpt: str | Path, device: str) -> tuple[VariantGPT, int, Path]:
    """Warm start: rebuild the VariantGPT from the base checkpoint (a file, or a run dir whose
    latest ckpt is picked). Loads to CPU first so the bundled optimizer state (~2x model size)
    never touches the GPU (same rationale as load_variant_from_run)."""
    p = Path(base_ckpt)
    ckpt_path = latest_checkpoint(p) if p.is_dir() else p
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, norm=cfg.norm, pos=cfg.pos,
        mlp=cfg.mlp, n_kv_head=getattr(cfg, "n_kv_head", None)))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt["step"], ckpt_path


def cosine_lr(step: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    """Linear warmup 0 -> base_lr over `warmup` steps, then cosine decay to min_lr (matches
    scripts/sft.py and the pretraining Trainer schedule)."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (base_lr - min_lr)


def _score_pairs(model: RewardModel, pairs: list[tuple[list[int], list[int]]], idx: list[int],
                 device: str, use_amp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward over a micro-batch of pairs: chosen and rejected are collated TOGETHER
    (2m sequences, shared padding) and split after. Returns (r_chosen, r_rejected), fp32."""
    seqs = [pairs[i][0] for i in idx] + [pairs[i][1] for i in idx]
    batch = collate_reward(seqs, PAD_ID)
    input_ids = batch["input_ids"].to(device)
    lengths = batch["lengths"].to(device)
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scores = model(input_ids, lengths)
    else:
        scores = model(input_ids, lengths)
    scores = scores.float()  # loss/margins in fp32 regardless of autocast
    return scores[:len(idx)], scores[len(idx):]


@torch.no_grad()
def evaluate_pairs(model: RewardModel, pairs: list[tuple[list[int], list[int]]], device: str,
                   batch_size: int, use_amp: bool) -> dict:
    """Pairwise accuracy (strict wins), mean margin, and count over a pair set."""
    was_training = model.training
    model.eval()
    wins, margin_sum = 0, 0.0
    for start in range(0, len(pairs), batch_size):
        idx = list(range(start, min(start + batch_size, len(pairs))))
        r_c, r_r = _score_pairs(model, pairs, idx, device, use_amp)
        wins += (r_c > r_r).sum().item()
        margin_sum += (r_c - r_r).sum().item()
    if was_training:
        model.train()
    return {"acc": wins / len(pairs), "margin_mean": margin_sum / len(pairs), "n": len(pairs)}


def run_train_reward(base_ckpt: str | Path, prefs: str | Path, out: str | Path,
                     tokenizer: str | Path, uf_prefs: str | Path, epochs: int = 2,
                     lr: float = 1e-5, batch_size: int = 4, grad_accum: int = 4,
                     holdout: int = 231, uf_n: int = 231, seed: int = 1337,
                     device: str = "cpu", limit: int | None = None,
                     log_interval: int = 10) -> dict:
    """Train the reward model, evaluate per epoch, keep the best-holdout checkpoint, and write
    eval.json. Returns {"steps", "train_acc_final", "eval", "ckpt_path", "out_dir", ...}."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {device!r} requested but CUDA is unavailable")
    torch.manual_seed(seed)

    tok = FastTokenizer.load(str(tokenizer))
    rows = load_prefs(prefs, limit=limit)
    if not rows:
        raise ValueError(f"no preference pairs in {prefs}")
    train_idx, hold_idx = split_holdout(len(rows), holdout, seed)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The exact held-out pairs, written BEFORE training so any future eval reuses this split.
    with (out_dir / "holdout.jsonl").open("w", encoding="utf-8") as f:
        for i in hold_idx:
            f.write(json.dumps({"index": i, **rows[i]}) + "\n")

    model_backbone, base_step, base_path = load_backbone(base_ckpt, device)
    block_size = model_backbone.config.block_size
    model = RewardModel(model_backbone).to(device)
    model.train()

    train_pairs, sk_train = build_reward_sequences(tok, [rows[i] for i in train_idx], block_size)
    hold_pairs, sk_hold = build_reward_sequences(tok, [rows[i] for i in hold_idx], block_size)
    uf_rows = load_prefs(uf_prefs, limit=uf_n)
    if not uf_rows:
        raise ValueError(f"no preference pairs in {uf_prefs}")
    uf_pairs, sk_uf = build_reward_sequences(tok, uf_rows, block_size)
    skipped = {"train": sk_train, "holdout": sk_hold, "uf": sk_uf}
    if any(skipped.values()):
        print(f"skipped pairs (response + sentinel fills block {block_size}): {skipped}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    eff_batch = batch_size * grad_accum
    steps_per_epoch = math.ceil(len(train_pairs) / eff_batch)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, total_steps // 20)
    min_lr = lr * 0.1
    use_amp = device.startswith("cuda")

    print(f"RM: {len(train_pairs)} train pairs (holdout {len(hold_pairs)}, uf {len(uf_pairs)}), "
          f"{epochs} epochs, {total_steps} steps (batch {batch_size} x accum {grad_accum} = "
          f"{eff_batch}, block {block_size}, lr {lr:g}) on {device}", flush=True)

    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")  # fresh log per run (appended progressively)

    def log_line(record: dict) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    rng = torch.Generator().manual_seed(seed)
    step = 0
    t0 = time.time()
    epoch_accs: list[float] = []
    per_epoch: list[dict] = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_pairs), generator=rng).tolist()
        epoch_wins, epoch_pairs_seen = 0, 0
        for start in range(0, len(train_pairs), eff_batch):
            eff_idx = order[start:start + eff_batch]
            cur_lr = cosine_lr(step, warmup, total_steps, lr, min_lr)
            for group in opt.param_groups:
                group["lr"] = cur_lr
            opt.zero_grad(set_to_none=True)
            micros = [eff_idx[m:m + batch_size] for m in range(0, len(eff_idx), batch_size)]
            loss_val, wins = 0.0, 0
            for idx in micros:
                r_c, r_r = _score_pairs(model, train_pairs, idx, device, use_amp)
                micro_loss = bradley_terry_loss(r_c, r_r)
                (micro_loss / len(micros)).backward()
                loss_val += micro_loss.item() / len(micros)
                wins += (r_c > r_r).sum().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            acc = wins / len(eff_idx)
            epoch_wins += wins
            epoch_pairs_seen += len(eff_idx)
            log_line({"kind": "step", "step": step, "epoch": epoch, "loss": loss_val,
                      "acc": acc, "lr": cur_lr, "elapsed_s": round(time.time() - t0, 1)})
            if step % log_interval == 0 or step == total_steps:
                print(f"epoch {epoch}/{epochs} step {step}/{total_steps} "
                      f"loss {loss_val:.4f} acc {acc:.3f} lr {cur_lr:.2e}", flush=True)

        train_acc = epoch_wins / epoch_pairs_seen
        hold_m = evaluate_pairs(model, hold_pairs, device, batch_size, use_amp)
        uf_m = evaluate_pairs(model, uf_pairs, device, batch_size, use_amp)
        ckpt_path = out_dir / f"ckpt_{step}.pt"
        save_reward_checkpoint(ckpt_path, model, step,
                               extra={"base_ckpt": str(base_path), "base_step": base_step})
        record = {"epoch": epoch, "step": step, "train_acc": train_acc,
                  "holdout_acc": hold_m["acc"], "holdout_margin_mean": hold_m["margin_mean"],
                  "uf_acc": uf_m["acc"], "uf_margin_mean": uf_m["margin_mean"],
                  "ckpt": ckpt_path.name}
        per_epoch.append(record)
        epoch_accs.append(train_acc)
        log_line({"kind": "epoch_eval", **record})
        print(f"epoch {epoch}: train_acc {train_acc:.3f} | holdout_acc {hold_m['acc']:.3f} "
              f"margin {hold_m['margin_mean']:.3f} | uf_acc {uf_m['acc']:.3f} -> {ckpt_path}",
              flush=True)

    # Keep the epoch with the best HOLDOUT accuracy (earliest wins ties — less overfit);
    # prune the other per-epoch checkpoints.
    best = max(per_epoch, key=lambda r: (r["holdout_acc"], -r["epoch"]))
    for record in per_epoch:
        if record is not best:
            (out_dir / record["ckpt"]).unlink()
    kept_path = out_dir / best["ckpt"]

    eval_report = {
        "holdout_acc": best["holdout_acc"], "holdout_margin_mean": best["holdout_margin_mean"],
        "uf_acc": best["uf_acc"], "uf_margin_mean": best["uf_margin_mean"],
        "n_holdout": len(hold_pairs), "n_uf": len(uf_pairs),
        "trained_pairs": len(train_pairs), "epochs": epochs, "lr": lr,
        "kept_epoch": best["epoch"], "kept_step": best["step"],
        "train_acc_kept_epoch": best["train_acc"], "holdout": holdout, "seed": seed,
        "skipped_pairs": skipped, "per_epoch": per_epoch,
        "base_ckpt": str(base_path), "base_step": base_step,
    }
    (out_dir / "eval.json").write_text(json.dumps(eval_report, indent=2) + "\n",
                                       encoding="utf-8")
    # Self-contained run dir: co-locate the tokenizer (no serve_config — an RM isn't chat).
    (out_dir / "tokenizer.json").write_text(
        Path(tokenizer).read_text(encoding="utf-8"), encoding="utf-8")

    print(f"\nkept epoch {best['epoch']} ({best['ckpt']})")
    print(f"holdout:       acc {best['holdout_acc']:.3f}  "
          f"mean margin {best['holdout_margin_mean']:.3f}  (n={len(hold_pairs)})")
    print(f"ultrafeedback: acc {best['uf_acc']:.3f}  "
          f"mean margin {best['uf_margin_mean']:.3f}  (n={len(uf_pairs)}, off-policy)")
    print(f"train-vs-holdout acc gap (kept epoch): "
          f"{best['train_acc'] - best['holdout_acc']:+.3f}")
    if device.startswith("cuda"):
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print(f"total time: {time.time() - t0:.0f}s -> {out_dir / 'eval.json'}")

    return {"steps": step, "train_acc_final": epoch_accs[-1], "eval": eval_report,
            "ckpt_path": str(kept_path), "out_dir": str(out_dir)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ckpt", default="runs/1b-sft-mix",
                    help="SFT checkpoint file, or a run dir (latest ckpt is used)")
    ap.add_argument("--prefs", default="data/corpora/rlaif_prefs_1b.jsonl")
    ap.add_argument("--out", default="runs/1b-rm")
    ap.add_argument("--tokenizer", default="runs/1b-sft-mix/tokenizer.json")
    ap.add_argument("--uf-prefs", default="data/corpora/uf_prefs_1b_2309.jsonl",
                    help="off-policy pairs for the cross-distribution eval")
    ap.add_argument("--uf-n", type=int, default=231, help="first N uf pairs to score")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="pairs per micro-batch (2x sequences per forward)")
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="micro-batches per optimizer step (effective batch = batch*accum)")
    ap.add_argument("--holdout", type=int, default=231,
                    help="pairs held out (last N of the seeded shuffle), never trained on")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="cap pairs (smoke runs)")
    ap.add_argument("--log-interval", type=int, default=10)
    args = ap.parse_args()

    run_train_reward(base_ckpt=args.base_ckpt, prefs=args.prefs, out=args.out,
                     tokenizer=args.tokenizer, uf_prefs=args.uf_prefs, epochs=args.epochs,
                     lr=args.lr, batch_size=args.batch_size, grad_accum=args.grad_accum,
                     holdout=args.holdout, uf_n=args.uf_n, seed=args.seed, device=args.device,
                     limit=args.limit, log_interval=args.log_interval)


if __name__ == "__main__":
    main()
