"""Direct Preference Optimization: nudge the SFT chat model toward the preferred (gold)
response and away from its own sampled one, relative to a FROZEN copy of itself.

    python scripts/dpo.py --sft-ckpt runs/350m-sft --prefs data/corpora/dpo_prefs.jsonl

Two models are built from the same SFT checkpoint: a trainable `policy` and a frozen
`reference` (eval + requires_grad_(False), all its forwards under no_grad). Each preference
pair is turned into two prompt-masked SFT-style sequences (chosen / rejected, each with the
`\\n### End` sentinel so the log-prob covers the stop). Per step we take the summed response
log-prob of all four (policy/ref x chosen/rejected) and minimise the DPO loss on the policy
only. The implicit-reward accuracy climbs toward 1.0 as the policy learns to prefer chosen.

The output run dir is servable exactly like the SFT one (ckpt + tokenizer + chat serve_config),
so no serving code changes are needed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.model.reference.checkpoint import latest_checkpoint  # noqa: E402
from microlab.model.reference.dpo import dpo_loss, ipo_loss, sequence_logprob  # noqa: E402
from microlab.model.reference.sft import build_sft_example, collate_sft  # noqa: E402
from microlab.model.reference.variants import VariantConfig, VariantGPT  # noqa: E402

# Appended to both chosen and rejected so the response log-prob covers the stop, matching SFT.
END_SENTINEL = "\n### End"
# Same chat serve config the SFT run writes, so the DPO run serves as a chat model unchanged.
SERVE_CONFIG = {"mode": "chat", "stop_strings": ["### End", "\n### Instruction:"]}
PAD_ID = 0  # matches collate_sft's default and the tokenizer convention here


def resolve_ckpt(sft_ckpt: str | Path) -> Path:
    """A checkpoint given as a file is used as-is; given as a run DIR, its latest ckpt_*.pt is
    picked (so ``--sft-ckpt runs/350m-sft`` also works)."""
    p = Path(sft_ckpt)
    return latest_checkpoint(p) if p.is_dir() else p


def _build_model(cfg, state_dict, device: str) -> VariantGPT:
    """Rebuild a VariantGPT from a saved cfg + weights (a fresh instance with its own params)."""
    model = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, norm=cfg.norm, pos=cfg.pos,
        mlp=cfg.mlp, n_kv_head=getattr(cfg, "n_kv_head", None),
    ))
    model.load_state_dict(state_dict)
    return model.to(device)


def load_policy_reference(
    sft_ckpt: str | Path, device: str
) -> tuple[VariantGPT, VariantGPT, object]:
    """Build the trainable policy and the frozen reference from the SAME SFT checkpoint. The
    reference is put in eval mode with grads disabled so its forwards store no activations and
    never move. Returns (policy, reference, cfg) — cfg is re-saved so the DPO run is servable."""
    ckpt = torch.load(resolve_ckpt(sft_ckpt), map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    policy = _build_model(cfg, ckpt["model"], device)
    reference = _build_model(cfg, ckpt["model"], device)
    reference.eval()
    reference.requires_grad_(False)
    return policy, reference, cfg


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


def build_pref_examples(tok, prefs: list[dict[str, str]]) -> list[tuple]:
    """For each pair, build the chosen and rejected SFT examples (prompt masked to IGNORE_INDEX,
    response + `\\n### End` supervised). Returns a list of (chosen_example, rejected_example),
    each example being (input_ids, labels)."""
    examples = []
    for p in prefs:
        chosen_ex = build_sft_example(tok, p["prompt"], p["chosen"] + END_SENTINEL)
        rejected_ex = build_sft_example(tok, p["prompt"], p["rejected"] + END_SENTINEL)
        examples.append((chosen_ex, rejected_ex))
    return examples


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


def _pair_logps(policy, reference, chosen, rejected, normalize=False):
    """Response log-probs for policy (with grad) and reference (no_grad) on both the chosen and
    rejected batches. Summed, or length-normalized per response token when `normalize`. Returns
    (pol_ch, pol_rej, ref_ch, ref_rej), each (N,)."""
    pol_ch = sequence_logprob(policy(chosen["input_ids"])[0], chosen["labels"], normalize)
    pol_rej = sequence_logprob(policy(rejected["input_ids"])[0], rejected["labels"], normalize)
    with torch.no_grad():
        ref_ch = sequence_logprob(reference(chosen["input_ids"])[0], chosen["labels"], normalize)
        ref_rej = sequence_logprob(
            reference(rejected["input_ids"])[0], rejected["labels"], normalize)
    return pol_ch, pol_rej, ref_ch, ref_rej


def run_dpo(sft_ckpt: str | Path, prefs: str | Path, out: str | Path, tokenizer: str | Path,
            epochs: int = 2, lr: float = 5e-6, beta: float = 0.1, batch_size: int = 8,
            block_size: int = 1024, device: str = "cpu", limit: int | None = None,
            log_interval: int = 10, seed: int = 1337, loss: str = "dpo",
            length_norm: bool = False, grad_accum: int = 1) -> dict:
    """Run preference optimization (DPO or IPO) and write a servable chat run dir. Returns
    {"final_loss", "final_acc", "steps", "loss_history", "acc_history", "ckpt_path", "out_dir"}."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(seed)
    loss_fn = {"dpo": dpo_loss, "ipo": ipo_loss}[loss]

    from microlab.tokenizer.fast import FastTokenizer

    tok = FastTokenizer.load(str(tokenizer))
    pref_rows = load_prefs(prefs, limit=limit)
    examples = build_pref_examples(tok, pref_rows)
    if not examples:
        raise ValueError(f"no preference pairs in {prefs}")

    policy, reference, cfg = load_policy_reference(sft_ckpt, device)
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)

    # micro-batch x grad_accum = effective batch; steps count OPTIMIZER steps (sft.py's
    # convention), so the LR schedule and ckpt names are invariant to how memory forced the
    # micro-batch to shrink.
    eff_batch = batch_size * grad_accum
    steps_per_epoch = math.ceil(len(examples) / eff_batch)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, total_steps // 20)
    min_lr = lr * 0.1
    use_amp = device.startswith("cuda")

    print(f"{loss.upper()}{' +length-norm' if length_norm else ''}: {len(examples)} pairs, "
          f"{epochs} epochs, {total_steps} steps (batch {batch_size}, block {block_size}, "
          f"lr {lr:g}, beta {beta}) on {device}")

    rng = torch.Generator().manual_seed(seed)
    step = 0
    loss_history: list[float] = []
    acc_history: list[float] = []
    for epoch in range(epochs):
        order = torch.randperm(len(examples), generator=rng).tolist()
        for start in range(0, len(examples), eff_batch):
            eff_idx = order[start:start + eff_batch]
            cur_lr = cosine_lr(step, warmup, total_steps, lr, min_lr)
            for group in opt.param_groups:
                group["lr"] = cur_lr
            opt.zero_grad(set_to_none=True)
            micros = [eff_idx[m:m + batch_size] for m in range(0, len(eff_idx), batch_size)]
            batch_loss_val, acc_num = 0.0, 0.0
            for idx in micros:
                chosen = collate_sft([examples[i][0] for i in idx], PAD_ID, block_size)
                rejected = collate_sft([examples[i][1] for i in idx], PAD_ID, block_size)
                chosen = {k: v.to(device) for k, v in chosen.items()}
                rejected = {k: v.to(device) for k, v in rejected.items()}

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pol_ch, pol_rej, ref_ch, ref_rej = _pair_logps(
                            policy, reference, chosen, rejected, length_norm)
                        micro_loss, acc = loss_fn(pol_ch, pol_rej, ref_ch, ref_rej, beta)
                else:
                    pol_ch, pol_rej, ref_ch, ref_rej = _pair_logps(
                        policy, reference, chosen, rejected, length_norm)
                    micro_loss, acc = loss_fn(pol_ch, pol_rej, ref_ch, ref_rej, beta)
                (micro_loss / len(micros)).backward()
                batch_loss_val += micro_loss.item() / len(micros)
                acc_num += acc * len(idx)

            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            acc = acc_num / len(eff_idx)
            loss_history.append(batch_loss_val)
            acc_history.append(acc)
            if step % log_interval == 0 or step == total_steps:
                print(f"epoch {epoch + 1}/{epochs} step {step}/{total_steps} "
                      f"loss {batch_loss_val:.4f} acc {acc:.3f} lr {cur_lr:.2e}")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"ckpt_{step}.pt"
    torch.save({"model": policy.state_dict(), "optimizer": opt.state_dict(),
                "step": step, "cfg": cfg}, ckpt_path)
    # Make the run dir self-contained + servable: co-locate the tokenizer and mark it chat.
    (out_dir / "tokenizer.json").write_text(
        Path(tokenizer).read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "serve_config.json").write_text(
        json.dumps(SERVE_CONFIG, indent=2) + "\n", encoding="utf-8")

    print(f"done: final_loss={loss_history[-1]:.4f} acc={acc_history[-1]:.3f} -> {ckpt_path}")
    print(f"wrote {out_dir / 'serve_config.json'} (chat mode) + tokenizer.json")
    return {"final_loss": loss_history[-1], "final_acc": acc_history[-1], "steps": step,
            "loss_history": loss_history, "acc_history": acc_history,
            "ckpt_path": str(ckpt_path), "out_dir": str(out_dir)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-ckpt", default="runs/350m-sft",
                    help="SFT checkpoint file, or a run dir (latest ckpt is used)")
    ap.add_argument("--prefs", default="data/corpora/dpo_prefs.jsonl")
    ap.add_argument("--out", default="runs/350m-dpo")
    ap.add_argument("--tokenizer", default="runs/350m-sft/tokenizer.json")
    ap.add_argument("--loss", choices=["dpo", "ipo"], default="dpo",
                    help="dpo: -logsigmoid(beta*margin); ipo: (margin - 1/(2*beta))^2, bounded")
    ap.add_argument("--length-norm", action="store_true",
                    help="length-normalize log-probs (SimPO-style) — margin O(1) regardless of "
                         "response length; stabilizes ipo on long responses (raise beta to match)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1,
                    help="dpo: higher = sharper; ipo: target margin is 1/(2*beta), higher = "
                         "closer to reference")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="cap pairs (smoke runs)")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per optimizer step (effective batch = batch*accum)")
    args = ap.parse_args()

    run_dpo(sft_ckpt=args.sft_ckpt, prefs=args.prefs, out=args.out, tokenizer=args.tokenizer,
            epochs=args.epochs, lr=args.lr, beta=args.beta, batch_size=args.batch_size,
            block_size=args.block_size, device=args.device, limit=args.limit, loss=args.loss,
            length_norm=args.length_norm,
            grad_accum=args.grad_accum)


if __name__ == "__main__":
    main()
