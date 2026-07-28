"""Supervised fine-tune the FineWeb base model into an instruction-following chat model.

    python scripts/sft.py --base-ckpt runs/350m/ckpt_13000.pt

Warm-starts from a pretrained VariantGPT checkpoint and trains with PROMPT LOSS MASKING
(only response tokens contribute to the loss — see model/reference/sft.py). Each response
is trained with a trailing ``\\n### End`` sentinel so the model LEARNS to signal
completion; serving stops on that string (the tokenizer has no EOS). Writes a run dir the
console can serve directly: ckpt_<step>.pt + tokenizer.json + serve_config.json
(``{"mode":"chat", ...}``), which flips the Playground into chat mode.

Data: JSONL, auto-detected per line. Old single-turn rows ({instruction, context,
response}, e.g. dolly15k/sft_mix) build exactly as before; multi-turn rows
({"turns": [{"user", "assistant", ...}, ...]}, e.g. chat_mix built by
scripts/build_chat_mix.py) supervise EVERY assistant turn — see
model/reference/chat_sft.py. Chat rows whose final turn alone exceeds --block-size are
skipped and counted (truncation already dropped their leading turns).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from microlab.model.reference.chat_sft import (
    END_SENTINEL,
    TurnTooLongError,
    build_chat_example,
)
from microlab.model.reference.checkpoint import latest_checkpoint
from microlab.model.reference.sft import (
    build_sft_example,
    collate_sft,
    format_chat,
    masked_cross_entropy,
)
from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.tokenizer.fast import FastTokenizer

# Dropped verbatim into the run dir; serve.py reads it to switch this run into chat mode.
SERVE_CONFIG = {"mode": "chat", "stop_strings": ["### End", "\n### Instruction:"]}
PAD_ID = 0  # matches collate_sft's default and the tokenizer convention here


def resolve_base_ckpt(base_ckpt: str | Path) -> Path:
    """A base checkpoint given as a file is used as-is; given as a run DIR, its latest
    ckpt_*.pt is picked (so ``--base-ckpt runs/350m`` also works)."""
    p = Path(base_ckpt)
    return latest_checkpoint(p) if p.is_dir() else p


def load_base_model(base_ckpt: str | Path, device: str) -> tuple[VariantGPT, object]:
    """Warm start: rebuild the VariantGPT from the base checkpoint's saved cfg and load its
    weights. Returns (model, cfg) — cfg is re-saved into the SFT checkpoint so the console's
    load_variant_from_run can serve the fine-tuned run without guessing the architecture."""
    ckpt_path = resolve_base_ckpt(base_ckpt)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, norm=cfg.norm, pos=cfg.pos,
        mlp=cfg.mlp, n_kv_head=getattr(cfg, "n_kv_head", None),
        # context-extended bases (runs/1b-4k) carry a raised RoPE base; dropping it here
        # would silently rebuild the RoPE cache at the 1e4 default and wreck long context
        rope_base=getattr(cfg, "rope_base", 10000.0),
    ))
    model.load_state_dict(ckpt["model"])
    return model.to(device), cfg


def build_examples(
    tok, rows: list[dict], block_size: int | None = None
) -> tuple[list[tuple[list[int], list[int]]], int]:
    """Turn data rows into (input_ids, labels), auto-detecting the schema per row:

    - {"turns": [...]}: multi-turn chat (chat_sft.build_chat_example) — every assistant
      turn supervised, leading turns dropped to fit block_size; a row whose final turn
      alone cannot fit is SKIPPED and counted (second return value), not silently clipped.
    - {"instruction", ...}: the original single-turn path, byte-identical to before
      (format_chat + response + END sentinel, prompt masked, empty responses skipped).

    Any other shape raises — unrecognized data should fail loudly, not train quietly."""
    examples: list[tuple[list[int], list[int]]] = []
    skipped_too_long = 0
    for i, row in enumerate(rows):
        if "turns" in row:
            if block_size is None:
                raise ValueError("block_size is required to build multi-turn chat examples")
            try:
                input_ids, labels, _ = build_chat_example(tok, row["turns"], block_size)
            except TurnTooLongError:
                skipped_too_long += 1
                continue
            examples.append((input_ids, labels))
        elif "instruction" in row:
            if not (row.get("response") or "").strip():
                continue
            prompt, _ = format_chat(row.get("instruction", ""), row.get("context", ""))
            input_ids, labels = build_sft_example(
                tok, prompt, row["response"] + END_SENTINEL)
            examples.append((input_ids, labels))
        else:
            raise ValueError(
                f"row {i}: unrecognized schema (expected 'turns' or 'instruction' key, "
                f"got {sorted(row)})")
    return examples, skipped_too_long


def load_rows(path: str | Path, limit: int | None = None) -> list[dict]:
    """Read data rows from JSONL as-is (schema is dispatched per row in build_examples,
    so single-turn and multi-turn rows can share one file)."""
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def cosine_lr(step: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    """Linear warmup 0 -> base_lr over `warmup` steps, then cosine decay to min_lr over the
    rest of `total` steps (nanoGPT schedule, matched to the pretraining Trainer)."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (base_lr - min_lr)


def run_sft(base_ckpt: str | Path, data: str | Path, out: str | Path, tokenizer: str | Path,
            epochs: int = 3, lr: float = 2e-5, batch_size: int = 16, block_size: int = 1024,
            device: str = "cpu", limit: int | None = None, log_interval: int = 20,
            seed: int = 1337, grad_accum: int = 1, save_every: int = 0) -> dict:
    """Fine-tune the base model on Dolly and write a servable chat run dir. Returns
    {"final_loss", "steps", "ckpt_path", "out_dir"}."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(seed)

    tok = FastTokenizer.load(str(tokenizer))
    rows = load_rows(data, limit=limit)
    examples, skipped_too_long = build_examples(tok, rows, block_size=block_size)
    if skipped_too_long:
        print(f"skipped {skipped_too_long} chat rows whose final turn alone exceeds "
              f"block_size={block_size}", flush=True)
    if not examples:
        raise ValueError(f"no usable SFT examples from {data} (all responses empty?)")

    model, cfg = load_base_model(base_ckpt, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    # `batch_size` is the per-forward micro-batch; grad_accum micro-batches make one
    # optimizer step, so effective batch = batch_size * grad_accum (memory-bound models keep
    # the recipe's effective batch by trading batch_size for accum). Steps = optimizer steps.
    eff_batch = batch_size * grad_accum
    steps_per_epoch = math.ceil(len(examples) / eff_batch)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, total_steps // 20)  # short warmup (~5%)
    min_lr = lr * 0.1
    use_amp = device.startswith("cuda")

    print(f"SFT: {len(examples)} examples, {epochs} epochs, {total_steps} steps "
          f"(batch {batch_size} x accum {grad_accum} = {eff_batch}, block {block_size}, "
          f"lr {lr:g}) on {device}", flush=True)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_periodic: Path | None = None

    def save_ckpt(tag_step: int) -> Path:
        path = out_dir / f"ckpt_{tag_step}.pt"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "step": tag_step, "cfg": cfg}, path)
        return path

    rng = torch.Generator().manual_seed(seed)
    step = 0
    last_loss = float("nan")
    for epoch in range(epochs):
        order = torch.randperm(len(examples), generator=rng).tolist()
        for start in range(0, len(examples), eff_batch):
            eff_idx = order[start:start + eff_batch]
            cur_lr = cosine_lr(step, warmup, total_steps, lr, min_lr)
            for group in opt.param_groups:
                group["lr"] = cur_lr
            opt.zero_grad(set_to_none=True)
            micros = [eff_idx[m:m + batch_size] for m in range(0, len(eff_idx), batch_size)]
            total_loss = 0.0
            for idx in micros:
                batch = collate_sft([examples[i] for i in idx], PAD_ID, block_size)
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits, _ = model(x)
                        loss = masked_cross_entropy(logits, y)
                else:
                    logits, _ = model(x)
                    loss = masked_cross_entropy(logits, y)
                (loss / len(micros)).backward()
                total_loss += loss.item() / len(micros)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            last_loss = total_loss
            if step % log_interval == 0 or step == total_steps:
                print(f"epoch {epoch + 1}/{epochs} step {step}/{total_steps} "
                      f"loss {last_loss:.4f} lr {cur_lr:.2e}", flush=True)
            # Progressive checkpointing: a crash hours in leaves a near-current model, not
            # nothing. Keep only the newest periodic ckpt (final save below is separate).
            if save_every > 0 and step % save_every == 0 and step != total_steps:
                path = save_ckpt(step)
                if last_periodic is not None and last_periodic.exists():
                    last_periodic.unlink()
                last_periodic = path

    ckpt_path = save_ckpt(step)
    if last_periodic is not None and last_periodic.exists() and last_periodic != ckpt_path:
        last_periodic.unlink()
    # Make the run dir self-contained + servable: co-locate the tokenizer and mark it chat.
    (out_dir / "tokenizer.json").write_text(
        Path(tokenizer).read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "serve_config.json").write_text(
        json.dumps(SERVE_CONFIG, indent=2) + "\n", encoding="utf-8")

    print(f"done: final_loss={last_loss:.4f} -> {ckpt_path}")
    print(f"wrote {out_dir / 'serve_config.json'} (chat mode) + tokenizer.json")
    return {"final_loss": last_loss, "steps": step,
            "ckpt_path": str(ckpt_path), "out_dir": str(out_dir)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-ckpt", default="runs/350m/ckpt_13000.pt",
                    help="pretrained checkpoint file, or a run dir (latest ckpt is used)")
    ap.add_argument("--data", default="data/corpora/dolly15k.jsonl")
    ap.add_argument("--out", default="runs/350m-sft")
    ap.add_argument("--tokenizer", default="runs/350m/tokenizer.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="cap examples (smoke runs)")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per optimizer step (effective batch = batch*accum)")
    ap.add_argument("--save-every", type=int, default=0,
                    help=">0: save a rolling checkpoint every N optimizer steps")
    args = ap.parse_args()

    run_sft(base_ckpt=args.base_ckpt, data=args.data, out=args.out, tokenizer=args.tokenizer,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            block_size=args.block_size, device=args.device, limit=args.limit,
            grad_accum=args.grad_accum, save_every=args.save_every)


if __name__ == "__main__":
    main()
