"""Supervised fine-tune the FineWeb base model into an instruction-following chat model.

    python scripts/sft.py --base-ckpt runs/350m/ckpt_13000.pt

Warm-starts from a pretrained VariantGPT checkpoint and trains on Dolly-15k with PROMPT
LOSS MASKING (only response tokens contribute to the loss — see model/reference/sft.py).
Each response is trained with a trailing ``\\n### End`` sentinel so the model LEARNS to
signal completion; serving stops on that string (the tokenizer has no EOS). Writes a
run dir the console can serve directly: ckpt_<step>.pt + tokenizer.json + serve_config.json
(``{"mode":"chat", ...}``), which flips the Playground into chat mode.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from microlab.data.reference.loaders import load_dolly
from microlab.model.reference.checkpoint import latest_checkpoint
from microlab.model.reference.sft import (
    build_sft_example,
    collate_sft,
    format_chat,
    masked_cross_entropy,
)
from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.tokenizer.fast import FastTokenizer

# Trained onto the end of every response so the model learns to emit it; serving stops here.
END_SENTINEL = "\n### End"
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
    ))
    model.load_state_dict(ckpt["model"])
    return model.to(device), cfg


def build_examples(tok, rows: list[dict]) -> list[tuple[list[int], list[int]]]:
    """Turn Dolly {instruction, context, response} rows into (input_ids, labels) with the
    prompt masked. The chat template is the single source of truth (format_chat); the END
    sentinel is appended to the response so it's part of what the model is supervised on.
    Rows with an empty response are skipped (nothing to learn to generate)."""
    examples: list[tuple[list[int], list[int]]] = []
    for row in rows:
        if not row["response"].strip():
            continue
        prompt, _ = format_chat(row["instruction"], row.get("context", ""))
        input_ids, labels = build_sft_example(tok, prompt, row["response"] + END_SENTINEL)
        examples.append((input_ids, labels))
    return examples


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
            seed: int = 1337) -> dict:
    """Fine-tune the base model on Dolly and write a servable chat run dir. Returns
    {"final_loss", "steps", "ckpt_path", "out_dir"}."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(seed)

    tok = FastTokenizer.load(str(tokenizer))
    rows = load_dolly(str(data), limit=limit)
    examples = build_examples(tok, rows)
    if not examples:
        raise ValueError(f"no usable SFT examples from {data} (all responses empty?)")

    model, cfg = load_base_model(base_ckpt, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    steps_per_epoch = math.ceil(len(examples) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, total_steps // 20)  # short warmup (~5%)
    min_lr = lr * 0.1
    use_amp = device.startswith("cuda")

    print(f"SFT: {len(examples)} examples, {epochs} epochs, {total_steps} steps "
          f"(batch {batch_size}, block {block_size}, lr {lr:g}) on {device}")

    rng = torch.Generator().manual_seed(seed)
    step = 0
    last_loss = float("nan")
    for epoch in range(epochs):
        order = torch.randperm(len(examples), generator=rng).tolist()
        for start in range(0, len(examples), batch_size):
            idx = order[start:start + batch_size]
            batch = collate_sft([examples[i] for i in idx], PAD_ID, block_size)
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            cur_lr = cosine_lr(step, warmup, total_steps, lr, min_lr)
            for group in opt.param_groups:
                group["lr"] = cur_lr
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, _ = model(x)
                    loss = masked_cross_entropy(logits, y)
            else:
                logits, _ = model(x)
                loss = masked_cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            last_loss = loss.item()
            if step % log_interval == 0 or step == total_steps:
                print(f"epoch {epoch + 1}/{epochs} step {step}/{total_steps} "
                      f"loss {last_loss:.4f} lr {cur_lr:.2e}")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"ckpt_{step}.pt"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "step": step, "cfg": cfg}, ckpt_path)
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
    args = ap.parse_args()

    run_sft(base_ckpt=args.base_ckpt, data=args.data, out=args.out, tokenizer=args.tokenizer,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            block_size=args.block_size, device=args.device, limit=args.limit)


if __name__ == "__main__":
    main()
