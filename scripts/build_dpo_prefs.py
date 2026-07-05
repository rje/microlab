"""Build DPO preference pairs from Dolly: chosen = the GOLD response, rejected = the SFT
model's OWN sampled generation for the same prompt.

    python scripts/build_dpo_prefs.py --sft-run runs/350m-sft --limit 3000

For each Dolly example we chat-template the instruction (+context), keep the gold answer as
`chosen`, and sample a `rejected` continuation from the SFT model (temperature > 0 so it's a
plausible-but-worse answer). The generation is truncated at the SFT stop strings. Pairs where
the model happened to reproduce the gold (or produced nothing) are dropped. Output is JSONL of
{"prompt", "chosen", "rejected"} that scripts/dpo.py trains on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.data.reference.loaders import load_dolly  # noqa: E402
from microlab.infer.reference.kv_cache import generate_cached  # noqa: E402
from microlab.model.reference.checkpoint import load_variant_from_run  # noqa: E402
from microlab.model.reference.sft import format_chat  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402

# Same stops the SFT run serves with; the rejected sample is cut at the earliest of these.
STOPS = ["### End", "\n### Instruction:"]


def truncate(text: str) -> str:
    """Cut a generated continuation at the earliest SFT stop string and strip it (the shape
    reused from scripts/eval_instructions.py)."""
    cut = min((text.find(s) for s in STOPS if s in text), default=-1)
    return (text[:cut] if cut >= 0 else text).strip()


def sample_rejected(model, tok, prompt: str, device: str, temp: float, max_new: int,
                    generator: torch.Generator) -> str:
    """Sample one continuation of `prompt` from the model and truncate it at the SFT stops."""
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = generate_cached(model, ids, max_new, temperature=temp, generator=generator)
    gen = tok.decode(out[0].tolist()[len(ids[0]):])
    return truncate(gen)


def build_prefs(model, tok, rows: list[dict], device: str, block_size: int, temp: float = 0.8,
                max_new: int = 80, seed: int = 0) -> list[dict[str, str]]:
    """Turn Dolly rows into DPO pairs. The sampling RNG is re-seeded per example (seed + index)
    so each rejected sample is distinct and the run is reproducible. Drops a pair when the gold
    is empty, the sample is empty, the sample matches the gold verbatim, or the templated prompt
    plus the generation budget won't fit in block_size (some Dolly rows carry long context)."""
    prefs: list[dict[str, str]] = []
    skipped_long = 0
    for i, row in enumerate(rows):
        chosen = row["response"]
        if not chosen.strip():
            continue
        prompt, _ = format_chat(row["instruction"], row.get("context", ""))
        # Skip prompts too long to generate a response from — a templated prompt over
        # block_size would fail the model's prefill assertion.
        if len(tok.encode(prompt)) + max_new > block_size:
            skipped_long += 1
            continue
        generator = torch.Generator(device=device).manual_seed(seed + i)
        rejected = sample_rejected(model, tok, prompt, device, temp, max_new, generator)
        if not rejected or rejected.strip() == chosen.strip():
            continue
        prefs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    if skipped_long:
        print(f"build_dpo_prefs: skipped {skipped_long} rows with prompt+{max_new} > "
              f"block_size {block_size}")
    return prefs


def write_prefs(prefs: list[dict[str, str]], out: str | Path) -> int:
    """Write pairs as JSONL (one object per line). Returns the count written."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in prefs:
            f.write(json.dumps(p) + "\n")
    return len(prefs)


def run_build(sft_run: str | Path, data: str | Path, out: str | Path, tokenizer: str | Path,
              limit: int | None = 3000, temp: float = 0.8, max_new: int = 80,
              device: str = "cpu", seed: int = 0) -> dict:
    """Load the SFT model + tokenizer, generate rejected samples over Dolly, and write the
    preference JSONL. Returns {"written", "considered", "out"}."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(seed)

    model, step = load_variant_from_run(Path(sft_run), device=device)
    tok = FastTokenizer.load(str(tokenizer))
    rows = load_dolly(str(data), limit=limit)
    print(f"build_dpo_prefs: {len(rows)} dolly rows, SFT step {step}, temp {temp}, "
          f"max_new {max_new} on {device}")

    prefs = build_prefs(model, tok, rows, device, model.config.block_size,
                        temp=temp, max_new=max_new, seed=seed)
    written = write_prefs(prefs, out)
    print(f"wrote {written} preference pairs (of {len(rows)} rows) -> {out}")
    return {"written": written, "considered": len(rows), "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-run", default="runs/350m-sft",
                    help="SFT run dir (latest ckpt is loaded as the sampler)")
    ap.add_argument("--data", default="data/corpora/dolly15k.jsonl")
    ap.add_argument("--out", default="data/corpora/dpo_prefs.jsonl")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json (default: <sft-run>/tokenizer.json)")
    ap.add_argument("--limit", type=int, default=3000, help="cap dolly rows considered")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer = args.tokenizer or str(Path(args.sft_run) / "tokenizer.json")
    run_build(sft_run=args.sft_run, data=args.data, out=args.out, tokenizer=tokenizer,
              limit=args.limit, temp=args.temp, max_new=args.max_new, device=args.device,
              seed=args.seed)


if __name__ == "__main__":
    main()
