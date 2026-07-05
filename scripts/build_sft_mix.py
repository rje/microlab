"""Build a mixed instruction-tuning corpus from several public sources, normalized to the
{instruction, context, response} JSONL schema that scripts/sft.py consumes (via load_dolly).

Diversity of instruction *types* is what SFT needs, so a mix of human-written and templated
sources beats any single set. All sources here are permissive/CC:

  - Databricks Dolly-15k  (local JSONL, human-written)               -> data/corpora/dolly15k.jsonl
  - Stanford Alpaca 52k   (HF tatsu-lab/alpaca, {instruction,input,output})
  - No Robots 10k         (HF HuggingFaceH4/no_robots, human, chat)  single-turn rows only

    python scripts/build_sft_mix.py --out data/corpora/sft_mix.jsonl

The mapping functions (normalize_alpaca / normalize_no_robots) are pure so they can be unit
tested without touching the network; the load_* wrappers are thin HF streaming adapters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.data.reference.loaders import load_dolly  # noqa: E402

Row = dict[str, str]


def normalize_alpaca(row: dict) -> Row | None:
    """Alpaca {instruction, input, output} -> {instruction, context, response}. The Alpaca
    'input' is optional context. Returns None if the instruction or output is empty."""
    instruction = (row.get("instruction") or "").strip()
    response = (row.get("output") or "").strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "context": (row.get("input") or "").strip(),
            "response": response}


def normalize_no_robots(row: dict) -> Row | None:
    """No Robots stores a `messages` list of {role, content}. Keep only clean single-turn
    rows (one user prompt, one assistant reply) and map user->instruction, assistant->response.
    Returns None for multi-turn rows or when either side is empty."""
    messages = row.get("messages") or []
    if len(messages) != 2 or messages[0].get("role") != "user" \
            or messages[1].get("role") != "assistant":
        return None
    instruction = (messages[0].get("content") or "").strip()
    response = (messages[1].get("content") or "").strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "context": "", "response": response}


def _load_hf(dataset: str, split: str, normalize, limit: int | None) -> list[Row]:
    """Stream an HF dataset and apply a normalizer, dropping rows it rejects (returns None)."""
    from datasets import load_dataset  # optional/heavy dep

    rows: list[Row] = []
    for raw in load_dataset(dataset, split=split, streaming=True):
        norm = normalize(raw)
        if norm is not None:
            rows.append(norm)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def build_mix(dolly_path: str, limit_per_source: int | None = None,
              no_robots_split: str = "train", seed: int = 0) -> tuple[list[Row], dict]:
    """Assemble the normalized mix and a per-source count. Deterministically shuffled so the
    SFT batches interleave sources rather than seeing all of one set then all of the next."""
    import random

    dolly = load_dolly(dolly_path, limit=limit_per_source)
    alpaca = _load_hf("tatsu-lab/alpaca", "train", normalize_alpaca, limit_per_source)
    no_robots = _load_hf("HuggingFaceH4/no_robots", no_robots_split, normalize_no_robots,
                         limit_per_source)
    counts = {"dolly": len(dolly), "alpaca": len(alpaca), "no_robots": len(no_robots)}
    mix = dolly + alpaca + no_robots
    random.Random(seed).shuffle(mix)
    counts["total"] = len(mix)
    return mix, counts


def write_jsonl(rows: list[Row], out: str | Path) -> int:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dolly", default="data/corpora/dolly15k.jsonl")
    ap.add_argument("--out", default="data/corpora/sft_mix.jsonl")
    ap.add_argument("--limit-per-source", type=int, default=None,
                    help="cap rows per source (smoke runs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mix, counts = build_mix(args.dolly, limit_per_source=args.limit_per_source, seed=args.seed)
    n = write_jsonl(mix, args.out)
    print(f"sources: {counts}")
    print(f"wrote {n} instruction examples -> {args.out}")


if __name__ == "__main__":
    main()
