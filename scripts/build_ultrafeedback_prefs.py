"""Build an OFF-POLICY preference set from UltraFeedback for the data-source comparison
(on-policy RLAIF pairs vs public preference pairs, trained with identical IPO configs).

Normalizes HF `HuggingFaceH4/ultrafeedback_binarized` (split train_prefs; rows carry
{prompt, chosen: [messages], rejected: [messages]}) into the {prompt, chosen, rejected}
JSONL that scripts/dpo.py consumes. The prompt goes through format_chat so BOTH arms train
on the exact same template the SFT model saw. Pairs whose longer side would overflow
--block-size after tokenization are dropped (dpo.py asserts on overflow; filtering here
keeps the training set clean rather than masking failures at train time).

    python scripts/build_ultrafeedback_prefs.py --out data/corpora/uf_prefs.jsonl \
        --tokenizer runs/1b-sft-mix/tokenizer.json --limit 2500

normalize_ultrafeedback / fits_block are pure (unit-tested, no network); the HF loader is a
thin streaming wrapper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.model.reference.sft import format_chat  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402

Pair = dict[str, str]


def _last_assistant(messages: list) -> str:
    """Content of the final assistant message, '' when the shape is off."""
    if not messages:
        return ""
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return ""
    return (last.get("content") or "").strip()


def normalize_ultrafeedback(row: dict) -> Pair | None:
    """ultrafeedback_binarized row -> {prompt, chosen, rejected} with the chat template
    applied. None when either side is empty, or chosen == rejected (no signal)."""
    instruction = (row.get("prompt") or "").strip()
    chosen = _last_assistant(row.get("chosen") or [])
    rejected = _last_assistant(row.get("rejected") or [])
    if not instruction or not chosen or not rejected or chosen == rejected:
        return None
    prompt, _ = format_chat(instruction, "")
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def fits_block(pair: Pair, tok, block_size: int) -> bool:
    """True when prompt + the LONGER response tokenizes within block_size (dpo.py packs
    prompt+response into one block; the longer side is the binding constraint)."""
    prompt_len = len(tok.encode(pair["prompt"]))
    longest = max(len(tok.encode(pair["chosen"])), len(tok.encode(pair["rejected"])))
    return prompt_len + longest <= block_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpora/uf_prefs.jsonl")
    ap.add_argument("--tokenizer", default="runs/1b-sft-mix/tokenizer.json")
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=2500, help="pairs to keep")
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--split", default="train_prefs")
    args = ap.parse_args()

    from datasets import load_dataset  # optional/heavy dep

    tok = FastTokenizer.load(args.tokenizer)
    kept, seen, dropped_len, dropped_norm = [], 0, 0, 0
    for row in load_dataset(args.dataset, split=args.split, streaming=True):
        seen += 1
        pair = normalize_ultrafeedback(row)
        if pair is None:
            dropped_norm += 1
            continue
        if not fits_block(pair, tok, args.block_size):
            dropped_len += 1
            continue
        kept.append(pair)
        if len(kept) >= args.limit:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for pair in kept:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)} pairs (scanned {seen}: {dropped_norm} unusable, "
          f"{dropped_len} over block {args.block_size}) -> {out}")


if __name__ == "__main__":
    main()
