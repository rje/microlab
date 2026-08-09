"""Build the DISTILLED arm (B) for the distill-cost A/B: Magicoder-Evol-Instruct-110K
(GPT-4-authored) normalized and token-matched to the compliant arm's supervised-token budget.

    python scripts/build_code_sft_distilled.py \\
        --match data/corpora/code_sft_compliant.jsonl \\
        --out data/corpora/code_sft_distilled.jsonl

This arm INTENTIONALLY violates build-capability (its responses are GPT-4-authored, not
executor-verified or human-authored). It is a measurement instrument only: it is never
merged and never used to seed later training — it exists solely so the distill-cost A/B has
a distilled comparison point against arm A (build_code_sft.py). Same decontamination as arm
A (identical fingerprint source and n-gram size) so the comparison is controlled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.data.code_sft import (  # noqa: E402
    benchmark_fingerprints,
    decontaminate,
    token_match_subsample,
    total_supervised_tokens,
)


def normalize_magicoder(row: dict) -> dict | None:
    """ise-uiuc/Magicoder-Evol-Instruct-110K row {instruction, response} -> Row."""
    instruction = (row.get("instruction") or "").strip()
    response = (row.get("response") or "").strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "context": "", "response": response}


def build_distilled_mix(rows: list[dict], target_tokens: int, tok, seed: int = 0,
                        tolerance: float = 0.02):
    """Token-match `rows` down to `target_tokens` supervised tokens (arm A's budget).

    Raises ValueError if the (already-decontaminated) pool cannot reach the target within
    `tolerance` — token-matching is the A/B's only fairness control, so an under-filled
    distilled arm must fail loudly, not silently run an unfair comparison.
    """
    matched = token_match_subsample(rows, target_tokens, tok, seed=seed)
    matched_tokens = total_supervised_tokens(matched, tok)
    if matched_tokens < target_tokens * (1 - tolerance):
        raise ValueError(
            f"distilled pool has {matched_tokens} supervised tokens but arm A's budget is "
            f"{target_tokens} — cannot token-match a fair A/B (the compliant arm has more "
            f"supervised tokens than the whole decontaminated distilled pool). Reduce arm A "
            f"or change the matching direction before running.")
    return matched, {"target_tokens": target_tokens, "matched_tokens": matched_tokens,
                     "rows": len(matched)}


def _count_supervised_tokens_of_file(path: str, tok) -> int:  # pragma: no cover - IO
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    return total_supervised_tokens(rows, tok)


def main() -> None:  # pragma: no cover - network + IO
    from datasets import load_dataset

    from microlab.evals.code.tasks import load_humaneval, load_mbpp
    from microlab.tokenizer.fast import FastTokenizer

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", required=True,
                    help="arm A jsonl whose supervised-token count to match")
    ap.add_argument("--out", default="data/corpora/code_sft_distilled.jsonl")
    ap.add_argument("--tokenizer", default="runs/coder-1b-step40000/tokenizer.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = FastTokenizer.load(args.tokenizer)
    target = _count_supervised_tokens_of_file(args.match, tok)

    raw = []
    for r in load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train"):
        n = normalize_magicoder(r)
        if n:
            raw.append(n)

    # Decontaminate BEFORE matching so the matched token budget reflects usable rows. Identical
    # call shape to arm A (build_code_sft.py): same benchmark source, same n, so the A/B
    # comparison isn't biased by asymmetric decontamination.
    bench = [t.prompt + "\n" + t.test_program for t in (load_humaneval() + load_mbpp())]
    fp = benchmark_fingerprints(bench, n=10)
    raw, removed = decontaminate(raw, fp, n=10)

    matched, report = build_distilled_mix(raw, target, tok, seed=args.seed)
    report["decontaminated_removed"] = removed

    if not matched:
        raise SystemExit("empty distilled mix — refusing to proceed (verify by count)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in matched:
            f.write(json.dumps(r) + "\n")
    print(f"report: {json.dumps(report)}")
    print(f"wrote {len(matched)} rows -> {out}")


if __name__ == "__main__":
    main()
