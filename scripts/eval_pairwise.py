"""Pairwise A/B eval: greedy-complete N held-out instructions with two runs, judge each pair
with codex in BOTH orders (position-swapped), and report a win-rate. Far more rigorous than
eyeballing 15 completions, and reusable for any model comparison (base / SFT / RLAIF / ...).

    python scripts/eval_pairwise.py runs/350m-sft-mix runs/350m-rlaif-5k \\
        --data data/corpora/sft_mix.jsonl --skip 5000 --limit 120

Each instruction is judged twice (A-left/B-right and B-left/A-right); a model only "wins" that
instruction when it's preferred in BOTH orderings — inconsistent verdicts are position bias and
count as ties. Uses the same codex plumbing as rlaif_judge (schema-forced JSON, read-only).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.data.reference.loaders import load_dolly  # noqa: E402
from microlab.infer.reference.kv_cache import generate_cached  # noqa: E402
from microlab.model.reference.checkpoint import load_variant_from_run  # noqa: E402
from microlab.model.reference.sft import format_chat  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402

STOPS = ["### End", "\n### Instruction:"]

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["verdicts"],
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["item", "better"],
        "properties": {"item": {"type": "integer"},
                       "better": {"type": "string", "enum": ["left", "right", "tie"]}}}}},
}


def truncate(text: str) -> str:
    cut = min((text.find(s) for s in STOPS if s in text), default=-1)
    return (text[:cut] if cut >= 0 else text).strip()


def complete(model, tok, instruction: str, device: str, max_new: int = 80) -> str:
    """Greedy chat completion for one instruction, truncated at the SFT stops."""
    prompt = format_chat(instruction)[0]
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = generate_cached(model, ids, max_new, temperature=0.0)
    return truncate(tok.decode(out[0].tolist()[len(ids[0]):]))


def build_pair_prompt(batch: list[tuple[int, str, str, str]]) -> str:
    """batch: (item_id, instruction, left, right). Ask which response better follows each
    instruction, judged on correctness > instruction-following > coherence (penalize
    hallucination/looping); "tie" only if genuinely equal."""
    lines = ["You are comparing two responses (LEFT vs RIGHT) from small language models. For "
             "each item, say which better answers the instruction, by correctness > instruction-"
             "following > coherence (penalize hallucination and looping). Use 'tie' only if "
             "genuinely equal. Judge on merit, not position or length.", "", "Items:"]
    for item_id, instruction, left, right in batch:
        lines.append(f"item {item_id}: instruction={instruction!r}")
        lines.append(f"  LEFT:  {left!r}")
        lines.append(f"  RIGHT: {right!r}")
    ids = ", ".join(str(b[0]) for b in batch)
    lines += ["", f"Return JSON per the schema: verdicts=[{{item,better}}] where better is "
              f"'left'|'right'|'tie', exactly one entry for each item id: {ids}."]
    return "\n".join(lines)


def parse_pair_verdicts(text: str, valid_ids: set[int]) -> dict[int, str]:
    """Parse codex output into {item_id: 'left'|'right'|'tie'}, keeping only known item ids."""
    obj = json.loads(text)
    out = {}
    for v in obj["verdicts"]:
        item, better = v.get("item"), v.get("better")
        if item in valid_ids and better in ("left", "right", "tie"):
            out[item] = better
    return out


def resolve_pair(ab: str | None, ba: str | None) -> str:
    """Combine the two orderings into a per-instruction outcome. In order AB, LEFT is model A;
    in order BA, LEFT is model B. A (or B) wins only if preferred in BOTH orderings; anything
    else — disagreement or any tie/missing — is a tie (treats position bias as no-decision)."""
    a_ab = {"left": "A", "right": "B", "tie": "tie", None: "tie"}[ab]
    a_ba = {"left": "B", "right": "A", "tie": "tie", None: "tie"}[ba]
    if a_ab == "A" and a_ba == "A":
        return "A"
    if a_ab == "B" and a_ba == "B":
        return "B"
    return "tie"


def _codex_judge(prompt: str, schema_path: Path, timeout: int) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        out = Path(tf.name)
    cmd = ["codex", "exec", "-s", "read-only", "--skip-git-repo-check",
           "--output-schema", str(schema_path), "-o", str(out), "-"]
    proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"codex exit {proc.returncode}: {proc.stderr[-300:]}")
    text = out.read_text()
    out.unlink(missing_ok=True)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--data", default="data/corpora/sft_mix.jsonl")
    ap.add_argument("--skip", type=int, default=5000, help="skip the first N (RLAIF-trained) rows")
    ap.add_argument("--limit", type=int, default=120, help="held-out instructions to eval")
    ap.add_argument("--batch-size", type=int, default=20, help="items per codex call")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/pairwise_eval.json"))
    args = ap.parse_args()

    rows = load_dolly(args.data, limit=args.skip + args.limit)[args.skip:args.skip + args.limit]
    instrs = [r["instruction"] for r in rows]
    print(f"pairwise eval: {len(instrs)} held-out instructions, {args.run_a.name} vs "
          f"{args.run_b.name} on {args.device}", flush=True)

    ma, _ = load_variant_from_run(args.run_a, device=args.device)
    mb, _ = load_variant_from_run(args.run_b, device=args.device)
    tok = FastTokenizer.load(str(args.run_a / "tokenizer.json"))
    a = [complete(ma, tok, ins, args.device) for ins in instrs]
    b = [complete(mb, tok, ins, args.device) for ins in instrs]
    print("generated completions; judging (position-swapped)...", flush=True)

    # two orderings per instruction: even item id = AB (left=A), odd = BA (left=B)
    items = []
    for i, ins in enumerate(instrs):
        items.append((2 * i, ins, a[i], b[i]))
        items.append((2 * i + 1, ins, b[i], a[i]))

    schema_dir = Path(tempfile.mkdtemp())
    schema_path = schema_dir / "schema.json"
    schema_path.write_text(json.dumps(SCHEMA))
    verdicts: dict[int, str] = {}
    for start in range(0, len(items), args.batch_size):
        batch = items[start:start + args.batch_size]
        text = _codex_judge(build_pair_prompt(batch), schema_path, args.timeout)
        verdicts.update(parse_pair_verdicts(text, {it[0] for it in batch}))
        print(f"  judged items {start}-{start + len(batch) - 1}", flush=True)

    outcomes = Counter()
    per_instruction = []
    for i, ins in enumerate(instrs):
        outcome = resolve_pair(verdicts.get(2 * i), verdicts.get(2 * i + 1))
        outcomes[outcome] += 1
        per_instruction.append({"instruction": ins, "a": a[i], "b": b[i], "outcome": outcome})

    a_w, b_w, tie = outcomes["A"], outcomes["B"], outcomes["tie"]
    decided = a_w + b_w
    wr = 100 * a_w / decided if decided else 0.0
    print(f"\n{args.run_a.name}: {a_w} wins | {args.run_b.name}: {b_w} wins | {tie} ties "
          f"(of {len(instrs)})")
    print(f"{args.run_a.name} win-rate among decided: {wr:.0f}%  "
          f"({args.run_b.name}: {100 - wr:.0f}%)")
    args.out.write_text(json.dumps({"run_a": str(args.run_a), "run_b": str(args.run_b),
                                    "a_wins": a_w, "b_wins": b_w, "ties": tie,
                                    "per_instruction": per_instruction}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
