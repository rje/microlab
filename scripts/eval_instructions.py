"""Paired instruction completions for a before/after read: run the SAME chat-templated
instruction through two runs (e.g. a base model and its SFT'd sibling), greedy-decode, stop
at the SFT sentinels, and dump {instruction, a, b} pairs for a human/stronger-model judge.

    python scripts/eval_instructions.py runs/350m runs/350m-sft --out runs/sft_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.infer.reference.kv_cache import generate_cached  # noqa: E402
from microlab.model.reference.checkpoint import load_variant_from_run  # noqa: E402
from microlab.model.reference.sft import format_chat  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402

INSTRUCTIONS = [
    "What is the capital of France?",
    "List three primary colors.",
    "Explain what photosynthesis is.",
    "Rewrite this sentence to be more polite: Give me the report.",
    "What does the word 'ephemeral' mean?",
    "If a train travels 60 miles in 2 hours, how fast is it going?",
    "Write a short poem about the ocean.",
    "Name two planets in our solar system.",
    "Summarize the benefits of exercise in one sentence.",
    "How do you make a cup of tea?",
    "What is the difference between a cat and a dog?",
    "Give me one tip for staying focused while working.",
    "What is 15 plus 27?",
    "What is the opposite of 'happy'?",
    "Is a tomato a fruit or a vegetable?",
]
STOPS = ["### End", "\n### Instruction:"]


def truncate(text: str) -> str:
    cut = min((text.find(s) for s in STOPS if s in text), default=-1)
    return (text[:cut] if cut >= 0 else text).strip()


def complete(model, tok, instruction: str, device: str, max_new: int = 80) -> str:
    prompt = format_chat(instruction)[0]
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = generate_cached(model, ids, max_new, temperature=0.0)
    gen = tok.decode(out[0].tolist()[len(ids[0]):])
    return truncate(gen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path, help="e.g. runs/350m (base)")
    ap.add_argument("run_b", type=Path, help="e.g. runs/350m-sft")
    ap.add_argument("--out", type=Path, default=Path("runs/sft_eval.json"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ma, _ = load_variant_from_run(args.run_a, device=args.device)
    mb, _ = load_variant_from_run(args.run_b, device=args.device)
    tok = FastTokenizer.load(str(args.run_b / "tokenizer.json"))

    pairs = []
    for ins in INSTRUCTIONS:
        a = complete(ma, tok, ins, args.device)
        b = complete(mb, tok, ins, args.device)
        pairs.append({"instruction": ins, "a_base": a, "b_sft": b})
        print(f"\n### {ins}\n  [base] {a!r}\n  [sft ] {b!r}")

    args.out.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
