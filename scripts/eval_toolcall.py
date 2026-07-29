"""Tool-call eval: can a chat run pick the right tool and arguments for a request?

    python scripts/eval_toolcall.py --run runs/350m-sft-mix \\
        --out evals/code/350m-sft-mix-toolcall.jsonl

Each item shows a compact per-item tool list (see evals/toolcall/registry.json and
microlab.evals.code.toolcall for the prompt design + token budget), two fixed few-shot
turns in the trained chat template, then the request; the model must answer with
{"tool": ..., "arguments": {...}} — or the `clarify` / `none` markers. Decoding is
greedy and the few-shot is fixed, so the eval is deterministic.

Scoring: exact tool match + argument F1 (normalized values), reported overall and per
split ("pattern" = surface cues name the tool; "compositional" = inference, wrong-tool
distractors, missing-info clarify, out-of-scope none). Per-item records append to --out
(JSONL, resumable, config pinned in the header line); the aggregate summary lands next
to it as .summary.json.

The prompt is measured with the run's own tokenizer per item and the script RAISES if
prompt + --max-new exceeds the model's block size (fit matters: the 350M has block
1024)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.evals.code.gen import generate_until  # noqa: E402
from microlab.evals.code.prompts import CHAT_STOPS  # noqa: E402
from microlab.evals.code.toolcall import (  # noqa: E402
    aggregate,
    build_prompt,
    load_items,
    load_registry,
    score_item,
)
from microlab.model.reference.checkpoint import load_variant_from_run  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402


def read_resume(out_path: Path, header: dict) -> set[str]:
    """Item ids already recorded; the stored config header must match ours exactly."""
    if not out_path.exists():
        return set()
    lines = [json.loads(x) for x in
             out_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        return set()
    if "_header" not in lines[0] or lines[0]["_header"] != header:
        raise ValueError(
            f"{out_path} exists with a different config header; point --out somewhere "
            f"fresh (have {lines[0].get('_header')}, want {header})"
        )
    return {r["id"] for r in lines[1:]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="per-item results JSONL (appended progressively; resumable)")
    ap.add_argument("--registry", type=Path, default=Path("evals/toolcall/registry.json"))
    ap.add_argument("--items", type=Path, default=Path("evals/toolcall/items.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="first N items only")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    registry = load_registry(args.registry)
    items = load_items(args.items, registry)
    if args.limit is not None:
        items = items[: args.limit]

    header = {"run": str(args.run), "registry": str(args.registry),
              "items": str(args.items), "max_new": args.max_new, "limit": args.limit}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = read_resume(args.out, header)
    if not done and (not args.out.exists() or not args.out.read_text().strip()):
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"_header": header}) + "\n")

    model, step = load_variant_from_run(args.run, device=args.device)
    tok = FastTokenizer.load(str(args.run / "tokenizer.json"))
    block = model.config.block_size
    print(f"run={args.run} step={step} block={block} items={len(items)} "
          f"(resuming past {len(done)})")

    t0 = time.time()
    for i, item in enumerate(items):
        if item["id"] in done:
            continue
        prompt_ids = tok.encode(build_prompt(item, registry))
        if len(prompt_ids) + args.max_new > block:
            raise ValueError(
                f"item {item['id']}: prompt is {len(prompt_ids)} tokens; + max_new "
                f"{args.max_new} exceeds block_size {block} — trim the item's tool list"
            )
        reply = generate_until(model, tok, prompt_ids, max_new=args.max_new,
                               stops=CHAT_STOPS, device=args.device)
        rec = score_item(item, reply)
        rec["reply"] = reply.strip()
        rec["prompt_tokens"] = len(prompt_ids)
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        mark = "OK " if rec["strict"] else ("tool" if rec["tool_correct"] else "MISS")
        print(f"[{i + 1}/{len(items)}] {item['id']}: {mark} f1={rec['arg_f1']:.2f} "
              f"[{time.time() - t0:.0f}s]", flush=True)

    records = [json.loads(x) for x in
               args.out.read_text(encoding="utf-8").splitlines()[1:] if x.strip()]
    summary = {"header": header, "ckpt_step": step, **aggregate(records)}
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out} and {summary_path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
