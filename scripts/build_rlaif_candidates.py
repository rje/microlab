"""RLAIF stage 1 — sample K on-policy candidate responses per instruction from a chat model
(default runs/350m-sft-mix). Writes {instruction, prompt, candidates:[...]} JSONL. A judge then
ranks the candidates (stage 2, offline) and assemble_rlaif_prefs.py turns best/worst into
{prompt, chosen, rejected} pairs for dpo.py --loss ipo.

On-policy is the whole point: every candidate is the model's OWN sample, so a best-vs-worst
preference targets behavior the model can actually reach — unlike the off-policy gold-vs-self
pairs that made naive DPO regress on this tiny model.

    python scripts/build_rlaif_candidates.py --sft-run runs/350m-sft-mix --limit 400 --k 4
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

STOPS = ["### End", "\n### Instruction:"]


def truncate(text: str) -> str:
    """Cut a generation at the earliest SFT stop sentinel, then strip."""
    cut = min((text.find(s) for s in STOPS if s in text), default=-1)
    return (text[:cut] if cut >= 0 else text).strip()


def sample_candidates(model, tok, prompt: str, device: str, k: int, temp: float, max_new: int,
                      base_seed: int) -> list[str]:
    """Draw k continuations of `prompt` in ONE batched generate_cached call: the k rows are the
    same prompt tiled, so they need no padding, and batched multinomial samples each row from an
    independent draw off the shared generator (identical logits, different randomness -> distinct
    samples). Far faster than k sequential calls — generation is per-step-overhead-bound at batch
    1, so batching amortizes it near-linearly. Truncated at the SFT stops."""
    prompt_ids = tok.encode(prompt)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device).repeat(k, 1)
    gen = torch.Generator(device=device).manual_seed(base_seed)
    seqs = generate_cached(model, ids, max_new, temperature=temp, generator=gen)
    return [truncate(tok.decode(seqs[j].tolist()[len(prompt_ids):])) for j in range(k)]


def build_candidates(model, tok, rows: list[dict], device: str, block_size: int, k: int = 4,
                     temp: float = 0.8, max_new: int = 80, seed: int = 0,
                     writer=None, start_row: int = 0) -> list[dict]:
    """One record per instruction: {instruction, prompt, candidates, row}. Empty samples are
    dropped and exact duplicates collapsed; a record is skipped when fewer than 2 DISTINCT
    candidates survive (no preference can be formed) or the templated prompt won't fit
    block_size.

    `writer` (when given) receives each kept record AS IT IS PRODUCED, so a long run persists
    progress instead of buffering everything (a crash loses at most one instruction — the
    long-jobs-write-progressively lesson). `row` is the source index; resuming passes
    start_row = last done row + 1 and re-derives per-row seeds, so a resumed run samples the
    exact candidates an uninterrupted run would have."""
    items: list[dict] = []
    skipped_long = 0
    for i, row in enumerate(rows):
        if i < start_row:
            continue
        prompt, _ = format_chat(row["instruction"], row.get("context", ""))
        if len(tok.encode(prompt)) + max_new > block_size:
            skipped_long += 1
            continue
        cands = sample_candidates(model, tok, prompt, device, k, temp, max_new, seed + i * k)
        uniq = list(dict.fromkeys(c for c in cands if c.strip()))  # order-preserving dedup
        if len(uniq) < 2:
            continue
        item = {"instruction": row["instruction"], "prompt": prompt, "candidates": uniq,
                "row": i}
        if writer is not None:
            writer(item)
        items.append(item)
    if skipped_long:
        print(f"skipped {skipped_long} rows with prompt+{max_new} > block_size {block_size}")
    return items


def write_jsonl(items: list[dict], out: str | Path) -> int:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return len(items)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-run", default="runs/350m-sft-mix")
    ap.add_argument("--data", default="data/corpora/dolly15k.jsonl")
    ap.add_argument("--out", default="data/corpora/rlaif_candidates.jsonl")
    ap.add_argument("--tokenizer", default=None, help="default: <sft-run>/tokenizer.json")
    ap.add_argument("--limit", type=int, default=400, help="instructions to sample from")
    ap.add_argument("--k", type=int, default=4, help="candidates per instruction")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(args.seed)

    model, step = load_variant_from_run(Path(args.sft_run), device=device)
    tok = FastTokenizer.load(args.tokenizer or str(Path(args.sft_run) / "tokenizer.json"))
    rows = load_dolly(str(args.data), limit=args.limit)
    print(f"build_rlaif_candidates: {len(rows)} instructions, SFT step {step}, k {args.k}, "
          f"temp {args.temp}, max_new {args.max_new} on {device}")

    # Append + resume: pick up after the highest already-written source row, so a crash or
    # restart never re-generates (or loses) finished work.
    out_path = Path(args.out)
    start_row = 0
    if out_path.exists():
        done = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
        if done:
            start_row = max(rec.get("row", -1) for rec in done) + 1
            print(f"resuming: {len(done)} records on disk, continuing from row {start_row}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        def append(item: dict) -> None:
            f.write(json.dumps(item) + "\n")
            f.flush()

        build_candidates(model, tok, rows, device, model.config.block_size, k=args.k,
                         temp=args.temp, max_new=args.max_new, seed=args.seed,
                         writer=append, start_row=start_row)
    n = sum(1 for _ in out_path.open())  # total on disk (resumed runs: old + new)
    print(f"wrote {n} candidate records (of {len(rows)} instructions) -> {args.out}")


if __name__ == "__main__":
    main()
