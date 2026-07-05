"""RLAIF stage 2 — judge candidate records with the codex CLI (GPT), in parallel batches, with
resume. For each batch, codex picks the best and worst candidate per record (schema-forced JSON);
one verdict file per batch is written for assemble_rlaif_prefs.py to consume. A batch whose
verdict file already exists and validates is skipped, so the run resumes after an interruption or
a rate-limit stall.

    python scripts/rlaif_judge.py --candidates data/corpora/rlaif_candidates_5k.jsonl \\
        --out-dir data/corpora/rlaif_verdicts_5k --batch-size 30 --workers 4

codex quirk: with stdin piped it blocks reading it, so we pass the prompt on stdin explicitly
(prompt arg "-") and force structured output via --output-schema + -o <file>.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["verdicts"],
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["index", "best", "worst"],
        "properties": {"index": {"type": "integer"}, "best": {"type": "integer"},
                       "worst": {"type": "integer"}}}}},
}

RUBRIC = (
    "You are an impartial judge scoring short instruction-following responses from a SMALL "
    "language model. For each record, pick the BEST and the WORST candidate response, as 0-based "
    "indices into that record's candidate list. Judge by, in priority order: (1) correctness — "
    "heavily penalize hallucinations and confident nonsense; (2) instruction-following; (3) "
    "relevance and coherence — penalize looping/repetition. Candidate order is arbitrary. You "
    "should almost always separate a best from a worst; only if truly indistinguishable set them "
    "equal."
)


def build_judge_prompt(batch: list[tuple[int, dict]]) -> str:
    """Render one codex prompt for a batch of (global_index, record) pairs."""
    lines = [RUBRIC, "", "Records:"]
    for idx, rec in batch:
        lines.append(f"index {idx}: instruction={rec['instruction']!r}")
        for j, cand in enumerate(rec["candidates"]):
            lines.append(f"  [{j}] {cand!r}")
    idxs = [str(i) for i, _ in batch]
    lines.append("")
    lines.append(f"Return JSON per the schema: verdicts=[{{index,best,worst}}], exactly one entry "
                 f"for each of these indices: {', '.join(idxs)}.")
    return "\n".join(lines)


def parse_verdicts(text: str, batch: list[tuple[int, dict]]) -> list[dict]:
    """Parse codex's JSON and keep only well-formed verdicts: index belongs to the batch and
    best/worst are in range for that record's candidate count. Raises on unparseable output."""
    obj = json.loads(text)
    by_index = {idx: rec for idx, rec in batch}
    out = []
    for v in obj["verdicts"]:
        idx = v.get("index")
        rec = by_index.get(idx)
        if rec is None:
            continue
        n = len(rec["candidates"])
        best, worst = v.get("best"), v.get("worst")
        if not isinstance(best, int) or not isinstance(worst, int):
            continue
        if 0 <= best < n and 0 <= worst < n:
            out.append({"index": idx, "best": best, "worst": worst})
    return out


def _verdict_file_ok(path: Path, batch: list[tuple[int, dict]]) -> bool:
    """A batch is already done if its file parses and covers most of the batch (>=90%)."""
    if not path.exists():
        return False
    try:
        got = {v["index"] for v in json.loads(path.read_text())}
    except (json.JSONDecodeError, KeyError, TypeError):
        return False
    return len(got & {i for i, _ in batch}) >= 0.9 * len(batch)


def judge_batch(batch: list[tuple[int, dict]], out_path: Path, schema_path: Path,
                model: str | None, timeout: int) -> int:
    """Run codex on one batch and write its verdict list. Returns the count written."""
    prompt = build_judge_prompt(batch)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        codex_out = Path(tf.name)
    cmd = ["codex", "exec", "-s", "read-only", "--skip-git-repo-check",
           "--output-schema", str(schema_path), "-o", str(codex_out), "-"]
    if model:
        cmd[2:2] = ["-m", model]
    proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"codex exit {proc.returncode}: {proc.stderr[-500:]}")
    verdicts = parse_verdicts(codex_out.read_text(), batch)
    out_path.write_text(json.dumps(verdicts))
    codex_out.unlink(missing_ok=True)
    return len(verdicts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default="data/corpora/rlaif_candidates_5k.jsonl")
    ap.add_argument("--out-dir", default="data/corpora/rlaif_verdicts_5k")
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4, help="parallel codex processes")
    ap.add_argument("--model", default=None, help="codex model (-m); default = codex default")
    ap.add_argument("--timeout", type=int, default=900, help="per-codex-call seconds")
    args = ap.parse_args()

    records = [json.loads(x) for x in Path(args.candidates).read_text().splitlines() if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_path = out_dir / "_schema.json"
    schema_path.write_text(json.dumps(SCHEMA))

    batches = []
    for start in range(0, len(records), args.batch_size):
        batch = [(start + j, records[start + j])
                 for j in range(min(args.batch_size, len(records) - start))]
        batches.append((start, batch))

    todo = [(s, b, out_dir / f"verdicts_{s:05d}.json") for s, b in batches
            if not _verdict_file_ok(out_dir / f"verdicts_{s:05d}.json", b)]
    print(f"{len(records)} records, {len(batches)} batches (size {args.batch_size}); "
          f"{len(batches) - len(todo)} already done, judging {len(todo)} with {args.workers} "
          f"workers", flush=True)

    done = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(judge_batch, b, out, schema_path, args.model, args.timeout): s
                for s, b, out in todo}
        for fut in as_completed(futs):
            start = futs[fut]
            try:
                n = fut.result()
                done += 1
                print(f"  batch {start:05d}: {n} verdicts  ({done}/{len(todo)})", flush=True)
            except Exception as e:  # noqa: BLE001 — log and continue; resume re-runs failures
                fail += 1
                print(f"  batch {start:05d}: FAILED — {e}", file=sys.stderr, flush=True)

    print(f"done: {done} batches judged, {fail} failed (re-run to resume failures)", flush=True)


if __name__ == "__main__":
    main()
