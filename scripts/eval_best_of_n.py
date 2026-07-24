"""Best-of-n behavioral eval for the reward model: does picking the best of k on-policy
samples by RM score beat taking a single sample, per an independent position-swapped judge?
If the RM learned something real, best-of-k should win clearly; if it's noise, ~50/50.

    python scripts/eval_best_of_n.py --policy runs/1b-ipo-rlaif --rm runs/1b-rm \\
        --skip 5200 --limit 120 --k 8

Pipeline (every stage appends progressively and resumes, so a crash loses at most one unit):
  1. generate  k samples per held-out instruction from the policy — the same batched
     KV-cached sampler and per-row seeds as build_rlaif_candidates -> work/candidates.jsonl
  2. score     every candidate under the RM. Sequences are built by the EXACT training-time
     constructor from train_reward_model (left-truncated prompt + response + "\\n### End",
     scored at the last real token); a mismatched construction would be silently wrong, so
     it is reused, not re-implemented -> work/scores.jsonl
  3. judge     A = RM argmax of the k, B = candidate 0 (an unbiased single draw at the same
     settings), judged in BOTH orders by codex via eval_pairwise's machinery; A wins only if
     preferred in both orderings -> work/verdicts.jsonl
  4. report    win/loss/tie, RM-score stats, RM-vs-judge agreement -> --out

The policy and the RM are both ~1B; they are loaded SEQUENTIALLY (policy freed before the RM
loads) so only one is ever resident on the GPU."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.data.reference.loaders import load_dolly  # noqa: E402
from microlab.model.reference.checkpoint import (  # noqa: E402
    latest_checkpoint,
    load_variant_from_run,
)
from microlab.model.reference.sft import format_chat  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402
from microlab.train.reward import collate_reward, load_reward_checkpoint  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


brc = _load_script("build_rlaif_candidates")  # sample_candidates: batched KV-cached k-sampler
ep = _load_script("eval_pairwise")  # position-swapped codex judging machinery
trm = _load_script("train_reward_model")  # THE reward-sequence constructor + sentinel


def ensure_codex() -> str:
    """The judge is a hard dependency; a missing binary must kill the run, not degrade it."""
    path = shutil.which("codex")
    if path is None:
        raise RuntimeError("codex CLI not found on PATH — judging cannot run. Put its bin dir "
                           "on PATH (e.g. ~/.nvm/versions/node/*/bin) and rerun.")
    return path


def select_usable_rows(tok, rows: list[dict], skip: int, limit: int, max_new: int,
                       block_size: int) -> tuple[list[tuple[int, str, str]], int]:
    """First `limit` rows after index `skip` whose templated prompt leaves room for max_new
    tokens (the same guard candidate generation applies). Returns ([(abs_row, instruction,
    prompt)], n_skipped_by_guard); raises if the data runs out before `limit` usable rows."""
    picked: list[tuple[int, str, str]] = []
    skipped_long = 0
    for i in range(skip, len(rows)):
        prompt, _ = format_chat(rows[i]["instruction"], rows[i].get("context", ""))
        if len(tok.encode(prompt)) + max_new > block_size:
            skipped_long += 1
            continue
        picked.append((i, rows[i]["instruction"], prompt))
        if len(picked) == limit:
            return picked, skipped_long
    raise ValueError(f"only {len(picked)} usable rows after row {skip} (wanted {limit}; "
                     f"block-size guard removed {skipped_long})")


def build_candidate_sequences(tok, prompt: str, candidates: list[str],
                              block_size: int) -> list[list[int]]:
    """Token sequences for RM scoring, one per candidate, built by the training-time
    constructor itself (build_reward_sequences with chosen == rejected == candidate) so the
    construction can never drift from what the RM was trained on. Training SKIPS a pair whose
    response + sentinel fills the block; here that would silently misalign candidate indices
    with scores, so it raises instead (can't happen when max_new << block_size)."""
    rows = [{"prompt": prompt, "chosen": c, "rejected": c} for c in candidates]
    pairs, skipped = trm.build_reward_sequences(tok, rows, block_size)
    if skipped:
        raise ValueError(f"{skipped} candidate(s) + sentinel fill block_size {block_size}; "
                         f"scores would misalign with candidate indices")
    return [chosen for chosen, _ in pairs]


@torch.no_grad()
def score_sequences(rm, seqs: list[list[int]], device: str, use_amp: bool,
                    batch_size: int = 8) -> list[float]:
    """RM scores for token sequences, right-padded per micro-batch exactly like training
    (collate_reward + per-row lengths; pad sits after the scored position). fp32 outputs."""
    scores: list[float] = []
    for start in range(0, len(seqs), batch_size):
        batch = collate_reward(seqs[start:start + batch_size], trm.PAD_ID)
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = rm(input_ids, lengths)
        else:
            out = rm(input_ids, lengths)
        scores.extend(out.float().tolist())
    return scores


def pick_best(scores: list[float]) -> int:
    """Index of the max score; the EARLIEST index wins ties (deterministic)."""
    if not scores:
        raise ValueError("pick_best got an empty score list")
    return max(range(len(scores)), key=scores.__getitem__)


def aggregate(items: list[dict]) -> dict:
    """Roll per-item records ({outcome, scores, best_idx}) into the report: win counts and
    win-rate among decided; RM-score stats (picked vs first sample, within-k spread); and
    RM-vs-judge agreement — among decided items where the RM strictly preferred A (best_idx
    != 0), how often the judge sided with A. Items where the argmax IS the first sample carry
    no RM preference (A == B) and are excluded from the agreement denominator."""
    if not items:
        raise ValueError("aggregate got no items")
    outcomes = Counter(it["outcome"] for it in items)
    a_wins, b_wins, ties = outcomes["A"], outcomes["B"], outcomes["tie"]
    decided = a_wins + b_wins

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    agree = agree_n = 0
    for it in items:
        if it["outcome"] == "tie" or it["best_idx"] == 0:
            continue
        agree_n += 1
        agree += it["outcome"] == "A"
    return {
        "n_items": len(items),
        "a_wins": a_wins, "b_wins": b_wins, "ties": ties, "decided": decided,
        "win_rate_best_of_n": a_wins / decided if decided else None,
        "n_best_is_first": sum(1 for it in items if it["best_idx"] == 0),
        "rm_scores": {
            "mean_picked": mean([it["scores"][it["best_idx"]] for it in items]),
            "mean_first_sample": mean([it["scores"][0] for it in items]),
            "mean_spread": mean([max(it["scores"]) - min(it["scores"]) for it in items]),
        },
        "rm_judge_agreement": {
            "n": agree_n, "agree": agree,
            "rate": agree / agree_n if agree_n else None,
        },
    }


# ---------------------------------------------------------------- pipeline stages


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(li) for li in path.read_text(encoding="utf-8").splitlines()
            if li.strip()]


def _append_jsonl(f, obj: dict) -> None:
    f.write(json.dumps(obj) + "\n")
    f.flush()


def stage_generate(selected: list[tuple[int, str, str]], args, device: str) -> list[dict]:
    """k samples per instruction -> work/candidates.jsonl, one line per instruction AS IT IS
    PRODUCED. Per-row seed = seed + abs_row * k (build_rlaif_candidates' convention), so a
    resumed run draws exactly what an uninterrupted one would have. ALL k samples are kept in
    order — no dedup, no empty-drop — because index 0 is the single-sample baseline and the
    argmax must range over the full k."""
    path = args.work_dir / "candidates.jsonl"
    prompts = {i: p for i, _, p in selected}
    by_row: dict[int, dict] = {}
    for rec in _read_jsonl(path):
        if rec["row"] not in prompts or rec["prompt"] != prompts[rec["row"]] \
                or len(rec["candidates"]) != args.k:
            raise ValueError(f"{path} row {rec['row']} does not match the current selection/"
                             f"settings — it was produced with different arguments; move the "
                             f"work dir aside and rerun")
        by_row[rec["row"]] = rec
    todo = [(i, ins, p) for i, ins, p in selected if i not in by_row]
    print(f"generate: {len(by_row)} instructions already on disk, {len(todo)} to go", flush=True)
    if todo:
        model, step = load_variant_from_run(Path(args.policy), device=device)
        tok = FastTokenizer.load(str(Path(args.policy) / "tokenizer.json"))
        print(f"  policy step {step}, k {args.k}, temp {args.temp}, max_new {args.max_new}",
              flush=True)
        t0 = time.time()
        with path.open("a", encoding="utf-8") as f:
            for n, (i, ins, prompt) in enumerate(todo, 1):
                cands = brc.sample_candidates(model, tok, prompt, device, args.k, args.temp,
                                              args.max_new, args.seed + i * args.k)
                rec = {"row": i, "instruction": ins, "prompt": prompt, "candidates": cands}
                _append_jsonl(f, rec)
                by_row[i] = rec
                if n % 10 == 0 or n == len(todo):
                    print(f"  sampled {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        del model  # free the policy BEFORE the RM loads — only one 1B resident at a time
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return [by_row[i] for i, _, _ in selected]


def stage_score(records: list[dict], args, device: str) -> dict[int, dict]:
    """RM scores for every candidate -> work/scores.jsonl ({row, scores, best_idx} per line,
    written as produced). Returns {row: record} for all rows."""
    path = args.work_dir / "scores.jsonl"
    by_row: dict[int, dict] = {}
    for rec in _read_jsonl(path):
        if len(rec["scores"]) != args.k:
            raise ValueError(f"{path} row {rec['row']} has {len(rec['scores'])} scores, "
                             f"expected k={args.k} — stale work dir; move it aside and rerun")
        by_row[rec["row"]] = rec
    todo = [r for r in records if r["row"] not in by_row]
    print(f"score: {len(by_row)} instructions already scored, {len(todo)} to go", flush=True)
    if todo:
        rm_ckpt = latest_checkpoint(Path(args.rm))
        rm, rm_step = load_reward_checkpoint(rm_ckpt, device=device)
        tok = FastTokenizer.load(str(Path(args.rm) / "tokenizer.json"))
        block = rm.backbone.config.block_size
        use_amp = device.startswith("cuda")
        print(f"  RM {rm_ckpt.name} (step {rm_step}), block {block}, amp {use_amp}", flush=True)
        with path.open("a", encoding="utf-8") as f:
            for n, rec in enumerate(todo, 1):
                seqs = build_candidate_sequences(tok, rec["prompt"], rec["candidates"], block)
                scores = score_sequences(rm, seqs, device, use_amp,
                                         batch_size=args.score_batch)
                out = {"row": rec["row"], "scores": scores, "best_idx": pick_best(scores)}
                _append_jsonl(f, out)
                by_row[rec["row"]] = out
                if n % 20 == 0 or n == len(todo):
                    print(f"  scored {n}/{len(todo)}", flush=True)
        del rm
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return by_row


def stage_judge(items: list[tuple[int, str, str, str]], args) -> dict[int, str]:
    """Position-swapped codex judging -> work/verdicts.jsonl ({item, better} per line, a batch
    written as soon as it lands). Both orderings of an instruction are ADJACENT in the item
    list, so they fall in the same codex call (eval_pairwise's layout). A batch that comes
    back without a verdict for one of its items raises — rerunning retries only that batch."""
    path = args.work_dir / "verdicts.jsonl"
    done: dict[int, str] = {rec["item"]: rec["better"] for rec in _read_jsonl(path)}
    todo = [it for it in items if it[0] not in done]
    print(f"judge: {len(done)} verdicts already on disk, {len(todo)} items to go", flush=True)
    if not todo:
        return done
    ensure_codex()
    schema_path = Path(tempfile.mkdtemp()) / "schema.json"
    schema_path.write_text(json.dumps(ep.SCHEMA))
    batches = [todo[s:s + args.judge_batch] for s in range(0, len(todo), args.judge_batch)]
    lock = threading.Lock()

    def run_batch(batch: list[tuple[int, str, str, str]]) -> dict[int, str]:
        ids = {b[0] for b in batch}
        text = ep._codex_judge(ep.build_pair_prompt(batch), schema_path, args.timeout)
        got = ep.parse_pair_verdicts(text, ids)
        missing = ids - set(got)
        if missing:
            raise RuntimeError(f"judge returned no valid verdict for items {sorted(missing)}; "
                               f"rerun to retry (finished batches are saved)")
        with lock, path.open("a", encoding="utf-8") as f:
            for item_id in sorted(got):
                _append_jsonl(f, {"item": item_id, "better": got[item_id]})
        return got

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_batch, b) for b in batches]
        for n, fut in enumerate(as_completed(futures), 1):
            done.update(fut.result())  # raises the batch's error loudly
            print(f"  judged batch {n}/{len(batches)}", flush=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="runs/1b-ipo-rlaif")
    ap.add_argument("--rm", default="runs/1b-rm", help="reward-model run dir (latest ckpt)")
    ap.add_argument("--data", default="data/corpora/sft_mix.jsonl")
    ap.add_argument("--skip", type=int, default=5200,
                    help="rows to skip: clear of the 5000 preference-training rows AND the "
                         "standing 120-item eval window (5000-5119)")
    ap.add_argument("--limit", type=int, default=120, help="usable instructions to eval")
    ap.add_argument("--k", type=int, default=8, help="samples per instruction")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-batch", type=int, default=8, help="sequences per RM forward")
    ap.add_argument("--judge-batch", type=int, default=20, help="items per codex call")
    ap.add_argument("--workers", type=int, default=4, help="parallel codex calls")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/bestofn_eval.json"))
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="progressive/resumable stage files; default <out dir>/<out stem>_work")
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {args.device!r} requested but CUDA is unavailable")
    if args.work_dir is None:
        args.work_dir = args.out.parent / (args.out.stem + "_work")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    policy_ckpt = latest_checkpoint(Path(args.policy))
    cfg = torch.load(policy_ckpt, map_location="meta", weights_only=False)["cfg"]
    tok = FastTokenizer.load(str(Path(args.policy) / "tokenizer.json"))
    rows = load_dolly(args.data)
    selected, skipped_long = select_usable_rows(tok, rows, args.skip, args.limit,
                                                args.max_new, cfg.block_size)
    print(f"best-of-{args.k}: {len(selected)} usable instructions after row {args.skip} "
          f"({skipped_long} removed by the block guard), policy {policy_ckpt.name} vs a "
          f"single draw, device {args.device}", flush=True)

    records = stage_generate(selected, args, args.device)
    scores = stage_score(records, args, args.device)

    items: list[tuple[int, str, str, str]] = []
    per_item: list[dict] = []
    for idx, rec in enumerate(records):
        sc = scores[rec["row"]]
        a = rec["candidates"][sc["best_idx"]]  # A = RM argmax of the k
        b = rec["candidates"][0]  # B = unbiased single draw at the same settings
        items.append((2 * idx, rec["instruction"], a, b))
        items.append((2 * idx + 1, rec["instruction"], b, a))
        per_item.append({"row": rec["row"], "instruction": rec["instruction"],
                         "best_idx": sc["best_idx"], "scores": sc["scores"], "a": a, "b": b})

    verdicts = stage_judge(items, args)
    for idx, item in enumerate(per_item):
        item["outcome"] = ep.resolve_pair(verdicts.get(2 * idx), verdicts.get(2 * idx + 1))

    rm_eval_path = Path(args.rm) / "eval.json"
    report = {
        "question": f"does the RM's argmax of {args.k} on-policy samples beat a single "
                    f"sample, per the position-swapped codex judge?",
        "policy": str(args.policy), "policy_ckpt": policy_ckpt.name,
        "rm": str(args.rm), "rm_ckpt": latest_checkpoint(Path(args.rm)).name,
        "data": str(args.data), "skip": args.skip, "limit": args.limit,
        "k": args.k, "temp": args.temp, "max_new": args.max_new, "seed": args.seed,
        "skipped_long": skipped_long,
        **aggregate(per_item),
        "rm_holdout_acc_reference": (json.loads(rm_eval_path.read_text())["holdout_acc"]
                                     if rm_eval_path.exists() else None),
        "runtime_s": round(time.time() - t0, 1),
        "per_item": per_item,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    wr = report["win_rate_best_of_n"]
    rs = report["rm_scores"]
    agr = report["rm_judge_agreement"]
    print(f"\nbest-of-{args.k} (RM argmax): {report['a_wins']} wins | single sample: "
          f"{report['b_wins']} wins | {report['ties']} ties (of {report['n_items']})")
    print("no decided items" if wr is None else
          f"best-of-{args.k} win-rate among decided: {100 * wr:.0f}%")
    print(f"RM scores: mean picked {rs['mean_picked']:.3f} vs mean first sample "
          f"{rs['mean_first_sample']:.3f}, mean within-{args.k} spread {rs['mean_spread']:.3f}")
    print("RM-vs-judge agreement: n/a (no decided items with an RM preference)"
          if agr["rate"] is None else
          f"RM-vs-judge agreement on decided items: {agr['agree']}/{agr['n']} "
          f"({100 * agr['rate']:.0f}%) vs RM holdout acc "
          f"{report['rm_holdout_acc_reference']}")
    print(f"argmax was the first sample on {report['n_best_is_first']} items")
    print(f"wrote {args.out} ({report['runtime_s']}s)")


if __name__ == "__main__":
    main()
