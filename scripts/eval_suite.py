#!/usr/bin/env python
"""Checkpoint eval suite — the battery run against milestone checkpoints of a pretrain.

    python scripts/eval_suite.py --run runs/coder-1b --step 2000 \\
        --out evals/suite/coder-1b-2000.json

Appends to `evals/suite/<run>-trajectory.jsonl` so capability-vs-tokens is answerable
directly. That question is not decorative: our first 1B could not retrieve across its own
1,024-token window until ~17B tokens (0.04 at 8.4B, 0.87 at 16.8B), so a mid-run reading
of any capability is meaningless without the trajectory around it.

WHAT IS HERE, AND WHY EACH EARNS ITS PLACE

  per-slice val loss  The mix is six slices in fixed proportions. One aggregate val number
                      cannot tell you that math is diverging while code improves. Each
                      slice keeps its own val split, so this is nearly free and is the
                      most diagnostic single measurement available mid-run.
  fim                 We apply FIM to 50% of code documents. If the model does not learn
                      infilling, that is half the code budget spent on a capability we
                      never acquired — and nothing else in the suite would show it.
  repetition          Degenerate looping is the classic base-model failure and it is
                      invisible in val loss.
  humaneval / lmeval  Delegated to the existing scripts (sandboxed execution and
                      EleutherAI's harness respectively); wired here so one command
                      covers a milestone.

Passkey and length-generalisation are deliberately NOT in the default battery. They are
measured and reported, never gated — see docs/passkey-c2-verdict.md and
docs/small-model-long-retrieval-lit.md. Run them with --with-retrieval.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.data.fim import FIMConfig, split_documents  # noqa: E402
from microlab.model.reference.checkpoint import (  # noqa: E402
    resolve_checkpoint,
    variant_config_from_ckpt,
)
from microlab.model.reference.variants import VariantGPT  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402

# slice -> its own val shards. Built by the per-slice ingests, not by the mix builder.
SLICE_VAL = {
    "code": "data/shards/code-repo-32k",
    "web": "data/shards/web-49k",
    "math": "data/shards/math-49k",
    "markdown": "data/shards/markdown-49k",
    "arxiv": "data/shards/arxiv-49k",
    "commits": "data/shards/commits-49k",
}


@torch.no_grad()
def val_loss(model, tokens: np.ndarray, block: int, n_batches: int, device: str,
             seed: int = 0) -> float | None:
    """Mean next-token loss over `n_batches` windows drawn from `tokens`."""
    if len(tokens) < block + 1:
        return None
    g = np.random.default_rng(seed)
    total, n = 0.0, 0
    for _ in range(n_batches):
        i = int(g.integers(0, len(tokens) - block - 1))
        x = torch.from_numpy(tokens[i:i + block].astype(np.int64))[None].to(device)
        y = torch.from_numpy(tokens[i + 1:i + 1 + block].astype(np.int64))[None].to(device)
        _, loss = model(x, targets=y)
        total += float(loss)
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def fim_score(model, tokens: np.ndarray, cfg: FIMConfig, eot: int, block: int,
              device: str, n_docs: int = 64) -> dict:
    """Teacher-forced loss on the MIDDLE span of held-out FIM documents.

    Loss on the middle only, because that is the span infilling has to produce; loss over
    the whole document is dominated by the prefix and suffix the model can simply copy.
    Compared against the same documents' loss when read left-to-right, so the number says
    whether the FIM format was learned rather than merely how hard the code is.
    """
    # FIM is applied per CHUNK, so examples are EMBEDDED in documents rather than being
    # documents (the old `d[0] == cfg.prefix` filter would find zero examples on a
    # chunk-FIM corpus and report nothing). An example runs from a prefix sentinel to the
    # next prefix sentinel or end-of-document; by construction (fim_document, span 4096)
    # it fits the eval block comfortably.
    examples: list[list[int]] = []
    for d in split_documents(tokens, eot):
        ids = d.tolist()
        starts = [i for i, t in enumerate(ids) if t == cfg.prefix]
        for s, e in zip(starts, [*starts[1:], len(ids)], strict=False):
            if e - s > 32:
                examples.append(ids[s:e])
            if len(examples) >= n_docs:
                break
        if len(examples) >= n_docs:
            break
    if not examples:
        return {"n": 0, "note": "no FIM examples in this val split"}
    mid_tot, mid_n = 0.0, 0
    for ids in examples:
        ids = ids[:block]
        try:
            m = ids.index(cfg.middle)
        except ValueError:
            continue
        if m + 2 >= len(ids):
            continue
        x = torch.tensor(ids[:-1], device=device)[None]
        y = torch.tensor(ids[1:], device=device)[None]
        logits, _ = model(x)
        lp = torch.nn.functional.cross_entropy(
            logits[0].float(), y[0], reduction="none")
        mid_tot += float(lp[m:].mean())     # positions after <|fim_middle|>
        mid_n += 1
    # None, not 0.0, when nothing was scorable: 0.0 is a PERFECT score, and this metric
    # once printed it while structurally blind (6 of 64 documents scorable at the default
    # block; an unluckier sample gives zero). A missing measurement must read as missing,
    # and n is in the dict so the reader can judge how thin the evidence is.
    return {"n": mid_n,
            "middle_loss": mid_tot / mid_n if mid_n else None,
            "middle_ppl": math.exp(mid_tot / mid_n) if mid_n else None}


PY_STUBS = [
    "def fibonacci(n):\n",
    "def quicksort(arr):\n",
    "class Stack:\n",
    "import json\n\ndef load_config(path):\n",
    "def binary_search(xs, target):\n",
    "from dataclasses import dataclass\n\n@dataclass\nclass Point:\n",
]


@torch.no_grad()
def syntax_validity(model, tok, device: str, max_new: int = 120) -> dict:
    """Fraction of Python completions that PARSE.

    The early-signal code metric. HumanEval pass@1 stays at ~0 for most of a
    compute-optimal 1B's run — our own 1B needed ~17B tokens before an unrelated
    capability appeared at all — so a suite that only measures execution success reads
    zero for weeks and tells you nothing about whether training is working. Syntactic
    validity rises long before semantic correctness and is therefore the metric that
    actually moves during the phase where you still have decisions to make.
    """
    import ast

    from microlab.infer.reference.kv_cache import generate_cached
    ok, total, samples = 0, 0, []
    for stub in PY_STUBS:
        ids = tok.encode(stub)
        out = generate_cached(model, torch.tensor([ids], device=device), max_new,
                              temperature=0.0)
        text = stub + tok.decode(out[0, len(ids):].tolist())
        # Trim to the last complete line: a mid-line cut is a truncation artifact, not a
        # syntax error the model made.
        text = text[:text.rfind("\n") + 1] if "\n" in text else text
        total += 1
        try:
            ast.parse(text)
            ok += 1
        except SyntaxError:
            samples.append(text[:160])
    return {"n": total, "parsed": ok, "parse_rate": ok / max(total, 1),
            "failures": samples[:2]}


@torch.no_grad()
def repetition(model, tok, device: str, n: int = 8, max_new: int = 160) -> dict:
    """Fraction of greedy continuations that fall into a repeating loop.

    Degenerate repetition is the classic base-model failure and val loss does not see it:
    a model that loops still assigns high probability to the tokens it emits.
    """
    from microlab.infer.reference.kv_cache import generate_cached
    prompts = ["def ", "import ", "class ", "# ", "function ", "const ", "SELECT ", "/**"]
    looped = 0
    for p in prompts[:n]:
        ids = tok.encode(p)
        out = generate_cached(model, torch.tensor([ids], device=device), max_new,
                              temperature=0.0)
        gen = out[0, len(ids):].tolist()
        # a 12-gram that repeats is a loop, not style
        grams = [tuple(gen[i:i + 12]) for i in range(len(gen) - 12)]
        if grams and len(set(grams)) < 0.5 * len(grams):
            looped += 1
    return {"prompts": n, "looped": looped, "loop_rate": looped / max(n, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=None, help="milestone step; default latest")
    ap.add_argument("--out", default=None)
    ap.add_argument("--block", type=int, default=4096,
                    help="eval context; shorter than training block for speed")
    ap.add_argument("--val-batches", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--with-code", action="store_true",
                    help="HumanEval + MBPP + MultiPL-E js/ts (execution, sandboxed)")
    ap.add_argument("--with-math", action="store_true",
                    help="GSM8K + arithmetic via lm-eval-harness")
    ap.add_argument("--with-general", action="store_true",
                    help="hellaswag/arc/piqa — context for the code numbers, not the point")
    ap.add_argument("--no-probes", action="store_true",
                    help="skip track_probes.py (the qualitative + scored probe battery)")
    ap.add_argument("--with-retrieval", action="store_true",
                    help="passkey + length-gen; MEASURED AND REPORTED, never a gate")
    # multipl-js/ts are NOT in the default: the executor sandboxes Python only, and
    # eval_code raises NotImplementedError for them — every milestone recorded rc=1
    # noise. Re-add once a node executor exists.
    ap.add_argument("--code-datasets", default="humaneval,mbpp")
    ap.add_argument("--code-mode", default="base", choices=["base", "chat"],
                    help="passed to eval_code.py as --mode; a raw pretrain checkpoint "
                         "has no serve_config.json for --mode auto to read")
    ap.add_argument("--math-tasks", default="gsm8k,arithmetic_2da,arithmetic_2ds")
    a = ap.parse_args()

    run = Path(a.run)
    ckpt_path = resolve_checkpoint(run, a.step)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = ck["step"]
    model = VariantGPT(variant_config_from_ckpt(ck["cfg"], block_size=a.block)).eval()
    model.load_state_dict(ck["model"])
    model = model.to(a.device)
    train_cfg = ck["cfg"]
    del ck
    tok = FastTokenizer.load(str(run / "tokenizer.json"))
    eot = tok._tok.token_to_id("<|endoftext|>")
    # Pretrain/SFT checkpoints store the TRAINING config (batch_size/grad_accum present);
    # GRPO checkpoints store the model's VariantConfig, where tokens-seen is genuinely
    # unknowable — record None explicitly and say so, rather than crashing (this took the
    # whole suite down silently inside a gate script) or fabricating a number.
    if hasattr(train_cfg, "batch_size") and hasattr(train_cfg, "grad_accum"):
        tokens_seen = step * train_cfg.batch_size * train_cfg.grad_accum * train_cfg.block_size
        print(f"{ckpt_path.name}: step {step:,}, ~{tokens_seen/1e9:.2f}B tokens seen")
    else:
        tokens_seen = None
        print(f"{ckpt_path.name}: step {step:,} (tokens_seen unknown: ckpt stores a "
              f"model config, not a training config — post-trained run)")
    res = {"run": str(run), "ckpt": ckpt_path.name, "step": step,
           "tokens_seen": tokens_seen, "block": a.block}

    # --- per-slice val loss ---------------------------------------------------------
    t0 = time.time()
    per_slice = {}
    for name, d in SLICE_VAL.items():
        man = Path(d) / "val-manifest.json"
        if not man.exists():
            per_slice[name] = None
            continue
        shards = json.loads(man.read_text())["shards"]
        arr = np.fromfile(Path(d) / shards[0]["file"], dtype=np.uint16)
        loss = val_loss(model, arr, a.block, a.val_batches, a.device)
        per_slice[name] = {"loss": loss, "ppl": math.exp(loss) if loss else None}
        print(f"  val/{name:<9} {loss:.4f}" if loss else f"  val/{name:<9} n/a")
    res["per_slice_val"] = per_slice
    res["timing_val_s"] = round(time.time() - t0, 1)

    # --- FIM ------------------------------------------------------------------------
    # A run trained WITHOUT FIM has no infilling to measure — that is a fact about the
    # checkpoint, not a failure to hide, so it is recorded explicitly and distinguishably.
    # It is NOT a fallback: for a run whose config used the FIM tokenizer, missing
    # sentinels would mean the corpus was built wrong, and `trained_with_fim` in the
    # output is what makes those two cases tell apart at a glance.
    # The checkpoint's data_dir is the TRAINING box's path (/workspace/mix-v2 on cloud
    # episodes), which does not exist on the eval host. MICROLAB_MIX_DIR is the same
    # operator override the trainer honors — not a fallback: if neither points at a val
    # shard the result records the miss explicitly.
    # Resolve LAZILY: os.environ.get's default argument evaluates even when the env var is
    # set, and GRPO checkpoints' VariantConfig has no data_dir (same post-trained-ckpt drift
    # as tokens_seen above) — the eager default crashed the suite despite MICROLAB_MIX_DIR
    # being set correctly.
    mix_dir = os.environ.get("MICROLAB_MIX_DIR") or getattr(train_cfg, "data_dir", None)
    mix_val = Path(mix_dir) / "val-00000.bin" if mix_dir else None
    has_fim = all(tok._tok.token_to_id(t) is not None
                  for t in ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"))
    if not has_fim:
        res["fim"] = {"trained_with_fim": False,
                      "note": "checkpoint's tokenizer carries no FIM sentinels"}
        print("  fim n/a (model was not trained with FIM)")
    elif mix_val is None or not mix_val.exists():
        res["fim"] = {"trained_with_fim": True, "note": f"no val shard at {mix_val}"}
    else:
        fcfg = FIMConfig(tok._tok)
        arr = np.fromfile(mix_val, dtype=np.uint16)
        res["fim"] = {"trained_with_fim": True,
                      **fim_score(model, arr, fcfg, eot, a.block, a.device)}
        print(f"  fim middle_loss {res['fim'].get('middle_loss')} "
              f"(n={res['fim'].get('n')})")

    # --- repetition -----------------------------------------------------------------
    res["syntax_valid"] = syntax_validity(model, tok, a.device)
    print(f"  syntax parse_rate {res['syntax_valid']['parse_rate']:.2f} "
          f"({res['syntax_valid']['parsed']}/{res['syntax_valid']['n']})")

    res["repetition"] = repetition(model, tok, a.device)
    print(f"  repetition loop_rate {res['repetition']['loop_rate']:.2f}")

    del model
    torch.cuda.empty_cache()

    # --- delegated evals ------------------------------------------------------------
    def run_script(argv, key):
        print(f"  running {Path(argv[0]).name} ...", flush=True)
        p = subprocess.run([sys.executable, *argv], capture_output=True, text=True)
        res.setdefault("delegated", {})[key] = {
            "returncode": p.returncode, "tail": p.stdout[-600:] or p.stderr[-600:]}

    outdir = Path("evals/suite")
    outdir.mkdir(parents=True, exist_ok=True)
    if a.with_code:
        # Execution evals across four languages. Expect ~0 for most of a compute-optimal
        # run; they are the ceiling measurement, and `syntax_valid` above is what moves
        # early. Reading an early 0 here as "the model is broken" would be the same
        # mistake as gating on passkey at 32k.
        for ds in a.code_datasets.split(","):
            run_script(["scripts/eval_code.py", "--run", str(run), "--dataset", ds,
                        "--mode", a.code_mode,
                        "--out", str(outdir / f"{run.name}-{step}-{ds}.jsonl")], ds)
    if a.with_math:
        run_script(["scripts/lmeval_microlab.py", "--run", str(run), "--tasks",
                    a.math_tasks, "--out",
                    str(outdir / f"{run.name}-{step}-math.json")], "math")
    if a.with_general:
        run_script(["scripts/lmeval_microlab.py", "--run", str(run), "--tasks",
                    "hellaswag,arc_easy,piqa", "--out",
                    str(outdir / f"{run.name}-{step}-general.json")], "general")
    if not a.no_probes:
        # The probe tracker from the first 1B — 8 scored categories (capital, science,
        # commonsense, sequence, arithmetic, icl, code, math), a likelihood-based
        # multiple-choice set, and free-form completions to READ. It maintains its own
        # per-run probe_track.jsonl, which is the trajectory record that made the
        # capability-emergence sweep possible on the original 1B.
        run_script(["scripts/track_probes.py", "--run", str(run), str(step)], "probes")
    if a.with_retrieval:
        run_script(["scripts/eval_passkey.py", "--run", str(run), "--lengths",
                    "1024,4096,16384", "--depths", "0.1,0.5,0.9", "--n", "32",
                    "--out", str(outdir / f"{run.name}-{step}-passkey.json")], "passkey")

    out = Path(a.out) if a.out else outdir / f"{run.name}-{step}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    with (outdir / f"{run.name}-trajectory.jsonl").open("a") as f:
        f.write(json.dumps(res) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
