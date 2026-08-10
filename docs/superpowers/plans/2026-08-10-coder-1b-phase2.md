# coder-1b Phase 2 (free-lever program) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the sandbox executor in as a GRPO reward, build correctness-contrast IPO pairs and the self-gen loop, and train/stack the levers to make the best $0 1B code model.

**Architecture:** A pure oracle module `src/microlab/train/exec_reward.py` (solution extraction, per-case rewards, `score_texts` factory, sampling helper) feeds a new `--reward executor` mode in `scripts/train_grpo.py` (RM path untouched). Three thin CLIs build the pool, contrast pairs, and self-gen tranche; a summarizer computes rerank/delivered-correctness from existing `eval_code --n K` output. Operational training/eval runs last, gated by a pre-registered prediction.

**Tech Stack:** Python, PyTorch, existing `microlab` package (executor sandbox, GRPO loop, DPO/IPO, FastTokenizer, eval harness). No new dependencies.

## Global Constraints

- **Build capability, don't distill:** the executor is the only reward/filter ground truth; the codex CLI is the only external judge (pairwise, both orderings).
- **Reward = fraction of checked I/O cases passed** (dense). No extractable code → 0.0.
- **Pre-registered predictions committed before any training stage** (Task 8 before Task 9).
- **No silent fallbacks; verify by count:** pool/pair/tranche builders print counts and refuse to write empty output (guard BEFORE opening the file). Timeouts are NOT used as DPO "rejected" (slow-but-correct is a bad rejected).
- **Decontamination:** identical n=10-gram filter vs HumanEval+MBPP (`benchmark_fingerprints` / `decontaminate` from `microlab.data.code_sft`) on every new training text.
- Row/pair schemas: pool `{"instruction": str, "io": [{"input": str, "output": str}, ...]}`; prefs `{"prompt": <chat-formatted>, "chosen": str, "rejected": str}`; SFT rows `{"instruction","context","response"}`.
- Long jobs write progressively and resume (append JSONL, skip already-done keys).

## Interfaces this plan builds on (verified in-repo)

- `run_grpo(policy, reference, tok, prompts: list[str], score_texts, out, tokenizer_path, *, iters, prompts_per_iter=8, group_size=8, lr=1e-6, beta=0.04, clip_eps=0.2, temp=0.8, max_new=80, micro_batch=8, save_every=25, dump_every=10, seed=1337, device="cpu", warmup_iters=10) -> dict` — `score_texts(prompt, texts) -> list[float]` is injected (`src/microlab/train/grpo.py:229`).
- `scripts/train_grpo.py`: `build_prompt_pool(tok, rows, max_new, block_size) -> (prompts, n_skipped)` chat-formats rows via `format_chat`; `make_score_texts(rm, tok, ...)` is the RM oracle factory; main wires `--policy/--prefs/--out/...`.
- `microlab.evals.code.prompts.extract_code_block(reply) -> str` (sentinel-truncates, unfences).
- `microlab.data.code_sft`: `assemble_io_program(solution, stdin_data, expected_stdout) -> str`, `apps_problem/codecontests_problem/taco_problem(row) -> {"statement","solutions","io"}`, `benchmark_fingerprints`, `decontaminate`, `_loads_io`.
- `microlab.evals.code.executor.run_python(code, *, timeout_s=...) -> ExecResult` (`.passed`, `.timed_out`, `.exit_code`).
- `scripts/dpo.py`: prefs JSONL `{prompt, chosen, rejected}`, `--loss ipo`; prompt field is the already-chat-formatted string (`format_chat(...)[0]`).
- `model.reference.sft.format_chat(instruction, context="", response="") -> (prompt, response)`.
- `load_variant_from_run` + `generate_cached` (used by eval_code) for sampling; `FastTokenizer.load`.
- `scripts/eval_code.py --n K --temperature --top-k` writes per-sample rows `{task_id, sample, passed, ...}` + `.summary.json` with `pass@1`/`pass@K`.

---

### Task 1: Executor reward oracle (pure core)

**Files:**
- Create: `src/microlab/train/exec_reward.py`
- Test: `tests/train/test_exec_reward.py`

**Interfaces:**
- Produces: `extract_solution(reply: str) -> str`; `io_reward(solution: str, io_cases: list[dict], timeout_s: float = 5.0) -> float`; `make_exec_score_texts(io_by_prompt: dict[str, list[dict]], timeout_s: float = 5.0) -> callable` returning a `score_texts(prompt, texts) -> list[float]` that raises `KeyError` on an unknown prompt (no silent 0-reward for a wiring bug).

- [ ] **Step 1: Write the failing test**

```python
# tests/train/test_exec_reward.py
import pytest

from microlab.train.exec_reward import extract_solution, io_reward, make_exec_score_texts

CASES = [{"input": "21\n", "output": "42\n"}, {"input": "5\n", "output": "10\n"}]


def test_io_reward_fraction_of_cases():
    assert io_reward("n=int(input());print(n*2)", CASES) == 1.0
    # passes only the first case -> 0.5
    assert io_reward("print(42)", CASES) == 0.5
    assert io_reward("print('x')", CASES) == 0.0
    assert io_reward("", CASES) == 0.0          # no code -> 0, not an error


def test_extract_solution_unfences_and_truncates():
    reply = "Here you go:\n```python\nprint(1)\n```\n### End\njunk"
    assert extract_solution(reply) == "print(1)"


def test_score_texts_maps_prompts_and_rejects_unknown():
    score = make_exec_score_texts({"P1": CASES})
    got = score("P1", ["```python\nn=int(input());print(n*2)\n```", "nonsense"])
    assert got[0] == 1.0 and got[1] == 0.0
    with pytest.raises(KeyError):
        score("P-unknown", ["x"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_exec_reward.py -q`
Expected: FAIL (`ModuleNotFoundError: microlab.train.exec_reward`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/microlab/train/exec_reward.py
"""Executor-backed GRPO reward: rollout text -> extracted code -> sandbox -> reward.

The reward is the FRACTION of a problem's checked I/O cases the extracted solution passes
(dense — a binary all-pass reward zeroes the group advantage on most groups at ~14% pass
rates). This is the `score_texts` oracle run_grpo injects; the RM oracle in train_grpo.py is
the precedent. The executor is ground truth (build capability, don't distill).
"""
from __future__ import annotations

from microlab.data.code_sft import assemble_io_program
from microlab.evals.code.executor import run_python
from microlab.evals.code.prompts import extract_code_block


def extract_solution(reply: str) -> str:
    """Code from a chat rollout: sentinel-truncated, unfenced (reuses the eval extractor —
    training and eval must extract identically or reward would diverge from measurement)."""
    return extract_code_block(reply)


def io_reward(solution: str, io_cases: list[dict], timeout_s: float = 5.0) -> float:
    """Fraction of io_cases the solution passes. Empty/whitespace solution -> 0.0."""
    if not solution.strip() or not io_cases:
        return 0.0
    passed = 0
    for c in io_cases:
        prog = assemble_io_program(solution, c["input"], c["output"])
        if run_python(prog, timeout_s=timeout_s).passed:
            passed += 1
    return passed / len(io_cases)


def make_exec_score_texts(io_by_prompt: dict[str, list[dict]], timeout_s: float = 5.0):
    """score_texts(prompt, texts) -> rewards, keyed by the exact chat-formatted prompt.
    An unknown prompt raises KeyError — a pool/loop wiring bug must fail loudly, not train
    on silent zero rewards."""
    def score_texts(prompt: str, texts: list[str]) -> list[float]:
        cases = io_by_prompt[prompt]
        return [io_reward(extract_solution(t), cases, timeout_s=timeout_s) for t in texts]
    return score_texts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/train/test_exec_reward.py -q`
Expected: PASS (3 passed; the sandbox runs are real, a few seconds).

- [ ] **Step 5: Commit**

```bash
git add src/microlab/train/exec_reward.py tests/train/test_exec_reward.py
git commit -m "feat(phase2): executor reward oracle (fraction-of-cases, score_texts-compatible)"
```

---

### Task 2: GRPO pool builder

**Files:**
- Create: `scripts/build_grpo_pool.py`
- Test: `tests/scripts/test_build_grpo_pool.py`

**Interfaces:**
- Consumes: `apps_problem/codecontests_problem/taco_problem`, `benchmark_fingerprints`, `decontaminate` from `microlab.data.code_sft`.
- Produces: `pool_row(problem: dict, max_cases: int = 6) -> dict | None` (pure): normalized problem → `{"instruction": statement, "io": cases[:max_cases]}`, None when statement or cases are missing; a `main()` (`# pragma: no cover`) streaming the three competitive datasets (same hf:// sources as `build_code_sft._load_sources`), decontaminating (`_row_text` sees `instruction` only, so decontaminate on `[{"instruction": ..., "context": "", "response": ""}]`-shaped mirrors of the rows and keep survivors), writing `data/corpora/grpo_pool.jsonl` with a count report and an empty-guard BEFORE the write.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_build_grpo_pool.py
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_grpo_pool", Path(__file__).resolve().parents[2] / "scripts" / "build_grpo_pool.py")
bgp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bgp)


def test_pool_row_caps_cases_and_requires_statement_and_cases():
    p = {"statement": "double n", "solutions": ["x"],
         "io": [{"input": str(i), "output": str(i * 2)} for i in range(10)]}
    row = bgp.pool_row(p, max_cases=6)
    assert row["instruction"] == "double n" and len(row["io"]) == 6
    assert bgp.pool_row({"statement": "", "solutions": [], "io": p["io"]}) is None
    assert bgp.pool_row({"statement": "s", "solutions": [], "io": []}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/test_build_grpo_pool.py -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/build_grpo_pool.py
"""Build the GRPO prompt pool: competitive problems + capped I/O cases, decontaminated.

    python scripts/build_grpo_pool.py --out data/corpora/grpo_pool.jsonl \\
        --limit-per-dataset 4000 --max-cases 6

Emits {"instruction", "io": [{"input","output"}, ...]} per line. The policy pre-pass
(grpo_prepass.py) filters this to the signal-bearing subset before training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.data.code_sft import (  # noqa: E402
    apps_problem, benchmark_fingerprints, codecontests_problem, decontaminate, taco_problem,
)


def pool_row(problem: dict, max_cases: int = 6) -> dict | None:
    """Normalized problem -> pool row, or None when unusable (no statement / no cases)."""
    statement = (problem.get("statement") or "").strip()
    cases = (problem.get("io") or [])[:max_cases]
    if not statement or not cases:
        return None
    return {"instruction": statement, "io": cases}


def main() -> None:  # pragma: no cover - network + IO
    from datasets import load_dataset

    from microlab.evals.code.tasks import load_humaneval, load_mbpp

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/corpora/grpo_pool.jsonl")
    ap.add_argument("--limit-per-dataset", type=int, default=4000)
    ap.add_argument("--max-cases", type=int, default=6)
    args = ap.parse_args()

    adapters = [
        ("json", "hf://datasets/codeparrot/apps/train.jsonl", apps_problem),
        ("deepmind/code_contests", None, codecontests_problem),
        ("parquet", "hf://datasets/BAAI/TACO/ALL/train-*.parquet", taco_problem),
    ]
    rows, per_source = [], {}
    for fmt, data_files, adapt in adapters:
        it = load_dataset(fmt, data_files=data_files, split="train", streaming=True) \
            if data_files else load_dataset(fmt, split="train", streaming=True)
        n0, seen = len(rows), 0
        for r in it:
            seen += 1
            row = pool_row(adapt(r), max_cases=args.max_cases)
            if row:
                rows.append(row)
            if args.limit_per_dataset and seen >= args.limit_per_dataset:
                break
        per_source[adapt.__name__] = len(rows) - n0

    bench = [t.prompt + "\n" + t.test_program for t in (load_humaneval() + load_mbpp())]
    fp = benchmark_fingerprints(bench, n=10)
    mirrors = [{"instruction": r["instruction"], "context": "", "response": ""} for r in rows]
    kept_mirrors, removed = decontaminate(mirrors, fp, n=10)
    kept_ins = {m["instruction"] for m in kept_mirrors}
    rows = [r for r in rows if r["instruction"] in kept_ins]

    print(f"pool: {per_source} decontaminated_removed={removed} total={len(rows)}")
    if not rows:
        raise SystemExit("empty pool — refusing to proceed (verify by count)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} pool rows -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_build_grpo_pool.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_grpo_pool.py tests/scripts/test_build_grpo_pool.py
git commit -m "feat(phase2): GRPO pool builder (competitive problems + capped io, decontaminated)"
```

---

### Task 3: Sampling helper + policy pre-pass

**Files:**
- Modify: `src/microlab/train/exec_reward.py`
- Create: `scripts/grpo_prepass.py`
- Test: `tests/train/test_exec_reward.py`

**Interfaces:**
- Produces (in `exec_reward.py`): `signal_bearing(successes: int, k: int) -> bool` (pure: `0 < successes < k`); `sample_solutions(model, tok, prompt: str, k: int, *, max_new: int = 300, temp: float = 0.8, top_k: int = 40, seed: int = 0, device: str = "cuda") -> list[str]` — k sampled replies for one chat prompt via `generate_cached`, per-sample seeds `seed + i` (reproducible).
- Produces (script): `main()` — for each pool row: chat-format, sample k, reward each, append `{"instruction", "successes", "k", "rewards"}` to a progressive stats JSONL (resumable: skip instructions already present), then write the filtered signal pool `{"instruction","io"}` for rows with `signal_bearing(...)`. Reports counts; empty-guard before write.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/train/test_exec_reward.py
from microlab.train.exec_reward import signal_bearing


def test_signal_bearing_keeps_mixed_only():
    assert signal_bearing(0, 8) is False      # all-fail: zero advantage
    assert signal_bearing(8, 8) is False      # all-pass: zero advantage
    assert signal_bearing(1, 8) is True
    assert signal_bearing(7, 8) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_exec_reward.py -k signal -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/microlab/train/exec_reward.py
import torch as _torch


def signal_bearing(successes: int, k: int) -> bool:
    """Only mixed groups carry GRPO advantage: all-fail and all-pass both standardize to
    zero. The pre-pass keeps problems where the policy sometimes-but-not-always succeeds."""
    return 0 < successes < k


def sample_solutions(model, tok, prompt: str, k: int, *, max_new: int = 300,
                     temp: float = 0.8, top_k: int = 40, seed: int = 0,
                     device: str = "cuda") -> list[str]:
    """k sampled replies for one chat-formatted prompt (per-sample seed = seed+i so a rerun
    reproduces). Returns raw reply texts (caller extracts/rewards)."""
    from microlab.infer.reference.kv_cache import generate_cached
    ids = _torch.tensor([tok.encode(prompt)], device=device)
    outs = []
    for i in range(k):
        gen = _torch.Generator(device=device).manual_seed(seed + i)
        with _torch.no_grad():
            out = generate_cached(model, ids, max_new, temperature=temp, top_k=top_k,
                                  generator=gen)
        outs.append(tok.decode(out[0].tolist())[len(prompt):])
    return outs
```

```python
# scripts/grpo_prepass.py
"""Policy pre-pass: measure v1's per-problem success on the GRPO pool, keep the
signal-bearing subset (0 < successes < k).

    python scripts/grpo_prepass.py --policy runs/coder-1b-instruct-compliant \\
        --pool data/corpora/grpo_pool.jsonl --k 8 \\
        --stats data/corpora/grpo_prepass_stats.jsonl \\
        --out data/corpora/grpo_pool_signal.jsonl

Progressive + resumable: stats append per problem; already-measured instructions are
skipped on rerun. The signal pool is (re)written whole from the stats at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.model.reference.checkpoint import load_variant_from_run  # noqa: E402
from microlab.model.reference.sft import format_chat  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402
from microlab.train.exec_reward import (  # noqa: E402
    extract_solution, io_reward, sample_solutions, signal_bearing,
)


def main() -> None:  # pragma: no cover - GPU + sandbox operational script
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="runs/coder-1b-instruct-compliant")
    ap.add_argument("--pool", default="data/corpora/grpo_pool.jsonl")
    ap.add_argument("--stats", default="data/corpora/grpo_prepass_stats.jsonl")
    ap.add_argument("--out", default="data/corpora/grpo_pool_signal.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--timeout-s", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="pool rows (smoke)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pool = [json.loads(x) for x in Path(args.pool).read_text().splitlines()]
    if args.limit:
        pool = pool[:args.limit]
    done = set()
    stats_path = Path(args.stats)
    if stats_path.exists():
        done = {json.loads(x)["instruction"] for x in stats_path.read_text().splitlines()}

    model = load_variant_from_run(Path(args.policy), device=args.device).eval()
    tok = FastTokenizer.load(str(Path(args.policy) / "tokenizer.json"))

    with stats_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(pool):
            if row["instruction"] in done:
                continue
            prompt, _ = format_chat(row["instruction"], "")
            replies = sample_solutions(model, tok._tok, prompt, args.k,
                                       max_new=args.max_new, seed=args.seed,
                                       device=args.device)
            rewards = [io_reward(extract_solution(r), row["io"], timeout_s=args.timeout_s)
                       for r in replies]
            successes = sum(1 for r in rewards if r == 1.0)
            f.write(json.dumps({"instruction": row["instruction"],
                                "successes": successes, "k": args.k,
                                "rewards": rewards}) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  prepass {i + 1}/{len(pool)}", flush=True)

    stats = {json.loads(x)["instruction"]: json.loads(x)
             for x in stats_path.read_text().splitlines()}
    signal = [r for r in pool
              if r["instruction"] in stats
              and signal_bearing(stats[r["instruction"]]["successes"],
                                 stats[r["instruction"]]["k"])]
    print(f"prepass: pool={len(pool)} measured={len(stats)} signal={len(signal)}")
    if not signal:
        raise SystemExit("no signal-bearing problems — GRPO would starve (verify by count)")
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in signal:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(signal)} signal rows -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/train/test_exec_reward.py -q`
Expected: PASS (4 passed). (`grpo_prepass.py` main is operational; smoke-run in Task 9.)

- [ ] **Step 5: Commit**

```bash
git add src/microlab/train/exec_reward.py scripts/grpo_prepass.py tests/train/test_exec_reward.py
git commit -m "feat(phase2): sampling helper + signal-bearing pre-pass"
```

---

### Task 4: `--reward executor` mode in train_grpo

**Files:**
- Modify: `scripts/train_grpo.py` (argparse + oracle construction; RM path untouched)
- Test: `tests/scripts/test_train_grpo.py` (add one test)

**Interfaces:**
- Consumes: `make_exec_score_texts`, `run_grpo`, `build_prompt_pool`.
- Produces: CLI `--reward {rm,executor}` (default `rm`), `--pool <jsonl>`, `--timeout-s 5.0`; a module-level `build_executor_oracle(tok, pool_rows: list[dict], max_new: int, block_size: int, timeout_s: float) -> (prompts, score_texts)` that chat-formats each pool row (same block-fit guard as `build_prompt_pool`), builds `io_by_prompt`, and returns the filtered prompt list + oracle.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/scripts/test_train_grpo.py (reuse the module loader already in that file; it
# exposes the script as `tg` — if the loader binds a different name, follow the file's idiom)
def test_build_executor_oracle_maps_prompts_to_io():
    from microlab.model.reference.sft import format_chat

    class _ByteTok:
        def encode(self, s):
            return list(s.encode("utf-8"))

    pool = [{"instruction": "double n", "io": [{"input": "2\n", "output": "4\n"}]},
            {"instruction": "x" * 5000, "io": [{"input": "1\n", "output": "1\n"}]}]  # too long
    prompts, score = tg.build_executor_oracle(_ByteTok(), pool, max_new=64,
                                              block_size=2048, timeout_s=5.0)
    assert len(prompts) == 1                       # oversized row skipped (and counted)
    want_prompt, _ = format_chat("double n", "")
    assert prompts[0] == want_prompt
    got = score(prompts[0], ["```python\nn=int(input());print(n*2)\n```"])
    assert got == [1.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/test_train_grpo.py -k executor -q`
Expected: FAIL (`AttributeError: build_executor_oracle`).

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/train_grpo.py` (near `build_prompt_pool`):

```python
def build_executor_oracle(tok, pool_rows: list[dict], max_new: int, block_size: int,
                          timeout_s: float):
    """Executor-reward mode: chat-format each pool row (same block-fit guard as
    build_prompt_pool), key its I/O cases by the exact prompt string, and return
    (prompts, score_texts). Replaces the RM oracle behind the same interface."""
    from microlab.train.exec_reward import make_exec_score_texts
    sentinel_len = len(tok.encode(END_SENTINEL))
    io_by_prompt: dict[str, list[dict]] = {}
    skipped = 0
    for row in pool_rows:
        prompt, _ = format_chat(row["instruction"], "")
        if len(tok.encode(prompt)) + max_new + sentinel_len > block_size:
            skipped += 1
            continue
        io_by_prompt[prompt] = row["io"]
    if not io_by_prompt:
        raise ValueError(f"no usable pool rows: all {len(pool_rows)} exceed block_size")
    print(f"executor oracle: {len(io_by_prompt)} prompts ({skipped} skipped oversize)")
    return list(io_by_prompt), make_exec_score_texts(io_by_prompt, timeout_s=timeout_s)
```

In `main()`: add `ap.add_argument("--reward", default="rm", choices=["rm", "executor"])`, `ap.add_argument("--pool", default=None, help="pool jsonl (required for --reward executor)")`, `ap.add_argument("--timeout-s", type=float, default=5.0)`. In the wiring section, branch: for `executor`, require `--pool` (raise if absent), load its rows, call `build_executor_oracle(tok._tok, rows, args.max_new, policy_block_size, args.timeout_s)` for `(prompts, score_texts)`, and DO NOT load the RM; the `rm` branch is unchanged. (Follow the file's existing tokenizer/block-size variables — the RM branch shows which.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_train_grpo.py -q`
Expected: PASS (all, including the existing RM-path tests untouched).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_grpo.py tests/scripts/test_train_grpo.py
git commit -m "feat(phase2): --reward executor mode in train_grpo (RM path untouched)"
```

---

### Task 5: Correctness-contrast pairs builder

**Files:**
- Modify: `src/microlab/data/code_sft.py`
- Create: `scripts/build_contrast_prefs.py`
- Test: `tests/data/test_code_sft.py`

**Interfaces:**
- Produces (in `code_sft.py`): `contrast_pairs(problems: list[dict], max_cases: int = 6, max_solutions: int = 8, timeout_s: float = 5.0) -> tuple[list[dict], dict]` — per normalized problem, find one PASSING solution (passes all checked cases) and one WRONG-OUTPUT failing solution (fails ≥1 checked case with `exit_code != 0` and `timed_out == False`; timeouts excluded); emit `{"prompt": format_chat(statement)[0], "chosen": passing, "rejected": failing}`. Tally `{"problems","pairs","no_pair"}`.
- Produces (script): `main()` streaming the three competitive datasets (same sources as Task 2), building pairs, decontaminating (mirror rows with `instruction=statement, response=chosen+rejected` text), writing `data/corpora/contrast_prefs.jsonl` with counts + empty-guard.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/data/test_code_sft.py
from microlab.data.code_sft import contrast_pairs


def test_contrast_pairs_pairs_passing_with_wrong_output_not_timeout():
    problems = [{
        "statement": "double n",
        "solutions": ["n=int(input());print(n*2)",     # passes
                      "n=int(input());print(n+1)",      # wrong output -> valid rejected
                      "while True:\n    pass"],          # timeout -> NOT a valid rejected
        "io": [{"input": "21\n", "output": "42\n"}],
    }]
    pairs, tally = contrast_pairs(problems, timeout_s=2.0)
    assert tally == {"problems": 1, "pairs": 1, "no_pair": 0}
    assert pairs[0]["chosen"] == "n=int(input());print(n*2)"
    assert pairs[0]["rejected"] == "n=int(input());print(n+1)"
    assert "### Instruction" in pairs[0]["prompt"]      # chat-formatted


def test_contrast_pairs_skips_problem_without_both_sides():
    problems = [{"statement": "s", "solutions": ["print('right')"],
                 "io": [{"input": "", "output": "right\n"}]}]      # passing only, no failing
    pairs, tally = contrast_pairs(problems)
    assert pairs == [] and tally["no_pair"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_code_sft.py -k contrast -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/microlab/data/code_sft.py
from microlab.model.reference.sft import format_chat as _format_chat


def _io_outcome(solution: str, case: dict, timeout_s: float):
    """One sandbox run -> the ExecResult (callers need timed_out vs wrong-output)."""
    prog = assemble_io_program(solution, case["input"], case["output"])
    return run_python(prog, timeout_s=timeout_s)


def contrast_pairs(problems: list[dict], max_cases: int = 6, max_solutions: int = 8,
                   timeout_s: float = 5.0) -> tuple[list[dict], dict]:
    """Correctness-contrast DPO pairs: chosen = a solution passing ALL checked cases,
    rejected = one failing >=1 case by WRONG OUTPUT (timeouts excluded — slow-but-correct is
    a bad 'rejected'). Both sides are human solutions; the executor is the label."""
    pairs: list[dict] = []
    tally = {"problems": 0, "pairs": 0, "no_pair": 0}
    for p in problems:
        tally["problems"] += 1
        statement = (p.get("statement") or "").strip()
        cases = (p.get("io") or [])[:max_cases]
        sols = sorted(p.get("solutions") or [], key=len)[:max_solutions]
        if not statement or not cases or len(sols) < 2:
            tally["no_pair"] += 1
            continue
        chosen = rejected = None
        for sol in sols:
            outcomes = [_io_outcome(sol, c, timeout_s) for c in cases]
            if all(o.passed for o in outcomes):
                chosen = chosen or sol
            elif any((not o.passed) and (not o.timed_out) for o in outcomes) \
                    and not any(o.timed_out for o in outcomes):
                rejected = rejected or sol
            if chosen and rejected:
                break
        if chosen and rejected:
            pairs.append({"prompt": _format_chat(statement, "")[0],
                          "chosen": chosen.strip(), "rejected": rejected.strip()})
            tally["pairs"] += 1
        else:
            tally["no_pair"] += 1
    return pairs, tally
```

```python
# scripts/build_contrast_prefs.py
"""Correctness-contrast IPO pairs from competitive problems (chosen=passing,
rejected=wrong-output; executor-labeled, human-authored — build-capability compliant).

    python scripts/build_contrast_prefs.py --out data/corpora/contrast_prefs.jsonl \\
        --limit-per-dataset 3000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.data.code_sft import (  # noqa: E402
    apps_problem, benchmark_fingerprints, codecontests_problem, contrast_pairs,
    decontaminate, taco_problem,
)


def main() -> None:  # pragma: no cover - network + sandbox operational
    from datasets import load_dataset

    from microlab.evals.code.tasks import load_humaneval, load_mbpp

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/corpora/contrast_prefs.jsonl")
    ap.add_argument("--limit-per-dataset", type=int, default=3000)
    ap.add_argument("--max-cases", type=int, default=6)
    ap.add_argument("--max-solutions", type=int, default=8)
    ap.add_argument("--timeout-s", type=float, default=5.0)
    args = ap.parse_args()

    adapters = [
        ("json", "hf://datasets/codeparrot/apps/train.jsonl", apps_problem),
        ("deepmind/code_contests", None, codecontests_problem),
        ("parquet", "hf://datasets/BAAI/TACO/ALL/train-*.parquet", taco_problem),
    ]
    all_pairs, tally = [], {"problems": 0, "pairs": 0, "no_pair": 0}
    for fmt, data_files, adapt in adapters:
        it = load_dataset(fmt, data_files=data_files, split="train", streaming=True) \
            if data_files else load_dataset(fmt, split="train", streaming=True)
        probs, seen = [], 0
        for r in it:
            probs.append(adapt(r))
            seen += 1
            if args.limit_per_dataset and seen >= args.limit_per_dataset:
                break
        pairs, t = contrast_pairs(probs, max_cases=args.max_cases,
                                  max_solutions=args.max_solutions,
                                  timeout_s=args.timeout_s)
        all_pairs += pairs
        for k in tally:
            tally[k] += t[k]
        print(f"  {adapt.__name__}: {t}", flush=True)

    bench = [t.prompt + "\n" + t.test_program for t in (load_humaneval() + load_mbpp())]
    fp = benchmark_fingerprints(bench, n=10)
    mirrors = [{"instruction": p["prompt"], "context": "",
                "response": p["chosen"] + " " + p["rejected"]} for p in all_pairs]
    kept, removed = decontaminate(mirrors, fp, n=10)
    kept_prompts = {m["instruction"] for m in kept}
    all_pairs = [p for p in all_pairs if p["prompt"] in kept_prompts]

    print(f"contrast pairs: {tally} decontaminated_removed={removed} total={len(all_pairs)}")
    if not all_pairs:
        raise SystemExit("no contrast pairs — refusing to proceed (verify by count)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(all_pairs)} pairs -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_code_sft.py -k contrast -q`
Expected: PASS (2 passed; real sandbox runs, a few seconds).

- [ ] **Step 5: Commit**

```bash
git add src/microlab/data/code_sft.py scripts/build_contrast_prefs.py tests/data/test_code_sft.py
git commit -m "feat(phase2): correctness-contrast pairs (passing vs wrong-output, timeouts excluded)"
```

---

### Task 6: Delivered-correctness (rerank) summarizer

**Files:**
- Create: `scripts/eval_rerank.py`
- Test: `tests/scripts/test_eval_rerank.py`

**Interfaces:**
- Consumes: an `eval_code.py --n K` output JSONL (per-sample rows `{task_id, sample, passed}` + `_header` row).
- Produces: `delivered(rows: list[dict]) -> dict` (pure): `{"n_tasks", "k", "delivered_correct", "delivered_rate", "pass@1_first_sample"}` where a task counts as delivered-correct when ANY of its k samples passed (executor-rerank == pick the passer); `main()` reading a JSONL and printing/writing the summary.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_eval_rerank.py
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "eval_rerank", Path(__file__).resolve().parents[2] / "scripts" / "eval_rerank.py")
er = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(er)


def test_delivered_counts_any_passing_sample():
    rows = [{"_header": {}},
            {"task_id": "T1", "sample": 0, "passed": False},
            {"task_id": "T1", "sample": 1, "passed": True},
            {"task_id": "T2", "sample": 0, "passed": False},
            {"task_id": "T2", "sample": 1, "passed": False}]
    got = er.delivered(rows)
    assert got["n_tasks"] == 2 and got["k"] == 2
    assert got["delivered_correct"] == 1 and got["delivered_rate"] == 0.5
    assert got["pass@1_first_sample"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/test_eval_rerank.py -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/eval_rerank.py
"""Delivered correctness under executor reranking, from an eval_code --n K output.

With tests available, best-of-k + executor rerank delivers a task iff ANY sample passes —
so delivered_rate == pass@any-of-k, computed from the per-sample rows eval_code already
writes. pass@1_first_sample is the unbiased single-draw baseline for the same run.

    python scripts/eval_rerank.py evals/instruct/<run>-humaneval-sampled.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def delivered(rows: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if "_header" in r:
            continue
        by_task[r["task_id"]].append(r)
    n = len(by_task)
    if n == 0:
        raise ValueError("no task rows — wrong file? (verify by count)")
    k = max(len(v) for v in by_task.values())
    correct = sum(1 for v in by_task.values() if any(s["passed"] for s in v))
    first = sum(1 for v in by_task.values()
                if any(s["passed"] and s.get("sample") == 0 for s in v))
    return {"n_tasks": n, "k": k, "delivered_correct": correct,
            "delivered_rate": correct / n, "pass@1_first_sample": first / n}


def main() -> None:  # pragma: no cover - thin IO wrapper
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.jsonl.read_text().splitlines()]
    rep = delivered(rows)
    print(json.dumps(rep, indent=2))
    if args.out:
        args.out.write_text(json.dumps(rep, indent=2) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_eval_rerank.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_rerank.py tests/scripts/test_eval_rerank.py
git commit -m "feat(phase2): delivered-correctness summarizer (executor rerank == pass@any-of-k)"
```

---

### Task 7: Self-gen SFT builder

**Files:**
- Create: `scripts/build_selfgen_sft.py`
- Test: `tests/scripts/test_build_selfgen_sft.py`

**Interfaces:**
- Consumes: `sample_solutions`, `extract_solution`, `io_reward`, `benchmark_fingerprints`, `decontaminate`.
- Produces: `selfgen_row(instruction: str, solutions_with_rewards: list[tuple[str, float]]) -> dict | None` (pure): keep the SHORTEST solution with reward == 1.0 → `{"instruction", "context": "", "response"}`, None if no full pass; `main()` (`# pragma: no cover`): for each FULL-pool row, sample k from `--policy`, extract+reward, `selfgen_row`, progressive/resumable append to a work JSONL keyed by instruction, then dedup (exact-response dedup), decontaminate, write `data/corpora/selfgen_sft.jsonl` with counts + empty-guard.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_build_selfgen_sft.py
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_selfgen_sft", Path(__file__).resolve().parents[2] / "scripts" / "build_selfgen_sft.py")
bs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bs)


def test_selfgen_row_keeps_shortest_full_pass():
    sols = [("print(long_version(42))", 1.0), ("print(42)", 1.0), ("print(41)", 0.5)]
    row = bs.selfgen_row("emit 42", sols)
    assert row == {"instruction": "emit 42", "context": "", "response": "print(42)"}
    assert bs.selfgen_row("x", [("print(1)", 0.5)]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/test_build_selfgen_sft.py -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/build_selfgen_sft.py
"""Self-generated SFT tranche: the policy samples k solutions per pool problem; the
executor keeps full passers (SelfCodeAlign-style, own-model + executor labels — compliant).

    python scripts/build_selfgen_sft.py --policy runs/<best> \\
        --pool data/corpora/grpo_pool.jsonl --k 8 \\
        --work data/corpora/selfgen_work.jsonl --out data/corpora/selfgen_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def selfgen_row(instruction: str, solutions_with_rewards: list[tuple[str, float]]) -> dict | None:
    """Shortest full-pass (reward == 1.0) solution -> SFT row; None when nothing fully
    passes (a partial pass must NOT become training signal)."""
    full = sorted((s for s, r in solutions_with_rewards if r == 1.0 and s.strip()), key=len)
    if not full:
        return None
    return {"instruction": instruction, "context": "", "response": full[0]}


def main() -> None:  # pragma: no cover - GPU + sandbox operational
    import torch

    from microlab.data.code_sft import benchmark_fingerprints, decontaminate
    from microlab.evals.code.tasks import load_humaneval, load_mbpp
    from microlab.model.reference.checkpoint import load_variant_from_run
    from microlab.model.reference.sft import format_chat
    from microlab.tokenizer.fast import FastTokenizer
    from microlab.train.exec_reward import extract_solution, io_reward, sample_solutions

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--pool", default="data/corpora/grpo_pool.jsonl")
    ap.add_argument("--work", default="data/corpora/selfgen_work.jsonl")
    ap.add_argument("--out", default="data/corpora/selfgen_sft.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--timeout-s", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pool = [json.loads(x) for x in Path(args.pool).read_text().splitlines()]
    if args.limit:
        pool = pool[:args.limit]
    work = Path(args.work)
    done = {json.loads(x)["instruction"] for x in work.read_text().splitlines()} \
        if work.exists() else set()

    model = load_variant_from_run(Path(args.policy), device=args.device).eval()
    tok = FastTokenizer.load(str(Path(args.policy) / "tokenizer.json"))

    with work.open("a", encoding="utf-8") as f:
        for i, row in enumerate(pool):
            if row["instruction"] in done:
                continue
            prompt, _ = format_chat(row["instruction"], "")
            replies = sample_solutions(model, tok._tok, prompt, args.k,
                                       max_new=args.max_new, seed=args.seed,
                                       device=args.device)
            swr = [(extract_solution(r),
                    io_reward(extract_solution(r), row["io"], timeout_s=args.timeout_s))
                   for r in replies]
            out_row = selfgen_row(row["instruction"], swr)
            f.write(json.dumps({"instruction": row["instruction"], "row": out_row}) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  selfgen {i + 1}/{len(pool)}", flush=True)

    rows, seen_resp = [], set()
    for line in work.read_text().splitlines():
        d = json.loads(line)
        r = d.get("row")
        if r and r["response"] not in seen_resp:
            seen_resp.add(r["response"])
            rows.append(r)
    bench = [t.prompt + "\n" + t.test_program for t in (load_humaneval() + load_mbpp())]
    rows, removed = decontaminate(rows, benchmark_fingerprints(bench, n=10), n=10)
    print(f"selfgen: pool={len(pool)} kept={len(rows)} decontaminated_removed={removed}")
    if not rows:
        raise SystemExit("no self-gen rows — refusing to proceed (verify by count)")
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_build_selfgen_sft.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_selfgen_sft.py tests/scripts/test_build_selfgen_sft.py
git commit -m "feat(phase2): self-gen SFT builder (executor-filtered, shortest full pass)"
```

---

### Task 8: Pre-registered prediction doc

**Files:**
- Create: `docs/coder-1b-phase2-prediction.md`

Committed BEFORE Task 9 runs any training (the gate). Must contain, with numbers: the v1 baseline (HumanEval greedy 3.0%, sampled p@1 1.2% / p@10 6.1%; MBPP greedy 14.0%; FIM 0.6255; passkey 91.7%); per-stage bands — **GRPO:** MBPP greedy 18–22%, HumanEval sampled p@1 4–5% (ceiling 6.1% p@10), HumanEval greedy 4–9%; **contrast-IPO:** +1–3 MBPP over v1; **stack** (if both help): ≥ the best solo; **self-gen:** +1–4 MBPP over its base checkpoint; guardrails: FIM ≤ 0.68, passkey ≥ 82%, codex pairwise vs v1 ≥ 40% for every stage; falsifiers: any stage below v1 on BOTH benchmarks = lever failed (revert to previous checkpoint); reward > benchmark divergence (train reward rising while MBPP falls) = reward hacking, stop and inspect rollouts; a "too good" tripwire (MBPP > 28% = check pool/benchmark leakage). End with the honest caveat: GRPO gains are bounded by the policy's pass@k ceiling; these bands assume no new capability, only selection.

- [ ] **Step 1: Write the doc** (content per above, house style of `docs/coder-1b-instruct-prediction.md`).
- [ ] **Step 2: Commit**

```bash
git add docs/coder-1b-phase2-prediction.md
git commit -m "docs: pre-registered Phase 2 prediction (before any training)"
```

---

### Task 9: Operational run — pool, pre-pass, train, gate, stack, self-gen, milestone

**Files:** none (data + run dirs + milestone doc). Sequenced; each step names its command and expected output. PAUSE for the user before starting (GPU-hours) per the controller's instruction.

- [ ] **Step 1: Build pool.** `python scripts/build_grpo_pool.py --limit-per-dataset 4000 --max-cases 6` → expect per-source counts, total ≥ 3000, decontam small. Verify by count.
- [ ] **Step 2: Pre-pass (smoke then full).** Smoke: `python scripts/grpo_prepass.py --limit 20 --k 8` → confirm stats append + signal filter works. Full run (background, hours: k×pool generations): expect `signal=` several hundred+. If < 200, STOP — consult the prediction doc's starvation note before proceeding.
- [ ] **Step 3: GRPO smoke.** `python scripts/train_grpo.py --reward executor --pool data/corpora/grpo_pool_signal.jsonl --policy runs/coder-1b-instruct-compliant --out runs/coder-1b-grpo-exec --iters 3 --max-new 300 --group-size 8 --prompts-per-iter 4 --micro-batch 4` → confirm iterations log rewards with nonzero variance, samples.jsonl shows code rollouts, no OOM (watch VRAM; policy+ref at 1.2B).
- [ ] **Step 4: GRPO full.** Same command, `--iters 300 --save-every 25 --dump-every 10` (background; monitor grpo_log.jsonl reward trend + eyeball samples for hacking every ~50 iters). Expect mean reward rising over iters.
- [ ] **Step 5: GRPO eval gate.** Greedy HumanEval+MBPP (`eval_code --mode chat`), sampled HumanEval `--n 10`, `eval_rerank` on it, FIM+passkey guardrails (`eval_suite --no-probes`, `eval_passkey`), codex pairwise vs v1 on `code_sft_heldout.jsonl`. Score vs prediction bands.
- [ ] **Step 6: Contrast pairs + IPO.** `python scripts/build_contrast_prefs.py --limit-per-dataset 3000` (expect thousands of pairs); `python scripts/dpo.py --sft-ckpt runs/coder-1b-instruct-compliant --prefs data/corpora/contrast_prefs.jsonl --loss ipo --out runs/coder-1b-ipo-contrast` (background, ~1–2h); same eval gate.
- [ ] **Step 7: Stack per the spec rule.** If both beat v1: re-run the smaller-gain lever from the winner's checkpoint; eval; keep only if it beats the solo winner.
- [ ] **Step 8: Self-gen from the best.** `python scripts/build_selfgen_sft.py --policy runs/<best> --k 8` (background, hours); then `python scripts/sft.py --base-ckpt runs/<best> --data data/corpora/selfgen_sft.jsonl --epochs 2 --lr 1e-5 --batch-size 2 --grad-accum 8 --block-size 2048 --out runs/coder-1b-selfgen` (lighter than the main recipe — this is a top-up, not a fresh SFT); full eval gate.
- [ ] **Step 9: Milestone.** `docs/coder-1b-phase2-milestone.md`: every stage vs its band, error-mode taxonomy shift (did wrong-logic share fall?), delivered-correctness table, final named model. Commit doc + eval artifacts + updated ledger.

---

## Self-Review (completed during planning)

- **Spec coverage:** Unit 1 → Tasks 1–3 (oracle, pool, pre-pass); Unit 2 → Task 4 + Task 9 steps 3–5; Unit 3 → Task 5 + step 6; Unit 4 → Tasks 6–7 + steps 8; prediction → Task 8 (gated before step 3); stacking rule → step 7 (spec's explicit rule); codex judge → step 5/6/8 gates; risks (starvation floor, hacking telemetry, passkey watch) → steps 2/4/5.
- **Placeholder scan:** all code steps carry complete code; Task 8 is a doc with its required numbers enumerated; Task 9 is operational with concrete commands. Task 4's "follow the file's existing tokenizer/block-size variables" is a bounded instruction to match the RM branch's local idiom, not a TBD.
- **Type consistency:** `score_texts(prompt, texts) -> list[float]` everywhere; pool row `{"instruction","io"}` (Tasks 2→3→4→7); prefs `{prompt, chosen, rejected}` (Task 5 → dpo.py); `sample_solutions` consumed by Tasks 3 and 7 with the same signature; `io_reward`/`extract_solution` shared 1→3→7.

## Notes / risks carried from the spec

- GRPO wall-clock is dominated by rollout generation + sandbox scoring (prompts_per_iter × group_size executions/iter; at ~6 cases × ~50ms that's seconds/iter — fine).
- `max_new 300` for code rollouts (the prose default 80 truncates functions); block-fit guard handles long statements.
- The pre-pass and self-gen share sampling; both are resumable (the [[long-jobs-write-progressively]] rule).
- Reward-hacking telemetry is a human/controller loop (eyeball samples.jsonl), not automated — deliberate; the prose run caught hacking exactly this way.
