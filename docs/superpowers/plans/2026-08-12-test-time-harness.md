# Test-Time Execution Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harvest coder-1b v1's measured pass@k gap at inference: with-tests best-of-n (guaranteed win), probe-gated no-tests selection science on leakage-clean banks, and a tests-required `--best-of` CLI.

**Architecture:** A pure selection core (`src/microlab/infer/selection.py`) consuming pre-computed execution results; a small behavioral-execution helper (`src/microlab/infer/behavior.py`) that runs a candidate on one input in the sandbox and returns a comparable signature; thin runner scripts for the probe, the test-free bank, the selection experiment, and the 3a CLI. v1 is frozen; nothing trains.

**Tech Stack:** Python, existing `microlab` package (sandbox executor, batched sampler, eval harness, tokenizer). No new dependencies.

## Global Constraints

- **No training; v1 (`runs/coder-1b-instruct-compliant`) is frozen.** $0, local RTX 6000 Ada only (no renting).
- **Hidden benchmark tests are used ONLY for final scoring**, never for selection; additionally (generator-side rule) Unit 2 methods only consume banks whose generation prompts contained no gold tests — the **test-free MBPP bank** is the powered comparison; HumanEval is descriptive only.
- **Unit 0 measures DISCRIMINATION** (reference output differs from a known-wrong solution), not executability; probe prompts contain description+signature only, never gold asserts.
- Docstring-extracted vs synthesized inputs are **never pooled** in reported tables.
- No silent fallbacks; every method reports per-task coverage (verify by count); empty outputs raise before writing.
- Falsifiers (pre-registered in Task 7's doc, checked by code where possible): any method beating its bank's oracle = bug (raise); powered-bank recoverable gap < 15 tasks = report counts only, no %-headline.
- Row schema for banks matches `eval_code.py` output: `{"task_id", "sample", "passed", "solution", ...}` + one `_header` row, so `eval_rerank.delivered` works unchanged.

## Interfaces this plan builds on (verified in-repo this session)

- `microlab.evals.code.executor.run_python(code, *, timeout_s=...) -> ExecResult` (`.passed`, `.timed_out`, `.exit_code`, `.stdout`).
- `microlab.evals.code.tasks`: `CodeTask(task_id, prompt, instruction, entry_point, test_program)`, `assemble_program(solution, task)`, `humaneval_task(row)`, `mbpp_task(row)`, `load_humaneval()`, `load_mbpp()`. HumanEval rows: `prompt` (with docstring), `canonical_solution`, `test`, `entry_point`. MBPP sanitized rows: `prompt` (description), `code` (reference), `test_list`, `test_imports`.
- `microlab.train.exec_reward.sample_solutions(model, tok_raw, prompt, k, *, max_new, temp, top_k, seed, device)` (raw `tokenizers.Tokenizer`: `.encode(s).ids`/`.decode(ids)`); `extract_solution(reply)`.
- `microlab.model.reference.checkpoint.load_variant_from_run(run_dir, device=...) -> (model, step)`; `microlab.model.reference.sft.format_chat(instruction, context="")`; `FastTokenizer.load(path)` (`._tok` is the raw tokenizer).
- `scripts/eval_code.py --run --dataset --mode chat --n K --temperature --top-k --out` (writes bank + `.summary.json`); `scripts/eval_rerank.py`'s `delivered(rows) -> dict`.
- v1 chat template stops on `### End`; `serve_config.json` mode chat.

---

### Task 1: Pure selection core

**Files:**
- Create: `src/microlab/infer/selection.py`
- Test: `tests/infer/test_selection.py` (create `tests/infer/`)

**Interfaces:**
- Produces: `normalize_code(s: str) -> str`; `text_plurality(candidates: list[str]) -> int` (index); `behavior_clusters(signatures: list[tuple]) -> list[list[int]]` (index groups, largest first, deterministic); `pick_from_cluster(cluster: list[int], candidates: list[str], rule: str = "shortest", seed: int = 0) -> int`; `select_by_self_tests(assert_pass_counts: list[int]) -> int`; `first_sample() -> 0`. All pure — no sandbox, no IO.

- [ ] **Step 1: Write the failing test**

```python
# tests/infer/test_selection.py
from microlab.infer.selection import (
    behavior_clusters, normalize_code, pick_from_cluster, select_by_self_tests,
    text_plurality,
)


def test_normalize_and_text_plurality():
    cands = ["def f(x):\n    return x+1", "def f(x):\n\treturn x+1",  # same normalized
             "def f(x):\n    return x+2"]
    assert normalize_code(cands[0]) == normalize_code(cands[1])
    assert text_plurality(cands) in (0, 1)          # the duplicated variant wins


def test_behavior_clusters_groups_identical_signatures_largest_first():
    sigs = [("ok", "1"), ("ok", "2"), ("ok", "1"), ("err",), ("ok", "1")]
    clusters = behavior_clusters(sigs)
    assert clusters[0] == [0, 2, 4]                  # largest cluster first
    assert [1] in clusters and [3] in clusters


def test_pick_from_cluster_rules():
    cands = ["longer_candidate_text", "ab", "medium_one"]
    assert pick_from_cluster([0, 1, 2], cands, rule="shortest") == 1
    r = pick_from_cluster([0, 1, 2], cands, rule="random", seed=7)
    assert r in (0, 1, 2)
    assert pick_from_cluster([0, 1, 2], cands, rule="random", seed=7) == r  # deterministic


def test_select_by_self_tests_argmax_first_tie():
    assert select_by_self_tests([1, 3, 3, 0]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/infer/test_selection.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/microlab/infer/selection.py
"""Pure candidate-selection rules for the test-time harness.

Every function consumes ALREADY-COMPUTED results (strings, signatures, counts) — no
sandbox calls, no IO — so the selection science is hermetically testable. The impure
execution that produces behavioral signatures lives in microlab.infer.behavior.
"""
from __future__ import annotations

import random
import re

_WS = re.compile(r"\s+")


def normalize_code(s: str) -> str:
    """Whitespace-collapsed form for text-plurality (tabs/spaces/newlines equivalent)."""
    return _WS.sub(" ", s.strip())


def first_sample() -> int:
    """The floor selector: always the first draw (an unbiased single sample)."""
    return 0


def text_plurality(candidates: list[str]) -> int:
    """Index of the first candidate whose NORMALIZED text is most frequent. A floor
    baseline only — semantically equal code has many textual forms (see spec I4)."""
    norm = [normalize_code(c) for c in candidates]
    counts: dict[str, int] = {}
    for n in norm:
        counts[n] = counts.get(n, 0) + 1
    best = max(counts.values())
    for i, n in enumerate(norm):
        if counts[n] == best:
            return i
    raise AssertionError("unreachable: candidates nonempty")


def behavior_clusters(signatures: list[tuple]) -> list[list[int]]:
    """Group candidate indices by identical behavioral signature. Largest cluster first;
    ties broken by smallest first-index (deterministic)."""
    groups: dict[tuple, list[int]] = {}
    for i, s in enumerate(signatures):
        groups.setdefault(s, []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def pick_from_cluster(cluster: list[int], candidates: list[str], rule: str = "shortest",
                      seed: int = 0) -> int:
    """Pick one index from a cluster. rule='shortest' (Occam tiebreak — ablated, can favor
    degenerate code) or 'random' (seeded). Unknown rule raises."""
    if rule == "shortest":
        return min(cluster, key=lambda i: (len(candidates[i]), i))
    if rule == "random":
        return random.Random(seed).choice(sorted(cluster))
    raise ValueError(f"unknown pick rule {rule!r}")


def select_by_self_tests(assert_pass_counts: list[int]) -> int:
    """Index with the most self-test asserts passed; first index wins ties."""
    best = max(assert_pass_counts)
    return assert_pass_counts.index(best)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/infer/test_selection.py -q` — Expected: PASS (4 passed).
(Check: `src/microlab/infer/` already contains `reference/`; add `selection.py` beside it. If `tests/infer/` needs an `__init__` skip it — other test dirs have none.)

- [ ] **Step 5: Commit**

```bash
git add src/microlab/infer/selection.py tests/infer/test_selection.py
git commit -m "feat(harness): pure selection core (plurality, behavior clusters, pick rules)"
```

---

### Task 2: Behavioral execution helper

**Files:**
- Create: `src/microlab/infer/behavior.py`
- Test: `tests/infer/test_behavior.py`

**Interfaces:**
- Consumes: `run_python`.
- Produces: `behavior_signature(candidate: str, entry_point: str, input_expr: str, timeout_s: float = 3.0) -> tuple` — runs the candidate on ONE call-expression input, returns `("ok", stdout_repr)` on clean exit, `("err",)` on error, `("timeout",)` on timeout; `signatures_for(candidate, entry_point, input_exprs: list[str], timeout_s=3.0) -> tuple` — the tuple of per-input signatures (the clustering key).

- [ ] **Step 1: Write the failing test**

```python
# tests/infer/test_behavior.py
from microlab.infer.behavior import behavior_signature, signatures_for

ADD = "def add(a, b):\n    return a + b"
BAD = "def add(a, b):\n    return a - b"
BOOM = "def add(a, b):\n    raise ValueError"


def test_signature_distinguishes_right_from_wrong():
    s_good = behavior_signature(ADD, "add", "add(2, 3)")
    s_bad = behavior_signature(BAD, "add", "add(2, 3)")
    assert s_good[0] == "ok" and s_bad[0] == "ok" and s_good != s_bad


def test_signature_error_and_vector():
    assert behavior_signature(BOOM, "add", "add(1, 1)") == ("err",)
    vec = signatures_for(ADD, "add", ["add(1, 1)", "add(2, 2)"])
    assert vec == (("ok", "2"), ("ok", "4"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/infer/test_behavior.py -q` — Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/microlab/infer/behavior.py
"""Behavioral signatures: run a candidate on one input expression in the sandbox and
return a comparable outcome. The impure counterpart to microlab.infer.selection —
clustering compares these signatures; candidates that behave identically on shared
inputs land in the same cluster (AlphaCode-lineage behavioral equivalence)."""
from __future__ import annotations

from microlab.evals.code.executor import run_python

_HARNESS = "{candidate}\n\n_r = {input_expr}\nprint(repr(_r))\n"


def behavior_signature(candidate: str, entry_point: str, input_expr: str,
                       timeout_s: float = 3.0) -> tuple:
    """One (candidate, input) execution -> ("ok", repr) | ("err",) | ("timeout",).
    `entry_point` is accepted for interface clarity/logging; the input_expr already names
    the callable. Errors collapse to ("err",) deliberately: two candidates failing
    differently should not cluster as 'same behavior' by error-text accident."""
    prog = _HARNESS.format(candidate=candidate, input_expr=input_expr)
    res = run_python(prog, timeout_s=timeout_s)
    if res.timed_out:
        return ("timeout",)
    if res.exit_code != 0:
        return ("err",)
    return ("ok", res.stdout.strip())


def signatures_for(candidate: str, entry_point: str, input_exprs: list[str],
                   timeout_s: float = 3.0) -> tuple:
    """Signature VECTOR over all shared inputs — the clustering key."""
    return tuple(behavior_signature(candidate, entry_point, e, timeout_s=timeout_s)
                 for e in input_exprs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/infer/test_behavior.py -q` — Expected: PASS (real sandbox, ~2s).

- [ ] **Step 5: Commit**

```bash
git add src/microlab/infer/behavior.py tests/infer/test_behavior.py
git commit -m "feat(harness): behavioral signatures (sandboxed candidate-on-input outcomes)"
```

---

### Task 3: Test-free MBPP bank generator

**Files:**
- Create: `scripts/gen_testfree_bank.py`
- Test: `tests/scripts/test_gen_testfree_bank.py`

**Interfaces:**
- Consumes: `mbpp_task`, `assemble_program`, `run_python`, `sample_solutions`, `extract_solution`, `load_variant_from_run`, `format_chat`, `FastTokenizer`.
- Produces (pure, tested): `mbpp_signature(code: str) -> str | None` (first `def ...` line of the reference, sans trailing colon body); `testfree_instruction(prompt: str, signature: str) -> str` — description + bare signature, NO asserts; (script) `main()` writing an eval_code-compatible bank JSONL (`_header` + per-sample rows with `task_id/sample/passed/solution`) + `.summary.json` with pass@1/pass@10 via the unbiased estimator already in `eval_rerank.delivered`-compatible form. Progressive + resumable by (task_id) key; empty-guard before final summary write.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_gen_testfree_bank.py
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "gen_testfree_bank", Path(__file__).resolve().parents[2] / "scripts" / "gen_testfree_bank.py")
gb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gb)


def test_mbpp_signature_extracts_def_line():
    code = "import math\ndef remove_Occ(s, ch):\n    return s\n"
    assert gb.mbpp_signature(code) == "def remove_Occ(s, ch):"
    assert gb.mbpp_signature("x = 1\n") is None


def test_testfree_instruction_contains_no_asserts():
    ins = gb.testfree_instruction("Write a python function to do X.", "def do_x(a):")
    assert "assert" not in ins
    assert "def do_x(a):" in ins and "Write a python function to do X." in ins
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/test_gen_testfree_bank.py -q` — Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/gen_testfree_bank.py
"""Generate the TEST-FREE MBPP sample bank (spec finding C1): prompts carry the task
description + bare function signature ONLY — never the gold asserts that mbpp_task's
standard chat instruction embeds. This bank is Unit 2's leakage-clean powered comparison;
its pass@k vs the standard bank's also measures test-conditioning.

    python scripts/gen_testfree_bank.py --policy runs/coder-1b-instruct-compliant \\
        --out evals/harness/mbpp-testfree-bank.jsonl --n 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def mbpp_signature(code: str) -> str | None:
    """First `def ...:` line of the reference solution (the bare signature), else None."""
    for line in (code or "").splitlines():
        if line.strip().startswith("def ") and line.rstrip().endswith(":"):
            return line.strip()
    return None


def testfree_instruction(prompt: str, signature: str) -> str:
    """Description + signature, no tests. The generator must stay blind to gold asserts."""
    return (f"{prompt.strip()}\n\nWrite the complete Python function "
            f"`{signature}` Reply with the full function definition.")


def main() -> None:  # pragma: no cover - GPU + sandbox operational
    import torch
    from datasets import load_dataset

    from microlab.evals.code.executor import run_python
    from microlab.evals.code.tasks import assemble_program, mbpp_task
    from microlab.model.reference.checkpoint import load_variant_from_run
    from microlab.model.reference.sft import format_chat
    from microlab.tokenizer.fast import FastTokenizer
    from microlab.train.exec_reward import extract_solution, sample_solutions

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="runs/coder-1b-instruct-compliant")
    ap.add_argument("--out", default="evals/harness/mbpp-testfree-bank.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            d = json.loads(line)
            if "task_id" in d:
                done.add(d["task_id"])
    model, _ = load_variant_from_run(Path(args.policy), device=args.device)
    tok = FastTokenizer.load(str(Path(args.policy) / "tokenizer.json"))

    with out.open("a", encoding="utf-8") as f:
        if not done:
            f.write(json.dumps({"_header": {
                "run": args.policy, "dataset": "mbpp-testfree", "n": args.n,
                "temperature": args.temp, "top_k": args.top_k, "seed": args.seed,
                "note": "generation prompts contain NO gold asserts (spec C1)"}}) + "\n")
        skipped_sig = 0
        for i, row in enumerate(rows):
            task = mbpp_task(row)
            if task.task_id in done:
                continue
            sig = mbpp_signature(row.get("code", ""))
            if sig is None:
                skipped_sig += 1
                continue
            prompt, _ = format_chat(testfree_instruction(row["prompt"], sig), "")
            replies = sample_solutions(model, tok._tok, prompt, args.n,
                                       max_new=args.max_new, temp=args.temp,
                                       top_k=args.top_k, seed=args.seed,
                                       device=args.device)
            for s_idx, rep in enumerate(replies):
                sol = extract_solution(rep)
                ok = bool(sol.strip()) and run_python(
                    assemble_program(sol, task), timeout_s=args.timeout_s).passed
                f.write(json.dumps({"task_id": task.task_id, "sample": s_idx,
                                    "passed": ok, "solution": sol}) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"  bank {i + 1}/{len(rows)}", flush=True)
    print(f"skipped (no signature): {skipped_sig}")

    bank = [json.loads(x) for x in out.read_text().splitlines()]
    tasks_written = {r["task_id"] for r in bank if "task_id" in r}
    if not tasks_written:
        raise SystemExit("empty bank — refusing to summarize (verify by count)")
    # summary via the shared delivered() logic
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "eval_rerank", Path(__file__).resolve().parent / "eval_rerank.py")
    er = ilu.module_from_spec(spec)
    spec.loader.exec_module(er)
    rep = er.delivered(bank)
    Path(str(out) + ".summary.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(f"summary: {rep}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_gen_testfree_bank.py -q` — Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_testfree_bank.py tests/scripts/test_gen_testfree_bank.py
git commit -m "feat(harness): test-free MBPP bank generator (no gold asserts at generation)"
```

---

### Task 4: Input-discrimination probe

**Files:**
- Create: `scripts/probe_input_discrimination.py`
- Test: `tests/scripts/test_probe_input_discrimination.py`

**Interfaces:**
- Consumes: `behavior_signature` (Task 2), `sample_solutions`, `extract_solution`, task loaders.
- Produces (pure, tested): `parse_input_exprs(reply: str, entry_point: str) -> list[str]` — lines/fragments that are call-expressions of `entry_point` (regex `entry_point\s*\(.*\)`), deduped, max 3; `make_mutant(code: str) -> str | None` — first successful mechanical mutation (swap `<`→`<=`, `>`→`>=`, `+`→`-`, `==`→`!=`, one at a time, first that changes the text), else None; `is_discriminating(sig_ref: tuple, sig_wrong: tuple) -> bool` — both ran (`ref[0]=="ok"`) and differ. (Script) `main()`: for N tasks (50 HE / 50 MBPP), synthesis prompt = description+signature ONLY; wrong solution drawn from an existing bank's failed candidates for that task if available else `make_mutant(reference)` (labeled); appends per-task JSONL `{task_id, n_inputs, n_discriminating, wrong_source}`; prints the gate fraction + which soft band it lands in.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_probe_input_discrimination.py
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "probe_id", Path(__file__).resolve().parents[2] / "scripts" / "probe_input_discrimination.py")
pid = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pid)


def test_parse_input_exprs_extracts_calls_only():
    reply = "Here:\nadd(1, 2)\nprint(add(3,4))\nnot_a_call\nadd(5, 6)\nadd(1, 2)"
    got = pid.parse_input_exprs(reply, "add")
    assert got == ["add(1, 2)", "add(3,4)", "add(5, 6)"]   # deduped, max 3, call-only


def test_make_mutant_changes_code_or_none():
    assert pid.make_mutant("def f(a):\n    return a + 1") == "def f(a):\n    return a - 1"
    assert pid.make_mutant("def f():\n    pass") is None


def test_is_discriminating_requires_ref_ok_and_difference():
    assert pid.is_discriminating(("ok", "1"), ("ok", "2")) is True
    assert pid.is_discriminating(("ok", "1"), ("err",)) is True
    assert pid.is_discriminating(("ok", "1"), ("ok", "1")) is False
    assert pid.is_discriminating(("err",), ("ok", "2")) is False   # ref must run
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/test_probe_input_discrimination.py -q` — Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/probe_input_discrimination.py
"""Unit 0: can v1 synthesize inputs that SEPARATE right code from wrong code?

Validity criterion is DISCRIMINATION (spec finding I1), not executability: an input
counts iff the reference runs it cleanly AND its output differs from a known-wrong
solution's. Synthesis prompts carry description+signature only — never gold asserts.

    python scripts/probe_input_discrimination.py --n-tasks 100 \\
        --out evals/harness/probe_discrimination.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_input_exprs(reply: str, entry_point: str) -> list[str]:
    """Call-expressions of entry_point found in the reply: deduped, first 3."""
    pat = re.compile(re.escape(entry_point) + r"\s*\([^\n]*\)")
    out: list[str] = []
    for m in pat.finditer(reply):
        expr = m.group(0)
        if expr not in out:
            out.append(expr)
        if len(out) == 3:
            break
    return out


_MUTATIONS = [("<=", ">="), ("<", "<="), (">", ">="), ("==", "!="), ("+", "-")]


def make_mutant(code: str) -> str | None:
    """First mechanical mutation that changes the text (labeled fallback wrong-solution
    when no failed bank candidate exists for a task); None if nothing mutates."""
    for old, new in _MUTATIONS:
        if old in code:
            mutated = code.replace(old, new, 1)
            if mutated != code:
                return mutated
    return None


def is_discriminating(sig_ref: tuple, sig_wrong: tuple) -> bool:
    """Reference must RUN on the input; discrimination = the wrong solution behaves
    differently (different output, error, or timeout)."""
    return sig_ref[0] == "ok" and sig_ref != sig_wrong


def main() -> None:  # pragma: no cover - GPU + sandbox operational
    import torch
    from datasets import load_dataset

    from microlab.evals.code.tasks import load_humaneval
    from microlab.infer.behavior import behavior_signature
    from microlab.model.reference.checkpoint import load_variant_from_run
    from microlab.model.reference.sft import format_chat
    from microlab.tokenizer.fast import FastTokenizer
    from microlab.train.exec_reward import extract_solution, sample_solutions

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="runs/coder-1b-instruct-compliant")
    ap.add_argument("--out", default="evals/harness/probe_discrimination.jsonl")
    ap.add_argument("--n-tasks", type=int, default=100, help="split evenly HE/MBPP")
    ap.add_argument("--wrong-bank", default="evals/instruct/coder-1b-instruct-v2-humaneval-sampled.jsonl",
                    help="existing bank searched for a FAILED candidate per HumanEval task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=3.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    half = args.n_tasks // 2
    mbpp_rows = list(load_dataset("google-research-datasets/mbpp", "sanitized",
                                  split="test"))[:half]
    he_rows = list(load_dataset("openai/openai_humaneval", split="test"))[:half]

    # wrong-candidate lookup from an existing bank (failed samples), HumanEval only
    wrong_by_task: dict[str, str] = {}
    wb = Path(args.wrong_bank)
    if wb.exists():
        for line in wb.read_text().splitlines():
            d = json.loads(line)
            if "task_id" in d and not d.get("passed") and d.get("solution", "").strip():
                wrong_by_task.setdefault(d["task_id"], d["solution"])

    # probe entries: (task_id, description, signature, entry_point, reference, wrong, src)
    probes = []
    for row in he_rows:
        ref = row["prompt"] + row["canonical_solution"]
        wrong = wrong_by_task.get(row["task_id"]) or make_mutant(ref)
        if wrong is None:
            continue
        src = "bank" if row["task_id"] in wrong_by_task else "mutant"
        sig = f"def {row['entry_point']}(...)"
        probes.append((row["task_id"], row["prompt"], sig, row["entry_point"],
                       ref, wrong, src))
    for row in mbpp_rows:
        code = row.get("code", "")
        sigline = next((ln.strip() for ln in code.splitlines()
                        if ln.strip().startswith("def ") and ln.rstrip().endswith(":")),
                       None)
        if sigline is None:
            continue
        entry = sigline.split("def ", 1)[1].split("(", 1)[0].strip()
        wrong = make_mutant(code)
        if wrong is None:
            continue
        probes.append((f"Mbpp/{row['task_id']}", row["prompt"], sigline, entry,
                       code, wrong, "mutant"))
\n    model, _ = load_variant_from_run(Path(args.policy), device=args.device)
    tok = FastTokenizer.load(str(Path(args.policy) / "tokenizer.json"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["task_id"] for x in out.read_text().splitlines()} \
        if out.exists() else set()

    ok_tasks = total = 0
    with out.open("a", encoding="utf-8") as f:
        for task_id, desc, sig, entry, ref, wrong, src in probes:
            if task_id in done:
                continue
            total += 1
            instr = (f"{desc.strip()}\n\nGive 3 example calls to `{entry}` (one per "
                     f"line, plain expressions like `{entry}(...)` with concrete "
                     f"arguments). Signature: {sig}")
            prompt, _ = format_chat(instr, "")
            reply = sample_solutions(model, tok._tok, prompt, 1, max_new=120,
                                     temp=0.7, top_k=40, seed=args.seed,
                                     device=args.device)[0]
            exprs = parse_input_exprs(reply, entry)
            n_disc = 0
            for e in exprs:
                s_ref = behavior_signature(ref, entry, e, timeout_s=args.timeout_s)
                s_wr = behavior_signature(wrong, entry, e, timeout_s=args.timeout_s)
                if is_discriminating(s_ref, s_wr):
                    n_disc += 1
            f.write(json.dumps({"task_id": task_id, "n_inputs": len(exprs),
                                "n_discriminating": n_disc, "wrong_source": src}) + "\n")
            f.flush()
            if n_disc >= 2:
                ok_tasks += 1

    rows = [json.loads(x) for x in out.read_text().splitlines()]
    frac = sum(1 for r in rows if r["n_discriminating"] >= 2) / len(rows)
    band = ("SYNTH-OK (>=56%)" if frac >= 0.56 else
            "AMBIGUOUS (44-56%): run both regimes" if frac >= 0.44 else
            "SUBSET-ONLY (24-44%)" if frac >= 0.24 else
            "SYNTHESIS DEAD (<24%)")
    print(f"PROBE: {sum(1 for r in rows if r['n_discriminating'] >= 2)}/{len(rows)} "
          f"tasks with >=2 discriminating inputs = {frac:.2f} -> {band}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_probe_input_discrimination.py -q` — Expected: PASS (3 passed). Also `ruff check scripts/probe_input_discrimination.py` clean (the stray lines removed).

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_input_discrimination.py tests/scripts/test_probe_input_discrimination.py
git commit -m "feat(harness): input-discrimination probe (separation vs known-wrong, no gold asserts)"
```

---

### Task 5: Selection experiment runner

**Files:**
- Create: `scripts/eval_selection.py`
- Test: `tests/scripts/test_eval_selection.py`

**Interfaces:**
- Consumes: Task 1 selectors, Task 2 `signatures_for`, banks (eval_code-row schema), probe outputs, `eval_rerank.delivered`.
- Produces (pure, tested): `bank_by_task(rows: list[dict]) -> dict[str, list[dict]]` (ordered samples, header dropped, raises on empty); `run_methods(candidates: list[str], passed: list[bool], signatures: list[tuple] | None, assert_counts: list[int] | None, seed: int = 0) -> dict[str, int | None]` — per-method selected INDEX (`first`, `text_plurality`, `cluster_shortest`, `cluster_random`, `self_tests`, `oracle`), None where inputs unavailable; **raises `RuntimeError` if any non-oracle method's selected candidate passes while oracle finds none** (beat-the-oracle leakage check); `summarize(per_task: dict[str, dict[str, bool | None]]) -> dict` — per-method pass counts, coverage, and `recoverable_gap` (oracle-passes minus first-passes), with `"underpowered": recoverable_gap < 15`.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_eval_selection.py
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_selection", Path(__file__).resolve().parents[2] / "scripts" / "eval_selection.py")
es = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(es)


def test_bank_by_task_orders_and_drops_header():
    rows = [{"_header": {}},
            {"task_id": "T", "sample": 1, "passed": False, "solution": "b"},
            {"task_id": "T", "sample": 0, "passed": True, "solution": "a"}]
    got = es.bank_by_task(rows)
    assert [r["sample"] for r in got["T"]] == [0, 1]
    with pytest.raises(ValueError):
        es.bank_by_task([{"_header": {}}])


def test_run_methods_selects_and_flags_oracle():
    cands = ["def f(): return 1", "def f(): return 2", "def f(): return 1"]
    passed = [False, True, False]
    sigs = [("ok", "1"), ("ok", "2"), ("ok", "1")]
    got = es.run_methods(cands, passed, sigs, None, seed=0)
    assert got["first"] == 0 and got["oracle"] == 1
    assert got["cluster_shortest"] in (0, 2)      # largest cluster = the wrong pair
    assert got["self_tests"] is None              # not provided -> None, not fabricated


def test_run_methods_raises_on_beat_the_oracle():
    # oracle sees no pass, but 'passed' claims the plurality pick passes -> leakage bug
    cands = ["a", "a"]
    with pytest.raises(RuntimeError):
        es.run_methods(cands, [True, False], None, None, oracle_override=[False, False])


def test_summarize_power_gate():
    per_task = {f"t{i}": {"first": False, "oracle": True} for i in range(10)}
    rep = es.summarize(per_task)
    assert rep["recoverable_gap"] == 10 and rep["underpowered"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/test_eval_selection.py -q` — Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/eval_selection.py
"""Unit 2 runner: no-tests selection methods vs the oracle, per bank.

Selection is blind to hidden tests; hidden tests only SCORE the selected candidate.
The beat-the-oracle check is a leakage falsifier IN CODE: no method may pass where the
oracle (any-of-k on hidden tests) finds no passing candidate.

    python scripts/eval_selection.py --bank evals/harness/mbpp-testfree-bank.jsonl \\
        --inputs evals/harness/synth_inputs.jsonl --dataset mbpp \\
        --out evals/harness/selection-mbpp-testfree.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.infer.selection import (  # noqa: E402
    behavior_clusters, first_sample, pick_from_cluster, select_by_self_tests,
    text_plurality,
)


def bank_by_task(rows: list[dict]) -> dict[str, list[dict]]:
    """Group bank rows by task, ordered by sample index. Raises on empty (verify by count)."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        if "_header" in r:
            continue
        by.setdefault(r["task_id"], []).append(r)
    if not by:
        raise ValueError("no task rows in bank — wrong file? (verify by count)")
    for v in by.values():
        v.sort(key=lambda r: r["sample"])
    return by


def run_methods(candidates: list[str], passed: list[bool],
                signatures: list[tuple] | None, assert_counts: list[int] | None,
                seed: int = 0, oracle_override: list[bool] | None = None) -> dict:
    """Selected index per method; None where the method's inputs are unavailable.
    oracle_override exists only for the leakage self-test."""
    oracle_passed = oracle_override if oracle_override is not None else passed
    out: dict[str, int | None] = {"first": first_sample(),
                                  "text_plurality": text_plurality(candidates)}
    if signatures is not None:
        clusters = behavior_clusters(signatures)
        out["cluster_shortest"] = pick_from_cluster(clusters[0], candidates, "shortest")
        out["cluster_random"] = pick_from_cluster(clusters[0], candidates, "random", seed)
    else:
        out["cluster_shortest"] = out["cluster_random"] = None
    out["self_tests"] = (select_by_self_tests(assert_counts)
                         if assert_counts is not None else None)
    out["oracle"] = next((i for i, p in enumerate(oracle_passed) if p), None)
    if out["oracle"] is None:
        for m, idx in out.items():
            if m != "oracle" and idx is not None and passed[idx]:
                raise RuntimeError(
                    f"method {m!r} selected a passing candidate where the oracle found "
                    f"none — hidden tests leaked into selection (falsifier)")
    return out


def summarize(per_task: dict[str, dict[str, bool | None]]) -> dict:
    """per_task: task -> method -> did-the-selected-candidate-pass (None = no coverage).
    Reports counts, coverage, the recoverable gap, and the power verdict."""
    methods = sorted({m for d in per_task.values() for m in d})
    rep: dict = {"n_tasks": len(per_task), "methods": {}}
    for m in methods:
        vals = [d.get(m) for d in per_task.values()]
        rep["methods"][m] = {"passes": sum(1 for v in vals if v),
                             "coverage": sum(1 for v in vals if v is not None)}
    first_p = rep["methods"].get("first", {}).get("passes", 0)
    oracle_p = rep["methods"].get("oracle", {}).get("passes", 0)
    rep["recoverable_gap"] = oracle_p - first_p
    rep["underpowered"] = rep["recoverable_gap"] < 15
    return rep


def main() -> None:  # pragma: no cover - sandbox operational
    from microlab.infer.behavior import signatures_for

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--inputs", default=None,
                    help="jsonl {task_id, entry_point, exprs:[...]} of SYNTHESIZED inputs")
    ap.add_argument("--docstring-inputs", default=None,
                    help="jsonl of docstring-extracted inputs — reported SEPARATELY")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=3.0)
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.bank).read_text().splitlines()]
    banks = bank_by_task(rows)

    def load_inputs(path):
        if not path or not Path(path).exists():
            return {}
        return {json.loads(x)["task_id"]: json.loads(x)
                for x in Path(path).read_text().splitlines()}

    input_sets = {"synth": load_inputs(args.inputs),
                  "docstring": load_inputs(args.docstring_inputs)}
    report: dict = {}
    for src, inputs in input_sets.items():
        if not inputs and src == "docstring":
            continue
        per_task: dict[str, dict] = {}
        for task_id, samples in banks.items():
            cands = [s["solution"] for s in samples]
            passed = [bool(s["passed"]) for s in samples]
            sigs = None
            meta = inputs.get(task_id)
            if meta and meta.get("exprs"):
                sigs = [signatures_for(c, meta["entry_point"], meta["exprs"][:4],
                                       timeout_s=args.timeout_s) for c in cands]
            sel = run_methods(cands, passed, sigs, None, seed=args.seed)
            per_task[task_id] = {m: (passed[i] if i is not None else None)
                                 for m, i in sel.items()}
        report[src] = summarize(per_task)
        print(f"[{src}] {json.dumps(report[src], indent=1)}")

    if not report:
        raise SystemExit("no input source produced a report (verify by count)")
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_eval_selection.py -q` — Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_selection.py tests/scripts/test_eval_selection.py
git commit -m "feat(harness): selection experiment runner (leakage check in code, power gate)"
```

---

### Task 6: `best_of` CLI (deliverable 3a — tests-required)

**Files:**
- Create: `scripts/generate_best_of.py`
- Test: `tests/scripts/test_generate_best_of.py`

**Interfaces:**
- Consumes: `sample_solutions`, `extract_solution`, `run_python`, `format_chat`.
- Produces (pure, tested): `pick_best(solutions: list[str], results: list[bool]) -> int | None` — first passing index, None if none pass; (script) CLI: `--instruction "..." --asserts-file tests.py --k 8` → samples k, runs each against the caller-provided asserts (solution + file contents concatenated, exit 0 = pass), prints the first passer (or the first sample with a NO-PASS warning to stderr and exit code 3 — the caller must be able to TELL).

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_generate_best_of.py
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "generate_best_of", Path(__file__).resolve().parents[2] / "scripts" / "generate_best_of.py")
gbo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gbo)


def test_pick_best_first_passer_or_none():
    assert gbo.pick_best(["a", "b", "c"], [False, True, True]) == 1
    assert gbo.pick_best(["a"], [False]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/test_generate_best_of.py -q` — Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/generate_best_of.py
"""Best-of-k with caller-provided tests (harness deliverable 3a, tests-required).

    python scripts/generate_best_of.py --instruction "Write add(a,b)" \\
        --asserts-file my_asserts.py --k 8

Samples k candidates from the frozen v1, executes each against the asserts, prints the
first passer. Exit 0 = a passing solution was printed; exit 3 = NONE passed (the first
sample is printed anyway, with a stderr warning) — callers must check the exit code.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pick_best(solutions: list[str], results: list[bool]) -> int | None:
    """First index whose result is True; None when nothing passed."""
    for i, ok in enumerate(results):
        if ok:
            return i
    return None


def main() -> int:  # pragma: no cover - GPU + sandbox operational
    import torch

    from microlab.evals.code.executor import run_python
    from microlab.model.reference.checkpoint import load_variant_from_run
    from microlab.model.reference.sft import format_chat
    from microlab.tokenizer.fast import FastTokenizer
    from microlab.train.exec_reward import extract_solution, sample_solutions

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="runs/coder-1b-instruct-compliant")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--asserts-file", required=True, type=Path,
                    help="python file of asserts run AFTER the candidate definition")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=8.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    asserts = args.asserts_file.read_text()
    model, _ = load_variant_from_run(Path(args.policy), device=args.device)
    tok = FastTokenizer.load(str(Path(args.policy) / "tokenizer.json"))
    prompt, _ = format_chat(args.instruction, "")
    replies = sample_solutions(model, tok._tok, prompt, args.k, max_new=args.max_new,
                               temp=args.temp, top_k=args.top_k, seed=args.seed,
                               device=args.device)
    sols = [extract_solution(r) for r in replies]
    results = [bool(s.strip()) and run_python(s + "\n\n" + asserts,
                                              timeout_s=args.timeout_s).passed
               for s in sols]
    best = pick_best(sols, results)
    if best is None:
        print(sols[0])
        print(f"WARNING: none of {args.k} candidates passed the provided asserts",
              file=sys.stderr)
        return 3
    print(sols[best])
    print(f"# passed provided asserts (candidate {best + 1}/{args.k})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_generate_best_of.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_best_of.py tests/scripts/test_generate_best_of.py
git commit -m "feat(harness): best-of-k CLI with caller asserts (3a, exit-code honest)"
```

---

### Task 7: Pre-registered prediction doc

**Files:**
- Create: `docs/coder-1b-harness-prediction.md`

No code test; the gate is commit-before-Task-8. Contents (exact numbers, house style):
v1 HumanEval pass@10 within **±4 pts of 6.1%** (v2 proxy; SE of the difference 2.6 pts);
v1 MBPP standard-bank pass@10 = **2–3× the same run's pass@1**; test-free MBPP pass@1
**20–60% relative BELOW** the standard bank's (test-conditioning); probe discrimination
fraction unknown-wildcard with soft bands 24/44/56; clustering (synthesized inputs,
test-free MBPP) recovers **10–50%** of that bank's oracle−floor gap; text-plurality <10%;
`cluster_random` within noise of `cluster_shortest` (ablation, no directional claim).
Falsifiers verbatim from the spec: power (<15-task gap → counts only), selector-leak
(beat-the-oracle → bug), generator-leak (no-tests ≈ oracle on assert-embedded prompts →
contamination), probe false-positive (gate passed but singleton clusters everywhere),
clustering ≤ text-plurality (execution adds nothing → ship tests-required), all-agree-wrong
domination (caps k-scaling). Caveat: HumanEval descriptive only.

- [ ] **Step 1: Write the doc per above.**
- [ ] **Step 2: Commit**

```bash
git add docs/coder-1b-harness-prediction.md
git commit -m "docs: pre-registered harness prediction (before any measurement)"
```

---

### Task 8: Operational run

**Files:** none (banks, probe outputs, selection tables, milestone doc). Sequenced.

- [ ] **Step 1: Standard banks (Unit 1).** `eval_code.py --run runs/coder-1b-instruct-compliant --mode chat --n 10 --temperature 0.7 --top-k 40` for humaneval AND mbpp → `evals/harness/` (~2.5h GPU). Then `eval_rerank.py` on both → v1 true pass@1/pass@10 + delivered table. Score the Task-7 bands that are now measurable.
- [ ] **Step 2: Test-free MBPP bank.** `gen_testfree_bank.py --n 10` (~45 min). Report the test-conditioning delta (standard vs test-free pass@1/@10). Verify by count (rows = tasks×10 + header, minus skipped-signature count).
- [ ] **Step 3: Probe (Unit 0).** `probe_input_discrimination.py --n-tasks 100` (~45 min; smoke `--n-tasks 10` first). Print gate band; record. Also WRITE the synthesized-inputs file for Unit 2 (`{task_id, entry_point, exprs}` for probed+new tasks as the gate allows) and the docstring-extracted inputs file for HumanEval (a small extraction pass over prompts, `>>> entry(...)` regex — same parser as the probe's).
- [ ] **Step 4: Selection experiment (Unit 2).** Per the gate band: `eval_selection.py` on the test-free MBPP bank (powered) and the HumanEval bank (descriptive), synthesized and docstring input files passed separately. Check `underpowered` and the falsifier outputs; error analysis (all-agree-wrong vs fragmentation) via a short notebook-style pass over the largest-cluster compositions in the report JSONs.
- [ ] **Step 5: 3a smoke.** `generate_best_of.py` on 2 hand-written instruction+asserts pairs; verify exit codes 0 and 3 behave as documented.
- [ ] **Step 6: Milestone.** `docs/coder-1b-harness-milestone.md`: delivered-correctness table, test-conditioning delta, probe verdict, selection table (sources separate), every Task-7 band scored, ship/no-ship for 3b. Commit + merge per house flow (final whole-branch review first).

---

## Self-Review (completed during planning)

- **Spec coverage:** Unit 0 → Task 4; Unit 1 → Task 8 steps 1–2 (+ Task 3 builder); Unit 2 → Tasks 1, 2, 5 + step 4; 3a → Task 6 (+ step 5); 3b explicitly ABSENT (gated); prediction → Task 7 before Task 8; C1 (test-free bank) → Task 3; I1 (discrimination) → Task 4; I2 (never pooled) → Task 5's separate input files + report keys; I4 (plurality as floor) → Task 1/5; power falsifier → `summarize.underpowered`; selector-leak falsifier → `run_methods` RuntimeError.
- **Placeholder scan:** Task 4's Step-3 sketch contains two explicitly flagged transcription-guard lines with DELETE instructions — intentional and called out, not silent placeholders; all other code steps complete.
- **Type consistency:** signature tuples `("ok", s)|("err",)|("timeout",)` shared Tasks 2/4/5; bank row schema shared Tasks 3/5 with `eval_code.py`; `exprs` input-file schema shared Tasks 4(step-3 writer in Task 8)/5; selector names (`first`, `text_plurality`, `cluster_shortest`, `cluster_random`, `self_tests`, `oracle`) consistent across Tasks 1/5.
