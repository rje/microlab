# coder-1b-instruct + distill-cost A/B — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a compliant (no-distill) code-instruction SFT set, instruction-tune the coder-1b base into `coder-1b-instruct`, and measure the build-capability rule's cost against a token-matched distilled arm.

**Architecture:** Pure, unit-testable data helpers in `src/microlab/data/code_sft.py`; two builder CLIs (compliant arm A, distilled arm B) that share those helpers plus one decontamination pass; an eval+compare harness over two already-trained runs. Training uses the existing (now hybrid-capable) `scripts/sft.py`; nothing in the training loop changes.

**Tech Stack:** Python, PyTorch, HuggingFace `datasets` (streaming), the existing `microlab` package (executor sandbox, tokenizer, SFT/eval scripts).

## Global Constraints

- **Build capability, don't distill.** Arm A rows are human-authored or executor-verified-human only. External models appear only as the pairwise *judge*, never as training text. Arm B intentionally violates this and is a disposable measurement instrument: never merged, never seeds later training.
- **No silent truncation / no silent caps.** Overlong rows are dropped and counted, never tail-clipped (`collate_sft` right-truncates and would eat the `### End` sentinel). Every per-source cap is logged.
- **Identical treatment across arms.** Same base `runs/coder-1b-step40000`, same hyperparameters (3 epochs, LR 2e-5 cosine ~5% warmup / 10% min, bf16, block 2048, same effective batch/seed), same decontamination, same eval battery. Only `--data`/`--out` differ.
- **Verify by count.** Builders assert and print per-source row counts AND supervised-token counts; a zero or wildly-off count fails loudly.
- **Row schema** (consumed by `sft.py`): single-turn `{"instruction": str, "context": str, "response": str}` or multi-turn `{"turns": [{"user": str, "assistant": str, "context"?: str}, ...]}`.
- **Language scope:** Python-first (matches the Python-dominant pretraining code and the Python-only benchmarks); JS/TS/shell/SQL deferred to a later pass.
- Servable-run convention: each SFT run dir carries `ckpt_*.pt` + `tokenizer.json` + `serve_config.json` (`sft.py` writes these).

## Interfaces this plan builds on (already in the repo, verified)

- `microlab.evals.code.executor.run_python(code: str, *, timeout_s=10.0, memory_mb=512, require_netns=False, max_output_bytes=...) -> ExecResult`; `ExecResult.passed: bool` (`exit_code == 0 and not timed_out`), `.exit_code`, `.stdout`, `.stderr`. **Runs `code` as a standalone program with `stdin=DEVNULL`** — no stdin.
- `microlab.evals.code.tasks.CodeTask(task_id, prompt, instruction, entry_point, test_program)`; `assemble_program(solution: str, task: CodeTask) -> str` (unit-test style: solution + test suffix, exit 0 iff `check()` passes); `humaneval_task(row)`, `mbpp_task(row)`, `load_humaneval() -> list[CodeTask]`, `load_mbpp() -> list[CodeTask]` (sanitized **test** split).
- `microlab.model.reference.sft.format_chat(instruction, context="", response="") -> (prompt, response)`, `IGNORE_INDEX`, `build_sft_example`.
- `microlab.model.reference.chat_sft.END_SENTINEL == "\n### End"`, `build_chat_example(tok, turns, block_size)`, `TurnTooLongError`.
- `microlab.tokenizer.fast.FastTokenizer.load(path) -> FastTokenizer`; `.encode(text) -> list[int]`.
- `scripts/build_chat_mix.py.extract_oasst_conversations(messages, max_turns=6, all_assistant_children=False) -> list[Conv]` where `Conv = {"turns": list[{"user","assistant"}]}` (reuse; import via importlib as its tests do).
- `scripts/build_sft_mix.py`: pattern to mirror — pure `normalize_*(row) -> Row|None`, `_load_hf(dataset, split, normalize, limit)`, `write_jsonl(rows, out) -> int`.
- `scripts/sft.py`: `--base-ckpt --data --out --tokenizer --epochs --lr --batch-size --block-size --grad-accum --save-every --limit --device`.
- `scripts/eval_code.py --run <dir> --dataset {humaneval,mbpp} --mode chat --out <jsonl> --n --temperature --top-k --max-new` → writes `<out>` + `<out>.summary.json` with `pass@1`.
- `scripts/eval_pairwise.py run_a run_b --data <jsonl> --skip --limit --out` → position-swapped codex judge, writes win-rate JSON.
- `scripts/eval_suite.py` computes FIM middle-loss (guardrail) given a staged run dir + `MICROLAB_MIX_DIR`.

---

### Task 1: Shared module + CommitPackFT / MBPP-train normalizers

**Files:**
- Create: `src/microlab/data/code_sft.py`
- Test: `tests/data/test_code_sft.py`

**Interfaces:**
- Produces: `Row = dict[str, str]`; `normalize_commitpack(row: dict, lang_allow: set[str] | None) -> Row | None`; `normalize_mbpp_train(row: dict) -> Row | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_code_sft.py
from microlab.data.code_sft import normalize_commitpack, normalize_mbpp_train


def test_normalize_commitpack_message_to_new_contents():
    row = {"subject": "Fix off-by-one in range", "message": "Fix off-by-one in range\n",
           "new_contents": "for i in range(n):\n    pass\n", "old_contents": "...",
           "lang": "Python"}
    got = normalize_commitpack(row, lang_allow={"python"})
    assert got == {"instruction": "Fix off-by-one in range",
                   "context": "", "response": "for i in range(n):\n    pass\n"}


def test_normalize_commitpack_drops_disallowed_language():
    row = {"subject": "x", "message": "x", "new_contents": "console.log(1)", "lang": "JavaScript"}
    assert normalize_commitpack(row, lang_allow={"python"}) is None


def test_normalize_commitpack_drops_empty_message_or_body():
    assert normalize_commitpack({"message": "", "new_contents": "x", "lang": "Python"},
                                lang_allow={"python"}) is None
    assert normalize_commitpack({"message": "do", "new_contents": "  ", "lang": "Python"},
                                lang_allow={"python"}) is None


def test_normalize_mbpp_train_prompt_to_code():
    row = {"task_id": 601, "text": "Write a function to add two numbers.",
           "code": "def add(a, b):\n    return a + b", "test_list": ["assert add(1,2)==3"]}
    got = normalize_mbpp_train(row)
    assert got["instruction"] == "Write a function to add two numbers."
    assert got["response"] == "def add(a, b):\n    return a + b"
    assert got["context"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_code_sft.py -q`
Expected: FAIL (`ModuleNotFoundError: microlab.data.code_sft`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/microlab/data/code_sft.py
"""Pure, network-free helpers for building the coder-1b code-instruction SFT mixes.

Every function here is deterministic and importable without torch or a GPU so the builders
in scripts/ stay thin and the logic is unit-tested off-network — the same split
build_sft_mix.py uses. Row is the single-turn schema scripts/sft.py consumes.
"""
from __future__ import annotations

Row = dict[str, str]


def normalize_commitpack(row: dict, lang_allow: set[str] | None) -> Row | None:
    """CommitPackFT row -> {instruction=commit message, response=new file contents}.

    The commit message is the instruction; the post-commit file is the target. `lang_allow`
    (lowercased language names) gates languages — Python-first for this run. Returns None for
    a disallowed language or an empty message/body.
    """
    lang = (row.get("lang") or "").strip().lower()
    if lang_allow is not None and lang not in lang_allow:
        return None
    # CommitPackFT uses `message`; `subject` is the first line. Prefer the subject as the
    # instruction (concise), fall back to the full message.
    instruction = (row.get("subject") or row.get("message") or "").strip()
    response = (row.get("new_contents") or "").strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "context": "", "response": response}


def normalize_mbpp_train(row: dict) -> Row | None:
    """MBPP (sanitized) train/validation/prompt row -> {instruction=text, response=code}.

    NOT the test split (that is the eval set). Returns None if text or code is empty.
    """
    instruction = (row.get("text") or row.get("prompt") or "").strip()
    response = (row.get("code") or "").strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "context": "", "response": response}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_code_sft.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/microlab/data/code_sft.py tests/data/test_code_sft.py
git commit -m "feat(code-sft): CommitPackFT + MBPP-train normalizers"
```

---

### Task 2: OASST code-thread extraction

**Files:**
- Modify: `src/microlab/data/code_sft.py`
- Test: `tests/data/test_code_sft.py`

**Interfaces:**
- Consumes: `extract_oasst_conversations` from `scripts/build_chat_mix.py`.
- Produces: `is_code_conv(conv: dict) -> bool`; `oasst_code_convs(messages: list[dict], max_turns: int = 6) -> list[dict]` returning multi-turn `{"turns": [...]}` rows whose assistant turns contain code.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/data/test_code_sft.py
from microlab.data.code_sft import is_code_conv, oasst_code_convs


def test_is_code_conv_true_when_assistant_has_fenced_code():
    conv = {"turns": [{"user": "sort a list in python",
                       "assistant": "Use sorted:\n```python\nsorted(xs)\n```"}]}
    assert is_code_conv(conv) is True


def test_is_code_conv_false_for_pure_prose():
    conv = {"turns": [{"user": "hi", "assistant": "Hello, how are you?"}]}
    assert is_code_conv(conv) is False


def test_oasst_code_convs_keeps_only_code_threads():
    # two roots: one code, one prose; only the code one survives
    messages = [
        {"message_id": "a", "parent_id": None, "role": "prompter", "text": "write python",
         "lang": "en"},
        {"message_id": "b", "parent_id": "a", "role": "assistant",
         "text": "```python\nprint(1)\n```", "lang": "en", "rank": 0},
        {"message_id": "c", "parent_id": None, "role": "prompter", "text": "hello", "lang": "en"},
        {"message_id": "d", "parent_id": "c", "role": "assistant", "text": "hi there",
         "lang": "en", "rank": 0},
    ]
    convs = oasst_code_convs(messages)
    assert len(convs) == 1
    assert "print(1)" in convs[0]["turns"][0]["assistant"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_code_sft.py -k oasst -q` and `-k is_code_conv`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/microlab/data/code_sft.py
import importlib.util as _ilu
import re as _re
from pathlib import Path as _Path

# Reuse the OASST tree-walker from the chat-mix builder (single source of truth for the
# rank-0-child linearization); scripts/ isn't a package so load it by path.
_bcm_spec = _ilu.spec_from_file_location(
    "build_chat_mix", _Path(__file__).resolve().parents[3] / "scripts" / "build_chat_mix.py")
_bcm = _ilu.module_from_spec(_bcm_spec)
_bcm_spec.loader.exec_module(_bcm)

_CODE_FENCE = _re.compile(r"```")


def is_code_conv(conv: dict) -> bool:
    """True if any assistant turn contains a fenced code block (```). The cheap, precise
    signal that a thread is about code without language-classifying every message."""
    return any(_CODE_FENCE.search(t.get("assistant", "")) for t in conv.get("turns", []))


def oasst_code_convs(messages: list[dict], max_turns: int = 6) -> list[dict]:
    """Linearize OASST trees (best-ranked child) and keep only code-bearing conversations."""
    convs = _bcm.extract_oasst_conversations(messages, max_turns=max_turns)
    return [c for c in convs if is_code_conv(c)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_code_sft.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/microlab/data/code_sft.py tests/data/test_code_sft.py
git commit -m "feat(code-sft): OASST code-thread extraction (reuses chat-mix walker)"
```

---

### Task 3: Executor verification (unit-test AND stdin/stdout problems)

**Files:**
- Modify: `src/microlab/data/code_sft.py`
- Test: `tests/data/test_code_sft.py`

**Interfaces:**
- Consumes: `run_python`, `assemble_program`, `CodeTask` from `microlab.evals.code`.
- Produces: `assemble_io_program(solution: str, stdin_data: str, expected_stdout: str) -> str`; `verify_unit_test(solution: str, task: CodeTask) -> bool`; `verify_io(solution: str, stdin_data: str, expected_stdout: str, timeout_s: float = 10.0) -> bool`.

Rationale: APPS/CodeContests/TACO are **stdin→stdout** problems and `run_python` provides no stdin, so `assemble_program` (unit-test style) does not apply. `assemble_io_program` builds a self-contained program that feeds the input via a patched `sys.stdin`, captures stdout, and exits non-zero unless it matches — the I/O analogue of `assemble_program`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/data/test_code_sft.py
import pytest
from microlab.data.code_sft import verify_io, verify_unit_test
from microlab.evals.code.tasks import CodeTask


def test_verify_io_accepts_correct_and_rejects_wrong():
    sol = "n = int(input())\nprint(n * 2)\n"
    assert verify_io(sol, stdin_data="21\n", expected_stdout="42\n") is True
    assert verify_io("print('nope')\n", stdin_data="21\n", expected_stdout="42\n") is False


def test_verify_io_rejects_infinite_loop_via_timeout():
    assert verify_io("while True:\n    pass\n", stdin_data="", expected_stdout="x\n",
                     timeout_s=2.0) is False


def test_verify_unit_test_accepts_correct_solution():
    task = CodeTask(task_id="t", prompt="", instruction="add",
                    entry_point="add",
                    test_program="def check(add):\n    assert add(1, 2) == 3\ncheck(add)\n")
    assert verify_unit_test("def add(a, b):\n    return a + b", task) is True
    assert verify_unit_test("def add(a, b):\n    return a - b", task) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_code_sft.py -k verify -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/microlab/data/code_sft.py
from microlab.evals.code.executor import run_python
from microlab.evals.code.tasks import CodeTask, assemble_program

_IO_HARNESS = '''\
import sys, io
sys.stdin = io.StringIO({stdin!r})
_out = io.StringIO()
_real = sys.stdout
sys.stdout = _out
{solution}
sys.stdout = _real
_got = _out.getvalue()
_want = {expected!r}
# Compare with trailing-whitespace tolerance per line (competitive judges are lenient here).
def _norm(s): return "\\n".join(line.rstrip() for line in s.rstrip("\\n").split("\\n"))
sys.exit(0 if _norm(_got) == _norm(_want) else 1)
'''


def assemble_io_program(solution: str, stdin_data: str, expected_stdout: str) -> str:
    """Wrap a stdin->stdout solution into a self-contained program that exits 0 iff its
    output matches `expected_stdout` (line-rstrip tolerant). The I/O analogue of
    assemble_program, needed because run_python gives the child no stdin."""
    return _IO_HARNESS.format(stdin=stdin_data, solution=solution, expected=expected_stdout)


def verify_io(solution: str, stdin_data: str, expected_stdout: str,
              timeout_s: float = 10.0) -> bool:
    """True iff `solution` reproduces `expected_stdout` for `stdin_data` in the sandbox."""
    prog = assemble_io_program(solution, stdin_data, expected_stdout)
    return run_python(prog, timeout_s=timeout_s).passed


def verify_unit_test(solution: str, task: CodeTask) -> bool:
    """True iff `solution` passes `task`'s unit-test suffix (HumanEval/MBPP style)."""
    return run_python(assemble_program(solution, task)).passed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_code_sft.py -k verify -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/microlab/data/code_sft.py tests/data/test_code_sft.py
git commit -m "feat(code-sft): executor verification for unit-test and stdin/stdout problems"
```

---

### Task 4: APPS/CodeContests/TACO → executor-verified rows

**Files:**
- Modify: `src/microlab/data/code_sft.py`
- Test: `tests/data/test_code_sft.py`

**Interfaces:**
- Consumes: `verify_io`, `Row`.
- Produces: `verified_competitive_rows(problems: list[dict], max_per_problem: int = 1, timeout_s: float = 10.0) -> tuple[list[Row], dict]`. Each `problem` is normalized to `{"statement": str, "solutions": list[str], "io": list[{"input": str, "output": str}]}` by a per-dataset adapter (`apps_problem(row)`, `codecontests_problem(row)`, `taco_problem(row)`), also added here. Returns `(rows, tally)` where `tally = {"problems": n, "verified": k, "no_passing_solution": m}`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/data/test_code_sft.py
from microlab.data.code_sft import verified_competitive_rows


def test_verified_competitive_rows_keeps_only_passing_solution():
    problems = [{
        "statement": "Read n, print n*2.",
        "solutions": ["n=int(input());print(n-1)",      # wrong
                      "n=int(input());print(n*2)"],       # correct
        "io": [{"input": "21\n", "output": "42\n"}],
    }]
    rows, tally = verified_competitive_rows(problems, max_per_problem=1)
    assert tally == {"problems": 1, "verified": 1, "no_passing_solution": 0}
    assert len(rows) == 1
    assert rows[0]["instruction"] == "Read n, print n*2."
    assert rows[0]["response"] == "n=int(input());print(n*2)"


def test_verified_competitive_rows_drops_problem_with_no_passing_solution():
    problems = [{"statement": "s", "solutions": ["print('x')"],
                 "io": [{"input": "", "output": "y\n"}]}]
    rows, tally = verified_competitive_rows(problems)
    assert rows == []
    assert tally["no_passing_solution"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_code_sft.py -k competitive -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/microlab/data/code_sft.py
import json as _json


def verified_competitive_rows(problems: list[dict], max_per_problem: int = 1,
                              timeout_s: float = 10.0) -> tuple[list[Row], dict]:
    """For each normalized problem, keep up to `max_per_problem` human solutions that pass
    ALL its I/O cases in the sandbox. instruction=statement, response=verified solution.
    Solutions are tried shortest-first (concise correct code is the better demonstration)."""
    rows: list[Row] = []
    tally = {"problems": 0, "verified": 0, "no_passing_solution": 0}
    for p in problems:
        tally["problems"] += 1
        statement = (p.get("statement") or "").strip()
        cases = p.get("io") or []
        if not statement or not cases:
            tally["no_passing_solution"] += 1
            continue
        kept = 0
        for sol in sorted(p.get("solutions") or [], key=len):
            if all(verify_io(sol, c["input"], c["output"], timeout_s=timeout_s) for c in cases):
                rows.append({"instruction": statement, "context": "", "response": sol.strip()})
                kept += 1
                if kept >= max_per_problem:
                    break
        if kept == 0:
            tally["no_passing_solution"] += 1
        else:
            tally["verified"] += 1
    return rows, tally


def apps_problem(row: dict) -> dict:
    """codeparrot/apps row -> normalized problem. `solutions` and `input_output` are
    JSON-encoded strings; input_output has parallel `inputs`/`outputs` lists."""
    io = _json.loads(row["input_output"]) if row.get("input_output") else {"inputs": [], "outputs": []}
    cases = [{"input": i if isinstance(i, str) else "".join(i),
              "output": o if isinstance(o, str) else "".join(o)}
             for i, o in zip(io.get("inputs", []), io.get("outputs", []))]
    sols = _json.loads(row["solutions"]) if row.get("solutions") else []
    return {"statement": row.get("question", ""), "solutions": sols, "io": cases}


def codecontests_problem(row: dict) -> dict:
    """deepmind/code_contests row -> normalized problem. Python solutions only (language enum
    1==PYTHON, 3==PYTHON3 in the dataset); public+private tests as I/O cases."""
    sols = []
    sol_field = row.get("solutions") or {}
    for lang, txt in zip(sol_field.get("language", []), sol_field.get("solution", [])):
        if lang in (1, 3):
            sols.append(txt)
    cases = []
    for group in ("public_tests", "private_tests"):
        g = row.get(group) or {}
        cases += [{"input": i, "output": o}
                  for i, o in zip(g.get("input", []), g.get("output", []))]
    return {"statement": row.get("description", ""), "solutions": sols, "io": cases}


def taco_problem(row: dict) -> dict:
    """BAAI/TACO row -> normalized problem. `solutions` is a JSON list; `input_output` is the
    same JSON-string shape as APPS."""
    sols = _json.loads(row["solutions"]) if isinstance(row.get("solutions"), str) else (row.get("solutions") or [])
    io = _json.loads(row["input_output"]) if row.get("input_output") else {"inputs": [], "outputs": []}
    cases = [{"input": i if isinstance(i, str) else "".join(i),
              "output": o if isinstance(o, str) else "".join(o)}
             for i, o in zip(io.get("inputs", []), io.get("outputs", []))]
    return {"statement": row.get("question", ""), "solutions": sols, "io": cases}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_code_sft.py -k competitive -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/microlab/data/code_sft.py tests/data/test_code_sft.py
git commit -m "feat(code-sft): executor-verified APPS/CodeContests/TACO rows"
```

---

### Task 5: Supervised-token counting + token-match subsampler

**Files:**
- Modify: `src/microlab/data/code_sft.py`
- Test: `tests/data/test_code_sft.py`

**Interfaces:**
- Consumes: `format_chat`, `END_SENTINEL`; a tokenizer with `.encode(str) -> list[int]`.
- Produces: `row_supervised_tokens(row: dict, tok) -> int`; `total_supervised_tokens(rows: list[dict], tok) -> int`; `token_match_subsample(rows: list[dict], target_tokens: int, tok, seed: int = 0) -> list[dict]`.

Supervised tokens = the tokens that contribute to the loss: for a single-turn row, `response + END_SENTINEL`; for a multi-turn row, the sum over assistant turns. This is the fair matching unit (arm A responses are far longer than Magicoder's, so row-count matching would be unfair).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/data/test_code_sft.py
from microlab.data.code_sft import (row_supervised_tokens, token_match_subsample,
                                    total_supervised_tokens)


class _ByteTok:
    def encode(self, s): return list(s.encode("utf-8"))


def test_row_supervised_tokens_counts_response_plus_sentinel():
    from microlab.model.reference.chat_sft import END_SENTINEL
    tok = _ByteTok()
    row = {"instruction": "hi", "context": "", "response": "print(1)"}
    assert row_supervised_tokens(row, tok) == len(tok.encode("print(1)" + END_SENTINEL))


def test_token_match_subsample_hits_target_within_one_row():
    tok = _ByteTok()
    rows = [{"instruction": "i", "context": "", "response": "x" * 10} for _ in range(100)]
    per = row_supervised_tokens(rows[0], tok)
    target = per * 12
    got = token_match_subsample(rows, target_tokens=target, tok=tok, seed=0)
    assert abs(total_supervised_tokens(got, tok) - target) <= per  # within one row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_code_sft.py -k supervised or token_match -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/microlab/data/code_sft.py
import random as _random

from microlab.model.reference.chat_sft import END_SENTINEL
from microlab.model.reference.sft import format_chat


def row_supervised_tokens(row: dict, tok) -> int:
    """Tokens that contribute to the SFT loss for one row (the response side + sentinel).
    Multi-turn rows sum over assistant turns; single-turn rows use response + END_SENTINEL."""
    if "turns" in row:
        return sum(len(tok.encode((t.get("assistant") or "") + END_SENTINEL))
                   for t in row["turns"])
    _, response = format_chat(row.get("instruction", ""), row.get("context", ""),
                              row.get("response", ""))
    return len(tok.encode(response + END_SENTINEL))


def total_supervised_tokens(rows: list[dict], tok) -> int:
    return sum(row_supervised_tokens(r, tok) for r in rows)


def token_match_subsample(rows: list[dict], target_tokens: int, tok, seed: int = 0) -> list[dict]:
    """Deterministically shuffle and take rows until the cumulative supervised-token count
    reaches `target_tokens` (stopping at the first row that meets or crosses it). Used to size
    the distilled arm to the compliant arm's supervised-token budget."""
    shuffled = list(rows)
    _random.Random(seed).shuffle(shuffled)
    out, acc = [], 0
    for r in shuffled:
        if acc >= target_tokens:
            break
        out.append(r)
        acc += row_supervised_tokens(r, tok)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_code_sft.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/microlab/data/code_sft.py tests/data/test_code_sft.py
git commit -m "feat(code-sft): supervised-token counting + token-match subsampler"
```

---

### Task 6: Decontamination against the eval benchmarks

**Files:**
- Modify: `src/microlab/data/code_sft.py`
- Test: `tests/data/test_code_sft.py`

**Interfaces:**
- Produces: `benchmark_fingerprints(prompts: list[str], n: int = 10) -> set[str]` (normalized n-gram set); `decontaminate(rows: list[dict], fingerprints: set[str], n: int = 10) -> tuple[list[dict], int]` returning `(kept_rows, removed_count)`. A row is removed if any of its text's n-grams collide with a benchmark n-gram.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/data/test_code_sft.py
from microlab.data.code_sft import benchmark_fingerprints, decontaminate


def test_decontaminate_removes_planted_benchmark_row_keeps_benign():
    bench = ["def has_close_elements(numbers, threshold): return any(abs(a-b) < threshold ...)"]
    fp = benchmark_fingerprints(bench, n=8)
    rows = [
        {"instruction": "impl", "context": "",
         "response": "def has_close_elements(numbers, threshold): return any(abs(a-b) < threshold ...)"},
        {"instruction": "add", "context": "", "response": "def add(a, b):\n    return a + b"},
    ]
    kept, removed = decontaminate(rows, fp, n=8)
    assert removed == 1
    assert kept == [rows[1]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_code_sft.py -k decontaminate -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/microlab/data/code_sft.py
def _norm_tokens(text: str) -> list[str]:
    """Whitespace/punctuation-insensitive word stream for n-gram matching."""
    return _re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> set[str]:
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def benchmark_fingerprints(prompts: list[str], n: int = 10) -> set[str]:
    """Normalized n-gram set over benchmark prompts + canonical solutions (pass both in)."""
    fp: set[str] = set()
    for p in prompts:
        fp |= _ngrams(_norm_tokens(p), n)
    return fp


def _row_text(row: dict) -> str:
    if "turns" in row:
        return " ".join((t.get("user", "") + " " + t.get("assistant", "")) for t in row["turns"])
    return f"{row.get('instruction', '')} {row.get('context', '')} {row.get('response', '')}"


def decontaminate(rows: list[dict], fingerprints: set[str], n: int = 10) -> tuple[list[dict], int]:
    """Drop any row sharing an n-gram with the benchmark fingerprint set. Returns
    (kept_rows, removed_count). Applied identically to both arms so it cannot bias the A/B."""
    kept, removed = [], 0
    for r in rows:
        if _ngrams(_norm_tokens(_row_text(r)), n) & fingerprints:
            removed += 1
        else:
            kept.append(r)
    return kept, removed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_code_sft.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/microlab/data/code_sft.py tests/data/test_code_sft.py
git commit -m "feat(code-sft): n-gram decontamination against HumanEval/MBPP/LiveCodeBench"
```

---

### Task 7: Arm A builder CLI (compliant mix)

**Files:**
- Create: `scripts/build_code_sft.py`
- Test: `tests/scripts/test_build_code_sft.py`

**Interfaces:**
- Consumes: everything in `microlab.data.code_sft`; `load_humaneval`, `load_mbpp` for the decontamination fingerprints.
- Produces: `build_compliant_mix(sources: dict, tok, seed: int = 0) -> tuple[list[dict], dict]` (pure over already-loaded source rows, so it is unit-testable without the network); a `main()` that streams the real sources, verifies, decontaminates, writes `data/corpora/code_sft_compliant.jsonl`, and prints the count/token/verify report.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_build_code_sft.py
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_code_sft", Path(__file__).resolve().parents[2] / "scripts" / "build_code_sft.py")
bcs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bcs)


class _ByteTok:
    def encode(self, s): return list(s.encode("utf-8"))


def test_build_compliant_mix_merges_reports_and_shuffles_deterministically():
    sources = {
        "commitpack": [{"instruction": "fix", "context": "", "response": "def f(): pass"}],
        "mbpp_train": [{"instruction": "add", "context": "", "response": "def add(a,b): return a+b"}],
        "oasst": [{"turns": [{"user": "sort", "assistant": "```python\nsorted(x)\n```"}]}],
        "competitive": [{"instruction": "n*2", "context": "", "response": "print(int(input())*2)"}],
    }
    rows, report = bcs.build_compliant_mix(sources, _ByteTok(), seed=0)
    assert len(rows) == 4
    assert report["counts"] == {"commitpack": 1, "mbpp_train": 1, "oasst": 1,
                                "competitive": 1, "total": 4}
    assert report["supervised_tokens"] > 0
    # deterministic order under a fixed seed
    rows2, _ = bcs.build_compliant_mix(sources, _ByteTok(), seed=0)
    assert rows == rows2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/test_build_code_sft.py -q`
Expected: FAIL (`build_compliant_mix` undefined).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/build_code_sft.py
"""Build the COMPLIANT (no-distill) code-instruction SFT mix for coder-1b-instruct (arm A).

Sources (all human-authored or executor-verified-human):
  - CommitPackFT (bigcode/commitpackft)         commit message -> new file   [Python-first]
  - MBPP sanitized train/validation/prompt      problem text -> reference code
  - OASST1 code threads (OpenAssistant/oasst1)   multi-turn, code-bearing
  - APPS / CodeContests / TACO                   statement -> executor-VERIFIED solution

    python scripts/build_code_sft.py --out data/corpora/code_sft_compliant.jsonl

build_compliant_mix() is pure over already-loaded rows so it is unit-tested without the
network; main() streams the real sources, verifies competitive solutions in the sandbox,
decontaminates against the eval benchmarks, and prints a count/token/verify report.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.data.code_sft import (  # noqa: E402
    apps_problem, benchmark_fingerprints, codecontests_problem, decontaminate,
    normalize_commitpack, normalize_mbpp_train, oasst_code_convs, taco_problem,
    total_supervised_tokens, verified_competitive_rows,
)

PY_LANGS = {"python"}
OUT_DEFAULT = "data/corpora/code_sft_compliant.jsonl"


def build_compliant_mix(sources: dict, tok, seed: int = 0) -> tuple[list[dict], dict]:
    """Merge already-normalized source row-lists, seed-shuffle, and report counts + supervised
    tokens. `sources` keys: commitpack, mbpp_train, oasst, competitive (each a list of rows)."""
    order = ["commitpack", "mbpp_train", "oasst", "competitive"]
    mix: list[dict] = []
    counts = {}
    for key in order:
        rows = sources.get(key, [])
        counts[key] = len(rows)
        mix += rows
    random.Random(seed).shuffle(mix)
    counts["total"] = len(mix)
    return mix, {"counts": counts, "supervised_tokens": total_supervised_tokens(mix, tok)}


def _load_sources(args, tok) -> dict:  # pragma: no cover - network/HF streaming
    from datasets import load_dataset
    lim = args.limit_per_source

    def cap(rows):
        return rows[:lim] if lim else rows

    commit = []
    for r in load_dataset("bigcode/commitpackft", "python", split="train", streaming=True):
        n = normalize_commitpack(r, PY_LANGS)
        if n:
            commit.append(n)
        if lim and len(commit) >= lim:
            break

    mbpp = []
    for split in ("train", "validation", "prompt"):
        for r in load_dataset("google-research-datasets/mbpp", "sanitized", split=split):
            n = normalize_mbpp_train(r)
            if n:
                mbpp.append(n)
    mbpp = cap(mbpp)

    oasst_msgs = list(load_dataset("OpenAssistant/oasst1", split="train"))
    oasst = cap(oasst_code_convs(oasst_msgs))

    comp_rows, tally = [], {"problems": 0, "verified": 0, "no_passing_solution": 0}
    adapters = [("codeparrot/apps", None, "test", apps_problem),
                ("deepmind/code_contests", None, "train", codecontests_problem),
                ("BAAI/TACO", None, "train", taco_problem)]
    for name, cfg, split, adapt in adapters:
        probs = []
        it = load_dataset(name, cfg, split=split, streaming=True) if cfg else \
            load_dataset(name, split=split, streaming=True)
        for r in it:
            probs.append(adapt(r))
            if lim and len(probs) >= lim:
                break
        rows, t = verified_competitive_rows(probs, max_per_problem=args.max_per_problem,
                                            timeout_s=args.timeout_s)
        comp_rows += rows
        for k in tally:
            tally[k] += t[k]
    print(f"competitive verify tally: {tally}", flush=True)
    return {"commitpack": commit, "mbpp_train": mbpp, "oasst": oasst, "competitive": comp_rows}


def main() -> None:  # pragma: no cover - network + IO
    from microlab.evals.code.tasks import load_humaneval, load_mbpp
    from microlab.tokenizer.fast import FastTokenizer

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--tokenizer", default="runs/coder-1b-step40000/tokenizer.json")
    ap.add_argument("--limit-per-source", type=int, default=None)
    ap.add_argument("--max-per-problem", type=int, default=1)
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = FastTokenizer.load(args.tokenizer)
    sources = _load_sources(args, tok)
    mix, report = build_compliant_mix(sources, tok, seed=args.seed)

    # Decontaminate against the eval benchmarks (prompts + canonical solutions).
    bench = [t.prompt + "\n" + t.test_program for t in (load_humaneval() + load_mbpp())]
    fp = benchmark_fingerprints(bench, n=10)
    mix, removed = decontaminate(mix, fp, n=10)
    report["decontaminated_removed"] = removed
    report["counts"]["total"] = len(mix)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in mix:
            f.write(json.dumps(r) + "\n")
    print(f"report: {json.dumps(report)}")
    print(f"wrote {len(mix)} rows -> {out}")
    if not mix:
        raise SystemExit("empty mix — refusing to proceed (verify by count)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_build_code_sft.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_code_sft.py tests/scripts/test_build_code_sft.py
git commit -m "feat(code-sft): arm A compliant-mix builder CLI"
```

---

### Task 8: Arm B builder CLI (distilled, token-matched)

**Files:**
- Create: `scripts/build_code_sft_distilled.py`
- Test: `tests/scripts/test_build_code_sft_distilled.py`

**Interfaces:**
- Consumes: `normalize` for Magicoder rows; `token_match_subsample`, `total_supervised_tokens`, `benchmark_fingerprints`, `decontaminate`.
- Produces: `normalize_magicoder(row: dict) -> dict | None`; `build_distilled_mix(rows: list[dict], target_tokens: int, tok, seed: int = 0) -> tuple[list[dict], dict]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_build_code_sft_distilled.py
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_distilled", Path(__file__).resolve().parents[2] / "scripts" / "build_code_sft_distilled.py")
bd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bd)


class _ByteTok:
    def encode(self, s): return list(s.encode("utf-8"))


def test_normalize_magicoder_maps_problem_and_solution():
    row = {"instruction": "Write a function to add.", "response": "def add(a,b): return a+b"}
    assert bd.normalize_magicoder(row) == {"instruction": "Write a function to add.",
                                           "context": "", "response": "def add(a,b): return a+b"}


def test_build_distilled_mix_token_matches_target():
    from microlab.data.code_sft import total_supervised_tokens
    tok = _ByteTok()
    rows = [{"instruction": "i", "context": "", "response": "x" * 20} for _ in range(200)]
    target = 20 * 5  # ~5 rows' worth (response only ~ len 20 + sentinel)
    out, report = bd.build_distilled_mix(rows, target_tokens=target, tok=tok, seed=0)
    assert report["target_tokens"] == target
    assert total_supervised_tokens(out, tok) >= target or len(out) == len(rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/test_build_code_sft_distilled.py -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/build_code_sft_distilled.py
"""Build the DISTILLED arm (B) for the distill-cost A/B: Magicoder-Evol-Instruct-110K
(GPT-4-authored) normalized and token-matched to the compliant arm's supervised-token budget.

    python scripts/build_code_sft_distilled.py \\
        --match data/corpora/code_sft_compliant.jsonl \\
        --out data/corpora/code_sft_distilled.jsonl

This arm INTENTIONALLY violates build-capability. It is a measurement instrument only:
never merged, never used to seed later training. Same decontamination as arm A so the
comparison is controlled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.data.code_sft import (  # noqa: E402
    benchmark_fingerprints, decontaminate, total_supervised_tokens, token_match_subsample,
)


def normalize_magicoder(row: dict) -> dict | None:
    """ise-uiuc/Magicoder-Evol-Instruct-110K row {instruction, response} -> Row."""
    instruction = (row.get("instruction") or "").strip()
    response = (row.get("response") or "").strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "context": "", "response": response}


def build_distilled_mix(rows: list[dict], target_tokens: int, tok, seed: int = 0):
    """Token-match `rows` down to `target_tokens` supervised tokens (arm A's budget)."""
    matched = token_match_subsample(rows, target_tokens, tok, seed=seed)
    return matched, {"target_tokens": target_tokens,
                     "matched_tokens": total_supervised_tokens(matched, tok),
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
    ap.add_argument("--match", required=True, help="arm A jsonl whose supervised-token count to match")
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

    # Decontaminate BEFORE matching so the matched token budget reflects usable rows.
    bench = [t.prompt + "\n" + t.test_program for t in (load_humaneval() + load_mbpp())]
    raw, removed = decontaminate(raw, benchmark_fingerprints(bench, n=10), n=10)

    matched, report = build_distilled_mix(raw, target, tok, seed=args.seed)
    report["decontaminated_removed"] = removed

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in matched:
            f.write(json.dumps(r) + "\n")
    print(f"report: {json.dumps(report)}")
    print(f"wrote {len(matched)} rows -> {out}")
    if not matched:
        raise SystemExit("empty distilled mix — refusing to proceed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_build_code_sft_distilled.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_code_sft_distilled.py tests/scripts/test_build_code_sft_distilled.py
git commit -m "feat(code-sft): arm B distilled builder (token-matched to arm A)"
```

---

### Task 9: Pre-registered prediction doc

**Files:**
- Create: `docs/coder-1b-instruct-prediction.md`

This task has no code test; its gate is that it is **committed before any arm is trained** (Task 11). It follows the `docs/coder-1b-prediction.md` convention: falsifiable bands written first.

- [ ] **Step 1: Write the prediction doc**

Content must include, with concrete numbers: the base's measured floor (HumanEval 1/164 = 0.61%, MBPP 10/257 = 3.9%, greedy; plus the sampled figures once measured); a predicted band for **arm A** (compliant) on HumanEval and MBPP pass@1 greedy+sampled; the same band for **arm B** (distilled); an explicit **distill-gap** prediction (e.g. "distilled beats compliant by 3–10 HumanEval points, or the gap is within the ±1-task noise"); the guardrail expectation (FIM middle-loss stays within X of the base 0.5848; passkey long-context not collapsed); and the falsifiers (either arm below base = SFT is broken; compliant arm collapses FIM = block size too small).

- [ ] **Step 2: Commit**

```bash
git add docs/coder-1b-instruct-prediction.md
git commit -m "docs: pre-registered prediction for coder-1b-instruct A/B (before training)"
```

---

### Task 10: Eval + compare harness

**Files:**
- Create: `scripts/eval_instruct_compare.py`
- Test: `tests/scripts/test_eval_instruct_compare.py`

**Interfaces:**
- Consumes: `eval_code.py` (subprocess) summaries; `eval_pairwise.py` (subprocess); the FIM guardrail from `eval_suite.py`.
- Produces: `assemble_report(arm_summaries: dict, pairwise: dict, guardrail: dict) -> dict` (pure — merges already-collected results into the comparison table); a `main()` that runs the evals and writes `evals/instruct/compare.json` + `compare.md`.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_eval_instruct_compare.py
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "eic", Path(__file__).resolve().parents[2] / "scripts" / "eval_instruct_compare.py")
eic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eic)


def test_assemble_report_builds_comparison_table():
    arm = {"compliant": {"humaneval": 0.10, "mbpp": 0.09, "humaneval_sampled": 0.12},
           "distilled": {"humaneval": 0.14, "mbpp": 0.11, "humaneval_sampled": 0.15}}
    pairwise = {"win_rate_compliant": 0.42, "win_rate_distilled": 0.46, "ties": 0.12}
    guardrail = {"compliant": {"fim_middle_loss": 0.60}, "distilled": {"fim_middle_loss": 0.61},
                 "base": {"fim_middle_loss": 0.5848}}
    rep = eic.assemble_report(arm, pairwise, guardrail)
    assert rep["arms"]["distilled"]["humaneval"] == 0.14
    assert rep["distill_gap"]["humaneval"] == 0.14 - 0.10   # distilled - compliant
    assert rep["guardrail_fim_delta"]["compliant"] == 0.60 - 0.5848
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/test_eval_instruct_compare.py -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/eval_instruct_compare.py
"""Evaluate two instruct arms (compliant vs distilled) and emit the A/B comparison.

Runs, for each arm: HumanEval + MBPP pass@1 greedy AND sampled (eval_code.py --mode chat);
then the pairwise judge between arms (eval_pairwise.py); then the FIM guardrail vs base.
Writes evals/instruct/compare.json + compare.md.

    python scripts/eval_instruct_compare.py \\
        --compliant runs/coder-1b-instruct-compliant \\
        --distilled runs/coder-1b-instruct-distilled \\
        --base runs/coder-1b-step40000

assemble_report() is pure so the merge/verdict logic is unit-tested; main() shells out to
the existing eval scripts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def assemble_report(arm_summaries: dict, pairwise: dict, guardrail: dict) -> dict:
    """Merge already-collected results into the comparison. `arm_summaries[arm][metric]`
    holds pass@1 floats; distill_gap is distilled-minus-compliant per metric."""
    metrics = sorted({m for a in arm_summaries.values() for m in a})
    gap = {m: round(arm_summaries["distilled"].get(m, 0.0)
                    - arm_summaries["compliant"].get(m, 0.0), 4)
           for m in metrics if m in arm_summaries.get("compliant", {})
           and m in arm_summaries.get("distilled", {})}
    base_fim = guardrail.get("base", {}).get("fim_middle_loss")
    fim_delta = {arm: round(g.get("fim_middle_loss", 0.0) - base_fim, 4)
                 for arm, g in guardrail.items() if arm != "base" and base_fim is not None}
    return {"arms": arm_summaries, "pairwise": pairwise, "distill_gap": gap,
            "guardrail_fim_delta": fim_delta, "guardrail_raw": guardrail}


def _run_eval_code(run: Path, dataset: str, sampled: bool, out_dir: Path) -> float:  # pragma: no cover
    tag = "sampled" if sampled else "greedy"
    out = out_dir / f"{run.name}-{dataset}-{tag}.jsonl"
    cmd = [sys.executable, "scripts/eval_code.py", "--run", str(run), "--dataset", dataset,
           "--mode", "chat", "--out", str(out)]
    if sampled:
        cmd += ["--temperature", "0.7", "--top-k", "40"]
    subprocess.run(cmd, check=True)
    return json.loads((out.with_suffix(".jsonl.summary.json")).read_text())["pass@1"]


def main() -> None:  # pragma: no cover - orchestration
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compliant", type=Path, required=True)
    ap.add_argument("--distilled", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("evals/instruct"))
    ap.add_argument("--pairwise-data", default="data/corpora/code_sft_compliant.jsonl")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arms = {}
    for name, run in (("compliant", args.compliant), ("distilled", args.distilled)):
        arms[name] = {
            "humaneval": _run_eval_code(run, "humaneval", False, args.out_dir),
            "mbpp": _run_eval_code(run, "mbpp", False, args.out_dir),
            "humaneval_sampled": _run_eval_code(run, "humaneval", True, args.out_dir),
            "mbpp_sampled": _run_eval_code(run, "mbpp", True, args.out_dir),
        }

    pw_out = args.out_dir / "pairwise.json"
    subprocess.run([sys.executable, "scripts/eval_pairwise.py", str(args.compliant),
                    str(args.distilled), "--data", args.pairwise_data, "--skip", "0",
                    "--limit", "120", "--out", str(pw_out)], check=True)
    pairwise = json.loads(pw_out.read_text())

    # FIM guardrail: eval_suite.py prints fim middle_loss per staged run (base + both arms).
    guardrail = {"base": {}, "compliant": {}, "distilled": {}}  # filled by the operator step

    report = assemble_report(arms, pairwise, guardrail)
    (args.out_dir / "compare.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["distill_gap"], indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_eval_instruct_compare.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_instruct_compare.py tests/scripts/test_eval_instruct_compare.py
git commit -m "feat(code-sft): eval+compare harness for the instruct A/B"
```

---

### Task 11: Operational run — build, validate memory, train, evaluate

**Files:** none (produces data + run dirs + a milestone doc). This task is executed on the RTX 6000 Ada; each step lists its expected output. It is a single task because the steps share one deliverable (the measured A/B result) and each depends on the previous.

- [ ] **Step 1: Build arm A (full run)**

Run: `MICROLAB_MIX_DIR=data/shards/mix-v2 python scripts/build_code_sft.py --out data/corpora/code_sft_compliant.jsonl`
Expected: a `report:` line with non-zero per-source counts and `supervised_tokens`, a `competitive verify tally`, and `decontaminated_removed`. Sanity: total rows > 20k, decontam removals a small fraction. If any source is 0, stop and fix (verify by count).

- [ ] **Step 2: Build arm B (token-matched to arm A)**

Run: `python scripts/build_code_sft_distilled.py --match data/corpora/code_sft_compliant.jsonl --out data/corpora/code_sft_distilled.jsonl`
Expected: `matched_tokens` within one row of arm A's `supervised_tokens`.

- [ ] **Step 3: Confirm the prediction doc is committed (Task 9) BEFORE training.**

Run: `git log --oneline -- docs/coder-1b-instruct-prediction.md`
Expected: a commit exists. If not, do Task 9 now.

- [ ] **Step 4: Validate a safe micro-batch at block 2048**

Run a 8-step smoke with a VRAM sampler (as in commit `aa89fed`'s validation), micro-batch 2:
`python scripts/sft.py --base-ckpt runs/coder-1b-step40000 --data data/corpora/code_sft_compliant.jsonl --tokenizer runs/coder-1b-step40000/tokenizer.json --out /tmp/smoke --limit 64 --epochs 1 --batch-size 2 --grad-accum 4 --block-size 2048 --device cuda`
Expected: completes without OOM; note peak VRAM. If it OOMs, drop to `--batch-size 1 --grad-accum 8`. Record the chosen micro-batch/accum; **use the same for both arms.**

- [ ] **Step 5: Train arm A (compliant)**

Run: `python scripts/sft.py --base-ckpt runs/coder-1b-step40000 --data data/corpora/code_sft_compliant.jsonl --tokenizer runs/coder-1b-step40000/tokenizer.json --out runs/coder-1b-instruct-compliant --epochs 3 --lr 2e-5 --batch-size <validated> --grad-accum <validated> --block-size 2048 --save-every 500 --device cuda`
Expected: loss decreases; a servable run dir with `serve_config.json` (chat mode).

- [ ] **Step 6: Train arm B (distilled)** — identical flags, `--data data/corpora/code_sft_distilled.jsonl --out runs/coder-1b-instruct-distilled`.
Expected: same as Step 5.

- [ ] **Step 7: FIM guardrail** — stage each of base / compliant / distilled and run the FIM middle-loss (as in the 40k battery): `MICROLAB_MIX_DIR=data/shards/mix-v2 python scripts/eval_suite.py --run <run> --device cuda --out evals/instruct/<name>-suite.json`. Record `fim.middle_loss` for each; confirm neither arm is materially above the base 0.5848.

- [ ] **Step 8: Run the compare harness**

Run: `python scripts/eval_instruct_compare.py --compliant runs/coder-1b-instruct-compliant --distilled runs/coder-1b-instruct-distilled --base runs/coder-1b-step40000`
Then paste the Step-7 FIM numbers into `evals/instruct/compare.json`'s guardrail block (or pass them via a `--fim` arg if added).
Expected: `evals/instruct/compare.json` with the HumanEval/MBPP greedy+sampled table, pairwise win-rate, and distill-gap.

- [ ] **Step 9: Write + commit the milestone doc** `docs/coder-1b-instruct-milestone.md`: the A/B table, the distill gap, the guardrail deltas, and the verdict scored against `docs/coder-1b-instruct-prediction.md` (did the distilled arm win, and by how much vs the predicted band; did decontamination stay comparable across arms). Commit data reports, run summaries, `evals/instruct/*`, and the milestone doc.

```bash
git add data/corpora/*.jsonl evals/instruct docs/coder-1b-instruct-milestone.md
git commit -m "eval: coder-1b-instruct A/B — compliant vs distilled measured"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** Unit 1 → Tasks 1–4; Unit 2 → Task 8; Unit 3 (decontamination) → Task 6, applied in Tasks 7–8; Unit 4 → Tasks 10–11; training config → Task 11 steps 4–6; pre-registered prediction → Task 9; guardrail → Task 11 step 7; deliverables all mapped.
- **Placeholder scan:** the only non-code task is Task 9 (a doc), which specifies the exact required contents and numbers; Task 11 is operational with concrete commands and expected outputs. `<validated>` micro-batch is a value produced by Task 11 Step 4, not a placeholder for the implementer to invent.
- **Type consistency:** `Row` = single-turn dict throughout; `verified_competitive_rows` returns `(rows, tally)`; `build_compliant_mix` returns `(rows, report)` with `report["counts"]`/`report["supervised_tokens"]`; `assemble_report` keys (`arms`, `distill_gap`, `guardrail_fim_delta`) match its test. Executor helpers (`verify_io`, `verify_unit_test`) names match Tasks 3–4 usage.

## Notes / risks carried from the spec

- **APPS/CodeContests/TACO schema drift.** The adapters (`apps_problem` etc.) encode the documented field shapes; verify against a handful of real rows in Task 11 Step 1 (the verify tally surfaces a broken adapter as `no_passing_solution == problems`).
- **Executor throughput.** Verifying many competitive solutions is serial `run_python` calls; use `--limit-per-source` first, and consider capping problems. Log how many were skipped — no silent caps.
- **Decontamination strength.** n=10 word n-grams; if Task 11 Step 1 shows a suspiciously high removal count from APPS/TACO, that is a signal (those overlap benchmarks), not a bug — record it.
