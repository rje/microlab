"""Pure, network-free helpers for building the coder-1b code-instruction SFT mixes.

Every function here is deterministic and importable without torch or a GPU so the builders
in scripts/ stay thin and the logic is unit-tested off-network — the same split
build_sft_mix.py uses. Row is the single-turn schema scripts/sft.py consumes.
"""
from __future__ import annotations

import importlib.util as _ilu
import json as _json
import re as _re
from pathlib import Path as _Path

from microlab.evals.code.executor import run_python
from microlab.evals.code.tasks import CodeTask, assemble_program

Row = dict[str, str]

# Reuse the OASST tree-walker from the chat-mix builder (single source of truth for the
# rank-0-child linearization); scripts/ isn't a package so load it by path.
_bcm_spec = _ilu.spec_from_file_location(
    "build_chat_mix", _Path(__file__).resolve().parents[3] / "scripts" / "build_chat_mix.py")
_bcm = _ilu.module_from_spec(_bcm_spec)
_bcm_spec.loader.exec_module(_bcm)

_CODE_FENCE = _re.compile(r"```")


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
    # Responses are stripped to match the other normalizers (normalize_alpaca /
    # normalize_no_robots) and because END_SENTINEL ("\n### End") supplies the trailing boundary.
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


def is_code_conv(conv: dict) -> bool:
    """True if any assistant turn contains a fenced code block (```). The cheap, precise
    signal that a thread is about code without language-classifying every message."""
    return any(_CODE_FENCE.search(t.get("assistant", "")) for t in conv.get("turns", []))


def oasst_code_convs(messages: list[dict], max_turns: int = 6) -> list[dict]:
    """Linearize OASST trees (best-ranked child) and keep only code-bearing conversations."""
    convs = _bcm.extract_oasst_conversations(messages, max_turns=max_turns)
    return [c for c in convs if is_code_conv(c)]


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
    if row.get("input_output"):
        io = _json.loads(row["input_output"])
    else:
        io = {"inputs": [], "outputs": []}
    cases = [{"input": i if isinstance(i, str) else "".join(i),
              "output": o if isinstance(o, str) else "".join(o)}
             for i, o in zip(io.get("inputs", []), io.get("outputs", []), strict=False)]
    sols = _json.loads(row["solutions"]) if row.get("solutions") else []
    return {"statement": row.get("question", ""), "solutions": sols, "io": cases}


def codecontests_problem(row: dict) -> dict:
    """deepmind/code_contests row -> normalized problem. Python solutions only (language enum
    1==PYTHON, 3==PYTHON3 in the dataset); public+private tests as I/O cases."""
    sols = []
    sol_field = row.get("solutions") or {}
    for lang, txt in zip(
        sol_field.get("language", []), sol_field.get("solution", []), strict=False
    ):
        if lang in (1, 3):
            sols.append(txt)
    cases = []
    for group in ("public_tests", "private_tests"):
        g = row.get(group) or {}
        cases += [{"input": i, "output": o}
                  for i, o in zip(g.get("input", []), g.get("output", []), strict=False)]
    return {"statement": row.get("description", ""), "solutions": sols, "io": cases}


def taco_problem(row: dict) -> dict:
    """BAAI/TACO row -> normalized problem. `solutions` is a JSON list; `input_output` is the
    same JSON-string shape as APPS."""
    if isinstance(row.get("solutions"), str):
        sols = _json.loads(row["solutions"])
    else:
        sols = row.get("solutions") or []
    if row.get("input_output"):
        io = _json.loads(row["input_output"])
    else:
        io = {"inputs": [], "outputs": []}
    cases = [{"input": i if isinstance(i, str) else "".join(i),
              "output": o if isinstance(o, str) else "".join(o)}
             for i, o in zip(io.get("inputs", []), io.get("outputs", []), strict=False)]
    return {"statement": row.get("question", ""), "solutions": sols, "io": cases}
