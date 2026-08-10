"""Pure, network-free helpers for building the coder-1b code-instruction SFT mixes.

Every function here is deterministic and importable without torch or a GPU so the builders
in scripts/ stay thin and the logic is unit-tested off-network — the same split
build_sft_mix.py uses. Row is the single-turn schema scripts/sft.py consumes.
"""
from __future__ import annotations

import importlib.util as _ilu
import json as _json
import random as _random
import re as _re
from pathlib import Path as _Path

from microlab.evals.code.executor import run_python
from microlab.evals.code.tasks import CodeTask, assemble_program
from microlab.model.reference.chat_sft import END_SENTINEL
from microlab.model.reference.sft import format_chat
from microlab.model.reference.sft import format_chat as _format_chat

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
try:
    exec(compile({solution!r}, "<solution>", "exec"), {{"__name__": "__main__"}})
except SystemExit:
    pass
finally:
    sys.stdout = _real
_got = _out.getvalue()
_want = {expected!r}
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
                              timeout_s: float = 10.0, max_cases: int | None = None,
                              max_solutions: int | None = None,
                              progress_every: int = 0) -> tuple[list[Row], dict]:
    """For each normalized problem, keep up to `max_per_problem` human solutions that pass
    its I/O cases in the sandbox. instruction=statement, response=verified solution.
    Solutions are tried shortest-first (concise correct code is the better demonstration).

    The solutions are ALREADY human-accepted submissions (they passed the origin judge), so
    running them here is a SANITY FILTER — does this code run and reproduce outputs in our
    sandbox — not a correctness discovery. That makes two bounds safe and necessary, because
    each I/O case is a separate sandbox subprocess and competitive problems carry dozens of
    (often large) hidden cases:
      - `max_cases`: check only the first N cases per solution (the leading cases are the
        small sample cases; the large hidden ones cost seconds each and add little signal
        for an already-accepted solution).
      - `max_solutions`: try only the N shortest solutions per problem.
    `progress_every>0` prints a running count every N problems (long jobs must be observable).
    """
    rows: list[Row] = []
    tally = {"problems": 0, "verified": 0, "no_passing_solution": 0}
    for p in problems:
        tally["problems"] += 1
        if progress_every and tally["problems"] % progress_every == 0:
            print(f"    competitive: {tally['problems']} problems, "
                  f"{tally['verified']} verified", flush=True)
        statement = (p.get("statement") or "").strip()
        cases = p.get("io") or []
        if max_cases is not None:
            cases = cases[:max_cases]
        if not statement or not cases:
            tally["no_passing_solution"] += 1
            continue
        sols = sorted(p.get("solutions") or [], key=len)
        if max_solutions is not None:
            sols = sols[:max_solutions]
        kept = 0
        for sol in sols:
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


def _io_item_to_str(v) -> str:
    """Coerce one APPS/TACO input/output item to a string. Items are usually the raw
    stdin/stdout text, but some rows wrap them in a list of lines and a few carry non-string
    scalars — str() them so a stray type can't crash the whole build (observed: TACO outputs
    containing a bool)."""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return "".join(_io_item_to_str(x) for x in v)
    return str(v)


def _io_cases(io: dict) -> list[dict]:
    """Build (input, output) string cases from a parsed APPS/TACO input_output dict.
    Call-based problems (carrying `fn_name`) are stdin/stdout-incompatible with the executor's
    verify_io, so they yield no cases and the problem drops out as unverifiable rather than
    being falsely tested against stdout."""
    if not isinstance(io, dict) or io.get("fn_name"):
        return []
    return [{"input": _io_item_to_str(i), "output": _io_item_to_str(o)}
            for i, o in zip(io.get("inputs", []), io.get("outputs", []), strict=False)]


def _loads_io(s: str):
    """json.loads for competitive input_output/solutions, keeping integer literals as strings.

    Competitive test cases carry arbitrarily large integers (observed: a 9,131-digit value),
    which exceed Python 3.11's int()-from-str digit cap and crash a plain json.loads. We never
    need these as ints — I/O values are coerced to stdin/stdout strings — so parse them as
    strings and sidestep the limit entirely."""
    return _json.loads(s, parse_int=str)


def apps_problem(row: dict) -> dict:
    """codeparrot/apps row -> normalized problem. `solutions` and `input_output` are
    JSON-encoded strings; input_output has parallel `inputs`/`outputs` lists."""
    io = _loads_io(row["input_output"]) if row.get("input_output") else {}
    sols = _loads_io(row["solutions"]) if row.get("solutions") else []
    return {"statement": row.get("question", ""), "solutions": sols, "io": _io_cases(io)}


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
        sols = _loads_io(row["solutions"])
    else:
        sols = row.get("solutions") or []
    io = _loads_io(row["input_output"]) if row.get("input_output") else {}
    return {"statement": row.get("question", ""), "solutions": sols, "io": _io_cases(io)}


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
    """Sum supervised tokens across all rows."""
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


def _norm_tokens(text: str) -> list[str]:
    """Whitespace/punctuation-insensitive word stream for n-gram matching."""
    return _re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> set[str]:
    """n-gram set from normalized tokens; single token if fewer than n tokens."""
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
    """Extract text from single-turn or multi-turn row schema."""
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
