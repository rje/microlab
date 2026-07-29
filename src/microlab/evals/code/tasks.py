"""HumanEval / MBPP rows -> CodeTask: one shape the runner scores, whichever dataset or
prompting mode produced the completion.

The row->task converters and the test-program assembly are pure functions so the tests
exercise them on tiny inline fixtures with no network; only the two ``load_*`` helpers
touch the HF hub (cached after the first call).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class CodeTask:
    """One executable coding problem.

    ``prompt`` is the completion-style prefix a BASE model continues (for HumanEval the
    function signature + docstring, verbatim). ``instruction`` is the chat-mode wording.
    ``test_program`` is a self-contained suffix: appended to the candidate solution it
    exits 0 iff the solution is correct."""

    task_id: str
    prompt: str
    instruction: str
    entry_point: str
    test_program: str


def humaneval_task(row: dict) -> CodeTask:
    """openai/openai_humaneval row: prompt IS the canonical completion prefix; the test
    field defines check(candidate) which we call on the entry point."""
    for key in ("task_id", "prompt", "test", "entry_point"):
        if not row.get(key):
            raise ValueError(f"HumanEval row {row.get('task_id')!r}: missing {key!r}")
    test_program = f"{row['test']}\n\ncheck({row['entry_point']})\n"
    return CodeTask(
        task_id=str(row["task_id"]),
        prompt=row["prompt"],
        instruction=(
            "Complete the following Python function. "
            "Reply with the full function definition.\n\n" + row["prompt"].rstrip()
        ),
        entry_point=row["entry_point"],
        test_program=test_program,
    )


def _mbpp_entry_point(test_list: list[str]) -> str:
    """The function name under test, recovered from the first assert (canonical MBPP
    practice — the dataset has no entry_point field)."""
    first = test_list[0]
    marker = "assert "
    if marker not in first:
        raise ValueError(f"MBPP test without assert: {first!r}")
    expr = first.split(marker, 1)[1].lstrip()
    if expr.startswith("("):  # e.g. assert (fn(...) == x)
        expr = expr[1:].lstrip()
    name = ""
    for ch in expr:
        if ch.isalnum() or ch == "_":
            name += ch
        else:
            break
    if not name:
        raise ValueError(f"cannot recover entry point from {first!r}")
    return name


def mbpp_task(row: dict) -> CodeTask:
    """google-research-datasets/mbpp (sanitized) row. Base-mode prompt is the standard
    docstring form: description + the first assert (so the model sees the exact function
    name and call shape), then the model writes the function."""
    for key in ("task_id", "prompt", "test_list"):
        if not row.get(key):
            raise ValueError(f"MBPP row {row.get('task_id')!r}: missing {key!r}")
    tests: list[str] = list(row["test_list"])
    imports: list[str] = list(row.get("test_imports") or [])
    entry = _mbpp_entry_point(tests)
    prompt = f'"""\n{row["prompt"].strip()}\n{tests[0]}\n"""\n'
    instruction = (
        f"{row['prompt'].strip()}\n\n"
        "Write a Python function that satisfies these tests. "
        "Reply with the full function definition.\n"
        + "\n".join(tests)
    )
    test_program = "\n".join([*imports, *tests]) + "\n"
    return CodeTask(
        task_id=f"Mbpp/{row['task_id']}",
        prompt=prompt,
        instruction=instruction,
        entry_point=entry,
        test_program=test_program,
    )


def assemble_program(solution: str, task: CodeTask) -> str:
    """Candidate solution + the task's tests as one runnable program (exit 0 == pass)."""
    return solution.rstrip() + "\n\n" + task.test_program


def load_humaneval() -> list[CodeTask]:
    from datasets import load_dataset

    rows: Iterable[dict] = load_dataset("openai/openai_humaneval", split="test")
    return [humaneval_task(r) for r in rows]


def load_mbpp() -> list[CodeTask]:
    from datasets import load_dataset

    rows: Iterable[dict] = load_dataset(
        "google-research-datasets/mbpp", "sanitized", split="test"
    )
    return [mbpp_task(r) for r in rows]


def load_tasks(dataset: str) -> list[CodeTask]:
    if dataset == "humaneval":
        return load_humaneval()
    if dataset == "mbpp":
        return load_mbpp()
    if dataset in ("multipl-js", "multipl-ts"):
        raise NotImplementedError(
            "MultiPL-E JS/TS needs a sandboxed node runtime (the executor only caps "
            "Python today); the CodeTask interface is language-agnostic — add a "
            "multipl_task converter plus a node executor to enable it"
        )
    raise ValueError(f"unknown dataset {dataset!r} (humaneval, mbpp)")
