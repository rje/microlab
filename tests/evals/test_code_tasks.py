"""CodeTask conversion + test-program assembly on tiny inline fixtures (no network),
including end-to-end execution of a correct and an incorrect candidate through the
sandboxed executor."""

from __future__ import annotations

import pytest

from microlab.evals.code.executor import run_python
from microlab.evals.code.tasks import (
    assemble_program,
    humaneval_task,
    load_tasks,
    mbpp_task,
)

HE_ROW = {
    "task_id": "HumanEval/0",
    "prompt": 'def add(a, b):\n    """Return a + b."""\n',
    "canonical_solution": "    return a + b\n",
    "test": ("def check(candidate):\n    assert candidate(2, 3) == 5\n"
             "    assert candidate(-1, 1) == 0\n"),
    "entry_point": "add",
}

MBPP_ROW = {
    "task_id": 12,
    "prompt": "Write a function to double a number.",
    "code": "def double(n):\n    return 2 * n",
    "test_imports": ["import math"],
    "test_list": ["assert double(2) == 4", "assert double(-3) == -6"],
}


def test_humaneval_task_fields():
    t = humaneval_task(HE_ROW)
    assert t.task_id == "HumanEval/0"
    assert t.prompt == HE_ROW["prompt"]  # completion prefix is verbatim
    assert t.entry_point == "add"
    assert t.test_program.endswith("check(add)\n")
    assert HE_ROW["prompt"].rstrip() in t.instruction


def test_humaneval_missing_field_raises():
    with pytest.raises(ValueError, match="missing 'entry_point'"):
        humaneval_task({**HE_ROW, "entry_point": ""})


def test_mbpp_task_fields():
    t = mbpp_task(MBPP_ROW)
    assert t.task_id == "Mbpp/12"
    assert t.entry_point == "double"
    # base prompt is the docstring form: description + first assert
    assert t.prompt == '"""\nWrite a function to double a number.\nassert double(2) == 4\n"""\n'
    assert "import math" in t.test_program
    assert "assert double(-3) == -6" in t.test_program
    assert "assert double(2) == 4" in t.instruction


def test_mbpp_entry_point_handles_parenthesized_assert():
    row = {**MBPP_ROW, "test_list": ["assert (double(2) == 4)"]}
    assert mbpp_task(row).entry_point == "double"


def test_mbpp_entry_point_unrecoverable_raises():
    row = {**MBPP_ROW, "test_list": ["double(2) == 4"]}  # no assert
    with pytest.raises(ValueError, match="without assert"):
        mbpp_task(row)


def test_mbpp_entry_point_recovers_inner_call_from_math_isclose():
    row = {**MBPP_ROW, "test_list": ["assert math.isclose(find_area(2), 12.56)"]}
    assert mbpp_task(row).entry_point == "find_area"


def test_mbpp_entry_point_recovers_inner_call_from_set_wrapper():
    row = {**MBPP_ROW, "test_list": ["assert set(common(1)) == set([2])"]}
    assert mbpp_task(row).entry_point == "common"


def test_mbpp_entry_point_no_call_assert_raises():
    row = {**MBPP_ROW, "test_list": ["assert True"]}
    with pytest.raises(ValueError, match="cannot recover entry point"):
        mbpp_task(row)


def test_mbpp_entry_point_keeps_shadowed_builtin_under_test():
    # Mbpp/126's reference is literally `def sum(a,b)` — a user function shadowing a
    # builtin, called directly. It must NOT be skipped as a wrapper.
    row = {**MBPP_ROW, "test_list": ["assert sum(10,15) == 6"]}
    assert mbpp_task(row).entry_point == "sum"


def test_assemble_and_execute_correct_solution_passes():
    t = humaneval_task(HE_ROW)
    program = assemble_program(HE_ROW["prompt"] + HE_ROW["canonical_solution"], t)
    assert run_python(program).passed


def test_assemble_and_execute_wrong_solution_fails():
    t = humaneval_task(HE_ROW)
    program = assemble_program("def add(a, b):\n    return a - b\n", t)
    res = run_python(program)
    assert not res.passed
    assert "AssertionError" in res.stderr


def test_mbpp_execute_roundtrip():
    t = mbpp_task(MBPP_ROW)
    assert run_python(assemble_program(MBPP_ROW["code"], t)).passed
    assert not run_python(assemble_program("def double(n):\n    return n\n", t)).passed


def test_load_tasks_multipl_stub_and_unknown():
    with pytest.raises(NotImplementedError, match="node"):
        load_tasks("multipl-js")
    with pytest.raises(ValueError, match="unknown dataset"):
        load_tasks("rust")
