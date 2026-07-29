"""Prompt construction + completion extraction: base-mode truncation semantics differ by
dataset (HumanEval continues a signature, MBPP writes the whole function), chat replies
are truncated at the trained sentinels and mined for code blocks."""

from __future__ import annotations

from microlab.evals.code.prompts import (
    BASE_STOPS,
    base_solution,
    chat_prompt,
    chat_solution,
    extract_code_block,
    truncate_at,
)
from microlab.evals.code.tasks import humaneval_task

HE_ROW = {
    "task_id": "HumanEval/0",
    "prompt": 'def add(a, b):\n    """Return a + b."""\n',
    "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
    "entry_point": "add",
}


def test_truncate_at_earliest_stop():
    assert truncate_at("abcSTOPdefHALTx", ["HALT", "STOP"]) == "abc"
    assert truncate_at("no stops here", ["STOP"]) == "no stops here"


def test_base_solution_humaneval_cuts_next_toplevel_def_and_keeps_prompt():
    t = humaneval_task(HE_ROW)
    completion = "    return a + b\n\ndef sneaky():\n    pass\n"
    sol = base_solution("humaneval", t, completion)
    assert sol == t.prompt + "    return a + b\n"


def test_base_solution_mbpp_keeps_leading_def_cuts_asserts():
    t_prompt_unused = None  # mbpp solution ignores the task prompt
    completion = "def double(n):\n    return 2 * n\n\nassert double(2) == 4\n"
    sol = base_solution("mbpp", t_prompt_unused, completion)
    assert sol == "def double(n):\n    return 2 * n\n"
    assert "\ndef " not in BASE_STOPS["mbpp"]  # the model's own def must survive


def test_chat_prompt_uses_sft_template():
    p = chat_prompt("Write a function.")
    assert p.startswith("### Instruction:\n")
    assert p.endswith("### Response:\n")


def test_extract_code_block_fenced_with_tag():
    reply = "Sure!\n```python\ndef f():\n    return 1\n```\nHope that helps."
    assert extract_code_block(reply) == "def f():\n    return 1"


def test_extract_code_block_fenced_no_tag_and_unclosed():
    assert extract_code_block("```\nx = 1\n```") == "x = 1"
    # generation cut by max_new before the closing fence: keep what's there
    assert extract_code_block("```python\nx = 2\n") == "x = 2"


def test_extract_code_block_no_fence_returns_reply():
    assert extract_code_block("def f():\n    return 1\n") == "def f():\n    return 1"


def test_chat_solution_stops_at_sentinel():
    reply = "def f():\n    return 1\n### End\n### Instruction:\ngarbage"
    assert chat_solution(reply) == "def f():\n    return 1"
