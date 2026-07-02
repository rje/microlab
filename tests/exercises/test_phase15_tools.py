"""Spec + validation for the hand-written Phase-15 tool-call parsing/validation primitives.

Implement ``microlab.exercises.phase15_tools`` until these pass. Differential tests grade you
against ``microlab.model.reference.tools``.
"""

import pytest

from microlab.exercises.phase15_tools import (
    parse_tool_call,
    run_tool_loop,
    schema_validity_rate,
    validate_tool_call,
)
from microlab.model.reference.tools import parse_tool_call as ref_parse
from microlab.model.reference.tools import run_tool_loop as ref_loop
from microlab.model.reference.tools import schema_validity_rate as ref_rate
from microlab.model.reference.tools import validate_tool_call as ref_validate

SCHEMA = {"calc": {"required": ["expr"]}}


class _ScriptedModel:
    """A fake model callable that replays a fixed sequence of outputs, ignoring context."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self._i = 0

    def __call__(self, context: str) -> str:
        out = self._outputs[self._i]
        self._i += 1
        return out


# a valid tool call, then an invalid one (unknown tool), then a final answer
_SCRIPT = [
    '<tool>{"name": "calc", "arguments": {"expr": "2+2"}}</tool>',
    '<tool>{"name": "nope", "arguments": {}}</tool>',
    "<answer>4</answer>",
]


def test_parse_tool_call_known_values():
    assert parse_tool_call('<tool>{"name": "calc", "arguments": {"expr": "2+2"}}</tool>') == {
        "name": "calc",
        "arguments": {"expr": "2+2"},
    }
    assert parse_tool_call("no tool here") is None
    assert parse_tool_call("<tool>{bad json}</tool>") is None
    assert parse_tool_call('<tool>{"arguments": {}}</tool>') is None  # no name


def test_parse_tool_call_matches_reference():
    samples = [
        '<tool>{"name": "calc", "arguments": {"expr": "1+1"}}</tool>',
        "junk",
        '<tool>{"name":"calc","arguments":{}}</tool>',
        '<tool>{"name":"unknown"}</tool>',
        "<tool>{bad json}</tool>",
    ]
    for s in samples:
        assert parse_tool_call(s) == ref_parse(s)


def test_validate_tool_call_known_values():
    assert validate_tool_call({"name": "calc", "arguments": {"expr": "2+2"}}, SCHEMA)
    assert not validate_tool_call({"name": "nope", "arguments": {}}, SCHEMA)
    assert not validate_tool_call({"name": "calc", "arguments": {}}, SCHEMA)  # missing expr


def test_validate_tool_call_matches_reference():
    calls = [
        {"name": "calc", "arguments": {"expr": "1"}},
        {"name": "calc", "arguments": {}},
        {"name": "nope", "arguments": {}},
    ]
    for c in calls:
        assert validate_tool_call(c, SCHEMA) == ref_validate(c, SCHEMA)


def test_schema_validity_rate_matches_reference():
    outs = [
        '<tool>{"name":"calc","arguments":{"expr":"1"}}</tool>',
        "junk",
        '<tool>{"name":"calc","arguments":{}}</tool>',
    ]
    assert schema_validity_rate(outs, SCHEMA) == ref_rate(outs, SCHEMA)
    assert schema_validity_rate(outs, SCHEMA) == pytest.approx(1 / 3)


def test_run_tool_loop_matches_reference():
    # drive BOTH loops with the same scripted model + tool registry; the fresh model per loop
    # replays the same outputs, so the full result dict must match the oracle exactly.
    tools = {"calc": lambda args: "4"}
    stu = run_tool_loop(_ScriptedModel(_SCRIPT), tools, SCHEMA, "solve 2+2", max_steps=5)
    ref = ref_loop(_ScriptedModel(_SCRIPT), tools, SCHEMA, "solve 2+2", max_steps=5)
    assert stu == ref
    # and it terminates on the final answer after all three steps (not at max_steps)
    assert stu == {"answer": "4", "steps": 3, "transcript": _SCRIPT}

pytestmark = pytest.mark.exercise
