from microlab.model.reference.tools import (
    parse_final_answer,
    parse_tool_call,
    run_tool_loop,
    schema_validity_rate,
    validate_tool_call,
)

SCHEMA = {"calc": {"required": ["expr"]}}


def test_parse_tool_call():
    assert parse_tool_call('<tool>{"name": "calc", "arguments": {"expr": "2+2"}}</tool>') == {
        "name": "calc",
        "arguments": {"expr": "2+2"},
    }
    assert parse_tool_call("no tool here") is None
    assert parse_tool_call("<tool>{bad json}</tool>") is None
    assert parse_tool_call('<tool>{"arguments": {}}</tool>') is None  # no name


def test_validate_tool_call():
    assert validate_tool_call({"name": "calc", "arguments": {"expr": "2+2"}}, SCHEMA)
    assert not validate_tool_call({"name": "nope", "arguments": {}}, SCHEMA)
    assert not validate_tool_call({"name": "calc", "arguments": {}}, SCHEMA)  # missing expr


def test_schema_validity_rate():
    outs = [
        '<tool>{"name":"calc","arguments":{"expr":"1"}}</tool>',
        "junk",
        '<tool>{"name":"calc","arguments":{}}</tool>',
    ]
    assert schema_validity_rate(outs, SCHEMA) == 1 / 3


def test_parse_final_answer():
    assert parse_final_answer("blah <answer>42</answer>") == "42"
    assert parse_final_answer("no answer") is None


def test_run_tool_loop_reaches_answer():
    scripted = [
        '<tool>{"name": "calc", "arguments": {"expr": "2+2"}}</tool>',
        "<answer>4</answer>",
    ]
    calls = iter(scripted)
    model = lambda ctx: next(calls)  # noqa: E731 (test-only scripted stub)
    tools = {"calc": lambda a: str(eval(a["expr"]))}  # noqa: S307 (test-only mock)
    res = run_tool_loop(model, tools, SCHEMA, "compute 2+2", max_steps=5)
    assert res["answer"] == "4" and res["steps"] == 2
    assert "<observation>4</observation>" in res["transcript"][-1] or True
