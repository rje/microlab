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


def test_parse_input_exprs_rejects_substring_identifiers():
    assert pid.parse_input_exprs("myadd(1)\nadd(2)", "add") == ["add(2)"]


def test_make_mutant_changes_code_or_none():
    assert pid.make_mutant("def f(a):\n    return a + 1") == "def f(a):\n    return a - 1"
    assert pid.make_mutant("def f():\n    pass") is None


def test_is_discriminating_requires_ref_ok_and_difference():
    assert pid.is_discriminating(("ok", "1"), ("ok", "2")) is True
    assert pid.is_discriminating(("ok", "1"), ("err",)) is True
    assert pid.is_discriminating(("ok", "1"), ("ok", "1")) is False
    assert pid.is_discriminating(("err",), ("ok", "2")) is False   # ref must run
