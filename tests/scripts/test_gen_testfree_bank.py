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


def test_plan_resume_completes_partials_and_dedups_header():
    rows = [{"_header": {"n": 2}}, {"_header": {"n": 2}},
            {"task_id": "A", "sample": 0, "passed": True, "solution": "x"},
            {"task_id": "A", "sample": 1, "passed": False, "solution": "y"},
            {"task_id": "B", "sample": 0, "passed": False, "solution": "z"}]  # partial
    done, compact = gb.plan_resume(rows, n=2)
    assert done == {"A"}
    assert sum(1 for r in compact if "_header" in r) == 1
    assert all(r.get("task_id") != "B" for r in compact)


def test_mbpp_signature_prefers_entry_point_over_first_def():
    code = "def helper(x):\n    return x\ndef target(a, b):\n    return a + b\n"
    assert gb.mbpp_signature(code, entry_point="target") == "def target(a, b):"
    assert gb.mbpp_signature(code) == "def helper(x):"          # legacy behavior, no entry
    assert gb.mbpp_signature(code, entry_point="missing") is None
