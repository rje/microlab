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
