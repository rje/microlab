import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_selfgen_sft", Path(__file__).resolve().parents[2] / "scripts" / "build_selfgen_sft.py")
bs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bs)


def test_selfgen_row_keeps_shortest_full_pass():
    sols = [("print(long_version(42))", 1.0), ("print(42)", 1.0), ("print(41)", 0.5)]
    row = bs.selfgen_row("emit 42", sols)
    assert row == {"instruction": "emit 42", "context": "", "response": "print(42)"}
    assert bs.selfgen_row("x", [("print(1)", 0.5)]) is None
