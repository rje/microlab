import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "generate_best_of", Path(__file__).resolve().parents[2] / "scripts" / "generate_best_of.py")
gbo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gbo)


def test_pick_best_first_passer_or_none():
    assert gbo.pick_best(["a", "b", "c"], [False, True, True]) == 1
    assert gbo.pick_best(["a"], [False]) is None
