import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_grpo_pool", Path(__file__).resolve().parents[2] / "scripts" / "build_grpo_pool.py")
bgp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bgp)


def test_pool_row_caps_cases_and_requires_statement_and_cases():
    p = {"statement": "double n", "solutions": ["x"],
         "io": [{"input": str(i), "output": str(i * 2)} for i in range(10)]}
    row = bgp.pool_row(p, max_cases=6)
    assert row["instruction"] == "double n" and len(row["io"]) == 6
    assert bgp.pool_row({"statement": "", "solutions": [], "io": p["io"]}) is None
    assert bgp.pool_row({"statement": "s", "solutions": [], "io": []}) is None
