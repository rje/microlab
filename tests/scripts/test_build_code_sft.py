import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_code_sft", Path(__file__).resolve().parents[2] / "scripts" / "build_code_sft.py")
bcs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bcs)


class _ByteTok:
    def encode(self, s): return list(s.encode("utf-8"))


def test_build_compliant_mix_merges_reports_and_shuffles_deterministically():
    sources = {
        "commitpack": [{"instruction": "fix", "context": "", "response": "def f(): pass"}],
        "mbpp_train": [{"instruction": "add", "context": "",
                       "response": "def add(a,b): return a+b"}],
        "oasst": [{"turns": [{"user": "sort", "assistant": "```python\nsorted(x)\n```"}]}],
        "competitive": [{"instruction": "n*2", "context": "", "response": "print(int(input())*2)"}],
    }
    rows, report = bcs.build_compliant_mix(sources, _ByteTok(), seed=0)
    assert len(rows) == 4
    assert report["counts"] == {"commitpack": 1, "mbpp_train": 1, "oasst": 1,
                                "competitive": 1, "total": 4}
    assert report["supervised_tokens"] > 0
    # deterministic order under a fixed seed
    rows2, _ = bcs.build_compliant_mix(sources, _ByteTok(), seed=0)
    assert rows == rows2
