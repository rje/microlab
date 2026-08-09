import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_code_sft_distilled.py"
_SPEC = importlib.util.spec_from_file_location("build_distilled", _SCRIPT)
bd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bd)


class _ByteTok:
    def encode(self, s): return list(s.encode("utf-8"))


def test_normalize_magicoder_maps_problem_and_solution():
    row = {"instruction": "Write a function to add.", "response": "def add(a,b): return a+b"}
    assert bd.normalize_magicoder(row) == {"instruction": "Write a function to add.",
                                           "context": "", "response": "def add(a,b): return a+b"}


def test_build_distilled_mix_token_matches_target():
    from microlab.data.code_sft import total_supervised_tokens
    tok = _ByteTok()
    rows = [{"instruction": "i", "context": "", "response": "x" * 20} for _ in range(200)]
    target = 20 * 5  # ~5 rows' worth (response only ~ len 20 + sentinel)
    out, report = bd.build_distilled_mix(rows, target_tokens=target, tok=tok, seed=0)
    assert report["target_tokens"] == target
    assert total_supervised_tokens(out, tok) >= target or len(out) == len(rows)


def test_build_distilled_mix_raises_when_pool_too_small():
    import pytest
    tok = _ByteTok()
    rows = [{"instruction": "i", "context": "", "response": "x" * 5} for _ in range(3)]
    with pytest.raises(ValueError, match="cannot token-match"):
        bd.build_distilled_mix(rows, target_tokens=100000, tok=tok, seed=0)
