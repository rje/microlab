import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "eic", Path(__file__).resolve().parents[2] / "scripts" / "eval_instruct_compare.py")
eic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eic)


def test_assemble_report_builds_comparison_table():
    arm = {"compliant": {"humaneval": 0.10, "mbpp": 0.09, "humaneval_sampled": 0.12},
           "distilled": {"humaneval": 0.14, "mbpp": 0.11, "humaneval_sampled": 0.15}}
    pairwise = {"win_rate_compliant": 0.42, "win_rate_distilled": 0.46, "ties": 0.12}
    guardrail = {"compliant": {"fim_middle_loss": 0.60}, "distilled": {"fim_middle_loss": 0.61},
                 "base": {"fim_middle_loss": 0.5848}}
    rep = eic.assemble_report(arm, pairwise, guardrail)
    assert rep["arms"]["distilled"]["humaneval"] == 0.14
    assert rep["distill_gap"]["humaneval"] == 0.14 - 0.10   # distilled - compliant
    assert rep["guardrail_fim_delta"]["compliant"] == 0.60 - 0.5848
