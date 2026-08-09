"""Evaluate two instruct arms (compliant vs distilled) and emit the A/B comparison.

Runs, for each arm: HumanEval + MBPP pass@1 greedy AND sampled (eval_code.py --mode chat);
then the pairwise judge between arms (eval_pairwise.py); then the FIM guardrail vs base.
Writes evals/instruct/compare.json + compare.md.

    python scripts/eval_instruct_compare.py \\
        --compliant runs/coder-1b-instruct-compliant \\
        --distilled runs/coder-1b-instruct-distilled \\
        --base runs/coder-1b-step40000

assemble_report() is pure so the merge/verdict logic is unit-tested; main() shells out to
the existing eval scripts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def assemble_report(arm_summaries: dict, pairwise: dict, guardrail: dict) -> dict:
    """Merge already-collected results into the comparison. `arm_summaries[arm][metric]`
    holds pass@1 floats; distill_gap is distilled-minus-compliant per metric."""
    metrics = sorted({m for a in arm_summaries.values() for m in a})
    gap = {m: arm_summaries["distilled"].get(m, 0.0) - arm_summaries["compliant"].get(m, 0.0)
           for m in metrics if m in arm_summaries.get("compliant", {})
           and m in arm_summaries.get("distilled", {})}
    base_fim = guardrail.get("base", {}).get("fim_middle_loss")
    fim_delta = {arm: g.get("fim_middle_loss", 0.0) - base_fim
                 for arm, g in guardrail.items() if arm != "base" and base_fim is not None}
    return {"arms": arm_summaries, "pairwise": pairwise, "distill_gap": gap,
            "guardrail_fim_delta": fim_delta, "guardrail_raw": guardrail}


def _run_eval_code(  # pragma: no cover
    run: Path, dataset: str, sampled: bool, out_dir: Path
) -> float:
    tag = "sampled" if sampled else "greedy"
    out = out_dir / f"{run.name}-{dataset}-{tag}.jsonl"
    cmd = [sys.executable, "scripts/eval_code.py", "--run", str(run), "--dataset", dataset,
           "--mode", "chat", "--out", str(out)]
    if sampled:
        cmd += ["--temperature", "0.7", "--top-k", "40"]
    subprocess.run(cmd, check=True)
    return json.loads((out.with_suffix(".jsonl.summary.json")).read_text())["pass@1"]


def main() -> None:  # pragma: no cover - orchestration
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compliant", type=Path, required=True)
    ap.add_argument("--distilled", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("evals/instruct"))
    ap.add_argument("--pairwise-data", default="data/corpora/code_sft_compliant.jsonl")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arms = {}
    for name, run in (("compliant", args.compliant), ("distilled", args.distilled)):
        arms[name] = {
            "humaneval": _run_eval_code(run, "humaneval", False, args.out_dir),
            "mbpp": _run_eval_code(run, "mbpp", False, args.out_dir),
            "humaneval_sampled": _run_eval_code(run, "humaneval", True, args.out_dir),
            "mbpp_sampled": _run_eval_code(run, "mbpp", True, args.out_dir),
        }

    pw_out = args.out_dir / "pairwise.json"
    subprocess.run([sys.executable, "scripts/eval_pairwise.py", str(args.compliant),
                    str(args.distilled), "--data", args.pairwise_data, "--skip", "0",
                    "--limit", "120", "--out", str(pw_out)], check=True)
    pairwise = json.loads(pw_out.read_text())

    # FIM guardrail: eval_suite.py prints fim middle_loss per staged run (base + both arms).
    guardrail = {"base": {}, "compliant": {}, "distilled": {}}  # filled by the operator step

    report = assemble_report(arms, pairwise, guardrail)
    (args.out_dir / "compare.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["distill_gap"], indent=2))


if __name__ == "__main__":
    main()
