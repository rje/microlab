import json

from microlab.evals.backends import FixtureBackend
from microlab.evals.runner import run_eval_suite


def test_runner_writes_records_and_summary(tmp_path):
    suite_path = tmp_path / "suite.jsonl"
    suite_path.write_text(
        json.dumps(
            {
                "id": "exact-001",
                "category": "math",
                "prompt": "Answer only with 4.",
                "checks": [{"type": "exact_match", "value": "4"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    summary = run_eval_suite(
        suite_path=suite_path,
        backend=FixtureBackend({"exact-001": "4"}),
        output_dir=output_dir,
        run_config={"backend": {"type": "fixture"}},
    )

    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert (output_dir / "records.jsonl").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "config.json").exists()
