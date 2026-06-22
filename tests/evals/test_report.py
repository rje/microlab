import json

from microlab.evals.report import write_markdown_report


def test_write_markdown_report_includes_summary_and_failures(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps({"total": 1, "passed": 0, "failed": 1, "pass_rate": 0.0, "by_category": {}}),
        encoding="utf-8",
    )
    (run_dir / "records.jsonl").write_text(
        json.dumps(
            {
                "task": {"id": "exact-001", "category": "math", "prompt": "Answer only with 4."},
                "output": {"text": "5", "latency_seconds": 0.01},
                "checks": [{"passed": False, "message": "expected exact match"}],
                "passed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report_path = write_markdown_report(run_dir)

    report = report_path.read_text(encoding="utf-8")
    assert "# Evaluation Report" in report
    assert "Pass rate: 0.000" in report
    assert "exact-001" in report
