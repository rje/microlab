import json
import subprocess
import sys


def test_cli_runs_fixture_eval(tmp_path):
    suite_path = tmp_path / "suite.jsonl"
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "run"
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
    config_path.write_text(
        json.dumps({"backend": {"type": "fixture", "answers": {"exact-001": "4"}}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "microlab.evals.cli",
            "--suite",
            str(suite_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "pass_rate=1.000" in result.stdout
    assert (output_dir / "report.md").exists()
