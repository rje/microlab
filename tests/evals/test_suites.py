import json

import pytest

from microlab.evals.suites import load_suite, write_suite


def test_load_suite_reads_jsonl_tasks(tmp_path):
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

    tasks = load_suite(suite_path)

    assert len(tasks) == 1
    assert tasks[0].id == "exact-001"


def test_load_suite_reports_line_number_for_bad_json(tmp_path):
    suite_path = tmp_path / "bad.jsonl"
    valid_line = json.dumps(
        {
            "id": "ok-001",
            "category": "test",
            "prompt": "Answer only with 4.",
            "checks": [{"type": "exact_match", "value": "4"}],
        }
    )
    suite_path.write_text(valid_line + "\nnot json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_suite(suite_path)


def test_write_suite_round_trips(tmp_path):
    source = tmp_path / "source.jsonl"
    target = tmp_path / "target.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "contains-001",
                "category": "instruction",
                "prompt": "Name a primary color.",
                "checks": [{"type": "contains", "value": "red"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    tasks = load_suite(source)
    write_suite(target, tasks)

    assert load_suite(target) == tasks
