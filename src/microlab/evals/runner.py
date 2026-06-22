from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microlab.evals.backends import ModelBackend
from microlab.evals.checks import score_text
from microlab.evals.schema import TaskResult
from microlab.evals.suites import load_suite


def summarize_results(results: list[TaskResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_category.setdefault(result.task.category, {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(result.passed)
    for bucket in by_category.values():
        bucket["pass_rate"] = bucket["passed"] / bucket["total"] if bucket["total"] else 0.0
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "by_category": by_category,
        "created_at": datetime.now(UTC).isoformat(),
    }


def run_eval_suite(
    suite_path: str | Path,
    backend: ModelBackend,
    output_dir: str | Path,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    from microlab.evals.report import write_markdown_report

    tasks = load_suite(suite_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: list[TaskResult] = []
    records_path = out / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as records:
        for task in tasks:
            model_output = backend.generate(task)
            check_results = score_text(task, model_output)
            task_result = TaskResult(task=task, output=model_output, checks=check_results)
            results.append(task_result)
            records.write(json.dumps(task_result.to_record(), sort_keys=True) + "\n")

    summary = summarize_results(results)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown_report(out)
    return summary
