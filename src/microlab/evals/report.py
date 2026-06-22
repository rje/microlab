from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_markdown_report(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    records = _load_jsonl(path / "records.jsonl")
    failed = [record for record in records if not record["passed"]]

    lines = [
        "# Evaluation Report",
        "",
        f"Total tasks: {summary['total']}",
        f"Passed: {summary['passed']}",
        f"Failed: {summary['failed']}",
        f"Pass rate: {summary['pass_rate']:.3f}",
        "",
        "## Categories",
        "",
    ]
    for category, bucket in sorted(summary.get("by_category", {}).items()):
        lines.append(
            f"- {category}: {bucket['passed']}/{bucket['total']} "
            f"({bucket['pass_rate']:.3f})"
        )
    if not summary.get("by_category"):
        lines.append("- No category data")

    lines.extend(["", "## Failed Tasks", ""])
    if not failed:
        lines.append("No failed tasks.")
    for record in failed:
        task = record["task"]
        output = record["output"]
        lines.extend(
            [
                f"### {task['id']}",
                "",
                f"Category: `{task['category']}`",
                "",
                "Prompt:",
                "",
                "```text",
                task["prompt"],
                "```",
                "",
                "Output:",
                "",
                "```text",
                output["text"],
                "```",
                "",
                "Checks:",
            ]
        )
        for check in record["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- {status}: {check['message']}")
        lines.append("")

    report_path = path / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
