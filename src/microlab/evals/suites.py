from __future__ import annotations

import json
from pathlib import Path

from microlab.evals.schema import EvalTask


def load_suite(path: str | Path) -> list[EvalTask]:
    suite_path = Path(path)
    tasks: list[EvalTask] = []
    with suite_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{suite_path}: invalid JSON on line {line_number}") from error
            try:
                tasks.append(EvalTask.from_dict(payload))
            except ValueError as error:
                raise ValueError(
                    f"{suite_path}: invalid task on line {line_number}: {error}"
                ) from error
    if not tasks:
        raise ValueError(f"{suite_path}: suite contains no tasks")
    return tasks


def write_suite(path: str | Path, tasks: list[EvalTask]) -> None:
    suite_path = Path(path)
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    with suite_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), sort_keys=True) + "\n")
