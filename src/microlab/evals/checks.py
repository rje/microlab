from __future__ import annotations

import json
import re
from typing import Any

from microlab.evals.schema import Check, CheckResult, EvalTask, ModelOutput


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _maybe_casefold(text: str, check: Check) -> str:
    if "ignore_case" in check.flags:
        return text.casefold()
    return text


def _load_json(text: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        return False, None


def _read_path(payload: Any, path: str) -> tuple[bool, Any]:
    current = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def score_check(check: Check, output: ModelOutput) -> CheckResult:
    text = output.text
    if check.type == "exact_match":
        expected = normalize_text(str(check.value))
        actual = normalize_text(text)
        passed = _maybe_casefold(actual, check) == _maybe_casefold(expected, check)
        return CheckResult(
            check=check, passed=passed, message=f"expected exact match: {expected!r}"
        )

    if check.type == "contains":
        expected = str(check.value)
        haystack = _maybe_casefold(text, check)
        needle = _maybe_casefold(expected, check)
        passed = needle in haystack
        return CheckResult(
            check=check, passed=passed, message=f"expected output to contain: {expected!r}"
        )

    if check.type == "regex":
        pattern = str(check.value)
        passed = re.search(pattern, text, flags=re.MULTILINE) is not None
        return CheckResult(check=check, passed=passed, message=f"expected regex match: {pattern!r}")

    if check.type == "json_valid":
        passed, _ = _load_json(text)
        return CheckResult(check=check, passed=passed, message="expected valid JSON")

    if check.type == "json_field_equals":
        passed, payload = _load_json(text)
        if not passed:
            return CheckResult(check=check, passed=False, message="expected valid JSON")
        if not check.path:
            return CheckResult(check=check, passed=False, message="json_field_equals requires path")
        exists, actual = _read_path(payload, check.path)
        if not exists:
            return CheckResult(check=check, passed=False, message=f"missing path: {check.path}")
        passed = actual == check.value
        return CheckResult(
            check=check,
            passed=passed,
            message=f"expected {check.path} to equal {check.value!r}; got {actual!r}",
        )

    return CheckResult(check=check, passed=False, message=f"unsupported check type: {check.type}")


def score_text(task: EvalTask, output: ModelOutput) -> list[CheckResult]:
    return [score_check(check, output) for check in task.checks]
