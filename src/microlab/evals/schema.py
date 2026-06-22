from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SUPPORTED_CHECK_TYPES = {
    "exact_match",
    "contains",
    "regex",
    "json_valid",
    "json_field_equals",
}


@dataclass(frozen=True)
class Check:
    type: str
    value: Any | None = None
    path: str | None = None
    flags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Check:
        check_type = payload.get("type")
        if check_type not in SUPPORTED_CHECK_TYPES:
            raise ValueError(f"unsupported check type: {check_type}")
        return cls(
            type=check_type,
            value=payload.get("value"),
            path=payload.get("path"),
            flags=list(payload.get("flags", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type}
        if self.value is not None:
            result["value"] = self.value
        if self.path is not None:
            result["path"] = self.path
        if self.flags:
            result["flags"] = self.flags
        return result


@dataclass(frozen=True)
class EvalTask:
    id: str
    category: str
    prompt: str
    checks: list[Check]
    max_new_tokens: int = 128
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalTask:
        checks = [Check.from_dict(item) for item in payload.get("checks", [])]
        if not checks:
            raise ValueError("EvalTask requires at least one check")
        task_id = str(payload.get("id", "")).strip()
        category = str(payload.get("category", "")).strip()
        prompt = str(payload.get("prompt", "")).strip()
        if not task_id:
            raise ValueError("EvalTask requires id")
        if not category:
            raise ValueError("EvalTask requires category")
        if not prompt:
            raise ValueError("EvalTask requires prompt")
        return cls(
            id=task_id,
            category=category,
            prompt=prompt,
            checks=checks,
            max_new_tokens=int(payload.get("max_new_tokens", 128)),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "checks": [check.to_dict() for check in self.checks],
            "max_new_tokens": self.max_new_tokens,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ModelOutput:
    task_id: str
    text: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class CheckResult:
    check: Check
    passed: bool
    message: str


@dataclass(frozen=True)
class TaskResult:
    task: EvalTask
    output: ModelOutput
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_record(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "output": {
                "task_id": self.output.task_id,
                "text": self.output.text,
                "latency_seconds": self.output.latency_seconds,
                "prompt_tokens": self.output.prompt_tokens,
                "completion_tokens": self.output.completion_tokens,
            },
            "checks": [
                {
                    "check": check_result.check.to_dict(),
                    "passed": check_result.passed,
                    "message": check_result.message,
                }
                for check_result in self.checks
            ],
            "passed": self.passed,
        }
