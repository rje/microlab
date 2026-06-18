# Phase 0 Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, reproducible evaluation harness that can score local LLM generations before any Microlab training work begins.

**Architecture:** The harness is a Python package under `src/microlab/evals/` with focused modules for suite loading, model backends, metrics, run execution, and report writing. It uses JSONL eval suites, deterministic test backends for fast development, an Ollama backend for the first real local baselines, and a Hugging Face causal-LM backend for later direct-transformers comparisons.

**Tech Stack:** Python 3.11, the `microlab` conda environment, Ollama local API, PyTorch, Transformers, pytest, ruff, JSONL, Markdown reports.

---

## Phase Scope

This phase builds evaluation infrastructure, not training code. It should answer four questions:

1. Can we define repeatable eval tasks in a simple file format?
2. Can we run the same eval suite against different model backends?
3. Can we score exact answers, containment, regex checks, valid JSON, and simple JSON-field checks?
4. Can we save run artifacts that make future model comparisons honest?

The first production baseline should use the local Ollama model `qwen3.6:27b`. After that baseline works, run the same suite against `qwen3.6:35b` as a comparison. Keep the Hugging Face backend available for later experiments, but do not make Phase 0 depend on downloading a model from Hugging Face.

## Required Environment Rule

All Python commands must use the project conda environment:

```bash
/home/rje/anaconda3/bin/conda run -n microlab python ...
/home/rje/anaconda3/bin/conda run -n microlab pytest
/home/rje/anaconda3/bin/conda run -n microlab ruff check .
```

Do not use `base`, system Python, or bare `pip`.

## File Structure

Create or modify these files in `~/src/python/microlab`:

- Create: `.gitignore` - ignores caches, model artifacts, datasets, and generated runs.
- Create: `pyproject.toml` - package metadata, pytest config, ruff config.
- Create: `src/microlab/__init__.py` - package marker.
- Create: `src/microlab/evals/__init__.py` - eval package exports.
- Create: `src/microlab/evals/schema.py` - dataclasses and validation for tasks, checks, model outputs, and scored results.
- Create: `src/microlab/evals/suites.py` - JSONL suite loader and writer.
- Create: `src/microlab/evals/metrics.py` - scoring functions.
- Create: `src/microlab/evals/backends.py` - model backend interface, fixture backend, Ollama backend, and Hugging Face backend.
- Create: `src/microlab/evals/runner.py` - orchestration for loading tasks, generating answers, scoring, and saving records.
- Create: `src/microlab/evals/report.py` - aggregate summary and Markdown report generation.
- Create: `src/microlab/evals/cli.py` - command line entry point.
- Create: `evals/suites/smoke.jsonl` - tiny deterministic suite for fast tests.
- Create: `evals/suites/phase0-core.jsonl` - initial human-readable LLM eval suite.
- Create: `configs/eval/smoke-fixture.json` - deterministic backend config.
- Create: `configs/eval/ollama-qwen3_6_27b.json` - first real local model baseline config.
- Create: `configs/eval/ollama-qwen3_6_35b.json` - larger local model comparison config.
- Create: `scripts/run_phase0_smoke.sh` - smoke run command.
- Create: `scripts/run_phase0_baseline.sh` - local Ollama 27B baseline command.
- Create: `scripts/run_phase0_ollama_35b.sh` - local Ollama 35B comparison command.
- Create: `notes/phase0-reading.md` - reading notes template for the evaluation papers.
- Create: `tests/evals/test_schema.py` - schema validation tests.
- Create: `tests/evals/test_suites.py` - JSONL loader tests.
- Create: `tests/evals/test_metrics.py` - metric behavior tests.
- Create: `tests/evals/test_runner.py` - end-to-end fixture backend test.
- Create: `tests/evals/test_cli.py` - CLI smoke test.

Generated files after runs:

- `runs/evals/phase0-*/config.json`
- `runs/evals/phase0-*/records.jsonl`
- `runs/evals/phase0-*/summary.json`
- `runs/evals/phase0-*/report.md`

## Dataset Format

Each eval task is one JSON object per line:

```json
{
  "id": "json-valid-001",
  "category": "json",
  "prompt": "Return exactly one JSON object with an integer field named answer set to 4.",
  "checks": [
    {"type": "json_valid"},
    {"type": "json_field_equals", "path": "answer", "value": 4}
  ],
  "max_new_tokens": 64,
  "metadata": {"skill": "structured-output"}
}
```

Supported check types for Phase 0:

- `exact_match`
- `contains`
- `regex`
- `json_valid`
- `json_field_equals`

## Task 1: Initialize Project Hygiene

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`

- [ ] **Step 1: Initialize git repository if needed**

Run:

```bash
cd ~/src/python/microlab
git rev-parse --is-inside-work-tree || git init
```

Expected: either `true` or output from `git init`.

- [ ] **Step 2: Create `.gitignore`**

Create `.gitignore` with:

```gitignore
__pycache__/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.ipynb_checkpoints/
*.pyc

.env
.venv/
venv/

runs/
data/raw/
data/processed/
models/
checkpoints/
wandb/

*.log
*.tmp
```

- [ ] **Step 3: Create `pyproject.toml`**

Create `pyproject.toml` with:

```toml
[project]
name = "microlab"
version = "0.1.0"
description = "Single-GPU LLM training and evaluation lab"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

- [ ] **Step 4: Run formatting/lint discovery**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab ruff check .
```

Expected: ruff may report no Python files yet, or only issues in future-created files. There should be no conda/base errors.

- [ ] **Step 5: Commit**

Run:

```bash
cd ~/src/python/microlab
git add .gitignore pyproject.toml AGENTS.md environment.yml plans/
git commit -m "chore: initialize microlab project hygiene"
```

Expected: commit succeeds. If git user identity is not configured, set local repo identity with `git config user.name "rje"` and `git config user.email "rje@rje.ai"`, then rerun the commit.

## Task 2: Define Core Schema

**Files:**
- Create: `src/microlab/__init__.py`
- Create: `src/microlab/evals/__init__.py`
- Create: `src/microlab/evals/schema.py`
- Test: `tests/evals/test_schema.py`

- [ ] **Step 1: Create package directories**

Run:

```bash
cd ~/src/python/microlab
mkdir -p src/microlab/evals tests/evals
touch src/microlab/__init__.py src/microlab/evals/__init__.py
```

- [ ] **Step 2: Write failing schema tests**

Create `tests/evals/test_schema.py`:

```python
import pytest

from microlab.evals.schema import Check, EvalTask


def test_eval_task_accepts_valid_payload():
    task = EvalTask.from_dict(
        {
            "id": "exact-001",
            "category": "math",
            "prompt": "Answer only with 4.",
            "checks": [{"type": "exact_match", "value": "4"}],
            "max_new_tokens": 8,
            "metadata": {"difficulty": "smoke"},
        }
    )

    assert task.id == "exact-001"
    assert task.checks == [Check(type="exact_match", value="4")]
    assert task.max_new_tokens == 8


def test_eval_task_rejects_missing_checks():
    with pytest.raises(ValueError, match="at least one check"):
        EvalTask.from_dict(
            {
                "id": "bad-001",
                "category": "math",
                "prompt": "Answer only with 4.",
                "checks": [],
            }
        )


def test_eval_task_rejects_unknown_check_type():
    with pytest.raises(ValueError, match="unsupported check type"):
        EvalTask.from_dict(
            {
                "id": "bad-002",
                "category": "math",
                "prompt": "Answer only with 4.",
                "checks": [{"type": "semantic_vibes", "value": "good"}],
            }
        )
```

- [ ] **Step 3: Run the failing schema tests**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_schema.py -v
```

Expected: FAIL because `microlab.evals.schema` does not exist.

- [ ] **Step 4: Implement schema dataclasses**

Create `src/microlab/evals/schema.py`:

```python
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
    def from_dict(cls, payload: dict[str, Any]) -> "Check":
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
    def from_dict(cls, payload: dict[str, Any]) -> "EvalTask":
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
```

- [ ] **Step 5: Run schema tests**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_schema.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
cd ~/src/python/microlab
git add src/microlab tests/evals/test_schema.py
git commit -m "feat: add evaluation schema"
```

## Task 3: Implement JSONL Suite Loading

**Files:**
- Create: `src/microlab/evals/suites.py`
- Test: `tests/evals/test_suites.py`

- [ ] **Step 1: Write failing suite loader tests**

Create `tests/evals/test_suites.py`:

```python
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
    suite_path.write_text('{"id": "ok"}\nnot json\n', encoding="utf-8")

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
```

- [ ] **Step 2: Run failing suite tests**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_suites.py -v
```

Expected: FAIL because `microlab.evals.suites` does not exist.

- [ ] **Step 3: Implement suite loading**

Create `src/microlab/evals/suites.py`:

```python
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
                raise ValueError(f"{suite_path}: invalid task on line {line_number}: {error}") from error
    if not tasks:
        raise ValueError(f"{suite_path}: suite contains no tasks")
    return tasks


def write_suite(path: str | Path, tasks: list[EvalTask]) -> None:
    suite_path = Path(path)
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    with suite_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), sort_keys=True) + "\n")
```

- [ ] **Step 4: Run suite tests**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_suites.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
cd ~/src/python/microlab
git add src/microlab/evals/suites.py tests/evals/test_suites.py
git commit -m "feat: load evaluation suites from jsonl"
```

## Task 4: Implement Metrics

**Files:**
- Create: `src/microlab/evals/metrics.py`
- Test: `tests/evals/test_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Create `tests/evals/test_metrics.py`:

```python
from microlab.evals.metrics import score_text
from microlab.evals.schema import Check, EvalTask, ModelOutput


def make_task(checks):
    return EvalTask(
        id="task-001",
        category="test",
        prompt="Prompt",
        checks=checks,
    )


def make_output(text):
    return ModelOutput(task_id="task-001", text=text, latency_seconds=0.01)


def test_exact_match_normalizes_whitespace():
    results = score_text(make_task([Check(type="exact_match", value="hello world")]), make_output(" hello world\n"))
    assert results[0].passed


def test_contains_is_case_insensitive_when_flagged():
    results = score_text(
        make_task([Check(type="contains", value="Paris", flags=["ignore_case"])]),
        make_output("the answer is paris"),
    )
    assert results[0].passed


def test_regex_supports_multiline_output():
    results = score_text(
        make_task([Check(type="regex", value=r"def add\\(a, b\\):\\n\\s+return a \\+ b")]),
        make_output("def add(a, b):\n    return a + b"),
    )
    assert results[0].passed


def test_json_valid_passes_for_valid_json():
    results = score_text(make_task([Check(type="json_valid")]), make_output('{"answer": 4}'))
    assert results[0].passed


def test_json_field_equals_reads_dot_path():
    results = score_text(
        make_task([Check(type="json_field_equals", path="answer.value", value=4)]),
        make_output('{"answer": {"value": 4}}'),
    )
    assert results[0].passed


def test_json_field_equals_fails_on_missing_path():
    results = score_text(
        make_task([Check(type="json_field_equals", path="answer.value", value=4)]),
        make_output('{"answer": {}}'),
    )
    assert not results[0].passed
    assert "missing path" in results[0].message
```

- [ ] **Step 2: Run failing metric tests**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_metrics.py -v
```

Expected: FAIL because `microlab.evals.metrics` does not exist.

- [ ] **Step 3: Implement metrics**

Create `src/microlab/evals/metrics.py`:

```python
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
        return CheckResult(check=check, passed=passed, message=f"expected exact match: {expected!r}")

    if check.type == "contains":
        expected = str(check.value)
        haystack = _maybe_casefold(text, check)
        needle = _maybe_casefold(expected, check)
        passed = needle in haystack
        return CheckResult(check=check, passed=passed, message=f"expected output to contain: {expected!r}")

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
```

- [ ] **Step 4: Run metric tests**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_metrics.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
cd ~/src/python/microlab
git add src/microlab/evals/metrics.py tests/evals/test_metrics.py
git commit -m "feat: add phase zero eval metrics"
```

## Task 5: Implement Model Backends

**Files:**
- Create: `src/microlab/evals/backends.py`
- Test: `tests/evals/test_backends.py`

- [ ] **Step 1: Write failing backend tests**

Create `tests/evals/test_backends.py`:

```python
from microlab.evals.backends import FixtureBackend, OllamaBackend, create_backend
from microlab.evals.schema import EvalTask


def test_fixture_backend_returns_configured_answer():
    backend = FixtureBackend({"task-001": "4"})
    task = EvalTask(
        id="task-001",
        category="math",
        prompt="Answer only with 4.",
        checks=[],
    )

    output = backend.generate(task)

    assert output.task_id == "task-001"
    assert output.text == "4"
    assert output.latency_seconds >= 0


def test_create_backend_builds_fixture_backend():
    backend = create_backend({"type": "fixture", "answers": {"task-001": "4"}})
    assert isinstance(backend, FixtureBackend)


def test_create_backend_builds_ollama_backend():
    backend = create_backend(
        {
            "type": "ollama",
            "model": "qwen3.6:27b",
            "host": "http://127.0.0.1:11434",
            "temperature": 0,
        }
    )

    assert isinstance(backend, OllamaBackend)
    assert backend.model == "qwen3.6:27b"
```

- [ ] **Step 2: Run failing backend tests**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_backends.py -v
```

Expected: FAIL because `microlab.evals.backends` does not exist.

- [ ] **Step 3: Implement backends**

Create `src/microlab/evals/backends.py`:

```python
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from microlab.evals.schema import EvalTask, ModelOutput


class ModelBackend(ABC):
    @abstractmethod
    def generate(self, task: EvalTask) -> ModelOutput:
        raise NotImplementedError


class FixtureBackend(ModelBackend):
    def __init__(self, answers: dict[str, str]):
        self.answers = answers

    def generate(self, task: EvalTask) -> ModelOutput:
        start = time.perf_counter()
        text = self.answers.get(task.id, "")
        return ModelOutput(
            task_id=task.id,
            text=text,
            latency_seconds=time.perf_counter() - start,
        )


class OllamaBackend(ModelBackend):
    def __init__(
        self,
        model: str,
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
        timeout_seconds: int = 600,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def generate(self, task: EvalTask) -> ModelOutput:
        start = time.perf_counter()
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": task.prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": task.max_new_tokens,
                },
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("response", "")).strip()
        return ModelOutput(
            task_id=task.id,
            text=text,
            latency_seconds=time.perf_counter() - start,
            prompt_tokens=payload.get("prompt_eval_count"),
            completion_tokens=payload.get("eval_count"),
        )


class HuggingFaceCausalLMBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        dtype = torch_dtype
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    def generate(self, task: EvalTask) -> ModelOutput:
        start = time.perf_counter()
        messages = [{"role": "user", "content": task.prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = task.prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=task.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion_ids = generated[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return ModelOutput(
            task_id=task.id,
            text=text,
            latency_seconds=time.perf_counter() - start,
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            completion_tokens=int(completion_ids.shape[-1]),
        )


def create_backend(config: dict[str, Any]) -> ModelBackend:
    backend_type = config.get("type")
    if backend_type == "fixture":
        return FixtureBackend(dict(config.get("answers", {})))
    if backend_type == "ollama":
        return OllamaBackend(
            model=str(config["model"]),
            host=str(config.get("host", "http://127.0.0.1:11434")),
            temperature=float(config.get("temperature", 0.0)),
            timeout_seconds=int(config.get("timeout_seconds", 600)),
        )
    if backend_type == "hf_causal_lm":
        return HuggingFaceCausalLMBackend(
            model_id=str(config["model_id"]),
            device_map=str(config.get("device_map", "auto")),
            torch_dtype=str(config.get("torch_dtype", "auto")),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
    raise ValueError(f"unsupported backend type: {backend_type}")
```

- [ ] **Step 4: Run backend tests**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_backends.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
cd ~/src/python/microlab
git add src/microlab/evals/backends.py tests/evals/test_backends.py
git commit -m "feat: add eval model backends"
```

## Task 6: Implement Runner and Artifact Writing

**Files:**
- Create: `src/microlab/evals/runner.py`
- Test: `tests/evals/test_runner.py`

- [ ] **Step 1: Write failing runner test**

Create `tests/evals/test_runner.py`:

```python
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
```

- [ ] **Step 2: Run failing runner test**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_runner.py -v
```

Expected: FAIL because `microlab.evals.runner` does not exist.

- [ ] **Step 3: Implement runner**

Create `src/microlab/evals/runner.py`:

```python
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microlab.evals.backends import ModelBackend
from microlab.evals.metrics import score_text
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
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
```

- [ ] **Step 4: Run runner test**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_runner.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
cd ~/src/python/microlab
git add src/microlab/evals/runner.py tests/evals/test_runner.py
git commit -m "feat: run eval suites and save artifacts"
```

## Task 7: Implement Markdown Reports

**Files:**
- Create: `src/microlab/evals/report.py`
- Modify: `src/microlab/evals/runner.py`
- Test: `tests/evals/test_report.py`

- [ ] **Step 1: Write failing report test**

Create `tests/evals/test_report.py`:

```python
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
```

- [ ] **Step 2: Run failing report test**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_report.py -v
```

Expected: FAIL because `microlab.evals.report` does not exist.

- [ ] **Step 3: Implement report writer**

Create `src/microlab/evals/report.py`:

```python
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
```

- [ ] **Step 4: Modify runner to write report**

Update `src/microlab/evals/runner.py` by adding:

```python
from microlab.evals.report import write_markdown_report
```

Then add this before `return summary`:

```python
    write_markdown_report(out)
```

- [ ] **Step 5: Run report and runner tests**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_report.py tests/evals/test_runner.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
cd ~/src/python/microlab
git add src/microlab/evals/report.py src/microlab/evals/runner.py tests/evals/test_report.py tests/evals/test_runner.py
git commit -m "feat: write eval markdown reports"
```

## Task 8: Add CLI

**Files:**
- Create: `src/microlab/evals/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/evals/test_cli.py`

- [ ] **Step 1: Write failing CLI test**

Create `tests/evals/test_cli.py`:

```python
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "pass_rate=1.000" in result.stdout
    assert (output_dir / "report.md").exists()
```

- [ ] **Step 2: Run failing CLI test**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_cli.py -v
```

Expected: FAIL because `microlab.evals.cli` does not exist.

- [ ] **Step 3: Implement CLI**

Create `src/microlab/evals/cli.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from microlab.evals.backends import create_backend
from microlab.evals.runner import run_eval_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Microlab evaluation suite")
    parser.add_argument("--suite", required=True, help="Path to JSONL eval suite")
    parser.add_argument("--config", required=True, help="Path to JSON eval config")
    parser.add_argument("--output-dir", required=True, help="Directory for run artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    backend = create_backend(config["backend"])
    summary = run_eval_suite(
        suite_path=args.suite,
        backend=backend,
        output_dir=args.output_dir,
        run_config=config,
    )
    print(
        f"total={summary['total']} passed={summary['passed']} "
        f"failed={summary['failed']} pass_rate={summary['pass_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add console script to `pyproject.toml`**

Append this to `pyproject.toml`:

```toml
[project.scripts]
microlab-eval = "microlab.evals.cli:main"
```

- [ ] **Step 5: Run CLI test**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
cd ~/src/python/microlab
git add src/microlab/evals/cli.py pyproject.toml tests/evals/test_cli.py
git commit -m "feat: add eval command line runner"
```

## Task 9: Add Initial Eval Suites and Configs

**Files:**
- Create: `evals/suites/smoke.jsonl`
- Create: `evals/suites/phase0-core.jsonl`
- Create: `configs/eval/smoke-fixture.json`
- Create: `configs/eval/ollama-qwen3_6_27b.json`
- Create: `configs/eval/ollama-qwen3_6_35b.json`

- [ ] **Step 1: Create suite and config directories**

Run:

```bash
cd ~/src/python/microlab
mkdir -p evals/suites configs/eval
```

- [ ] **Step 2: Create smoke suite**

Create `evals/suites/smoke.jsonl`:

```jsonl
{"id":"smoke-exact-001","category":"math","prompt":"Answer only with 4.","checks":[{"type":"exact_match","value":"4"}],"max_new_tokens":8,"metadata":{"skill":"exact-answer"}}
{"id":"smoke-json-001","category":"json","prompt":"Return exactly one JSON object with an integer field named answer set to 4.","checks":[{"type":"json_valid"},{"type":"json_field_equals","path":"answer","value":4}],"max_new_tokens":64,"metadata":{"skill":"structured-output"}}
```

- [ ] **Step 3: Create core Phase 0 suite**

Create `evals/suites/phase0-core.jsonl`:

```jsonl
{"id":"exact-math-001","category":"math","prompt":"Answer only with the final integer: What is 17 + 25?","checks":[{"type":"exact_match","value":"42"}],"max_new_tokens":16,"metadata":{"skill":"arithmetic"}}
{"id":"exact-math-002","category":"math","prompt":"Answer only with the final integer: What is 9 * 8?","checks":[{"type":"exact_match","value":"72"}],"max_new_tokens":16,"metadata":{"skill":"arithmetic"}}
{"id":"json-valid-001","category":"json","prompt":"Return exactly one JSON object with fields city and country for Paris, France.","checks":[{"type":"json_valid"},{"type":"json_field_equals","path":"city","value":"Paris"},{"type":"json_field_equals","path":"country","value":"France"}],"max_new_tokens":96,"metadata":{"skill":"structured-output"}}
{"id":"instruction-001","category":"instruction","prompt":"Write one sentence that contains the words alpha and omega. Do not use a list.","checks":[{"type":"contains","value":"alpha","flags":["ignore_case"]},{"type":"contains","value":"omega","flags":["ignore_case"]}],"max_new_tokens":64,"metadata":{"skill":"instruction-following"}}
{"id":"regex-code-001","category":"code","prompt":"Write a Python function named add that returns the sum of parameters a and b. Output only code.","checks":[{"type":"regex","value":"def add\\(a, b\\):\\n\\s+return a \\+ b"}],"max_new_tokens":96,"metadata":{"skill":"code-format"}}
{"id":"format-001","category":"format","prompt":"Answer with exactly this text and nothing else: MICROLAB_READY","checks":[{"type":"exact_match","value":"MICROLAB_READY"}],"max_new_tokens":16,"metadata":{"skill":"format-control"}}
```

- [ ] **Step 4: Create fixture config**

Create `configs/eval/smoke-fixture.json`:

```json
{
  "name": "smoke-fixture",
  "backend": {
    "type": "fixture",
    "answers": {
      "smoke-exact-001": "4",
      "smoke-json-001": "{\"answer\": 4}"
    }
  }
}
```

- [ ] **Step 5: Create first local Ollama baseline config**

Create `configs/eval/ollama-qwen3_6_27b.json`:

```json
{
  "name": "ollama-qwen3.6-27b",
  "backend": {
    "type": "ollama",
    "model": "qwen3.6:27b",
    "host": "http://127.0.0.1:11434",
    "temperature": 0,
    "timeout_seconds": 600
  }
}
```

- [ ] **Step 6: Create larger local Ollama comparison config**

Create `configs/eval/ollama-qwen3_6_35b.json`:

```json
{
  "name": "ollama-qwen3.6-35b",
  "backend": {
    "type": "ollama",
    "model": "qwen3.6:35b",
    "host": "http://127.0.0.1:11434",
    "temperature": 0,
    "timeout_seconds": 900
  }
}
```

- [ ] **Step 7: Verify Ollama has the configured models**

Run:

```bash
ollama list | grep -E 'qwen3\.6:(27b|35b)'
```

Expected: output includes both `qwen3.6:27b` and `qwen3.6:35b`.

- [ ] **Step 8: Validate suites with loader**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab python -c "from microlab.evals.suites import load_suite; print(len(load_suite('evals/suites/smoke.jsonl'))); print(len(load_suite('evals/suites/phase0-core.jsonl')))"
```

Expected:

```text
2
6
```

- [ ] **Step 9: Commit**

Run:

```bash
cd ~/src/python/microlab
git add evals/suites configs/eval
git commit -m "feat: add phase zero eval suites"
```

## Task 10: Add Run Scripts

**Files:**
- Create: `scripts/run_phase0_smoke.sh`
- Create: `scripts/run_phase0_baseline.sh`
- Create: `scripts/run_phase0_ollama_35b.sh`

- [ ] **Step 1: Create smoke run script**

Create `scripts/run_phase0_smoke.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_ID="phase0-smoke-$(date -u +%Y%m%dT%H%M%SZ)"

/home/rje/anaconda3/bin/conda run -n microlab python -m microlab.evals.cli \
  --suite evals/suites/smoke.jsonl \
  --config configs/eval/smoke-fixture.json \
  --output-dir "runs/evals/${RUN_ID}"

echo "Wrote runs/evals/${RUN_ID}"
```

- [ ] **Step 2: Create 27B baseline run script**

Create `scripts/run_phase0_baseline.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_ID="phase0-ollama-qwen3-6-27b-$(date -u +%Y%m%dT%H%M%SZ)"

/home/rje/anaconda3/bin/conda run -n microlab python -m microlab.evals.cli \
  --suite evals/suites/phase0-core.jsonl \
  --config configs/eval/ollama-qwen3_6_27b.json \
  --output-dir "runs/evals/${RUN_ID}"

echo "Wrote runs/evals/${RUN_ID}"
```

- [ ] **Step 3: Create 35B comparison run script**

Create `scripts/run_phase0_ollama_35b.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_ID="phase0-ollama-qwen3-6-35b-$(date -u +%Y%m%dT%H%M%SZ)"

/home/rje/anaconda3/bin/conda run -n microlab python -m microlab.evals.cli \
  --suite evals/suites/phase0-core.jsonl \
  --config configs/eval/ollama-qwen3_6_35b.json \
  --output-dir "runs/evals/${RUN_ID}"

echo "Wrote runs/evals/${RUN_ID}"
```

- [ ] **Step 4: Make scripts executable**

Run:

```bash
cd ~/src/python/microlab
chmod +x scripts/run_phase0_smoke.sh scripts/run_phase0_baseline.sh scripts/run_phase0_ollama_35b.sh
```

- [ ] **Step 5: Run smoke script**

Run:

```bash
cd ~/src/python/microlab
./scripts/run_phase0_smoke.sh
```

Expected: output includes `total=2 passed=2 failed=0 pass_rate=1.000` and prints the run directory.

- [ ] **Step 6: Commit**

Run:

```bash
cd ~/src/python/microlab
git add scripts/run_phase0_smoke.sh scripts/run_phase0_baseline.sh scripts/run_phase0_ollama_35b.sh
git commit -m "feat: add phase zero eval run scripts"
```

## Task 11: Run Full Test and Lint Pass

**Files:**
- No new files.

- [ ] **Step 1: Run all tests**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run ruff**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab ruff check .
```

Expected: no lint failures.

- [ ] **Step 3: Commit any fixes**

If tests or ruff required fixes, run:

```bash
cd ~/src/python/microlab
git add src tests pyproject.toml
git commit -m "fix: clean up phase zero eval harness"
```

Expected: commit succeeds only if there were changes.

## Task 12: Run First Local Model Baseline

**Files:**
- Generated: `runs/evals/phase0-*/config.json`
- Generated: `runs/evals/phase0-*/records.jsonl`
- Generated: `runs/evals/phase0-*/summary.json`
- Generated: `runs/evals/phase0-*/report.md`

- [ ] **Step 1: Verify GPU from inside conda env**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected:

```text
True
NVIDIA RTX 6000 Ada Generation
```

- [ ] **Step 2: Verify Ollama models and API**

Run:

```bash
ollama list | grep -E 'qwen3\.6:(27b|35b)'
curl -s http://127.0.0.1:11434/api/tags | /home/rje/anaconda3/bin/conda run -n microlab python -m json.tool >/tmp/ollama-tags.json
```

Expected: `ollama list` includes both `qwen3.6:27b` and `qwen3.6:35b`; the API response parses as JSON.

- [ ] **Step 3: Run 27B baseline script**

Run:

```bash
cd ~/src/python/microlab
./scripts/run_phase0_baseline.sh
```

Expected: a run directory under `runs/evals/` with `summary.json`, `records.jsonl`, `report.md`, and `config.json`. This uses the already-installed Ollama model `qwen3.6:27b`.

- [ ] **Step 4: Inspect 27B report**

Run:

```bash
cd ~/src/python/microlab
latest_run="$(find runs/evals -maxdepth 1 -type d -name 'phase0-ollama-qwen3-6-27b-*' | sort | tail -1)"
sed -n '1,160p' "${latest_run}/report.md"
```

Expected: Markdown report shows totals, category breakdown, and failed-task details if any tasks failed.

- [ ] **Step 5: Run 35B comparison script**

Run:

```bash
cd ~/src/python/microlab
./scripts/run_phase0_ollama_35b.sh
```

Expected: a second run directory under `runs/evals/` for `qwen3.6:35b`. This run may be slower than the 27B baseline.

- [ ] **Step 6: Record baseline summary in notes**

Run this command to append measured values from the latest baseline run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab python - <<'PY'
import json
from pathlib import Path

root = Path(".")
runs = sorted((root / "runs" / "evals").glob("phase0-ollama-qwen3-6-27b-*"))
if not runs:
    raise SystemExit("no phase0 qwen3.6 27b baseline run found")
run_dir = runs[-1]
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
records = [
    json.loads(line)
    for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
failures = [record for record in records if not record["passed"]]
failure_lines = []
for record in failures:
    messages = "; ".join(check["message"] for check in record["checks"] if not check["passed"])
    failure_lines.append(f"  - {record['task']['id']}: {messages}")
if not failure_lines:
    failure_lines.append("  - None")

note = "\n".join(
    [
        "",
        "## Baseline Run",
        "",
        "- Model: Ollama qwen3.6:27b",
        "- Suite: evals/suites/phase0-core.jsonl",
        f"- Run directory: {run_dir}",
        f"- Total tasks: {summary['total']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Pass rate: {summary['pass_rate']:.3f}",
        "- Notable failures:",
        *failure_lines,
        "",
    ]
)
notes_path = root / "notes" / "phase0-reading.md"
notes_path.parent.mkdir(parents=True, exist_ok=True)
with notes_path.open("a", encoding="utf-8") as handle:
    handle.write(note)
PY
```

- [ ] **Step 7: Record 35B comparison summary in notes**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab python - <<'PY'
import json
from pathlib import Path

root = Path(".")
runs = sorted((root / "runs" / "evals").glob("phase0-ollama-qwen3-6-35b-*"))
if not runs:
    raise SystemExit("no phase0 qwen3.6 35b comparison run found")
run_dir = runs[-1]
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
note = "\n".join(
    [
        "",
        "## 35B Comparison Run",
        "",
        "- Model: Ollama qwen3.6:35b",
        "- Suite: evals/suites/phase0-core.jsonl",
        f"- Run directory: {run_dir}",
        f"- Total tasks: {summary['total']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Pass rate: {summary['pass_rate']:.3f}",
        "",
    ]
)
notes_path = root / "notes" / "phase0-reading.md"
notes_path.parent.mkdir(parents=True, exist_ok=True)
with notes_path.open("a", encoding="utf-8") as handle:
    handle.write(note)
PY
```

- [ ] **Step 8: Commit baseline notes, not generated runs**

Run:

```bash
cd ~/src/python/microlab
git add notes/phase0-reading.md
git commit -m "docs: record phase zero baseline"
```

Expected: generated `runs/` files stay untracked because `.gitignore` excludes `runs/`.

## Task 13: Add Evaluation Reading Notes

**Files:**
- Create or modify: `notes/phase0-reading.md`

- [ ] **Step 1: Create notes directory**

Run:

```bash
cd ~/src/python/microlab
mkdir -p notes
```

- [ ] **Step 2: Create reading notes template**

Create `notes/phase0-reading.md`:

```markdown
# Phase 0 Reading Notes: Evaluation

## Reading Order

1. `papers/evaluation/2020-hendrycks-measuring-massive-multitask-language-understanding.pdf`
2. `papers/evaluation/2021-chen-evaluating-large-language-models-trained-on-code.pdf`
3. `papers/evaluation/2022-big-bench-authors-beyond-the-imitation-game.pdf`
4. `papers/evaluation/2022-liang-holistic-evaluation-of-language-models.pdf`
5. `papers/evaluation/2024-chiang-chatbot-arena.pdf`

## MMLU

- Core question:
- Method:
- What this teaches Microlab:
- What not to copy yet:

## HumanEval / Codex

- Core question:
- Method:
- What this teaches Microlab:
- What not to copy yet:

## BIG-bench

- Core question:
- Method:
- What this teaches Microlab:
- What not to copy yet:

## HELM

- Core question:
- Method:
- What this teaches Microlab:
- What not to copy yet:

## Chatbot Arena

- Core question:
- Method:
- What this teaches Microlab:
- What not to copy yet:

## Phase 0 Design Decisions

- We start with deterministic checks because they are easy to regression test.
- We store raw model outputs because aggregate scores are not enough for learning.
- We keep subjective judging out of Phase 0 and revisit it after SFT/preference-data phases.
```

- [ ] **Step 3: Commit reading notes template**

Run:

```bash
cd ~/src/python/microlab
git add notes/phase0-reading.md
git commit -m "docs: add phase zero evaluation reading notes"
```

## Task 14: Final Acceptance Check

**Files:**
- No new files.

- [ ] **Step 1: Verify required project files exist**

Run:

```bash
cd ~/src/python/microlab
test -f AGENTS.md
test -f environment.yml
test -f pyproject.toml
test -f evals/suites/smoke.jsonl
test -f evals/suites/phase0-core.jsonl
test -f configs/eval/smoke-fixture.json
test -f configs/eval/ollama-qwen3_6_27b.json
test -f configs/eval/ollama-qwen3_6_35b.json
test -x scripts/run_phase0_smoke.sh
test -x scripts/run_phase0_baseline.sh
test -x scripts/run_phase0_ollama_35b.sh
```

Expected: command exits 0.

- [ ] **Step 2: Verify tests and lint**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest -v
/home/rje/anaconda3/bin/conda run -n microlab ruff check .
```

Expected: tests pass and ruff exits 0.

- [ ] **Step 3: Verify smoke run**

Run:

```bash
cd ~/src/python/microlab
./scripts/run_phase0_smoke.sh
```

Expected: `total=2 passed=2 failed=0 pass_rate=1.000`.

- [ ] **Step 4: Verify baseline report exists**

Run:

```bash
cd ~/src/python/microlab
find runs/evals -maxdepth 2 -name report.md | sort | tail -5
```

Expected: at least one report path from the Phase 0 baseline or smoke run.

- [ ] **Step 5: Inspect git status**

Run:

```bash
cd ~/src/python/microlab
git status --short
```

Expected: no unexpected tracked-file changes. Generated `runs/` artifacts should not appear because `.gitignore` excludes them.

## Acceptance Criteria

Phase 0 is complete when:

- The project is a git repository.
- `AGENTS.md` requires the `microlab` conda environment.
- `pytest -v` passes from the `microlab` environment.
- `ruff check .` passes from the `microlab` environment.
- `./scripts/run_phase0_smoke.sh` produces a 100% pass report.
- `./scripts/run_phase0_baseline.sh` produces a real local-model report for Ollama `qwen3.6:27b`.
- `./scripts/run_phase0_ollama_35b.sh` produces a comparison report for Ollama `qwen3.6:35b`.
- `notes/phase0-reading.md` contains the evaluation reading list and the first baseline summary.
- Generated run artifacts are saved under `runs/evals/` and are not committed.

## Self-Review

- Spec coverage: This plan implements the Phase 0 overview deliverables: reproducible eval tasks, baseline results, and run-log/report format.
- Placeholder scan: The plan contains no deferred-work markers or angle-bracket placeholders.
- Type consistency: The plan consistently uses `EvalTask`, `Check`, `ModelOutput`, `CheckResult`, and `TaskResult` across schema, metrics, runner, and report modules.
- Scope check: Training, data cleaning, tokenization, reward models, and subjective LLM judging are intentionally outside Phase 0.
