from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_server_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "serve_site.py"
    spec = importlib.util.spec_from_file_location("serve_site", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_state_combines_phase_papers_synopses_and_eval_runs(tmp_path: Path):
    module = load_server_module()

    write_json(
        tmp_path / "site" / "content" / "phases.json",
        [
            {
                "id": "phase-0",
                "title": "Phase 0: Evaluation Harness",
                "status": "current",
                "goal": "Build a repeatable eval harness.",
                "tasks": [
                    {
                        "id": "schema",
                        "title": "Define eval task schema",
                        "status": "complete",
                        "why": "Every later result depends on stable task records.",
                        "links": ["plans/phase-0.md"],
                    }
                ],
                "readingPaperIds": ["mmlu"],
            }
        ],
    )
    write_json(
        tmp_path / "site" / "content" / "synopses" / "phase-0.json",
        {
            "mmlu": {
                "paperId": "mmlu",
                "oneSentence": "A broad benchmark for multitask knowledge.",
                "coreIdeas": ["Use many subjects to expose uneven capability."],
                "whyItMatters": "It demonstrates benchmark breadth.",
                "phaseConnection": "It informs the first knowledge eval tasks.",
                "suggestedReadingFocus": ["Study category construction."],
            }
        },
    )
    write_json(
        tmp_path / "papers" / "manifest.json",
        [
            {
                "topic": "evaluation",
                "title": "Measuring Massive Multitask Language Understanding",
                "authors": "Hendrycks et al.",
                "year": 2020,
                "source_url": "https://arxiv.org/abs/2009.03300",
                "pdf_url": "https://arxiv.org/pdf/2009.03300",
                "filename": "2020-hendrycks-mmlu.pdf",
            }
        ],
    )
    write_json(
        tmp_path / "runs" / "evals" / "phase0-smoke" / "summary.json",
        {
            "id": "phase0-smoke",
            "phaseId": "phase-0",
            "model": "fixture",
            "suite": "smoke",
            "createdAt": "2026-06-18T12:00:00Z",
            "metrics": {"passRate": 1.0},
            "artifactPaths": ["runs/evals/phase0-smoke/report.md"],
        },
    )

    state = module.load_state(tmp_path)

    assert state["phases"][0]["id"] == "phase-0"
    assert state["papers"][0] == {
        "id": "mmlu",
        "topic": "evaluation",
        "title": "Measuring Massive Multitask Language Understanding",
        "authors": "Hendrycks et al.",
        "year": 2020,
        "sourceUrl": "https://arxiv.org/abs/2009.03300",
        "pdfUrl": "/papers/evaluation/2020-hendrycks-mmlu.pdf",
        "filename": "2020-hendrycks-mmlu.pdf",
    }
    assert state["synopses"]["mmlu"]["phaseConnection"].startswith("It informs")
    assert state["evalRuns"][0]["artifactPaths"] == ["/artifacts/runs/evals/phase0-smoke/report.md"]


def test_resolve_safe_path_allows_nested_files_but_rejects_traversal(tmp_path: Path):
    module = load_server_module()

    root = tmp_path / "papers"
    allowed = root / "evaluation" / "paper.pdf"
    allowed.parent.mkdir(parents=True)
    allowed.write_bytes(b"%PDF")

    resolved = module.resolve_safe_path(root, "evaluation/paper.pdf")

    assert resolved == allowed
    with pytest.raises(ValueError, match="unsafe path"):
        module.resolve_safe_path(root, "../secrets.txt")
    with pytest.raises(ValueError, match="unsafe path"):
        module.resolve_safe_path(root, "/etc/passwd")
