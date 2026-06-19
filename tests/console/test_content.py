from __future__ import annotations

import json
from pathlib import Path

import pytest

from microlab.console import content


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_state_combines_phase_papers_synopses_and_eval_runs(tmp_path: Path):
    write_json(
        tmp_path / "site" / "content" / "phases.json",
        [
            {
                "id": "phase-0",
                "title": "Phase 0: Evaluation Harness",
                "status": "current",
                "goal": "Build a repeatable eval harness.",
                "tasks": [],
                "readingPaperIds": ["mmlu"],
            }
        ],
    )
    write_json(
        tmp_path / "site" / "content" / "synopses" / "phase-0.json",
        {"mmlu": {"paperId": "mmlu", "oneSentence": "x"}},
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

    state = content.load_state(tmp_path)

    assert state["phases"][0]["id"] == "phase-0"
    assert state["papers"][0]["id"] == "mmlu"
    assert state["papers"][0]["pdfUrl"] == "/papers/evaluation/2020-hendrycks-mmlu.pdf"
    assert state["synopses"]["mmlu"]["paperId"] == "mmlu"


def test_validate_state_rejects_unknown_reading_id(tmp_path: Path):
    write_json(
        tmp_path / "site" / "content" / "phases.json",
        [
            {
                "id": "phase-0",
                "title": "P0",
                "status": "current",
                "goal": "g",
                "tasks": [],
                "readingPaperIds": ["does-not-exist"],
            }
        ],
    )
    write_json(tmp_path / "papers" / "manifest.json", [])

    with pytest.raises(ValueError, match="unknown paper id"):
        content.load_state(tmp_path)


def test_resolve_safe_path_rejects_traversal(tmp_path: Path):
    root = tmp_path / "papers"
    allowed = root / "evaluation" / "paper.pdf"
    allowed.parent.mkdir(parents=True)
    allowed.write_bytes(b"%PDF")

    assert content.resolve_safe_path(root, "evaluation/paper.pdf") == allowed
    with pytest.raises(ValueError, match="unsafe path"):
        content.resolve_safe_path(root, "../secrets.txt")


def test_load_markdown_document_reads_allowed_markdown(tmp_path: Path):
    md = tmp_path / "plans" / "environment-setup.md"
    md.parent.mkdir(parents=True)
    md.write_text("# Environment Setup\n\nUse the env.\n", encoding="utf-8")

    document = content.load_markdown_document(tmp_path, "plans/environment-setup.md")
    assert document["title"] == "Environment Setup"
    assert document["path"] == "plans/environment-setup.md"
