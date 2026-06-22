from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote

PHASE_CONTENT = Path("site/content/phases.json")
SYNOPSES_CONTENT = Path("site/content/synopses")
PAPER_MANIFEST = Path("papers/manifest.json")
SITE_DIST = Path("site/dist")
EVAL_RUNS = Path("runs/evals")
MARKDOWN_ALLOWED_DIRS = {"ops", "papers", "plans"}
MARKDOWN_ALLOWED_ROOT_FILES = {"AGENTS.md", "README.md"}

SPECIAL_PAPER_IDS = {
    "Measuring Massive Multitask Language Understanding": "mmlu",
    "Evaluating Large Language Models Trained on Code": "humaneval-codex",
    "Holistic Evaluation of Language Models": "helm",
    (
        "Beyond the Imitation Game: Quantifying and extrapolating the capabilities "
        "of language models"
    ): "big-bench",
    "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference": "chatbot-arena",
}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    keep = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            keep.append(char)
            previous_dash = False
        elif not previous_dash:
            keep.append("-")
            previous_dash = True
    return "".join(keep).strip("-")


def paper_id_for(entry: dict[str, Any]) -> str:
    title = str(entry.get("title", ""))
    return SPECIAL_PAPER_IDS.get(title, slugify(title))


def load_papers(project_root: Path) -> list[dict[str, Any]]:
    manifest = read_json(project_root / PAPER_MANIFEST, [])
    papers = []
    for entry in manifest:
        topic = str(entry["topic"])
        filename = str(entry["filename"])
        papers.append(
            {
                "id": paper_id_for(entry),
                "topic": topic,
                "title": entry["title"],
                "authors": entry["authors"],
                "year": entry["year"],
                "sourceUrl": entry["source_url"],
                "pdfUrl": f"/papers/{topic}/{filename}",
                "filename": filename,
            }
        )
    return papers


def load_synopses(project_root: Path) -> dict[str, Any]:
    synopses_dir = project_root / SYNOPSES_CONTENT
    if not synopses_dir.exists():
        return {}

    synopses: dict[str, Any] = {}
    for path in sorted(synopses_dir.glob("*.json")):
        synopses.update(read_json(path, {}))
    return synopses


def artifact_url(path: str) -> str:
    if path.startswith("/artifacts/"):
        return path
    return f"/artifacts/{path.lstrip('/')}"


def load_eval_runs(project_root: Path) -> list[dict[str, Any]]:
    runs_root = project_root / EVAL_RUNS
    if not runs_root.exists():
        return []

    runs = []
    for summary_path in sorted(runs_root.glob("*/summary.json"), reverse=True):
        summary = read_json(summary_path, {})
        run_id = str(summary.get("id") or summary_path.parent.name)
        artifact_paths = summary.get("artifactPaths", [])
        runs.append(
            {
                "id": run_id,
                "phaseId": summary.get("phaseId", "phase-0"),
                "model": summary.get("model", "unknown"),
                "suite": summary.get("suite", summary_path.parent.name),
                "createdAt": summary.get("createdAt", ""),
                "metrics": summary.get("metrics", {}),
                "artifactPaths": [artifact_url(str(path)) for path in artifact_paths],
            }
        )
    return runs


def validate_state(state: dict[str, Any]) -> None:
    """Fail loudly on broken cross-references.

    The dashboard resolves ``readingPaperIds`` and synopsis keys against paper
    ids. A typo would otherwise be silently dropped by the client, so a paper
    would just disappear from a phase with no error. We would rather the console
    refuse to load and say exactly what is wrong.
    """
    paper_ids = {str(paper["id"]) for paper in state["papers"]}
    problems: list[str] = []

    for phase in state["phases"]:
        phase_id = phase.get("id", "<unknown phase>")
        for paper_id in phase.get("readingPaperIds", []):
            if paper_id not in paper_ids:
                problems.append(
                    f"phase '{phase_id}' references unknown paper id '{paper_id}'"
                )

    for synopsis_id, synopsis in state["synopses"].items():
        if synopsis_id not in paper_ids:
            problems.append(
                f"synopsis '{synopsis_id}' does not match any paper id"
            )
        declared = synopsis.get("paperId")
        if declared is not None and declared != synopsis_id:
            problems.append(
                f"synopsis '{synopsis_id}' has mismatched paperId '{declared}'"
            )

    if problems:
        raise ValueError(
            "Microlab content validation failed:\n  - " + "\n  - ".join(problems)
        )


def load_state(project_root: Path) -> dict[str, Any]:
    state = {
        "phases": read_json(project_root / PHASE_CONTENT, []),
        "papers": load_papers(project_root),
        "synopses": load_synopses(project_root),
        "evalRuns": load_eval_runs(project_root),
    }
    validate_state(state)
    return state


def title_from_markdown(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def resolve_safe_path(root: Path, requested_path: str) -> Path:
    decoded = unquote(requested_path)
    if decoded.startswith("/") or decoded.startswith("\\"):
        raise ValueError("unsafe path")

    parts = Path(decoded).parts
    if not parts or any(part in {"..", ""} for part in parts):
        raise ValueError("unsafe path")

    root = root.resolve()
    candidate = (root / decoded).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("unsafe path") from exc
    return candidate


def resolve_markdown_path(project_root: Path, requested_path: str) -> Path:
    decoded = unquote(requested_path)
    if decoded.startswith("/") or decoded.startswith("\\"):
        raise ValueError("unsafe markdown path")

    requested = Path(decoded)
    parts = requested.parts
    if not parts or any(part in {"..", ""} for part in parts):
        raise ValueError("unsafe markdown path")
    if requested.suffix.lower() != ".md":
        raise ValueError("only markdown files can be rendered")

    root = project_root.resolve()
    candidate = (root / requested).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("unsafe markdown path") from exc

    if len(relative.parts) == 1 and relative.name in MARKDOWN_ALLOWED_ROOT_FILES:
        return candidate
    if relative.parts[0] in MARKDOWN_ALLOWED_DIRS:
        return candidate
    if len(relative.parts) >= 3 and relative.parts[:2] == ("runs", "evals"):
        return candidate

    raise ValueError("markdown path is not published")


def load_markdown_document(project_root: Path, requested_path: str) -> dict[str, str]:
    path = resolve_markdown_path(project_root, requested_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)

    content = path.read_text(encoding="utf-8")
    relative_path = path.relative_to(project_root.resolve()).as_posix()
    fallback = path.stem.replace("-", " ").replace("_", " ").title()
    return {
        "path": relative_path,
        "title": title_from_markdown(content, fallback),
        "content": content,
    }


def resolve_artifact_path(project_root: Path, requested_path: str) -> Path:
    candidate = resolve_safe_path(project_root, requested_path)
    relative = candidate.relative_to(project_root.resolve())
    if len(relative.parts) < 2 or relative.parts[:2] != ("runs", "evals"):
        raise ValueError("unsafe path")
    return candidate


def cards_path(project_root: Path, paper_id: str) -> Path:
    return project_root / "content" / "papers" / paper_id / "cards.json"


def read_cards(project_root: Path, paper_id: str) -> list[dict[str, Any]]:
    path = cards_path(project_root, paper_id)
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("cards", []))


def all_cards(project_root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for paper in load_papers(project_root):
        for card in read_cards(project_root, paper["id"]):
            result.append(
                {
                    "id": card["id"],
                    "paperId": paper["id"],
                    "paperTitle": paper["title"],
                    "question": card["question"],
                    "answer": card["answer"],
                }
            )
    return result


def valid_paper_ids(project_root: Path) -> set[str]:
    return {paper["id"] for paper in load_papers(project_root)}


def notes_path(project_root: Path, paper_id: str) -> Path:
    return project_root / "content" / "papers" / paper_id / "notes.md"


def read_notes(project_root: Path, paper_id: str) -> str:
    path = notes_path(project_root, paper_id)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_notes(project_root: Path, paper_id: str, body: str) -> None:
    path = notes_path(project_root, paper_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def overview_path(project_root: Path, paper_id: str) -> Path:
    return project_root / "content" / "papers" / paper_id / "overview.json"


def read_overview(project_root: Path, paper_id: str) -> dict[str, Any] | None:
    path = overview_path(project_root, paper_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def public_library(project_root: Path) -> dict[str, Any]:
    papers = load_papers(project_root)
    by_id = {p["id"]: p for p in papers}

    def entry(paper: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": paper["id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "year": paper["year"],
            "topic": paper["topic"],
            "sourceUrl": paper["sourceUrl"],
            "pdfUrl": f"/public/pdf/{paper['id']}",
            "overview": read_overview(project_root, paper["id"]),
        }

    phases_raw = read_json(project_root / PHASE_CONTENT, [])
    used: set[str] = set()
    phases = []
    for phase in phases_raw:
        items = []
        for pid in phase.get("readingPaperIds", []):
            paper = by_id.get(pid)
            if paper is not None and pid not in used:
                items.append(entry(paper))
                used.add(pid)
        if items:
            phases.append({"id": phase["id"], "title": phase["title"], "papers": items})
    additional = [entry(p) for p in papers if p["id"] not in used]
    return {"phases": phases, "additional": additional}


def public_pdf_path(project_root: Path, paper_id: str) -> Path | None:
    manifest = read_json(project_root / PAPER_MANIFEST, [])
    for raw in manifest:
        if paper_id_for(raw) == paper_id:
            candidate = project_root / "papers" / str(raw["topic"]) / str(raw["filename"])
            return candidate if candidate.exists() else None
    return None
