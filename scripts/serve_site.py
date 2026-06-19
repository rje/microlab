from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE_CONTENT = Path("site/content/phases.json")
SYNOPSES_CONTENT = Path("site/content/synopses")
PAPER_MANIFEST = Path("papers/manifest.json")
SITE_DIST = Path("site/dist")
EVAL_RUNS = Path("runs/evals")

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


def load_state(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return {
        "phases": read_json(project_root / PHASE_CONTENT, []),
        "papers": load_papers(project_root),
        "synopses": load_synopses(project_root),
        "evalRuns": load_eval_runs(project_root),
    }


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


def resolve_artifact_path(project_root: Path, requested_path: str) -> Path:
    candidate = resolve_safe_path(project_root, requested_path)
    relative = candidate.relative_to(project_root.resolve())
    if len(relative.parts) < 2 or relative.parts[:2] != ("runs", "evals"):
        raise ValueError("unsafe path")
    return candidate


class MicrolabRequestHandler(BaseHTTPRequestHandler):
    project_root = PROJECT_ROOT

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/state":
            self.send_json(load_state(self.project_root))
            return

        if path.startswith("/papers/"):
            self.send_file_from_root(self.project_root / "papers", path.removeprefix("/papers/"))
            return

        if path.startswith("/artifacts/"):
            self.send_artifact(path.removeprefix("/artifacts/"))
            return

        self.send_site_file(path)

    def send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_text(self, status: int, message: str) -> None:
        body = f"{status} {message}\n".encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error_text(404, "Not Found")
            return

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file_from_root(self, root: Path, requested_path: str) -> None:
        try:
            self.send_file(resolve_safe_path(root, requested_path))
        except ValueError:
            self.send_error_text(400, "Bad Request")

    def send_artifact(self, requested_path: str) -> None:
        try:
            self.send_file(resolve_artifact_path(self.project_root, requested_path))
        except ValueError:
            self.send_error_text(400, "Bad Request")

    def send_site_file(self, requested_path: str) -> None:
        dist_root = self.project_root / SITE_DIST
        if requested_path in {"", "/"}:
            self.send_file(dist_root / "index.html")
            return

        try:
            candidate = resolve_safe_path(dist_root, requested_path.removeprefix("/"))
        except ValueError:
            self.send_error_text(400, "Bad Request")
            return

        if candidate.exists() and candidate.is_file():
            self.send_file(candidate)
            return

        self.send_file(dist_root / "index.html")

    def log_message(self, format: str, *args: object) -> None:
        return


def make_handler(project_root: Path) -> type[MicrolabRequestHandler]:
    class ProjectRequestHandler(MicrolabRequestHandler):
        pass

    ProjectRequestHandler.project_root = project_root.resolve()
    return ProjectRequestHandler


def run_server(host: str, port: int, project_root: Path) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(project_root))
    print(f"Microlab Console serving {project_root} at http://{host}:{port}", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Microlab Console.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    run_server(args.host, args.port, args.project_root.resolve())


if __name__ == "__main__":
    main()
