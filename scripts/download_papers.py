#!/usr/bin/env python3
"""Download the Microlab paper manifest into topic folders."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def is_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10_000:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def download(url: str, dest: Path) -> tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        part.unlink()

    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "4",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "20",
        "--max-time",
        "240",
        "-A",
        "microlab-paper-downloader/1.0",
        "-o",
        str(part),
        url,
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        if part.exists():
            part.unlink()
        return False, result.stderr.strip() or f"curl exited {result.returncode}"

    if not is_pdf(part):
        size = part.stat().st_size if part.exists() else 0
        if part.exists():
            part.unlink()
        return False, f"download was not a valid PDF, size={size}"

    part.replace(dest)
    return True, "downloaded"


def write_readme(root: Path, papers: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for paper in papers:
        grouped[paper["topic"]].append(paper)

    lines = [
        "# Microlab Paper Library",
        "",
        "This folder contains local PDF copies of the LLM papers referenced in the "
        "Microlab curriculum.",
        "The canonical machine-readable list is `manifest.json`.",
        "",
        "## Topics",
        "",
    ]

    for topic in sorted(grouped):
        lines.append(f"### {topic}")
        lines.append("")
        for paper in sorted(grouped[topic], key=lambda item: (item["year"], item["title"])):
            path = f"{paper['topic']}/{paper['filename']}"
            lines.append(
                f"- [{paper['title']}]({path}) ({paper['year']}) - "
                f"[source]({paper['source_url']})"
            )
        lines.append("")

    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    papers_root = root / "papers"
    manifest_path = papers_root / "manifest.json"
    papers = json.loads(manifest_path.read_text(encoding="utf-8"))

    write_readme(papers_root, papers)

    failures: list[str] = []
    downloaded = 0
    skipped = 0

    for paper in papers:
        dest = papers_root / paper["topic"] / paper["filename"]
        if is_pdf(dest):
            skipped += 1
            print(f"skip {dest.relative_to(root)}")
            continue

        ok, message = download(paper["pdf_url"], dest)
        if ok:
            downloaded += 1
            print(f"ok   {dest.relative_to(root)}")
        else:
            failures.append(f"{paper['title']} :: {paper['pdf_url']} :: {message}")
            print(f"fail {dest.relative_to(root)} :: {message}", file=sys.stderr)

    failure_path = papers_root / "download_failures.txt"
    if failures:
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
    elif failure_path.exists():
        failure_path.unlink()

    print(f"downloaded={downloaded} skipped={skipped} failures={len(failures)} total={len(papers)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
