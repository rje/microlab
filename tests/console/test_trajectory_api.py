"""The Run log endpoint: milestone docs + completion sweeps, read from the working tree."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def _app(tmp_path):
    from microlab.console.app import create_app
    (tmp_path / "evals" / "trajectory").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    rows = [
        {"step": 500, "prompt_id": "p1", "prompt_sha": "x", "completion": "aa"},
        {"step": 2000, "prompt_id": "p1", "prompt_sha": "x", "completion": "bb"},
    ]
    (tmp_path / "evals" / "trajectory" / "demo-completions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    (tmp_path / "docs" / "demo-milestone-2000.md").write_text("# m")
    (tmp_path / "docs" / "demo-prediction.md").write_text("# p")
    (tmp_path / "evals" / "trajectory_prompts.py").write_text(
        'PROMPTS = [{"id": "p1", "n": 8, "text": "hello"}]\n')
    return create_app(tmp_path)


def test_trajectory_lists_runs_steps_and_completions(tmp_path):
    app = _app(tmp_path)
    c = app.test_client()
    with c.session_transaction() as s:
        s["authed"] = True
    r = c.get("/api/trajectory")
    assert r.status_code == 200, r.data
    d = r.get_json()
    (run,) = d["runs"]
    assert run["run"] == "demo"
    assert run["steps"] == [500, 2000]
    assert run["completions"]["p1"] == {"500": "aa", "2000": "bb"}
    # prediction doc leads, milestones follow — the reading order of the run
    assert run["docs"] == ["docs/demo-prediction.md", "docs/demo-milestone-2000.md"]
    assert d["prompts"] == [{"id": "p1", "text": "hello"}]


def test_trajectory_serves_the_sampled_sibling_alongside_greedy(tmp_path):
    """The sampled track is a separate file that must NOT collide with the frozen greedy
    view. Both maps come back, and steps present only in one still appear."""
    app = _app(tmp_path)
    sampled = [
        {"step": 2000, "prompt_id": "p1", "prompt_sha": "x", "completion": "SS",
         "decoder": "sampled"},
        {"step": 4000, "prompt_id": "p1", "prompt_sha": "x", "completion": "TT",
         "decoder": "sampled"},
    ]
    (tmp_path / "evals" / "trajectory" / "demo-completions-sampled.jsonl").write_text(
        "\n".join(json.dumps(r) for r in sampled))
    c = app.test_client()
    with c.session_transaction() as s:
        s["authed"] = True
    (run,) = c.get("/api/trajectory").get_json()["runs"]
    # greedy is untouched...
    assert run["completions"]["p1"] == {"500": "aa", "2000": "bb"}
    # ...sampled is served in parallel...
    assert run["completions_sampled"]["p1"] == {"2000": "SS", "4000": "TT"}
    # ...and steps are the union of both tracks (4000 is sampled-only).
    assert run["steps"] == [500, 2000, 4000]


def test_trajectory_omits_sampled_when_absent(tmp_path):
    """A run with no sampled sweep yet returns an empty map, never an error."""
    app = _app(tmp_path)
    c = app.test_client()
    with c.session_transaction() as s:
        s["authed"] = True
    (run,) = c.get("/api/trajectory").get_json()["runs"]
    assert run["completions_sampled"] == {}


def test_trajectory_requires_auth(tmp_path):
    app = _app(tmp_path)
    r = app.test_client().get("/api/trajectory")
    assert r.status_code in (302, 401)
