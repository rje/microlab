"""Guards that keep the Microlab Console trustworthy as phases are added.

The console resolves phase reading lists and synopses against paper ids derived
from ``papers/manifest.json``. A broken reference would otherwise be silently
dropped in the UI, so these tests fail loudly at commit time instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from microlab.console import content

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASE_STATUSES = {"current", "planned", "complete"}
TASK_STATUSES = {"done", "active", "queued", "blocked"}


@pytest.fixture(scope="module")
def state():
    # load_state runs validate_state and raises on any broken cross-reference.
    return content.load_state(PROJECT_ROOT)


def test_project_state_loads_and_validates(state):
    assert state["phases"], "expected at least one phase"
    assert state["papers"], "expected the paper manifest to be non-empty"


def test_every_reading_id_resolves_to_a_paper(state):
    paper_ids = {paper["id"] for paper in state["papers"]}
    for phase in state["phases"]:
        for paper_id in phase["readingPaperIds"]:
            assert paper_id in paper_ids, f"{phase['id']} -> unknown paper '{paper_id}'"


def test_every_synopsis_matches_a_paper(state):
    paper_ids = {paper["id"] for paper in state["papers"]}
    for synopsis_id, synopsis in state["synopses"].items():
        assert synopsis_id in paper_ids, f"synopsis '{synopsis_id}' has no paper"
        assert synopsis["paperId"] == synopsis_id


def test_phase_ids_are_unique_and_well_formed(state):
    ids = [phase["id"] for phase in state["phases"]]
    assert len(ids) == len(set(ids)), "duplicate phase ids"
    for phase_id in ids:
        assert re.fullmatch(r"phase-\d+", phase_id), f"unexpected phase id '{phase_id}'"


def test_phase_and_task_statuses_are_known(state):
    for phase in state["phases"]:
        assert phase["status"] in PHASE_STATUSES, phase["id"]
        for task in phase["tasks"]:
            assert task["status"] in TASK_STATUSES, f"{phase['id']}:{task['id']}"
