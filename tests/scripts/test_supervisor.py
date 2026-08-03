"""Supervisor logic: progress detection, cost accounting, and the resume contract.

The supervisor exists so a preemption costs six minutes instead of the run. Every test
here targets a way that promise could fail QUIETLY — a stalled box that still bills, a
resume that rewinds, a cost counter that forgets what earlier instances spent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sup = _load("vast_supervisor")


class FakeS3:
    def __init__(self, keys):
        self._keys = keys

    def list_objects_v2(self, **kw):
        pfx = kw.get("Prefix", "")
        items = [{"Key": k, "Size": v} for k, v in self._keys.items() if k.startswith(pfx)]
        return {"Contents": items, "IsTruncated": False}


def test_progress_is_the_highest_step_not_the_newest_upload():
    """Re-uploading an older checkpoint must not rewind the run.

    B2 is object storage: a retried upload of ckpt_500 after ckpt_2000 exists would have a
    later mtime. Taking the newest object would silently resume 1,500 steps back and burn
    that compute again.
    """
    s3 = FakeS3({"coder-1b/ckpt_500.pt": 10, "coder-1b/ckpt_2000.pt": 10,
                 "coder-1b/ckpt_1000.pt": 10})
    assert sup.remote_step(s3, "b", "coder-1b") == 2000


def test_no_checkpoints_means_step_zero():
    assert sup.remote_step(FakeS3({}), "b", "coder-1b") == 0


def test_unrelated_objects_are_ignored():
    """The bucket also holds shakedown logs; they must not parse as progress."""
    s3 = FakeS3({"coder-1b/ckpt_100.pt": 1, "shakedown/shakedown-1.log": 1,
                 "coder-1b/tokenizer.json": 1})
    assert sup.remote_step(s3, "b", "coder-1b") == 100


def test_prefix_isolates_runs():
    """Two runs in one bucket must not read each other's progress."""
    s3 = FakeS3({"coder-1b/ckpt_100.pt": 1, "other-run/ckpt_9000.pt": 1})
    assert sup.remote_step(s3, "b", "coder-1b") == 100
    assert sup.remote_step(s3, "b", "other-run") == 9000


def test_spend_accumulates_across_episodes(tmp_path, monkeypatch):
    """Cost is cumulative over ALL instances. Resetting per instance would let an
    arbitrary number of preemptions each stay under the cap while the total ran away."""
    monkeypatch.setattr(sup, "STATE", tmp_path / "state.json")
    sup.save_state({"spent": 40.0, "episodes": [{"instance": 1}], "last_step": 3000})
    st = sup.load_state()
    st["spent"] += 35.0
    sup.save_state(st)
    assert sup.load_state()["spent"] == pytest.approx(75.0)
    assert sup.load_state()["last_step"] == 3000


def test_state_survives_a_supervisor_restart(tmp_path, monkeypatch):
    """The supervisor itself can be killed; the spend must not reset to zero."""
    monkeypatch.setattr(sup, "STATE", tmp_path / "state.json")
    sup.save_state({"spent": 123.45, "episodes": [], "last_step": 12000})
    assert json.loads((tmp_path / "state.json").read_text())["spent"] == 123.45
    assert sup.load_state()["spent"] == pytest.approx(123.45)


def test_missing_state_starts_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, "STATE", tmp_path / "none.json")
    st = sup.load_state()
    assert st == {"spent": 0.0, "episodes": [], "last_step": 0}


def test_onstart_runs_the_training_entrypoint():
    class A:
        repo = "https://example.com/r.git"
    s = sup.onstart(A())
    assert "cloud_train.sh" in s and "git clone" in s


def test_checkpoint_syncer_never_prunes_the_newest():
    """The syncer must not delete the file the next instance would resume from."""
    src = (SCRIPTS / "b2_ckpt_sync.py").read_text()
    assert "if p == local[-1]:" in src, "newest-checkpoint guard is missing"
    assert "confirmed" in src, "pruning must be gated on remote confirmation"
