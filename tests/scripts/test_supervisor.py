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


def test_crash_detection_ignores_a_stale_log(monkeypatch):
    """A previous episode's traceback must not condemn a healthy new instance.

    The log key is reused across episodes and a fresh box takes minutes to ship its first
    log. Reading it unguarded destroyed a healthy instance on the previous run's evidence —
    a false positive that costs more than the gap it closes.
    """
    import datetime as dt

    class S3:
        def __init__(self, when):
            self.when = when

        def get_object(self, Bucket, Key):
            class B:
                @staticmethod
                def read():
                    return b"MICROLAB_TRAIN_FAILED rc=1"
            return {"Body": B(), "LastModified": self.when}

    old = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)
    new = dt.datetime(2030, 1, 1, tzinfo=dt.UTC)
    episode_start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC).timestamp()

    assert sup.training_crashed(S3(old), "b", "p", since=episode_start) is None, \
        "a log older than the episode must be ignored"
    assert sup.training_crashed(S3(new), "b", "p", since=episode_start) is not None, \
        "a log written during this episode must still be read"


def test_crash_detection_without_a_since_still_reads():
    """Backwards compatible: no episode start means no freshness filter."""
    import datetime as dt

    class S3:
        def get_object(self, Bucket, Key):
            class B:
                @staticmethod
                def read():
                    return b"MICROLAB_TRAIN_FAILED rc=2"
            return {"Body": B(), "LastModified": dt.datetime.now(dt.UTC)}

    assert sup.training_crashed(S3(), "b", "p") is not None


def _s3_with(text: str, when=None):
    import datetime as dt

    class S3:
        def get_object(self, Bucket, Key):
            class B:
                @staticmethod
                def read():
                    return text.encode()
            return {"Body": B(), "LastModified": when or dt.datetime.now(dt.UTC)}
    return S3()


def test_benign_log_noise_is_not_a_crash():
    """The detector must not infer failure from prose.

    A heuristic scan for "Error" matched this exact line — optional NVML telemetry being
    absent, which the trainer handles and reports — and destroyed a healthy instance.
    """
    noise = ("GPU NVML telemetry unavailable (memory-only): ModuleNotFoundError: "
             "nvidia-ml-py does not seem to be installed or it can't be imported.\n"
             "Traceback (most recent call last):\n  handled during import probe\n"
             "step 5/40000 loss 8.1")
    assert sup.training_crashed(_s3_with(noise), "b", "p") is None


def test_explicit_sentinel_is_a_crash():
    log = "step 3/40000 loss 9.9\nMICROLAB_TRAIN_FAILED rc=1\n"
    got = sup.training_crashed(_s3_with(log), "b", "p")
    assert got and "MICROLAB_TRAIN_FAILED" in got and "rc=1" in got


def test_the_sentinel_is_actually_emitted_by_the_instance_script():
    """Both ends of the channel must agree, or the detector never fires at all."""
    src = (SCRIPTS / "cloud_train.sh").read_text()
    assert "MICROLAB_TRAIN_FAILED rc=$RC" in src
    assert 'if [ "$RC" -ne 0 ]' in src
