"""Tests for the HF text-dataset ingest, focused on CommitPack rendering.

The commits slice silently produced an EMPTY directory in the first mix build: the ingest
died on `RuntimeError: Dataset scripts are no longer supported, but found commitpackft.py`
and the wrapper script's `|| echo "commits FAILED"` wrote to a log nobody read. These tests
cover the rendering decision; the empty-slice class of failure is caught separately by the
mix builder's token-count assertion.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "ingest_text_dataset",
    Path(__file__).resolve().parents[2] / "scripts" / "ingest_text_dataset.py")
ingest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ingest)


def _rec(**over):
    r = {"message": "Declare queues when broker is instantiated\n",
         "subject": "Declare queues when broker is instantiated",
         "old_file": "sentry/queue/client.py",
         "new_file": "sentry/queue/client.py",
         "old_contents": "class Broker:\n    pass\n",
         "new_contents": "class Broker:\n    def __init__(self):\n        declare()\n"}
    r.update(over)
    return r


def test_render_commit_keeps_intent_and_both_sides():
    """The whole point of the slice: natural-language intent adjacent to the edit."""
    out = ingest.render_commit(_rec())
    assert "Declare queues when broker is instantiated" in out
    assert "class Broker:\n    pass" in out           # before
    assert "declare()" in out                          # after
    assert "--- a/sentry/queue/client.py" in out
    assert "+++ b/sentry/queue/client.py" in out
    # intent leads, so the model reads the description before the change
    assert out.index("Declare queues") < out.index("--- a/")


def test_render_commit_handles_renames():
    out = ingest.render_commit(_rec(old_file="a/old.py", new_file="b/new.py"))
    assert "--- a/a/old.py" in out
    assert "+++ b/b/new.py" in out


def test_render_commit_falls_back_to_subject_when_message_blank():
    out = ingest.render_commit(_rec(message="   "))
    assert out.startswith("Declare queues when broker is instantiated\n\n")


@pytest.mark.parametrize("field", ingest.COMMIT_FIELDS)
def test_render_commit_raises_on_missing_field(field):
    """No partial documents: a message-less 'commit' would drop the supervision we want."""
    bad = _rec()
    del bad[field]
    with pytest.raises(KeyError, match=field):
        ingest.render_commit(bad)


def test_render_commit_raises_on_non_string_field():
    with pytest.raises(KeyError, match="old_contents"):
        ingest.render_commit(_rec(old_contents=None))
