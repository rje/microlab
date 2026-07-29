"""scripts/eval_toolcall.py pure logic: resumable JSONL accounting with a pinned config
header. The prompt build, scoring, and budget machinery live in
microlab.evals.code.toolcall / .gen and are tested there."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_toolcall", Path(__file__).resolve().parents[2] / "scripts" / "eval_toolcall.py")
et = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(et)

HEADER = {"run": "runs/x", "registry": "evals/toolcall/registry.json",
          "items": "evals/toolcall/items.jsonl", "max_new": 64, "limit": None}


def test_read_resume_fresh_matching_and_mismatch(tmp_path):
    out = tmp_path / "r.jsonl"
    assert et.read_resume(out, HEADER) == set()
    out.write_text(json.dumps({"_header": HEADER}) + "\n"
                   + json.dumps({"id": "pat-001", "strict": True}) + "\n")
    assert et.read_resume(out, HEADER) == {"pat-001"}
    out2 = tmp_path / "r2.jsonl"
    out2.write_text(json.dumps({"_header": {**HEADER, "max_new": 32}}) + "\n")
    with pytest.raises(ValueError, match="different config header"):
        et.read_resume(out2, HEADER)


def test_read_resume_headerless_file_raises(tmp_path):
    out = tmp_path / "r.jsonl"
    out.write_text(json.dumps({"id": "pat-001"}) + "\n")
    with pytest.raises(ValueError, match="different config header"):
        et.read_resume(out, HEADER)
