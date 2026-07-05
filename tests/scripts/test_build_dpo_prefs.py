"""scripts/build_dpo_prefs.py: the stop-string truncation, the skip logic (empty gold, empty
sample, sample == gold), JSONL writing, and a tiny end-to-end run that loads a real VariantGPT
+ tokenizer and writes valid {prompt, chosen, rejected} lines. Loaded via importlib since
scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.tokenizer.fast import FastTokenizer

_SPEC = importlib.util.spec_from_file_location(
    "build_dpo_prefs", Path(__file__).resolve().parents[2] / "scripts" / "build_dpo_prefs.py")
bp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bp)


def test_truncate_cuts_at_earliest_stop():
    assert bp.truncate("hello world\n### End\nmore") == "hello world"
    assert bp.truncate("answer\n### Instruction:\nnext") == "answer"
    # earliest of the two stops wins
    assert bp.truncate("a\n### Instruction:\nb\n### End") == "a"
    # no stop -> whole (stripped) string
    assert bp.truncate("  plain answer  ") == "plain answer"


def test_build_prefs_skips_empty_and_identical(monkeypatch):
    rows = [
        {"instruction": "a", "context": "", "response": "gold-a"},  # kept
        {"instruction": "b", "context": "", "response": "   "},     # empty gold -> skip (no gen)
        {"instruction": "c", "context": "", "response": "gold-c"},  # sample empty -> skip
        {"instruction": "d", "context": "", "response": "gold-d"},  # sample == gold -> skip
    ]
    # The empty-gold row is skipped before generation, so sampling is only asked for a, c, d.
    scripted = iter(["model-a", "", "gold-d"])
    monkeypatch.setattr(bp, "sample_rejected", lambda *a, **k: next(scripted))

    prefs = bp.build_prefs(model=None, tok=None, rows=rows, device="cpu")
    assert len(prefs) == 1
    assert prefs[0]["chosen"] == "gold-a" and prefs[0]["rejected"] == "model-a"
    assert "### Instruction:" in prefs[0]["prompt"] and "### Response:" in prefs[0]["prompt"]


def test_write_prefs_roundtrips_jsonl(tmp_path):
    prefs = [{"prompt": "p1", "chosen": "c1", "rejected": "r1"},
             {"prompt": "p2", "chosen": "c2", "rejected": "r2"}]
    out = tmp_path / "sub" / "prefs.jsonl"
    n = bp.write_prefs(prefs, out)
    assert n == 2
    lines = out.read_text().splitlines()
    assert [json.loads(x) for x in lines] == prefs


def test_run_build_end_to_end_writes_valid_pairs(tmp_path):
    # A tiny servable SFT run (ckpt + tokenizer) to sample rejected responses from.
    tok = FastTokenizer.train(
        ["hello world", "the answer is four", "say something nice"] * 4,
        vocab_size=300, save_path=str(tmp_path / "tokenizer.json"))
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=2, n_head=2,
                        n_embd=16, norm="rms", pos="rope", mlp="swiglu")  # rope: kv-cache gen
    model = VariantGPT(cfg)
    run_dir = tmp_path / "sft-run"
    run_dir.mkdir()
    torch.save({"model": model.state_dict(), "step": 100, "cfg": cfg},
               run_dir / "ckpt_100.pt")
    (run_dir / "tokenizer.json").write_text((tmp_path / "tokenizer.json").read_text())

    rows = [{"instruction": f"question {i}", "context": "", "response": f"gold answer {i}"}
            for i in range(4)]
    data = tmp_path / "dolly.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))

    out = tmp_path / "dpo_prefs.jsonl"
    result = bp.run_build(sft_run=run_dir, data=data, out=out,
                          tokenizer=run_dir / "tokenizer.json", limit=4, temp=0.8,
                          max_new=8, device="cpu", seed=0)

    assert result["written"] >= 1
    lines = out.read_text().splitlines()
    assert len(lines) == result["written"]
    for line in lines:
        pair = json.loads(line)
        assert set(pair) == {"prompt", "chosen", "rejected"}
        assert "### Response:" in pair["prompt"]
        assert pair["chosen"].strip() and pair["rejected"].strip()
        assert pair["rejected"].strip() != pair["chosen"].strip()
