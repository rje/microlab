"""scripts/sft.py: example-building with prompt-masking + the END sentinel, and an
end-to-end tiny run that warm-starts, trains a few steps, and writes a servable chat run
(ckpt + tokenizer + serve_config). Loaded via importlib since scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import torch

from microlab.model.reference.checkpoint import load_variant_from_run
from microlab.model.reference.sft import IGNORE_INDEX, format_chat
from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.tokenizer.fast import FastTokenizer

_SPEC = importlib.util.spec_from_file_location(
    "sft_script", Path(__file__).resolve().parents[2] / "scripts" / "sft.py")
sft = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sft)


class _ByteTok:
    """Byte-level tokenizer: encode is exact per char, so prompt/response boundaries don't
    shift (a real BPE could merge across them)."""

    def encode(self, s):
        return list(s.encode("utf-8"))


def test_build_examples_masks_prompt_and_appends_sentinel():
    tok = _ByteTok()
    rows = [
        {"instruction": "Say hi", "context": "", "response": "hello"},
        {"instruction": "x", "context": "", "response": "   "},  # empty response -> skipped
    ]
    examples, skipped = sft.build_examples(tok, rows)
    assert len(examples) == 1  # the blank-response row contributes nothing to learn
    assert skipped == 0  # blank responses aren't "too long" skips
    input_ids, labels = examples[0]

    prompt, _ = format_chat("Say hi", "")
    n_prompt = len(tok.encode(prompt))
    supervised = tok.encode("hello" + sft.END_SENTINEL)
    assert labels[:n_prompt] == [IGNORE_INDEX] * n_prompt  # prompt fully masked out
    assert labels[n_prompt:] == supervised                # response + sentinel supervised
    assert input_ids[n_prompt:] == supervised             # sentinel is part of the sequence


def test_build_examples_includes_context_block():
    tok = _ByteTok()
    (example,), _ = sft.build_examples(
        tok, [{"instruction": "summarize", "context": "the ctx", "response": "ok"}])
    prompt, _ = format_chat("summarize", "the ctx")
    assert "### Input:" in prompt and "the ctx" in prompt
    assert len(example[0]) == len(tok.encode(prompt)) + len(tok.encode("ok" + sft.END_SENTINEL))


def test_end_sentinel_shared_with_chat_module_and_serve_config():
    # One source of truth: the sentinel scripts/sft.py trains and serves against is the
    # same constant the multi-turn builder renders, and serving stops on it.
    from microlab.model.reference.chat_sft import END_SENTINEL
    assert sft.END_SENTINEL is END_SENTINEL
    assert END_SENTINEL.strip() in sft.SERVE_CONFIG["stop_strings"]


def test_build_examples_auto_detects_turns_schema():
    # Old {instruction,...} rows and new {"turns": [...]} rows can share one file; each
    # line is dispatched by its schema and the chat rows match the reference builder.
    from microlab.model.reference.chat_sft import build_chat_example
    tok = _ByteTok()
    turns = [{"user": "U1", "assistant": "A1"}, {"user": "U2", "assistant": "A2"}]
    rows = [{"instruction": "Say hi", "context": "", "response": "hello"}, {"turns": turns}]
    examples, skipped = sft.build_examples(tok, rows, block_size=4096)
    assert len(examples) == 2 and skipped == 0
    want_ids, want_labels, _ = build_chat_example(tok, turns, 4096)
    assert examples[1] == (want_ids, want_labels)
    # a single-turn conversation row builds byte-identically to its old-schema twin
    (old_ex, new_ex), _ = sft.build_examples(
        tok, [{"instruction": "Say hi", "context": "", "response": "hello"},
              {"turns": [{"user": "Say hi", "assistant": "hello"}]}], block_size=4096)
    assert old_ex == new_ex


def test_build_examples_skips_oversized_chat_rows_with_count():
    tok = _ByteTok()
    rows = [{"turns": [{"user": "U", "assistant": "A" * 500}]},  # last turn alone > block
            {"turns": [{"user": "U", "assistant": "ok"}]}]
    examples, skipped = sft.build_examples(tok, rows, block_size=64)
    assert len(examples) == 1 and skipped == 1


def test_build_examples_raises_on_unknown_schema():
    try:
        sft.build_examples(_ByteTok(), [{"prompt": "?", "completion": "!"}], block_size=64)
        raise AssertionError("expected ValueError for unrecognized row schema")
    except ValueError as exc:
        assert "unrecognized" in str(exc)


def test_load_base_model_propagates_rope_base(tmp_path):
    # runs/1b-4k was context-extended with rope_base 1e5; rebuilding its VariantGPT
    # without the field would silently rebuild RoPE caches at the 1e4 default.
    cfg = VariantConfig(vocab_size=64, block_size=32, n_layer=1, n_head=2, n_embd=16,
                        norm="rms", pos="rope", mlp="swiglu", rope_base=100000.0)
    torch.save({"model": VariantGPT(cfg).state_dict(), "step": 1, "cfg": cfg},
               tmp_path / "ckpt_1.pt")
    model, loaded_cfg = sft.load_base_model(tmp_path / "ckpt_1.pt", "cpu")
    assert model.config.rope_base == 100000.0


def test_resolve_base_ckpt_accepts_file_or_dir(tmp_path):
    (tmp_path / "ckpt_5.pt").write_bytes(b"x")
    (tmp_path / "ckpt_9.pt").write_bytes(b"x")
    assert sft.resolve_base_ckpt(tmp_path / "ckpt_5.pt") == tmp_path / "ckpt_5.pt"
    assert sft.resolve_base_ckpt(tmp_path).name == "ckpt_9.pt"  # a dir -> its latest ckpt


def test_run_sft_end_to_end_writes_servable_chat_run(tmp_path):
    # A tiny pretrained base checkpoint to warm-start from (same dict shape the Trainer saves).
    tok = FastTokenizer.train(
        ["hello world", "say something nice", "the answer is four"] * 4,
        vocab_size=300, save_path=str(tmp_path / "tokenizer.json"))
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=2,
                        n_embd=16, norm="rms", pos="rope", mlp="swiglu")
    base = VariantGPT(cfg)
    torch.save({"model": base.state_dict(), "step": 100, "cfg": cfg}, tmp_path / "ckpt_100.pt")

    rows = [{"instruction": f"question {i}", "context": "", "response": f"answer {i}"}
            for i in range(6)]
    data = tmp_path / "dolly.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))

    out = tmp_path / "sft-run"
    result = sft.run_sft(base_ckpt=tmp_path / "ckpt_100.pt", data=data, out=out,
                         tokenizer=tmp_path / "tokenizer.json", epochs=1, lr=1e-3,
                         batch_size=2, block_size=64, device="cpu", limit=6, log_interval=1)

    assert math.isfinite(result["final_loss"])
    assert result["steps"] == 3  # 6 examples / batch 2 over 1 epoch

    ckpt = Path(result["ckpt_path"])
    assert ckpt.exists() and ckpt.name == "ckpt_3.pt"
    assert (out / "tokenizer.json").exists()
    serve_cfg = json.loads((out / "serve_config.json").read_text())
    assert serve_cfg["mode"] == "chat"
    assert "### End" in serve_cfg["stop_strings"]

    # The saved checkpoint loads through the console's serving loader (proves the run is
    # servable without any bespoke glue).
    model, step = load_variant_from_run(out, device="cpu")
    assert step == 3 and model.config.vocab_size == tok.vocab_size


def test_run_sft_raises_when_no_usable_examples(tmp_path):
    FastTokenizer.train(["a b c"] * 4, vocab_size=280,
                        save_path=str(tmp_path / "tokenizer.json"))
    cfg = VariantConfig(vocab_size=280, block_size=32, n_layer=1, n_head=2, n_embd=16,
                        norm="rms", pos="rope", mlp="swiglu")
    torch.save({"model": VariantGPT(cfg).state_dict(), "step": 1, "cfg": cfg},
               tmp_path / "ckpt_1.pt")
    data = tmp_path / "empty.jsonl"
    data.write_text(json.dumps({"instruction": "hi", "context": "", "response": "  "}))
    try:
        sft.run_sft(base_ckpt=tmp_path / "ckpt_1.pt", data=data, out=tmp_path / "o",
                    tokenizer=tmp_path / "tokenizer.json", epochs=1, device="cpu")
        raise AssertionError("expected a ValueError for no usable examples")
    except ValueError as exc:
        assert "no usable SFT examples" in str(exc)


def _tiny_base(tmp_path):
    tok = FastTokenizer.train(
        ["hello world", "say something nice", "the answer is four"] * 4,
        vocab_size=300, save_path=str(tmp_path / "tokenizer.json"))
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=2,
                        n_embd=16, norm="rms", pos="rope", mlp="swiglu")
    torch.save({"model": VariantGPT(cfg).state_dict(), "step": 100, "cfg": cfg},
               tmp_path / "ckpt_100.pt")
    rows = [{"instruction": f"question {i}", "context": "", "response": f"answer {i}"}
            for i in range(8)]
    data = tmp_path / "rows.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    return data


def test_run_sft_grad_accum_counts_optimizer_steps(tmp_path):
    # 8 examples, micro-batch 2, accum 2 -> effective batch 4 -> 2 optimizer steps/epoch.
    # `steps` (and the ckpt name) count OPTIMIZER steps, not micro-batches.
    data = _tiny_base(tmp_path)
    result = sft.run_sft(base_ckpt=tmp_path / "ckpt_100.pt", data=data,
                         out=tmp_path / "run", tokenizer=tmp_path / "tokenizer.json",
                         epochs=1, lr=1e-3, batch_size=2, grad_accum=2, block_size=64,
                         device="cpu", log_interval=1)
    assert result["steps"] == 2
    assert Path(result["ckpt_path"]).name == "ckpt_2.pt"
    assert math.isfinite(result["final_loss"])


def test_run_sft_save_every_writes_and_prunes_intermediates(tmp_path):
    # save_every=1 with 4 steps: intermediate ckpts appear during the run and are pruned so
    # only the latest periodic + final remain (crash-resilience without hoarding disk).
    data = _tiny_base(tmp_path)
    out = tmp_path / "run"
    result = sft.run_sft(base_ckpt=tmp_path / "ckpt_100.pt", data=data, out=out,
                         tokenizer=tmp_path / "tokenizer.json", epochs=1, lr=1e-3,
                         batch_size=2, block_size=64, device="cpu", log_interval=1,
                         save_every=1)
    assert result["steps"] == 4
    ckpts = sorted(out.glob("ckpt_*.pt"))
    assert [c.name for c in ckpts] == ["ckpt_4.pt"]  # intermediates pruned, final kept
    model, step = load_variant_from_run(out, device="cpu")
    assert step == 4


def test_run_sft_end_to_end_on_turns_data(tmp_path):
    # The full pipeline accepts the multi-turn schema (mixed with old rows in one file)
    # and still writes a servable chat run.
    tok = FastTokenizer.train(
        ["hello world", "say something nice", "the answer is four"] * 4,
        vocab_size=300, save_path=str(tmp_path / "tokenizer.json"))
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=2,
                        n_embd=16, norm="rms", pos="rope", mlp="swiglu")
    torch.save({"model": VariantGPT(cfg).state_dict(), "step": 100, "cfg": cfg},
               tmp_path / "ckpt_100.pt")
    rows = [{"turns": [{"user": f"q{i}", "assistant": f"a{i}"},
                       {"user": "more?", "assistant": "sure"}]} for i in range(4)]
    rows.append({"instruction": "old style", "context": "", "response": "still works"})
    data = tmp_path / "chat.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))

    out = tmp_path / "chat-run"
    result = sft.run_sft(base_ckpt=tmp_path / "ckpt_100.pt", data=data, out=out,
                         tokenizer=tmp_path / "tokenizer.json", epochs=1, lr=1e-3,
                         batch_size=5, block_size=64, device="cpu", log_interval=1)
    assert math.isfinite(result["final_loss"])
    assert result["steps"] == 1  # 5 examples / batch 5 over 1 epoch
    model, step = load_variant_from_run(out, device="cpu")
    assert step == 1
    serve_cfg = json.loads((out / "serve_config.json").read_text())
    assert serve_cfg["stop_strings"] == ["### End", "\n### Instruction:"]
