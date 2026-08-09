"""scripts/dpo.py: pair-example building (prompt masked, response + sentinel supervised), the
frozen-reference / trainable-policy split, and a tiny end-to-end DPO run on a trivially
separable synthetic pair set — proving the loss is finite, the implicit-reward accuracy climbs
toward 1.0, and the policy is saved as a servable chat run. Loaded via importlib since scripts/
isn't a package."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import torch

from microlab.model.reference.checkpoint import load_variant_from_run
from microlab.model.reference.dpo import sequence_logprob
from microlab.model.reference.sft import IGNORE_INDEX, format_chat
from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.tokenizer.fast import FastTokenizer

_SPEC = importlib.util.spec_from_file_location(
    "dpo_script", Path(__file__).resolve().parents[2] / "scripts" / "dpo.py")
dpo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dpo)


class _ByteTok:
    """Byte-level tokenizer: encode is exact per char so prompt/response boundaries don't shift."""

    def encode(self, s):
        return list(s.encode("utf-8"))


def test_build_pref_examples_masks_prompt_and_appends_sentinel():
    tok = _ByteTok()
    prompt, _ = format_chat("Say hi", "")
    prefs = [{"prompt": prompt, "chosen": "good", "rejected": "bad"}]
    (chosen_ex, rejected_ex) = dpo.build_pref_examples(tok, prefs)[0]

    n_prompt = len(tok.encode(prompt))
    for ex, resp in ((chosen_ex, "good"), (rejected_ex, "bad")):
        input_ids, labels = ex
        supervised = tok.encode(resp + dpo.END_SENTINEL)
        assert labels[:n_prompt] == [IGNORE_INDEX] * n_prompt  # prompt fully masked
        assert labels[n_prompt:] == supervised                 # response + sentinel supervised
        assert input_ids[n_prompt:] == supervised


def test_resolve_ckpt_accepts_file_or_dir(tmp_path):
    (tmp_path / "ckpt_5.pt").write_bytes(b"x")
    (tmp_path / "ckpt_9.pt").write_bytes(b"x")
    assert dpo.resolve_ckpt(tmp_path / "ckpt_5.pt") == tmp_path / "ckpt_5.pt"
    assert dpo.resolve_ckpt(tmp_path).name == "ckpt_9.pt"  # a dir -> its latest ckpt


def _tiny_sft_run(tmp_path):
    """Write a tiny servable SFT run (ckpt + tokenizer) and return (run_dir, tokenizer, cfg)."""
    tok = FastTokenizer.train(
        ["hello world", "the answer is four", "say something nice"] * 4,
        vocab_size=300, save_path=str(tmp_path / "tokenizer.json"))
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=2, n_head=2,
                        n_embd=16, norm="rms", pos="rope", mlp="swiglu")
    run_dir = tmp_path / "sft-run"
    run_dir.mkdir()
    torch.save({"model": VariantGPT(cfg).state_dict(), "step": 100, "cfg": cfg},
               run_dir / "ckpt_100.pt")
    (run_dir / "tokenizer.json").write_text((tmp_path / "tokenizer.json").read_text())
    return run_dir, run_dir / "tokenizer.json", cfg


def test_load_policy_reference_freezes_reference_only(tmp_path):
    run_dir, _, _ = _tiny_sft_run(tmp_path)
    policy, reference, _ = dpo.load_policy_reference(run_dir, device="cpu")

    assert all(p.requires_grad for p in policy.parameters())
    assert not any(p.requires_grad for p in reference.parameters())
    assert reference.training is False and policy.training is True
    # Same weights at init, but independent tensors.
    pp = next(policy.parameters())
    rp = next(reference.parameters())
    assert torch.equal(pp, rp) and pp.data_ptr() != rp.data_ptr()

    # One optimizer step on the policy must not move the frozen reference.
    ref_clone = rp.detach().clone()
    x = torch.randint(0, 300, (2, 16))
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    loss = sequence_logprob(policy(x)[0], x).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.equal(rp, ref_clone)          # reference unchanged
    assert not torch.equal(next(policy.parameters()), ref_clone)  # policy moved


def test_build_model_rebuilds_hybrid_frontier_fields(tmp_path):
    # coder-1b's field profile scaled down (KDA:MLA hybrid, peri-LN, NoPE, MLA latents,
    # qk-norm). The hand-listed config rebuild this guards against dropped every one of
    # these fields (and rope_base), so DPO could not warm-start a coder-1b SFT checkpoint.
    cfg = VariantConfig(vocab_size=64, block_size=32, n_layer=4, n_head=2, n_embd=16,
                        norm="rms", pos="nope", mlp="swiglu", block_norm="peri",
                        hybrid_every=4, gdn_gate="channel", global_attn="mla",
                        mla_kv_lora=8, qk_norm=True, gdn_fused=False)
    state = VariantGPT(cfg).state_dict()
    model = dpo._build_model(cfg, state, "cpu")  # strict load must pass
    got = model.config
    assert (got.hybrid_every, got.global_attn, got.block_norm) == (4, "mla", "peri")
    assert (got.mla_kv_lora, got.qk_norm, got.gdn_gate) == (8, True, "channel")


def test_run_dpo_raises_on_empty_prefs(tmp_path):
    run_dir, tok_path, _ = _tiny_sft_run(tmp_path)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    try:
        dpo.run_dpo(sft_ckpt=run_dir, prefs=empty, out=tmp_path / "o", tokenizer=tok_path,
                    device="cpu")
        raise AssertionError("expected a ValueError for empty prefs")
    except ValueError as exc:
        assert "no preference pairs" in str(exc)


def test_run_dpo_end_to_end_climbs_acc_and_writes_servable_run(tmp_path):
    run_dir, tok_path, cfg = _tiny_sft_run(tmp_path)

    # Trivially separable pairs: a clearly-preferred `chosen` vs a distinct `rejected`.
    prompts = [format_chat(f"question {i}")[0] for i in range(4)]
    prefs = [{"prompt": p, "chosen": "the correct and complete answer",
              "rejected": "no"} for p in prompts]
    prefs_path = tmp_path / "prefs.jsonl"
    prefs_path.write_text("\n".join(json.dumps(x) for x in prefs))

    out = tmp_path / "dpo-run"
    result = dpo.run_dpo(sft_ckpt=run_dir, prefs=prefs_path, out=out, tokenizer=tok_path,
                         epochs=8, lr=1e-3, beta=0.1, batch_size=2, block_size=128,
                         device="cpu", log_interval=1)

    assert math.isfinite(result["final_loss"])
    assert result["steps"] == 16  # 4 pairs / batch 2 = 2 steps/epoch * 8 epochs
    # DPO signal: accuracy starts at 0 (policy == reference) and climbs as the policy learns
    # to prefer chosen; on a trivially separable set it reaches 1.0.
    assert result["acc_history"][0] == 0.0
    assert result["acc_history"][-1] > result["acc_history"][0]
    assert result["acc_history"][-1] == 1.0
    assert result["loss_history"][-1] < result["loss_history"][0]  # loss falls

    # The saved policy is a servable chat run: ckpt + tokenizer + chat serve_config.
    ckpt = Path(result["ckpt_path"])
    assert ckpt.exists() and ckpt.name == "ckpt_16.pt"
    assert (out / "tokenizer.json").exists()
    serve_cfg = json.loads((out / "serve_config.json").read_text())
    assert serve_cfg["mode"] == "chat" and "### End" in serve_cfg["stop_strings"]

    model, step = load_variant_from_run(out, device="cpu")
    assert step == 16 and model.config.vocab_size == cfg.vocab_size


def test_run_dpo_grad_accum_counts_optimizer_steps(tmp_path):
    # 4 pairs, micro-batch 1, accum 2 -> effective batch 2 -> 2 optimizer steps/epoch.
    # Same convention as sft.py: `steps` (and the ckpt name) count OPTIMIZER steps.
    run_dir, tok_path, cfg = _tiny_sft_run(tmp_path)
    prompts = [format_chat(f"question {i}")[0] for i in range(4)]
    prefs = [{"prompt": p, "chosen": "the correct and complete answer",
              "rejected": "no"} for p in prompts]
    prefs_path = tmp_path / "prefs.jsonl"
    prefs_path.write_text("\n".join(json.dumps(x) for x in prefs))

    result = dpo.run_dpo(sft_ckpt=run_dir, prefs=prefs_path, out=tmp_path / "o",
                         tokenizer=tok_path, epochs=2, lr=1e-3, beta=0.1, batch_size=1,
                         grad_accum=2, block_size=128, device="cpu", log_interval=1)
    assert result["steps"] == 4  # 4 pairs / (1*2) = 2 opt steps/epoch * 2 epochs
    assert math.isfinite(result["final_loss"])
    assert result["acc_history"][-1] > 0.0  # still learns through accumulation
