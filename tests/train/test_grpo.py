"""microlab.train.grpo: the group-relative advantage math (incl. the zero-std group),
response-token masking (prompt positions contribute zero gradient), the clipped surrogate
(step-0 ratio == 1, gradient direction, clip counting), the k3 KL estimator (pointwise >= 0,
exactly 0 at identical models), the prompt sampler's resume determinism, and a tiny CPU
end-to-end run where a rigged reward RISES. Helpers copied from scripts (truncate_at_stops,
repetition_score) are pinned against their script originals via importlib."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from microlab.model.reference.sft import IGNORE_INDEX
from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.tokenizer.fast import FastTokenizer
from microlab.train.grpo import (
    clipped_surrogate,
    distinct_fraction,
    forward_logps,
    group_advantages,
    grpo_lr,
    iteration_prompt_indices,
    k3_kl,
    per_token_logps,
    repetition_score,
    run_grpo,
    truncate_at_stops,
)

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_model(vocab: int = 300, block: int = 64, seed: int = 0) -> VariantGPT:
    torch.manual_seed(seed)
    cfg = VariantConfig(vocab_size=vocab, block_size=block, n_layer=2, n_head=2, n_embd=32,
                        dropout=0.0, norm="rms", pos="rope", mlp="swiglu")
    return VariantGPT(cfg)


# ---------------------------------------------------------------- group_advantages


def test_group_advantages_standardizes_within_group():
    r = torch.tensor([[1.0, 2.0, 3.0], [10.0, 10.0, 40.0]])
    adv = group_advantages(r)
    # Row 0: mean 2, population std sqrt(2/3).
    s0 = math.sqrt(2.0 / 3.0)
    assert torch.allclose(adv[0], torch.tensor([-1.0 / s0, 0.0, 1.0 / s0]), atol=1e-5)
    # Row 1: mean 20, deviations (-10, -10, 20), population std sqrt(200).
    s1 = math.sqrt(200.0)
    assert torch.allclose(adv[1], torch.tensor([-10 / s1, -10 / s1, 20 / s1]), atol=1e-5)
    # Group means are ~0 after standardization.
    assert torch.allclose(adv.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_group_advantages_zero_std_group_is_zero_not_nan():
    adv = group_advantages(torch.tensor([[7.7, 7.7, 7.7], [1.0, 2.0, 3.0]]))
    assert (adv[0] == 0.0).all()  # all-same rewards = no signal, exactly zero
    assert torch.isfinite(adv).all()
    assert adv[1].abs().sum() > 0  # the informative group is untouched


def test_group_advantages_rejects_bad_shape():
    with pytest.raises(ValueError, match="group"):
        group_advantages(torch.tensor([1.0, 2.0]))


# ---------------------------------------------------------------- per_token_logps


def test_per_token_logps_matches_manual_and_masks():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, IGNORE_INDEX],
                           [IGNORE_INDEX, 2, 2, IGNORE_INDEX, IGNORE_INDEX]])
    logps, mask = per_token_logps(logits, labels)
    assert mask.tolist() == [[False, True, True, False], [True, True, False, False]]
    # Causal shift: logits position t predicts labels[t + 1].
    lsm = torch.log_softmax(logits, dim=-1)
    assert torch.allclose(logps[0, 1], lsm[0, 1, 3], atol=1e-6)
    assert torch.allclose(logps[0, 2], lsm[0, 2, 4], atol=1e-6)
    assert torch.allclose(logps[1, 0], lsm[1, 0, 2], atol=1e-6)
    assert (logps[~mask] == 0.0).all()  # masked positions are exactly zeroed


def test_per_token_logps_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        per_token_logps(torch.randn(1, 4, 5), torch.zeros(1, 3, dtype=torch.long))


def test_prompt_positions_contribute_zero_gradient():
    # Rigged sequence: only labels[3] and labels[4] are response tokens, so only logits
    # positions 2 and 3 (which predict them) may receive gradient — prompt and pad positions
    # must get EXACTLY zero.
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 11, requires_grad=True)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5, IGNORE_INDEX]])
    logps, mask = per_token_logps(logits, labels)
    surr, _ = clipped_surrogate(logps, logps.detach(), torch.tensor([1.0]), mask, 0.2)
    (-surr / mask.sum()).backward()
    g = logits.grad
    assert g[0, 2].abs().sum() > 0 and g[0, 3].abs().sum() > 0
    for t in (0, 1, 4, 5):
        assert g[0, t].abs().sum().item() == 0.0


# ---------------------------------------------------------------- clipped surrogate


def test_ratio_is_one_at_step_zero():
    # logp_old is captured by the SAME forward path (no_grad) the training pass uses, so at
    # the first update the importance ratio is exactly 1 on every response token.
    model = _tiny_model(seed=1)
    torch.manual_seed(2)
    input_ids = torch.randint(0, 300, (3, 12))
    labels = input_ids.clone()
    labels[:, :5] = IGNORE_INDEX
    with torch.no_grad():
        logp_old, mask = forward_logps(model, input_ids, labels, use_amp=False)
    logp_new, _ = forward_logps(model, input_ids, labels, use_amp=False)  # the training path
    assert logp_new.requires_grad
    ratio = torch.exp(logp_new - logp_old)[mask]
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6)


def _direction_delta(advantage: float) -> float:
    """Summed response-logp change after ONE optimizer step at the given advantage."""
    model = _tiny_model(seed=3)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5, 6]])
    with torch.no_grad():
        logp_old, mask = forward_logps(model, input_ids, labels, use_amp=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    logp_new, mask = forward_logps(model, input_ids, labels, use_amp=False)
    surr, _ = clipped_surrogate(logp_new, logp_old, torch.tensor([advantage]), mask, 0.2)
    (-surr / mask.sum()).backward()
    opt.step()
    with torch.no_grad():
        logp_after, _ = forward_logps(model, input_ids, labels, use_amp=False)
    return ((logp_after * mask).sum() - (logp_old * mask).sum()).item()


def test_positive_advantage_pushes_response_logp_up():
    assert _direction_delta(1.0) > 0


def test_negative_advantage_pushes_response_logp_down():
    assert _direction_delta(-1.0) < 0


def test_clipped_surrogate_values_and_clip_count():
    logp_new = torch.log(torch.tensor([[2.0, 1.0, 1.1]]))
    logp_old = torch.zeros(1, 3)  # ratios: 2.0, 1.0, 1.1
    mask = torch.ones(1, 3, dtype=torch.bool)
    surr, n_clip = clipped_surrogate(logp_new, logp_old, torch.tensor([1.0]), mask, 0.2)
    assert n_clip == 1  # only ratio 2.0 falls outside [0.8, 1.2]
    # Positive advantage: ratio 2.0 is capped at 1.2; the others pass through.
    assert surr.item() == pytest.approx(1.2 + 1.0 + 1.1, abs=1e-5)
    # Negative advantage: min() takes the UNCLIPPED branch (PPO pessimism).
    surr_neg, _ = clipped_surrogate(logp_new, logp_old, torch.tensor([-1.0]), mask, 0.2)
    assert surr_neg.item() == pytest.approx(-(2.0 + 1.0 + 1.1), abs=1e-5)


def test_clipped_surrogate_ignores_masked_positions():
    logp_new = torch.tensor([[0.0, 5.0]])  # wild ratio at the masked position
    logp_old = torch.zeros(1, 2)
    mask = torch.tensor([[True, False]])
    surr, n_clip = clipped_surrogate(logp_new, logp_old, torch.tensor([1.0]), mask, 0.2)
    assert surr.item() == pytest.approx(1.0)
    assert n_clip == 0


# ---------------------------------------------------------------- k3 KL


def test_k3_kl_pointwise_nonnegative_and_zero_at_identical():
    torch.manual_seed(0)
    ref, new = torch.randn(4, 9), torch.randn(4, 9)
    mask = torch.rand(4, 9) > 0.3
    assert k3_kl(ref, new, mask).item() >= 0.0
    assert k3_kl(new, new, mask).item() == 0.0
    one = torch.ones(1, 1, dtype=torch.bool)
    got = k3_kl(torch.tensor([[1.0]]), torch.tensor([[0.0]]), one)
    assert got.item() == pytest.approx(math.e - 2.0, abs=1e-6)  # e^1 - 1 - 1


def test_k3_kl_zero_between_identical_models():
    a, b = _tiny_model(seed=5), _tiny_model(seed=5)
    torch.manual_seed(6)
    input_ids = torch.randint(0, 300, (2, 10))
    labels = input_ids.clone()
    labels[:, :4] = IGNORE_INDEX
    with torch.no_grad():
        la, mask = forward_logps(a, input_ids, labels, use_amp=False)
        lb, _ = forward_logps(b, input_ids, labels, use_amp=False)
    assert k3_kl(la, lb, mask).item() == 0.0  # identical weights, identical path -> exact 0


# ---------------------------------------------------------------- schedule + prompt sampler


def test_grpo_lr_linear_warmup_then_constant():
    assert grpo_lr(1, 10, 1e-6) == pytest.approx(1e-7)
    assert grpo_lr(5, 10, 1e-6) == pytest.approx(5e-7)
    assert grpo_lr(10, 10, 1e-6) == pytest.approx(1e-6)
    assert grpo_lr(100, 10, 1e-6) == 1e-6  # constant after warmup, no cosine
    assert grpo_lr(1, 0, 1e-6) == 1e-6


def test_iteration_prompt_indices_pure_and_distinct():
    a = iteration_prompt_indices(1337, 3, 50, 8)
    # Pure in (seed, iteration): a restarted run redraws exactly these — resume determinism.
    assert a == iteration_prompt_indices(1337, 3, 50, 8)
    assert len(set(a)) == 8 and all(0 <= i < 50 for i in a)
    assert any(iteration_prompt_indices(1337, it, 50, 8) != a for it in (1, 2, 4, 5))
    with pytest.raises(ValueError, match="pool"):
        iteration_prompt_indices(0, 1, 4, 8)


# ---------------------------------------------------------------- script-pinned helpers


def test_truncate_at_stops_matches_candidate_script():
    brc = _load_script("build_rlaif_candidates")
    texts = ["plain text", "answer ### End trailing junk", "a\n### Instruction:\nb",
             "x ### End y\n### Instruction: z", "  spaced  ", ""]
    for text in texts:
        assert truncate_at_stops(text) == brc.truncate(text)


def test_repetition_score_matches_track_probes():
    tp = _load_script("track_probes")
    texts = ["a b c d a b c d a b c d", "all unique words right here today", ""]
    assert repetition_score(texts) == tp.repetition_score(texts)
    assert repetition_score(["w x y z"]) == 0.0
    assert repetition_score(["w x y z w x y z"]) > 0.0


def test_distinct_fraction():
    assert distinct_fraction([["a", "b", "c", "d"]]) == 1.0
    assert distinct_fraction([["a", "a", "a", "a"]]) == 0.25
    assert distinct_fraction([["a", "b"], ["c", "c"]]) == 0.75
    with pytest.raises(ValueError, match="group"):
        distinct_fraction([])


# ---------------------------------------------------------------- end-to-end (tiny, CPU)


def _train_tok(tmp_path):
    corpus = ["aaaa aaaa bbbb cccc aaaa dddd aaaa eeee"] * 8
    path = tmp_path / "tok.json"
    tok = FastTokenizer.train(corpus, vocab_size=280, save_path=str(path))
    return tok, path


def test_e2e_tiny_reward_rises(tmp_path):
    """The whole pipeline learns: with a rigged reward (count of 'a' characters), the mean
    reward over the last iterations must beat the first ones."""
    tok, tok_path = _train_tok(tmp_path)
    policy = _tiny_model(vocab=tok.vocab_size, seed=11)
    reference = _tiny_model(vocab=tok.vocab_size, seed=11)

    def score(prompt: str, texts: list[str]) -> list[float]:
        return [float(t.count("a")) for t in texts]

    out = tmp_path / "run"
    res = run_grpo(policy, reference, tok, ["say: ", "tell: "], score, out, tok_path,
                   iters=10, prompts_per_iter=2, group_size=8, lr=0.03, beta=0.0,
                   clip_eps=0.2, temp=1.0, max_new=8, micro_batch=8, save_every=0,
                   dump_every=5, seed=9, device="cpu", warmup_iters=2)

    records = [json.loads(li) for li in (out / "grpo_log.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in records] == list(range(1, 11))
    early = sum(r["reward_mean"] for r in records[:3]) / 3
    late = sum(r["reward_mean"] for r in records[-3:]) / 3
    assert late > early, f"reward did not rise: early {early:.3f} late {late:.3f}"
    for key in ("reward_max", "kl_mean", "adv_logp_shift", "resp_len_mean", "distinct_frac",
                "repetition", "clip_frac", "prompt_indices"):
        assert key in records[0]
    # Servable outputs at the end, like sft.py/dpo.py run dirs.
    assert (out / "ckpt_10.pt").exists()
    assert (out / "tokenizer.json").exists()
    assert json.loads((out / "serve_config.json").read_text())["mode"] == "chat"
    # Samples dumped at iteration 1 (baseline) and every dump_every after.
    samples = [json.loads(li) for li in (out / "samples.jsonl").read_text().splitlines()]
    assert {s["iter"] for s in samples} == {1, 5, 10}
    assert all("reward" in s and "response" in s and "prompt" in s for s in samples)
    assert res["ckpt_path"].endswith("ckpt_10.pt")


def test_resume_continues_iterations_and_matches_uninterrupted_run(tmp_path):
    tok, tok_path = _train_tok(tmp_path)

    def score(prompt: str, texts: list[str]) -> list[float]:
        return [float(len(t)) for t in texts]

    prompts = [f"p{i}: " for i in range(10)]
    kw = dict(prompts_per_iter=2, group_size=2, lr=1e-3, beta=0.04, clip_eps=0.2, temp=1.0,
              max_new=4, micro_batch=4, save_every=0, dump_every=0, seed=21, device="cpu",
              warmup_iters=1)

    def fresh():
        return (_tiny_model(vocab=tok.vocab_size, seed=2),
                _tiny_model(vocab=tok.vocab_size, seed=2))

    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    p, r = fresh()
    run_grpo(p, r, tok, prompts, score, a_dir, tok_path, iters=4, **kw)
    p, r = fresh()
    run_grpo(p, r, tok, prompts, score, b_dir, tok_path, iters=2, **kw)
    p, r = fresh()  # fresh objects: resume must restore weights + optimizer from ckpt_2
    run_grpo(p, r, tok, prompts, score, b_dir, tok_path, iters=4, **kw)

    la = [json.loads(x) for x in (a_dir / "grpo_log.jsonl").read_text().splitlines()]
    lb = [json.loads(x) for x in (b_dir / "grpo_log.jsonl").read_text().splitlines()]
    assert [rec["iter"] for rec in lb] == [1, 2, 3, 4]
    for ra, rb in zip(la, lb, strict=True):
        assert ra["prompt_indices"] == rb["prompt_indices"]  # sampler pure in (seed, iter)
    # The resumed run's trajectory matches the uninterrupted one (full state restored).
    assert la[-1]["reward_mean"] == pytest.approx(lb[-1]["reward_mean"])


def test_run_grpo_refuses_completed_out_dir(tmp_path):
    tok, tok_path = _train_tok(tmp_path)

    def score(prompt: str, texts: list[str]) -> list[float]:
        return [0.0 for _ in texts]

    p = _tiny_model(vocab=tok.vocab_size, seed=2)
    r = _tiny_model(vocab=tok.vocab_size, seed=2)
    out = tmp_path / "done"
    kw = dict(prompts_per_iter=1, group_size=2, lr=1e-3, beta=0.0, clip_eps=0.2, temp=1.0,
              max_new=2, micro_batch=2, save_every=0, dump_every=0, seed=3, device="cpu",
              warmup_iters=1)
    run_grpo(p, r, tok, ["q: "], score, out, tok_path, iters=1, **kw)
    with pytest.raises(ValueError, match="already"):
        run_grpo(p, r, tok, ["q: "], score, out, tok_path, iters=1, **kw)
