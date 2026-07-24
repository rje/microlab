"""scripts/lmeval_microlab.py: the lm-eval-harness LM adapter for VariantGPT. Everything
here runs on CPU with a tiny model + tiny trained FastTokenizer — no real 1B needed. The
manual scorers below implement the documented spec independently, so a bug in the adapter's
alignment/truncation/batching shows up as a mismatch, not as two copies of the same bug."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.tokenizer.fast import FastTokenizer

_SPEC = importlib.util.spec_from_file_location(
    "lmeval_microlab", Path(__file__).resolve().parents[2] / "scripts" / "lmeval_microlab.py")
lmeval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lmeval)

BLOCK = 32  # tiny so truncation and rolling windows are exercised with short texts


@pytest.fixture(scope="module")
def tiny():
    tok = FastTokenizer.train(
        ["the quick brown fox jumps over the lazy dog",
         "hello world says the fox", "the dog sleeps all day long"] * 4,
        vocab_size=300)
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=tok.vocab_size, block_size=BLOCK, n_layer=2, n_head=2,
                        n_embd=16, norm="rms", pos="rope", mlp="swiglu")
    model = VariantGPT(cfg).eval()
    lm = lmeval.MicrolabLM(model=model, tokenizer=tok, device="cpu", batch_size=4)
    return lm, tok, model


def _req(ctx: str, cont: str) -> SimpleNamespace:
    """Stand-in for lm_eval's Instance — the adapter only reads .args."""
    return SimpleNamespace(args=(ctx, cont))


def _manual_ll(model, ctx_toks, cont_toks, max_len):
    """Reference scorer: full sequence forward, log-softmax, sum ONLY the continuation
    positions. Left-truncates to the last max_len+1 tokens, exactly the documented policy."""
    full = (ctx_toks + cont_toks)[-(max_len + 1):]
    logits, _ = model(torch.tensor([full[:-1]], dtype=torch.long))
    logprobs = F.log_softmax(logits.float(), dim=-1)[0]
    contlen = len(cont_toks)
    rows = logprobs[len(full) - 1 - contlen:len(full) - 1]
    tgt = torch.tensor(cont_toks)
    ll = rows.gather(1, tgt.unsqueeze(1)).sum().item()
    return ll, bool(rows.argmax(dim=-1).eq(tgt).all())


def test_loglikelihood_scores_only_continuation_tokens(tiny):
    lm, tok, model = tiny
    ctx, cont = "the quick brown", " fox"
    (got_ll, got_greedy), = lm.loglikelihood([_req(ctx, cont)], disable_tqdm=True)
    ctx_toks, cont_toks = lm._encode_pair(ctx, cont)
    assert ctx_toks + cont_toks == tok.encode(ctx + cont)  # split covers the whole string
    want_ll, want_greedy = _manual_ll(model, ctx_toks, cont_toks, BLOCK)
    assert got_ll == pytest.approx(want_ll, rel=1e-5)
    assert got_greedy == want_greedy


def test_encode_pair_moves_trailing_context_space_into_continuation(tiny):
    lm, tok, _ = tiny
    ctx_toks, cont_toks = lm._encode_pair("the quick ", "fox")
    assert ctx_toks == tok.encode("the quick")
    assert ctx_toks + cont_toks == tok.encode("the quick fox")


def test_is_greedy_true_for_argmax_continuation(tiny):
    lm, tok, model = tiny
    ctx_toks = tok.encode("hello world")
    logits, _ = model(torch.tensor([ctx_toks]))
    greedy_tok = int(logits[0, -1].argmax())
    (_, is_greedy), = lm._loglikelihood_tokens(
        [(None, ctx_toks, [greedy_tok])], disable_tqdm=True)
    assert is_greedy is True
    not_greedy_tok = (greedy_tok + 1) % tok.vocab_size
    (_, is_greedy), = lm._loglikelihood_tokens(
        [(None, ctx_toks, [not_greedy_tok])], disable_tqdm=True)
    assert is_greedy is False


def test_empty_context_conditions_on_eot_prefix(tiny):
    lm, tok, model = tiny
    (got_ll, _), = lm.loglikelihood([_req("", "hello world")], disable_tqdm=True)
    want_ll, _ = _manual_ll(model, [tok.eot_token], tok.encode("hello world"), BLOCK)
    assert got_ll == pytest.approx(want_ll, rel=1e-5)


def test_long_context_is_left_truncated_to_block_size(tiny):
    lm, tok, model = tiny
    ctx = "the quick brown fox jumps over the lazy dog and " * 4  # well past 32 tokens
    cont = "sleeps"
    ctx_toks, cont_toks = lm._encode_pair(ctx, cont)
    assert len(ctx_toks) + len(cont_toks) > BLOCK + 1  # would exceed the model without truncation
    (got_ll, _), = lm.loglikelihood([_req(ctx, cont)], disable_tqdm=True)
    want_ll, _ = _manual_ll(model, ctx_toks, cont_toks, BLOCK)
    assert got_ll == pytest.approx(want_ll, rel=1e-5)


def test_batched_matches_one_at_a_time(tiny):
    lm, tok, model = tiny
    pairs = [("the quick", " brown fox"), ("hello", " world"), ("the dog", " sleeps"),
             ("the quick brown fox jumps", " over"), ("", "the lazy dog"),
             ("says the", " fox"), ("all day", " long")]
    reqs = [_req(c, k) for c, k in pairs]
    batched = lm.loglikelihood(reqs, disable_tqdm=True)  # batch_size=4 -> mixed-length batches
    single = [lm.loglikelihood([r], disable_tqdm=True)[0] for r in reqs]
    for (b_ll, b_greedy), (s_ll, s_greedy) in zip(batched, single, strict=True):
        assert b_ll == pytest.approx(s_ll, abs=1e-5)  # right-padding must not leak into scores
        assert b_greedy == s_greedy


def test_rolling_short_string_equals_empty_context_loglikelihood(tiny):
    lm, *_ = tiny
    s = "the quick brown fox"
    (rolling,) = lm.loglikelihood_rolling([SimpleNamespace(args=(s,))], disable_tqdm=True)
    (ll, _), = lm.loglikelihood([_req("", s)], disable_tqdm=True)
    assert rolling == pytest.approx(ll, rel=1e-5)


def test_rolling_long_string_covers_every_token_once(tiny):
    lm, tok, model = tiny
    s = "the quick brown fox jumps over the lazy dog and the dog sleeps all day long " * 3
    toks = tok.encode(s)
    assert len(toks) > BLOCK  # forces multiple rolling windows
    (got,) = lm.loglikelihood_rolling([SimpleNamespace(args=(s,))], disable_tqdm=True)
    from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
    windows = [make_disjoint_window(w) for w in get_rolling_token_windows(
        token_list=toks, prefix_token=tok.eot_token, max_seq_len=BLOCK, context_len=1)]
    assert sum(len(pred) for _, pred in windows) == len(toks)  # each token predicted once
    want = sum(_manual_ll(model, ctx, pred, BLOCK)[0] for ctx, pred in windows)
    assert got == pytest.approx(want, rel=1e-5)


def test_generate_until_is_not_implemented(tiny):
    lm, *_ = tiny
    with pytest.raises(NotImplementedError):
        lm.generate_until([SimpleNamespace(args=("hi", {}))])


def test_tok_encode_rejects_add_special_tokens_true(tiny):
    lm, *_ = tiny
    with pytest.raises(ValueError, match="special tokens"):
        lm.tok_encode("hi", add_special_tokens=True)


def test_continuation_longer_than_block_raises(tiny):
    lm, *_ = tiny
    with pytest.raises(ValueError, match="continuation"):
        lm._loglikelihood_tokens([(None, [1], list(range(BLOCK + 1)))], disable_tqdm=True)


def test_load_lm_rejects_vocab_mismatch(tmp_path, tiny):
    _, tok, _ = tiny
    cfg = VariantConfig(vocab_size=tok.vocab_size + 1, block_size=BLOCK, n_layer=1, n_head=2,
                        n_embd=16, norm="rms", pos="rope", mlp="swiglu")
    torch.save({"model": VariantGPT(cfg).state_dict(), "step": 1, "cfg": cfg},
               tmp_path / "ckpt_1.pt")
    tok._tok.save(str(tmp_path / "tokenizer.json"))
    with pytest.raises(ValueError, match="vocab"):
        lmeval.load_lm(tmp_path, device="cpu", batch_size=2)


def test_load_lm_builds_working_adapter_from_run_dir(tmp_path, tiny):
    _, tok, _ = tiny
    cfg = VariantConfig(vocab_size=tok.vocab_size, block_size=BLOCK, n_layer=1, n_head=2,
                        n_embd=16, norm="rms", pos="rope", mlp="swiglu")
    torch.save({"model": VariantGPT(cfg).state_dict(), "step": 7, "cfg": cfg},
               tmp_path / "ckpt_7.pt")
    tok._tok.save(str(tmp_path / "tokenizer.json"))
    lm, step = lmeval.load_lm(tmp_path, device="cpu", batch_size=2)
    assert step == 7
    assert lm.max_length == BLOCK
    (ll, _), = lm.loglikelihood([_req("the quick", " fox")], disable_tqdm=True)
    assert ll < 0.0
