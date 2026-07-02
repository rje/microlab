"""Serving core + endpoint: stream correctness, limits, auth, and failure modes."""

import pytest
import torch

from microlab.console import serve
from microlab.console.app import create_app
from microlab.model.reference.variants import VariantConfig, VariantGPT


class StubTok:
    """Minimal encode/decode over byte values — no tokenizer.json needed."""

    vocab_size = 64

    def encode(self, text):
        return [ord(c) % 64 for c in text]

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


def _tiny_state():
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=64, block_size=64, n_layer=2, n_head=4, n_embd=32,
                        norm="rms", pos="rope", mlp="swiglu")
    return serve.ServeState(model=VariantGPT(cfg).eval(), tokenizer=StubTok(), step=10,
                            device="cpu")


def test_stream_yields_text_and_respects_max_tokens():
    state = _tiny_state()
    pieces = list(serve.stream_generate(state, "hi", max_new_tokens=8, temperature=0.0))
    assert pieces and all(isinstance(p, str) for p in pieces)
    # deltas reassemble into the decode of exactly 8 generated ids
    assert len(state.tokenizer.encode("".join(pieces))) <= 8 + 2  # decode/encode slack


def test_stream_deterministic_with_seed():
    state = _tiny_state()
    a = "".join(serve.stream_generate(state, "hi", 8, temperature=1.0, seed=7))
    b = "".join(serve.stream_generate(state, "hi", 8, temperature=1.0, seed=7))
    assert a == b


def test_limits_raise():
    state = _tiny_state()
    with pytest.raises(ValueError):
        list(serve.stream_generate(state, "hi", max_new_tokens=513))
    with pytest.raises(ValueError):  # prompt + budget must fit block_size (64)
        list(serve.stream_generate(state, "x" * 60, max_new_tokens=32))


def test_endpoint_auth_and_stream(tmp_path, monkeypatch):
    app = create_app(str(tmp_path))  # empty project root: no ckpt -> 503 path too
    monkeypatch.setattr(serve, "get_state", lambda: _tiny_state())
    client = app.test_client()
    body = {"prompt": "hi", "max_new_tokens": 4, "temperature": 0.0}
    # unauthenticated -> redirect to login (302) or 401, never generation
    r = client.post("/api/generate", json=body)
    assert r.status_code in (302, 401)
    # bearer token path
    token = (tmp_path / "instance" / "api_token").read_text().strip()
    r = client.post("/api/generate", json=body,
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.data.decode()  # streamed body reassembled by test client
    # bad token -> 401
    r = client.post("/api/generate", json=body,
                    headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    # over-limit -> 400 with a real message
    r = client.post("/api/generate", json={"prompt": "hi", "max_new_tokens": 9999},
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 400


def test_endpoint_503_when_no_checkpoint(tmp_path):
    app = create_app(str(tmp_path))
    token = (tmp_path / "instance" / "api_token").read_text().strip()
    r = app.test_client().post("/api/generate", json={"prompt": "hi"},
                               headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 503
