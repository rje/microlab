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


@pytest.fixture(autouse=True)
def _reset_serve_state():
    """get_state caches a module-level singleton; reset it around every test so a real load
    in one test can't leak into the next (and vice versa)."""
    serve._state = None
    yield
    serve._state = None


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


@pytest.mark.gpu
def test_stream_on_cuda_with_seed_no_device_mismatch():
    # Regression: a seeded generation on a CUDA model needs a CUDA RNG generator — a CPU
    # generator against CUDA logits makes torch.multinomial raise. Only bites GPU+seed.
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=64, block_size=64, n_layer=2, n_head=4, n_embd=32,
                        norm="rms", pos="rope", mlp="swiglu")
    state = serve.ServeState(model=VariantGPT(cfg).eval().cuda(), tokenizer=StubTok(),
                             step=10, device="cuda")
    a = "".join(serve.stream_generate(state, "hi", 8, temperature=1.0, seed=7))
    b = "".join(serve.stream_generate(state, "hi", 8, temperature=1.0, seed=7))
    assert a and a == b  # runs without raising, and is deterministic


def test_limits_raise():
    state = _tiny_state()
    with pytest.raises(ValueError):
        list(serve.stream_generate(state, "hi", max_new_tokens=513))
    with pytest.raises(ValueError):  # prompt + budget must fit block_size (64)
        list(serve.stream_generate(state, "x" * 60, max_new_tokens=32))


def test_endpoint_auth_and_stream(tmp_path, monkeypatch):
    app = create_app(str(tmp_path))  # get_state is stubbed below, so no real ckpt is loaded
    monkeypatch.setattr(serve, "get_state", lambda root: _tiny_state())
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
    # non-ASCII token -> 401, never a 500 (compare_digest(str, str) rejects non-ASCII)
    r = client.post("/api/generate", json=body,
                    headers={"Authorization": "Bearer café"})
    assert r.status_code == 401
    # over-limit -> 400 with a real message
    r = client.post("/api/generate", json={"prompt": "hi", "max_new_tokens": 9999},
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 400
    # eager sampler validation: an invalid top_k -> 400 before any streaming commits
    r = client.post("/api/generate", json={"prompt": "hi", "top_k": -3},
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 400


def test_endpoint_503_when_no_checkpoint(tmp_path):
    # No get_state stub: the empty tmp_path project root has no runs/150m, so get_state
    # (which now anchors on PROJECT_ROOT, not CWD) raises FileNotFoundError -> 503.
    app = create_app(str(tmp_path))
    token = (tmp_path / "instance" / "api_token").read_text().strip()
    r = app.test_client().post("/api/generate", json={"prompt": "hi"},
                               headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 503


def test_endpoint_503_ignores_cwd(tmp_path, monkeypatch):
    """Regression for the CWD hazard: get_state must anchor on the app's PROJECT_ROOT, not
    the process CWD. Simulate the main-checkout case — a runs/150m sitting in the CWD — and
    prove it is NOT loaded when the app's project root is a different, empty dir. Pre-fix the
    stray checkpoint got picked up (a 500 from torch.load on garbage, or a 200 with the real
    model); post-fix the empty root wins -> a clean 503."""
    checkout = tmp_path / "checkout"
    (checkout / "runs" / "150m").mkdir(parents=True)
    (checkout / "runs" / "150m" / "ckpt_10.pt").write_bytes(b"not a real checkpoint")
    monkeypatch.chdir(checkout)
    root = tmp_path / "empty"
    root.mkdir()
    app = create_app(str(root))
    token = (root / "instance" / "api_token").read_text().strip()
    r = app.test_client().post("/api/generate", json={"prompt": "hi"},
                               headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 503
