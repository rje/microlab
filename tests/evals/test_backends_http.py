"""MicrolabHTTPBackend: one real HTTP round-trip against the console's /api/generate.

Spins the Flask app with a stubbed tiny serve-state (reusing test_serve.py's
_tiny_state/StubTok pattern) behind werkzeug.serving.make_server on port 0 in a
thread, points the backend at it with the token read from the app's instance dir,
and asserts a non-empty completion, a recorded latency, and that a wrong token
turns into an HTTPError via raise_for_status.
"""

import threading

import pytest
import requests
import torch
from werkzeug.serving import make_server

from microlab.console import serve
from microlab.console.app import create_app
from microlab.evals.backends import MicrolabHTTPBackend, create_backend
from microlab.evals.schema import EvalTask
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


def _task(prompt: str = "once upon a time") -> EvalTask:
    return EvalTask(id="t1", category="story", prompt=prompt, checks=[])


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    """Real HTTP server on an ephemeral port; token lives in the app's instance dir."""
    monkeypatch.setattr(serve, "get_state", lambda: _tiny_state())
    app = create_app(str(tmp_path))
    server = make_server("127.0.0.1", 0, app, threaded=True)
    host = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token_file = tmp_path / "instance" / "api_token"
    try:
        yield host, token_file
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_round_trip_reads_token_from_instance_dir(live_server):
    host, token_file = live_server
    backend = MicrolabHTTPBackend(host, token_file=str(token_file),
                                  max_new_tokens=8, temperature=0.0)
    output = backend.generate(_task())
    assert output.task_id == "t1"
    assert output.text  # non-empty completion streamed back over real HTTP
    assert output.latency_seconds >= 0


def test_wrong_token_raises_http_error(live_server):
    host, _ = live_server
    backend = MicrolabHTTPBackend(host, token="not-the-real-token")
    with pytest.raises(requests.HTTPError):
        backend.generate(_task())


def test_requires_token_or_token_file():
    with pytest.raises(ValueError):
        MicrolabHTTPBackend("http://127.0.0.1:9")


def test_create_backend_builds_microlab_http(tmp_path):
    token_file = tmp_path / "api_token"
    token_file.write_text("secret-token")
    backend = create_backend(
        {
            "type": "microlab_http",
            "host": "http://127.0.0.1:5000/",
            "token_file": str(token_file),
            "max_new_tokens": 64,
            "temperature": 0.0,
        }
    )
    assert isinstance(backend, MicrolabHTTPBackend)
    assert backend.host == "http://127.0.0.1:5000"  # trailing slash stripped
    assert backend.token == "secret-token"
    assert backend.max_new_tokens == 64
