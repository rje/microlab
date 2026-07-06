"""Serving core + endpoint: stream correctness, limits, auth, and failure modes."""

import threading
from pathlib import Path

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
    with pytest.raises(ValueError):  # repetition_penalty out of [1.0, 2.0]
        list(serve.stream_generate(state, "hi", max_new_tokens=8, repetition_penalty=3.0))
    with pytest.raises(ValueError):
        list(serve.stream_generate(state, "hi", max_new_tokens=8, repetition_penalty=0.5))


def test_repetition_penalty_generates_without_error():
    # a valid penalty threads through the loop (prev_ids grows each step) and still streams
    state = _tiny_state()
    out = "".join(serve.stream_generate(state, "hi", max_new_tokens=8, temperature=0.0,
                                        repetition_penalty=1.3))
    assert isinstance(out, str)


def test_endpoint_auth_and_stream(tmp_path, monkeypatch):
    app = create_app(str(tmp_path))  # get_state is stubbed below, so no real ckpt is loaded
    monkeypatch.setattr(serve, "get_state", lambda root, run=None, reload=False: _tiny_state())
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


# --- multi-run serving: list_runs, run-switching/eviction, reload, tokenizer resolution ---


class _FakeModel(torch.nn.Module):
    """Stand-in for a loaded VariantGPT — get_state never runs a forward pass on it."""


def _fake_run(root: Path, name: str, steps, tokenizer=True) -> Path:
    """A run dir with fake (non-loadable) ckpt_<step>.pt files; the model load is stubbed in
    these tests, so the file contents don't matter — only their presence and names do."""
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True)
    for step in steps:
        (run_dir / f"ckpt_{step}.pt").write_bytes(b"x")
    if tokenizer:
        (run_dir / "tokenizer.json").write_text("{}")
    return run_dir


@pytest.fixture
def stub_load(monkeypatch):
    """Patch the heavy model load + tokenizer load so get_state's control flow (selection,
    caching, eviction, reload) can be exercised without a real checkpoint. Returns the list of
    (run_name, step) loads so a test can count them."""
    loads: list[tuple[str, int]] = []

    def fake_load(run_dir, device="cpu"):
        step = max(int(p.stem.split("_")[1]) for p in Path(run_dir).glob("ckpt_*.pt"))
        loads.append((Path(run_dir).name, step))
        return _FakeModel(), step

    class _FakeFast:
        @staticmethod
        def load(path):
            return StubTok()

    monkeypatch.setattr(serve, "load_variant_from_run", fake_load)
    monkeypatch.setattr("microlab.tokenizer.fast.FastTokenizer", _FakeFast)
    return loads


def test_list_runs(tmp_path):
    _fake_run(tmp_path, "150m", [200, 6000, 400])
    _fake_run(tmp_path, "350m", [500, 4000])
    (tmp_path / "runs" / "empty").mkdir()  # a run dir with no checkpoints is skipped
    (tmp_path / "runs" / "notes.txt").write_text("x")  # a stray file is skipped
    base_dec = serve.DEFAULT_DECODING["base"]
    assert serve.list_runs(tmp_path) == [
        {"name": "150m", "latest_step": 6000, "mode": "base", "decoding": base_dec},
        {"name": "350m", "latest_step": 4000, "mode": "base", "decoding": base_dec},
    ]


def test_list_runs_no_runs_dir(tmp_path):
    assert serve.list_runs(tmp_path) == []


def test_serve_config_decoding_defaults_and_override(tmp_path):
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    # no config file -> base-mode decoding defaults
    assert serve._read_serve_config(run)["decoding"] == serve.DEFAULT_DECODING["base"]
    # chat mode -> chat decoding defaults
    (run / "serve_config.json").write_text('{"mode": "chat", "stop_strings": []}')
    assert serve._read_serve_config(run)["decoding"] == serve.DEFAULT_DECODING["chat"]
    # a run may override individual params, merged over the mode defaults
    (run / "serve_config.json").write_text(
        '{"mode": "chat", "stop_strings": [], "decoding": {"repetition_penalty": 1.5}}')
    dec = serve._read_serve_config(run)["decoding"]
    assert dec["repetition_penalty"] == 1.5  # overridden
    assert dec["temperature"] == serve.DEFAULT_DECODING["chat"]["temperature"]  # others kept


def test_get_state_selects_and_caches(tmp_path, stub_load):
    _fake_run(tmp_path, "150m", [10, 20])
    s1 = serve.get_state(tmp_path, run="150m")
    assert s1.run == "150m" and s1.step == 20
    s2 = serve.get_state(tmp_path, run="150m")  # same run, no newer ckpt -> reuse
    assert s2 is s1
    assert stub_load == [("150m", 20)]  # loaded exactly once
    assert serve.active() == {"name": "150m", "step": 20}


def test_get_state_switch_evicts_old_model(tmp_path, stub_load):
    _fake_run(tmp_path, "150m", [20])
    _fake_run(tmp_path, "350m", [5])
    s1 = serve.get_state(tmp_path, run="150m")
    s2 = serve.get_state(tmp_path, run="350m")  # different run -> evict + load
    assert s2.run == "350m" and s2.step == 5
    assert s1.model is None  # the evicted state had its model ref dropped (bounds VRAM)
    assert stub_load == [("150m", 20), ("350m", 5)]


def test_run_switch_waits_for_active_generation(tmp_path, stub_load):
    # Concurrency: generation and eviction share ONE lock, so a run switch must WAIT for an
    # in-flight stream instead of nulling the model under it. Hold _gen_lock to stand in for
    # an active generation, then prove a switching get_state blocks and doesn't evict.
    _fake_run(tmp_path, "a", [10])
    _fake_run(tmp_path, "b", [5])
    s1 = serve.get_state(tmp_path, run="a")
    result = {}

    def switch():
        result["state"] = serve.get_state(tmp_path, run="b")

    serve._gen_lock.acquire()
    worker = threading.Thread(target=switch)
    try:
        worker.start()
        worker.join(timeout=0.3)
        assert worker.is_alive()  # blocked on _gen_lock -> the switch waits for the stream
        assert s1.model is not None  # the in-use model was NOT evicted mid-generation
    finally:
        serve._gen_lock.release()  # the "generation" finishes and releases the lock
    worker.join(timeout=3)
    assert not worker.is_alive()  # no deadlock: the switch completes once the lock frees
    assert result["state"].run == "b" and s1.model is None


def test_get_state_reload_and_newer_checkpoint(tmp_path, stub_load):
    run_dir = _fake_run(tmp_path, "150m", [10])
    s1 = serve.get_state(tmp_path, run="150m")
    assert s1.step == 10
    (run_dir / "ckpt_20.pt").write_bytes(b"x")  # training writes a newer checkpoint
    s2 = serve.get_state(tmp_path, run="150m")  # auto-detects newer step -> reload
    assert s2 is not s1 and s2.step == 20
    s3 = serve.get_state(tmp_path, run="150m", reload=True)  # force even with no newer ckpt
    assert s3 is not s2 and s3.step == 20
    assert stub_load == [("150m", 10), ("150m", 20), ("150m", 20)]


def test_get_state_missing_checkpoint_keeps_resident(tmp_path, stub_load):
    _fake_run(tmp_path, "150m", [10])
    (tmp_path / "runs" / "350m").mkdir()  # exists but has no ckpt
    s1 = serve.get_state(tmp_path, run="150m")
    with pytest.raises(FileNotFoundError):
        serve.get_state(tmp_path, run="350m")
    assert s1.model is not None  # a failed switch didn't evict the working model
    assert serve.active() == {"name": "150m", "step": 10}


def test_tokenizer_resolution_fails_loudly(tmp_path, stub_load, monkeypatch):
    # A run with a checkpoint but NO resolvable tokenizer must raise, not silently decode with
    # some other run's tokenizer (which would produce garbage).
    monkeypatch.delenv("MICROLAB_SERVE_TOKENIZER", raising=False)
    monkeypatch.delenv("MICROLAB_SERVE_TOKENIZER_350M", raising=False)
    _fake_run(tmp_path, "350m", [5], tokenizer=False)
    with pytest.raises(FileNotFoundError, match="no tokenizer for run '350m'"):
        serve.get_state(tmp_path, run="350m")


def test_tokenizer_per_run_env_override(tmp_path, stub_load, monkeypatch):
    _fake_run(tmp_path, "350m", [5], tokenizer=False)
    tok = tmp_path / "elsewhere" / "tokenizer.json"
    tok.parent.mkdir()
    tok.write_text("{}")
    monkeypatch.setenv("MICROLAB_SERVE_TOKENIZER_350M", str(tok))
    s = serve.get_state(tmp_path, run="350m")  # resolves via the per-run env, no raise
    assert s.run == "350m"


def test_serve_runs_endpoint(tmp_path):
    _fake_run(tmp_path, "150m", [6000])
    _fake_run(tmp_path, "350m", [4000])
    app = create_app(str(tmp_path))
    token = (tmp_path / "instance" / "api_token").read_text().strip()
    client = app.test_client()
    assert client.get("/api/serve/runs").status_code in (302, 401)  # unauth -> no listing
    r = client.get("/api/serve/runs", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    payload = r.get_json()
    base_dec = serve.DEFAULT_DECODING["base"]
    assert payload["runs"] == [
        {"name": "150m", "latest_step": 6000, "mode": "base", "decoding": base_dec},
        {"name": "350m", "latest_step": 4000, "mode": "base", "decoding": base_dec},
    ]
    assert payload["active"] is None  # nothing loaded yet


def test_generate_passes_run_field(tmp_path, monkeypatch):
    app = create_app(str(tmp_path))
    captured = {}

    def fake_get_state(root, run=None, reload=False):
        captured["run"] = run
        return _tiny_state()

    monkeypatch.setattr(serve, "get_state", fake_get_state)
    token = (tmp_path / "instance" / "api_token").read_text().strip()
    r = app.test_client().post(
        "/api/generate",
        json={"prompt": "hi", "max_new_tokens": 4, "temperature": 0.0, "run": "350m"},
        headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert captured["run"] == "350m"
    # a non-string run is a 400, never handed to the serving layer
    r = app.test_client().post("/api/generate", json={"prompt": "hi", "run": 7},
                               headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 400


def test_reload_endpoint(tmp_path, monkeypatch):
    app = create_app(str(tmp_path))
    calls = {}

    def fake_get_state(root, run=None, reload=False):
        calls["run"], calls["reload"] = run, reload
        state = _tiny_state()
        state.run, state.step = run or "150m", 42
        return state

    monkeypatch.setattr(serve, "get_state", fake_get_state)
    token = (tmp_path / "instance" / "api_token").read_text().strip()
    client = app.test_client()
    assert client.post("/api/serve/reload", json={}).status_code in (302, 401)  # unauth
    r = client.post("/api/serve/reload", json={"run": "350m"},
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.get_json() == {"run": "350m", "step": 42}
    assert calls == {"run": "350m", "reload": True}


def test_reload_endpoint_503_when_unservable(tmp_path, monkeypatch):
    app = create_app(str(tmp_path))

    def boom(root, run=None, reload=False):
        raise FileNotFoundError("no checkpoint")

    monkeypatch.setattr(serve, "get_state", boom)
    token = (tmp_path / "instance" / "api_token").read_text().strip()
    r = app.test_client().post("/api/serve/reload", json={},
                               headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 503
    assert "error" in r.get_json()


def test_get_state_real_checkpoint_end_to_end(tmp_path):
    """A genuine on-disk checkpoint + a real run-local tokenizer.json, loaded through
    get_state and driven to a real streamed completion — proves list_runs -> get_state ->
    stream_generate works with real artifacts (no monkeypatching), and that the run-local
    tokenizer convention resolves. CPU-only, tiny, so it stays in the default suite."""
    from microlab.tokenizer.fast import FastTokenizer

    run_dir = tmp_path / "runs" / "mini"
    run_dir.mkdir(parents=True)
    tok = FastTokenizer.train(["hello world", "once upon a time", "the cat sat"] * 4,
                              vocab_size=300, save_path=str(run_dir / "tokenizer.json"))
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=4,
                        n_embd=32, norm="rms", pos="rope", mlp="swiglu")
    model = VariantGPT(cfg)
    torch.save({"model": model.state_dict(), "step": 30, "cfg": cfg},
               run_dir / "ckpt_30.pt")

    assert serve.list_runs(tmp_path) == [
        {"name": "mini", "latest_step": 30, "mode": "base",
         "decoding": serve.DEFAULT_DECODING["base"]}]
    state = serve.get_state(tmp_path, run="mini")
    assert state.run == "mini" and state.step == 30 and state.mode == "base"
    out = "".join(serve.stream_generate(state, "hello", max_new_tokens=6, temperature=0.0))
    assert isinstance(out, str) and out


# --- chat-aware serving: serve_config -> mode/stop_strings, template wrapping, stop-strings ---


class _WordTok:
    """Decode joins a per-id text fragment, so a scripted model can produce arbitrary text
    (including a stop string) deterministically. encode is irrelevant to these tests."""

    vocab_size = 32

    def __init__(self, frags: dict[int, str]) -> None:
        self.frags = frags

    def encode(self, text):
        return [0]

    def decode(self, ids):
        return "".join(self.frags.get(i, "") for i in ids)


class _ScriptedModel(torch.nn.Module):
    """Emits a fixed script of token ids (argmax at temperature 0). Ignores the KV cache for
    output; stream_generate only reads logits[:, -1, :]. With fill_cache=True it drives the
    cache to capacity so the block_size exit branch fires (the cache doesn't otherwise fill,
    since this stub never appends real K/V)."""

    def __init__(self, script: list[int], cfg: VariantConfig, fill_cache: bool = False) -> None:
        super().__init__()
        self.script = script
        self.config = cfg
        self.fill_cache = fill_cache
        self._i = 0

    def forward(self, idx, kv_cache=None):
        if self.fill_cache and kv_cache is not None:
            kv_cache.seq_len = kv_cache.capacity  # force the "out of context" break
        b, t = idx.shape
        logits = torch.zeros(b, t, self.config.vocab_size)
        logits[:, -1, self.script[min(self._i, len(self.script) - 1)]] = 10.0
        self._i += 1
        return logits, None


def _scripted_state(frags, script, *, mode="chat", stop_strings=("### End",), block_size=64,
                    fill_cache=False):
    cfg = VariantConfig(vocab_size=32, block_size=block_size, n_layer=1, n_head=2, n_embd=8,
                        norm="rms", pos="rope", mlp="swiglu")
    model = _ScriptedModel(script, cfg, fill_cache=fill_cache)
    return serve.ServeState(model=model, tokenizer=_WordTok(frags), step=1, device="cpu",
                            mode=mode, stop_strings=list(stop_strings))


def test_get_state_reads_chat_serve_config(tmp_path, stub_load):
    run_dir = _fake_run(tmp_path, "350m-sft", [5])
    (run_dir / "serve_config.json").write_text(
        '{"mode": "chat", "stop_strings": ["### End", "\\n### Instruction:"]}')
    state = serve.get_state(tmp_path, run="350m-sft")
    assert state.mode == "chat"
    assert state.stop_strings == ["### End", "\n### Instruction:"]
    # and list_runs surfaces the mode + chat decoding defaults for the UI
    assert {"name": "350m-sft", "latest_step": 5, "mode": "chat",
            "decoding": serve.DEFAULT_DECODING["chat"]} in serve.list_runs(tmp_path)


def test_chat_mode_wraps_prompt_and_stops_before_stop_string(monkeypatch):
    seen = []

    def spy_format_chat(prompt, context="", response=""):
        seen.append(prompt)
        return ("### Instruction:\n" + prompt + "\n\n### Response:\n", "")

    monkeypatch.setattr(serve, "format_chat", spy_format_chat)
    state = _scripted_state({1: "Hello", 2: " world", 3: "### End", 4: " leaked"},
                            script=[1, 2, 3, 4, 4])
    out = "".join(serve.stream_generate(state, "hi there", max_new_tokens=10, temperature=0.0))
    assert seen == ["hi there"]          # the chat template wrapped the user message
    assert out == "Hello world"          # truncated at the stop string, which is excluded


def test_chat_mode_does_not_leak_partial_stop_prefix(monkeypatch):
    # The stop string arrives across two tokens ("###" then " End"). The "###" prefix must be
    # held back, not streamed, or the client sees a half-emitted stop marker.
    monkeypatch.setattr(serve, "format_chat", lambda p, context="", response="": (p, ""))
    state = _scripted_state({1: "Hi ", 2: "###", 3: " End"}, script=[1, 2, 3])
    out = "".join(serve.stream_generate(state, "q", max_new_tokens=5, temperature=0.0))
    assert out == "Hi "  # neither the "###" partial nor the completed "### End" leaked


def test_base_mode_unchanged_ignores_stop_strings(monkeypatch):
    # A base run (no chat config) never stops on the marker text nor wraps the prompt.
    def boom(*a, **k):
        raise AssertionError("format_chat must not be called for a base run")

    monkeypatch.setattr(serve, "format_chat", boom)
    state = _scripted_state({1: "Hi ", 2: "### End", 3: " more"}, script=[1, 2, 3],
                            mode="base", stop_strings=[])
    out = "".join(serve.stream_generate(state, "q", max_new_tokens=3, temperature=0.0))
    assert out == "Hi ### End more"  # full completion, marker text and all


def test_raw_true_bypasses_chat_wrapping_and_stopping(monkeypatch):
    # raw=True forces base behavior on a chat model: no template, no stop truncation.
    def boom(*a, **k):
        raise AssertionError("format_chat must not be called when raw=True")

    monkeypatch.setattr(serve, "format_chat", boom)
    state = _scripted_state({1: "Hi ", 2: "### End", 3: " more"}, script=[1, 2, 3])
    out = "".join(serve.stream_generate(state, "q", max_new_tokens=3, temperature=0.0,
                                        raw=True))
    assert out == "Hi ### End more"


def test_chat_flushes_held_back_tail_at_token_cap(monkeypatch):
    # Regression: a reply cut off by max_new_tokens while a partial stop-prefix is held back
    # must still stream that tail — the reassembled stream must equal the full decode, not
    # silently drop the last few chars.
    monkeypatch.setattr(serve, "format_chat", lambda p, context="", response="": (p, ""))
    # "###" is a prefix of "### End" (held back); the reply never completes the stop marker.
    state = _scripted_state({1: "See ", 2: "###"}, script=[1, 2])
    out = "".join(serve.stream_generate(state, "q", max_new_tokens=2, temperature=0.0))
    assert out == "See ###"  # held-back "###" flushed at the cap, nothing dropped


def test_chat_flushes_held_back_newline_at_token_cap(monkeypatch):
    # A reply ending in "\n" (a prefix of the "\n### Instruction:" stop) is held back mid-
    # stream; hitting the cap must still flush it.
    monkeypatch.setattr(serve, "format_chat", lambda p, context="", response="": (p, ""))
    state = _scripted_state({1: "Answer", 2: "\n"}, script=[1, 2],
                            stop_strings=("### End", "\n### Instruction:"))
    out = "".join(serve.stream_generate(state, "q", max_new_tokens=2, temperature=0.0))
    assert out == "Answer\n"


def test_chat_flushes_held_back_tail_at_block_size(monkeypatch):
    # The OTHER exit path: running out of context (cache.seq_len >= block_size) with a partial
    # stop-prefix held back must also flush it.
    monkeypatch.setattr(serve, "format_chat", lambda p, context="", response="": (p, ""))
    state = _scripted_state({1: "###"}, script=[1], block_size=2, fill_cache=True)
    out = "".join(serve.stream_generate(state, "q", max_new_tokens=1, temperature=0.0))
    assert out == "###"  # flushed at the block_size break, not dropped


def test_chat_no_double_flush_after_stop_hit(monkeypatch):
    # When a stop string DID complete, the tail is intentionally dropped (truncated) — the new
    # flush logic must not resurrect it.
    monkeypatch.setattr(serve, "format_chat", lambda p, context="", response="": (p, ""))
    state = _scripted_state({1: "Hi ", 2: "### End", 3: " tail"}, script=[1, 2, 3])
    out = "".join(serve.stream_generate(state, "q", max_new_tokens=3, temperature=0.0))
    assert out == "Hi "  # stop hit -> truncated; " tail" and the marker stay dropped
