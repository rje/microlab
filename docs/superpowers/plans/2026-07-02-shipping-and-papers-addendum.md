# Shipping + Papers Addendum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 13 papers (with console reading content + phase reading-list updates) and the Phase 6 "Serve it" extension (streaming generation endpoint + Playground tab + eval-harness HTTP backend), per `docs/superpowers/specs/2026-07-02-shipping-and-papers-addendum-design.md`.

**Architecture:** Same delivery pattern as the merged expansion: feature-branch worktree, one implementer per task, per-task review, final review, merge+deploy. Serving reuses the Phase 6 inference reference (KVCache + sample_next) behind an authed Flask streaming route and a React Playground view.

**Tech Stack:** PyTorch, Flask (chunked streaming), React/TS console, pytest, `scripts/download_papers.py`.

## Global Constraints

- Run Python via `/home/rje/anaconda3/bin/conda run -n microlab <cmd>` from the worktree root.
- Worktree: `/home/rje/src/python/microlab-addendum` (branch `feat/serve-and-papers`); NEVER touch `/home/rje/src/python/microlab` (live training + live console).
- Ruff line length 100; default suite `pytest tests/ --ignore=tests/exercises -m "not gpu" -q` green after every commit (baseline: 269 passed).
- NEVER `git add -A` / `git add .` — stage explicit paths listed per task.
- Serving defaults: CPU device, lazy model load, single-generation lock, `max_new_tokens <= 512`. `/api/generate` requires session auth or bearer token — never open.
- Errors surface loudly (repo rule: no fallbacks that mask bugs). Missing checkpoint/tokenizer at serve time → 503 with instructions, not silent degradation.
- Commit trailers:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_018aPzfDwRdjhAE3v6CaCryE`

---

### Task 1: 13 papers — manifest + download + verified slugs

**Files:**
- Modify: `papers/manifest.json` (append 13 entries)
- Generated: PDFs + `papers/README.md` via `scripts/download_papers.py` (PDFs are gitignored; commit manifest+README only)

**Interfaces:**
- Produces: 13 paper ids (slugified titles) consumed by Tasks 2 and 5. Verify with `paper_id_for` and report the ACTUAL ids.

- [ ] **Step 1:** Append 13 entries matching the existing schema exactly (topic/title/authors/year/source_url/pdf_url/filename; `https://arxiv.org/abs/<id>` + `https://arxiv.org/pdf/<id>`; filename `YYYY-firstauthor-short-slug.pdf`):

| topic | title | authors | year | arXiv |
|---|---|---|---|---|
| modern-llm-recipes | Tulu 3: Pushing Frontiers in Open Language Model Post-Training | Lambert et al. | 2024 | 2411.15124 |
| modern-llm-recipes | 2 OLMo 2 Furious | OLMo Team et al. | 2025 | 2501.00656 |
| tokenizers-data | SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model | Allal et al. | 2025 | 2502.02737 |
| foundations | Muon is Scalable for LLM Training | Liu et al. | 2025 | 2502.16982 |
| foundations | Scaling Data-Constrained Language Models | Muennighoff et al. | 2023 | 2305.16264 |
| architecture | YaRN: Efficient Context Window Extension of Large Language Models | Peng et al. | 2023 | 2309.00071 |
| architecture | Better & Faster Large Language Models via Multi-token Prediction | Gloeckle et al. | 2024 | 2404.19737 |
| architecture | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention | Yuan et al. | 2025 | 2502.11089 |
| inference | Efficient Streaming Language Models with Attention Sinks | Xiao et al. | 2023 | 2309.17453 |
| inference | EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | Li et al. | 2024 | 2401.15077 |
| interpretability | Sparse Autoencoders Find Highly Interpretable Features in Language Models | Cunningham et al. | 2023 | 2309.08600 |
| evaluation | SWE-bench: Can Language Models Resolve Real-World GitHub Issues? | Jimenez et al. | 2023 | 2310.06770 |
| evaluation | tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains | Yao et al. | 2024 | 2406.12045 |

- [ ] **Step 2:** `conda run -n microlab python scripts/download_papers.py` → expect `failures=0 total=75`. THEN title-verify every new PDF: extract page-1 text (`pypdf`/`pdftotext`) and confirm it contains the manifest title (case-insensitive, punctuation-loose). An arXiv-ID/title mismatch = wrong paper: search arXiv for the exact title, fix the id/urls, re-download, and report the correction. Do not skip a failing paper.
- [ ] **Step 3:** Verify slugs: print `paper_id_for(e)` for the last 13 entries; report them.
- [ ] **Step 4:** Commit (`git add papers/manifest.json papers/README.md`): `feat: add 13 papers — Tulu3/OLMo2/SmolLM2, Muon, data-constrained, YaRN/MTP/NSA, sinks/EAGLE, SAE, SWE-bench/tau-bench`

---

### Task 2: Reading content for the 13 papers

Same fan-out as the expansion's Task 3 (controller dispatches one content agent per paper): each writes `content/papers/<id>/overview.json` + `cards.json` mirroring `content/papers/attention-is-all-you-need/*` key sets exactly, plus a one-key synopsis scratch file; controller assembles `site/content/synopses/addendum-2026-07.json`, validates (ids resolve, key-sets match, console loads), commits. Per-paper curriculum framing comes from the spec's phase mapping; EAGLE's synopsis covers EAGLE-1→3 + production adoption; NSA's notes the DeepSeek-V4 connection.

---

### Task 3: Checkpoint loader + serving core + endpoint

**Files:**
- Create: `src/microlab/model/reference/checkpoint.py`
- Modify: `scripts/interp_report.py`, `scripts/bench_inference.py` (import the shared loader, delete local copies)
- Create: `src/microlab/console/serve.py`
- Modify: `src/microlab/console/app.py` (`/api/generate` route + api-token helper)
- Test: `tests/console/test_serve.py` (mirror existing tests/console conventions — check `ls tests/` for where console tests live; if none exist, create `tests/console/` matching tests/interp's `__init__.py` convention)

**Interfaces:**
- Produces: `load_variant_from_run(run_dir: Path, device: str = "cpu") -> tuple[VariantGPT, int]`; `serve.get_state() -> ServeState` (fields: model, tokenizer, step, device, lock); `serve.stream_generate(state, prompt: str, max_new_tokens: int = 128, temperature: float = 0.8, top_k: int | None = None, top_p: float | None = None, seed: int | None = None) -> Iterator[str]`; route `POST /api/generate` (JSON in, chunked text/plain out); `load_or_create_api_token(instance_path) -> str` (file `instance/api_token`, 0600).

- [ ] **Step 1: Failing tests first** — `tests/console/test_serve.py`:

```python
"""Serving core + endpoint: stream correctness, limits, auth, and failure modes."""

import json

import pytest
import torch

from microlab.console import serve
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
    import sys
    sys.path.insert(0, "src")
    from microlab.console.app import create_app

    app = create_app(str(tmp_path))  # empty project root: no ckpt -> 503 path too
    monkeypatch.setattr(serve, "get_state", lambda: _tiny_state())
    client = app.test_client()
    body = {"prompt": "hi", "max_new_tokens": 4, "temperature": 0.0}
    # unauthenticated -> redirect to login (302), never generation
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
    import sys
    sys.path.insert(0, "src")
    from microlab.console.app import create_app

    app = create_app(str(tmp_path))
    token = (tmp_path / "instance" / "api_token").read_text().strip()
    r = app.test_client().post("/api/generate", json={"prompt": "hi"},
                               headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 503
```
(Adjust the module-import bootstrapping to however existing tests import `microlab.console` — check `tests/` for precedent; `pyproject`/conftest may already handle `src` on path. `get_state` in the 503 test is NOT monkeypatched — it must raise its FileNotFoundError naturally and the route must map it to 503.)

- [ ] **Step 2:** Run → fail (no `serve` module). **Step 3: Implement.**

`src/microlab/model/reference/checkpoint.py`:
```python
"""Load a trained VariantGPT from a run directory's latest checkpoint. Shared by the
interp report, the inference bench, and the console's serving endpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from microlab.model.reference.variants import VariantConfig, VariantGPT


def load_variant_from_run(run_dir: Path, device: str = "cpu") -> tuple[VariantGPT, int]:
    """Latest ckpt_*.pt by step number. Raises FileNotFoundError when none exists."""
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_*.pt in {run_dir}")
    ckpt = torch.load(ckpts[-1], map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, norm=cfg.norm, pos=cfg.pos,
        mlp=cfg.mlp,
    ))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["step"]
```
Refactor both scripts to `from microlab.model.reference.checkpoint import load_variant_from_run` (they print the loaded step themselves; keep their output lines).

`src/microlab/console/serve.py`:
```python
"""Serve the lab's own model from the console: lazy checkpoint load, a single-generation
lock, and a KV-cached streaming generator. The Phase 6 exercise stack (KVCache +
sample_next) IS the serving stack."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch

from microlab.infer.reference.kv_cache import KVCache
from microlab.infer.reference.sampling import sample_next
from microlab.model.reference.checkpoint import load_variant_from_run

MAX_NEW_TOKENS = 512


@dataclass
class ServeState:
    model: torch.nn.Module
    tokenizer: object
    step: int
    device: str
    lock: threading.Lock = field(default_factory=threading.Lock)


_state: ServeState | None = None
_state_lock = threading.Lock()


def get_state() -> ServeState:
    """Lazy singleton. Raises FileNotFoundError with setup instructions when the run
    dir or tokenizer is missing — the route maps that to a 503."""
    global _state
    with _state_lock:
        if _state is None:
            run_dir = Path(os.environ.get("MICROLAB_SERVE_RUN", "runs/150m"))
            tok_path = Path(os.environ.get(
                "MICROLAB_SERVE_TOKENIZER", "data/shards/tinystories/tokenizer.json"))
            device = os.environ.get("MICROLAB_SERVE_DEVICE", "cpu")
            model, step = load_variant_from_run(run_dir, device=device)
            if not tok_path.exists():
                raise FileNotFoundError(f"no tokenizer at {tok_path}")
            from microlab.tokenizer.fast import FastTokenizer

            _state = ServeState(model=model, tokenizer=FastTokenizer.load(str(tok_path)),
                                step=step, device=device)
        return _state


@torch.no_grad()
def stream_generate(state: ServeState, prompt: str, max_new_tokens: int = 128,
                    temperature: float = 0.8, top_k: int | None = None,
                    top_p: float | None = None, seed: int | None = None) -> Iterator[str]:
    """Yield text DELTAS. Accumulate ids and re-decode the full completion each step so
    byte-level BPE never splits a multi-byte character across chunks."""
    if not 0 < max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be in (0, {MAX_NEW_TOKENS}]")
    cfg = state.model.config
    prompt_ids = state.tokenizer.encode(prompt) or [0]
    if len(prompt_ids) + max_new_tokens > cfg.block_size:
        raise ValueError(
            f"prompt ({len(prompt_ids)} tokens) + max_new_tokens ({max_new_tokens}) "
            f"exceeds block_size ({cfg.block_size})")
    gen = None if seed is None else torch.Generator().manual_seed(seed)
    with state.lock:
        n_kv = getattr(cfg, "n_kv_head", None) or cfg.n_head
        cache = KVCache(cfg.n_layer, 1, n_kv, cfg.block_size,
                        cfg.n_embd // cfg.n_head, device=state.device)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=state.device)
        logits, _ = state.model(idx, kv_cache=cache)
        out_ids: list[int] = []
        emitted = ""
        for _ in range(max_new_tokens):
            nxt = sample_next(logits[:, -1, :], temperature=temperature, top_k=top_k,
                              top_p=top_p, generator=gen)
            out_ids.append(int(nxt[0, 0]))
            text = state.tokenizer.decode(out_ids)
            if len(text) > len(emitted):
                yield text[len(emitted):]
                emitted = text
            if cache.seq_len >= cfg.block_size:
                break
            logits, _ = state.model(nxt, kv_cache=cache)
```

`console/app.py` — token helper next to `_load_or_create_secret_key` (same pattern, file `api_token`), and the route (register with the other `/api/` routes):
```python
    @app.route("/api/generate", methods=["POST"])
    def api_generate():
        # Session auth OR bearer token (for the eval harness). Programmatic clients
        # can't do the login redirect dance; the token lives in instance/api_token.
        authed = bool(session.get("authed"))
        header = request.headers.get("Authorization", "")
        if not authed and header.startswith("Bearer "):
            if secrets.compare_digest(header.removeprefix("Bearer ").strip(),
                                      app.config["API_TOKEN"]):
                authed = True
            else:
                return jsonify({"error": "bad token"}), 401
        if not authed:
            return auth.unauthenticated_response()  # match login_required's behavior:
            # check auth.py for the exact helper/redirect used and mirror it
        body = request.get_json(force=True) or {}
        prompt = str(body.get("prompt", ""))
        if not prompt.strip():
            return jsonify({"error": "empty prompt"}), 400
        try:
            state = serve.get_state()
        except FileNotFoundError as exc:
            return jsonify({"error": f"model not servable: {exc}"}), 503
        try:
            stream = serve.stream_generate(
                state, prompt,
                max_new_tokens=int(body.get("max_new_tokens", 128)),
                temperature=float(body.get("temperature", 0.8)),
                top_k=int(body["top_k"]) if body.get("top_k") else None,
                top_p=float(body["top_p"]) if body.get("top_p") else None,
                seed=int(body["seed"]) if body.get("seed") is not None else None,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        headers = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
        return app.response_class(stream, mimetype="text/plain", headers=headers)
```
(Imports: `from microlab.console import serve`; `jsonify`; `secrets` already imported. `API_TOKEN` loaded in `create_app` via the new helper. IMPORTANT: `stream_generate` validates args EAGERLY (the ValueError raises at first `next()`, not at call) — either make it validate before returning the iterator by splitting into a validating wrapper that does the checks then returns the inner generator, or call the checks in the route; the tests pin 400-before-stream. Implement the split-wrapper form in serve.py: do limit checks in `stream_generate` BEFORE the `yield`-containing inner function and return that inner generator.)

- [ ] **Step 4:** Tests green; full default suite green; ruff clean. Verify no route-ordering conflict with the SPA catch-all (`/api/generate` is literal — fine, but confirm 405/404 behavior unaffected for other /api routes).
- [ ] **Step 5:** Commit (`git add src/microlab scripts tests`): `feat(serve): authed streaming generation endpoint backed by the phase-6 inference stack`

---

### Task 4: Playground tab

**Files:**
- Modify: `site/src/App.tsx` (view union gains `"playground"`, nav entry, `PlaygroundPanel`), `site/src/styles.css`

**Interfaces:**
- Consumes: `POST /api/generate` (Task 3), session-cookie auth (browser).

- [ ] **Step 1:** Extend the view state (`"phases" | "training" | "playground"`), add a nav button below Training (label "Playground", sub "Your model, live"), render `<main className="workspace workspace-full"><PlaygroundPanel /></main>` for the view (same grid-span treatment as Training).
- [ ] **Step 2:** `PlaygroundPanel` component:

```tsx
function PlaygroundPanel() {
  const [prompt, setPrompt] = useState("Once upon a time");
  const [output, setOutput] = useState("");
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(0); // 0 = off
  const [topP, setTopP] = useState(0); // 0 = off
  const [maxTokens, setMaxTokens] = useState(128);
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const generate = async () => {
    setRunning(true); setOutput(""); setStats(null); setError(null);
    const controller = new AbortController();
    abortRef.current = controller;
    const t0 = performance.now();
    let chars = 0;
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt, max_new_tokens: maxTokens, temperature,
          top_k: topK || null, top_p: topP || null,
        }),
        signal: controller.signal,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(body.error ?? `HTTP ${res.status}`);
      }
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const piece = decoder.decode(value, { stream: true });
        chars += piece.length;
        setOutput((prev) => prev + piece);
      }
      const secs = (performance.now() - t0) / 1000;
      setStats(`${(chars / Math.max(secs, 0.001)).toFixed(0)} chars/s · ${secs.toFixed(1)}s`);
    } catch (err) {
      if ((err as Error).name !== "AbortError") setError((err as Error).message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="playground-panel" aria-labelledby="playground-heading">
      {/* heading block matching TrainingPanel's structure */}
      {/* textarea for prompt; number/range inputs for temperature/topK/topP/maxTokens */}
      {/* Generate button (disabled while running) + Stop button (abortRef.current?.abort()) */}
      {/* <pre className="playground-output">{output}</pre>; stats line; error line */}
    </section>
  );
}
```
Fill in the JSX per the existing panel idioms (section-heading/eyebrow classes, same input styling family); this component logic block is normative, the markup follows house style. CSS: `.playground-panel` (flex column, gap), `.playground-output` (monospace, pre-wrap, bordered card, min-height ~40vh, scroll).

- [ ] **Step 3:** `cd site && npm run build` (green) and `npx vitest run` if the suite touches App. Commit (`git add site/src`): `feat(console): Playground tab — stream your own model with sampling controls`

---

### Task 5: HTTP eval backend + docs/content pass

**Files:**
- Modify: `src/microlab/evals/backends.py` (+`MicrolabHTTPBackend`, `create_backend` type `"microlab_http"`)
- Test: `tests/evals/test_backends_http.py` (spin the Flask test app with a stubbed serve state — reuse Task 3's fixtures pattern — and point the backend at it via `requests_mock`-free approach: use Flask test_client through a thin adapter, or run `app.test_server`? Simplest robust: use `werkzeug.serving.make_server` on port 0 in a thread for one real HTTP round-trip, stop in teardown)
- Modify: `docs/hand-write/phase6-inference.md` ("Serve it" section + EAGLE/NSA/sinks in readings), `site/content/phases.json` (readings updates for phases 1,2,4,5,6,8,9,13,15 + phase-6 summary sentence), `docs/curriculum.md` (row 6 run-for-real column), `plans/llm-lab-overview.md` (Phase 6 deliverables + key readings)

**Interfaces:**
- Consumes: paper ids from Task 1 (use the ACTUAL reported ids); `/api/generate` contract from Task 3.
- Produces: `MicrolabHTTPBackend(host, token=None, token_file=None, max_new_tokens=128, temperature=0.0, timeout_seconds=120)`.

- [ ] **Step 1:** Backend implementation (mirror OllamaBackend's shape):

```python
class MicrolabHTTPBackend(ModelBackend):
    """Evaluate the lab's own SERVED model over HTTP — the same harness that graded the
    Ollama baselines in Phase 0, pointed at /api/generate. Auth via the bearer token in
    instance/api_token."""

    def __init__(self, host: str, token: str | None = None, token_file: str | None = None,
                 max_new_tokens: int = 128, temperature: float = 0.0,
                 timeout_seconds: int = 120):
        self.host = host.rstrip("/")
        if token is None:
            if token_file is None:
                raise ValueError("provide token or token_file")
            token = Path(token_file).read_text(encoding="utf-8").strip()
        self.token = token
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def generate(self, task: EvalTask) -> ModelOutput:
        start = time.perf_counter()
        response = requests.post(
            f"{self.host}/api/generate",
            json={"prompt": task.prompt, "max_new_tokens": self.max_new_tokens,
                  "temperature": self.temperature, "seed": 0},
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return ModelOutput(task_id=task.id, text=response.text.strip(),
                           latency_seconds=time.perf_counter() - start)
```
(+ `from pathlib import Path` import; `create_backend` branch `"microlab_http"` mapping config keys host/token/token_file/max_new_tokens/temperature/timeout_seconds.)

- [ ] **Step 2:** Test — real HTTP round-trip against the app with a stubbed tiny state (werkzeug make_server on port 0, thread, teardown); assert non-empty text + latency recorded + wrong token raises HTTPError via raise_for_status.
- [ ] **Step 3:** Docs/content:
  - phases.json readings: phase-1 +SmolLM2; phase-2 +MTP; phase-4 +Muon, +data-constrained; phase-5 +SAE; phase-6 +sinks, +EAGLE, +NSA; phase-8 +YaRN, +OLMo2, +SmolLM2; phase-9 +Tulu3; phase-13 +Tulu3; phase-15 +SWE-bench, +tau-bench. Phase-6 summary: append a sentence that the phase ends by SERVING the model — the authed streaming Playground backed by the hand-written KV cache and sampler.
  - Guide "Serve it" section: endpoint + playground walkthrough, `microlab_http` eval config example, honest GGUF→Ollama stretch note (feasible: llama.cpp llama arch = RoPE/RMSNorm/SwiGLU; fiddly: weight mapping + byte-level BPE conversion; revisit after SFT), and one paragraph on the new readings: EAGLE (production speculative: draft head on target features, tree verification — what vLLM/SGLang ship), attention sinks (why streaming needs the first tokens), NSA (the post-GQA sparse-attention direction DeepSeek V4 productionized).
  - curriculum.md row 6 run-for-real: "inference bench + authed streaming Playground (console serves the 150M)". overview.md Phase 6: add served-endpoint deliverable + the three new readings.
- [ ] **Step 4:** Validations: phases.json ids resolve via `content.load_state`; full suite; ruff; `npm run build` still green. Commit (`git add src/microlab tests docs plans site/content`): `feat(evals+docs): microlab_http backend, serve-it guide, 13-paper reading lists`

---

### Task 5.5 (controller, fable): exercise-targeting audit

Read every `src/microlab/exercises/phaseNN_*.py` stub, its reference oracle, its
exercise tests, and its guide, across all 17 phases. Rubric per phase:
1. **Crux test:** is the hand-write the conceptually load-bearing mechanism of the
   phase (the thing you don't understand until you've written it), or is the hard part
   hiding oracle-only while the stub is a one-liner?
2. **Trivial-warmup check:** one-line stubs are fine as warm-ups ONLY next to a real
   crux in the same phase.
3. **Buildability:** does each stub have enough scaffolding (signatures, shapes,
   hints) to be attempted without reading the oracle first?
4. **New-material fit:** do the just-added readings (EAGLE, NSA, SAE, MTP, Muon…)
   suggest a missing exercise or a guide-documented stretch?
Output: a findings list (phase, verdict, proposed change: add-stub | move-from-oracle |
guide-stretch-note | leave), reviewed by the controller, then implemented as Task 5.6
by an opus subagent (stubs + exercise tests + guide notes, same conventions; oracle
implementations already exist for anything moved/added — no new oracles without
explicit controller sign-off).

### Task 5.6: implement accepted audit findings (opus)

Scope defined by 5.5's accepted findings. Same constraints as Tasks 3–5 (exercise
tests `exercise`-marked, NotImplementedError-only, default suite green, explicit-path
staging, standard trailers).

### Task 6 (controller): final review → merge → deploy → live verify

- Whole-branch review (fable) with per-task packages; fix wave if needed.
- Merge `feat/serve-and-papers` → main (no-ff); `scripts/download_papers.py` on main (13 new PDFs); `cd site && npm run build`; `systemctl --user restart microlab-site`.
- Live verification (authed): `/api/state` (17 phases, 75 papers); Playground via Playwright — log in, open Playground, generate ~40 tokens from the REAL model, assert streamed text appears; unauth POST `/api/generate` → 302/401; bearer-token curl generates text; `MicrolabHTTPBackend` smoke against the live endpoint with one fixture task.
- Training run still active; cleanup worktree+branch; ledger.
