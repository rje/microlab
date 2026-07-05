"""Serve the lab's own model from the console: lazy checkpoint load, a single-generation
lock, and a KV-cached streaming generator. The Phase 6 exercise stack (KVCache +
sample_next) IS the serving stack.

Multi-run: any run under ``<root>/runs/<name>`` that has a ``ckpt_*.pt`` can be served, but
only ONE model is resident at a time (training already uses ~24GB of the 48GB card, so a
second resident model would risk OOM). Switching runs, forcing a reload, or a newer
checkpoint appearing evicts the old model (drop refs + ``torch.cuda.empty_cache()``) before
loading the new one."""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch

from microlab.infer.reference.kv_cache import KVCache
from microlab.infer.reference.sampling import sample_next
from microlab.model.reference.checkpoint import latest_checkpoint, load_variant_from_run
from microlab.model.reference.sft import format_chat

MAX_NEW_TOKENS = 512


@dataclass
class ServeState:
    model: torch.nn.Module
    tokenizer: object
    step: int
    device: str
    run: str = ""
    # Set from <run_dir>/serve_config.json. A base run (no config) keeps raw-completion
    # behavior; a "chat" run wraps prompts with the SFT template and stops on stop_strings.
    mode: str = "base"
    stop_strings: list[str] = field(default_factory=list)


_state: ServeState | None = None
# Guards the resident-model slot: which run is loaded and swapping it out.
_state_lock = threading.Lock()
# Serializes the two operations that must never overlap on the single GPU: an in-flight
# generation, and evicting/loading a model in get_state. A per-ServeState lock could NOT do
# this — eviction nulls _state.model under _state_lock while a generation runs under a
# different lock, so it could free a model mid-stream. One shared lock makes a run switch
# WAIT for the active stream to finish instead. (Residual window documented in get_state.)
_gen_lock = threading.Lock()


def _anchor(root: Path, value: str) -> Path:
    """Resolve a serving path against the project root when it's relative; an absolute path
    (e.g. an explicit env override) is honored as-is."""
    p = Path(value)
    return p if p.is_absolute() else root / p


def _step_of(ckpt: Path) -> int:
    """Step number encoded in a ``ckpt_<step>.pt`` filename."""
    return int(ckpt.stem.split("_")[1])


def _read_serve_config(run_dir: Path) -> dict:
    """A run's ``serve_config.json`` (written by scripts/sft.py for chat runs), normalized to
    ``{"mode", "stop_strings"}``. No file (the pretraining runs) or an unset mode -> a base
    run: raw completion, no stop strings. A present-but-malformed file raises loudly rather
    than silently degrading to base (a broken config should be fixed, not masked)."""
    cfg_path = Path(run_dir) / "serve_config.json"
    if not cfg_path.exists():
        return {"mode": "base", "stop_strings": []}
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {"mode": data.get("mode", "base"), "stop_strings": data.get("stop_strings", [])}


def list_runs(root: Path) -> list[dict]:
    """Every run under ``<root>/runs/<name>`` that has at least one ``ckpt_*.pt``, sorted by
    name. Cheap: reads the filesystem only (no torch load). ``latest_step`` is taken from the
    newest checkpoint's filename — the same number the checkpoint records internally.
    ``mode`` ("base"/"chat") comes from the run's serve_config.json so the UI can tell them
    apart.

        [{"name": "150m", "latest_step": 6000, "mode": "base"}, ...]
    """
    runs_dir = Path(root) / "runs"
    if not runs_dir.is_dir():
        return []
    out: list[dict] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpts = list(run_dir.glob("ckpt_*.pt"))
        if not ckpts:
            continue
        out.append({"name": run_dir.name, "latest_step": max(_step_of(c) for c in ckpts),
                    "mode": _read_serve_config(run_dir)["mode"]})
    return out


def _default_run_name(root: Path) -> str:
    """The run served when the caller doesn't name one: the basename of ``MICROLAB_SERVE_RUN``
    if set (e.g. ``runs/150m`` -> ``150m``), else the first available run. Raises
    FileNotFoundError when nothing is trainable/servable so the route returns a 503."""
    configured = os.environ.get("MICROLAB_SERVE_RUN")
    if configured:
        return Path(configured).name
    runs = list_runs(root)
    if not runs:
        raise FileNotFoundError(
            f"no runs with a ckpt_*.pt under {Path(root) / 'runs'} — train one first "
            "(scripts/pretrain.py) or set MICROLAB_SERVE_RUN")
    return runs[0]["name"]


def _tokenizer_path(root: Path, run_name: str, run_dir: Path) -> Path:
    """Resolve the tokenizer for a run, failing LOUDLY rather than falling back to a wrong
    one (a 350m checkpoint decoded with the 150m tokenizer is garbage). First existing wins:

    1. run-local ``<run_dir>/tokenizer.json`` — co-located with the checkpoints, unambiguous;
       this is the convention to prefer (drop/symlink the training data-dir's tokenizer.json
       into the run dir).
    2. per-run env ``MICROLAB_SERVE_TOKENIZER_<NAME>`` (name upper-cased, non-alphanumerics
       -> ``_``; e.g. run ``350m`` -> ``MICROLAB_SERVE_TOKENIZER_350M``) — an explicit map for
       when the tokenizer can't live in the run dir.
    3. legacy ``MICROLAB_SERVE_TOKENIZER`` — honored ONLY for the default run, so the existing
       single-run deploy keeps working without pinning the wrong tokenizer onto other runs.

    Returns a path that EXISTS. Raises FileNotFoundError otherwise."""
    local = run_dir / "tokenizer.json"
    if local.exists():
        return local

    key = "MICROLAB_SERVE_TOKENIZER_" + re.sub(r"[^0-9A-Za-z]", "_", run_name).upper()
    override = os.environ.get(key)
    if override:
        p = _anchor(root, override)
        if not p.exists():
            raise FileNotFoundError(f"{key}={override} points at a missing tokenizer ({p})")
        return p

    legacy = os.environ.get("MICROLAB_SERVE_TOKENIZER")
    if legacy and run_name == _default_run_name(root):
        p = _anchor(root, legacy)
        if not p.exists():
            raise FileNotFoundError(
                f"MICROLAB_SERVE_TOKENIZER={legacy} points at a missing tokenizer ({p})")
        return p

    raise FileNotFoundError(
        f"no tokenizer for run '{run_name}': expected {local}, or set {key}=<path to the "
        "tokenizer.json that produced this run's shards>. Refusing to guess — a wrong "
        "tokenizer decodes to garbage.")


def active() -> dict | None:
    """The currently-resident run and the step of its loaded checkpoint, or None if nothing
    is loaded yet. Read-only; does not trigger a load."""
    with _state_lock:
        if _state is None:
            return None
        return {"name": _state.run, "step": _state.step}


def get_state(root: Path, run: str | None = None, reload: bool = False) -> ServeState:
    """Serve one run at a time, resident until superseded. Relative run/tokenizer paths (env
    or default) resolve against ``root`` — the app's PROJECT_ROOT — NOT the process CWD.

    ``run=None`` serves the default run (``MICROLAB_SERVE_RUN`` basename, else the first
    available). The resident model is reused UNLESS the requested run differs, ``reload`` is
    set, or a newer checkpoint exists than the loaded step — in which case the old model is
    evicted (refs dropped + CUDA cache emptied) BEFORE the new one loads, so peak VRAM stays
    at one model. Device from ``MICROLAB_SERVE_DEVICE`` (default cpu).

    Raises FileNotFoundError (missing run/checkpoint/tokenizer) with setup instructions — the
    route maps that to a 503. Both cheap checks (checkpoint present, tokenizer resolvable)
    happen BEFORE any eviction, so a bad request never destroys a working resident model."""
    global _state
    with _state_lock:
        run_name = run or _default_run_name(root)
        run_dir = Path(root) / "runs" / run_name
        device = os.environ.get("MICROLAB_SERVE_DEVICE", "cpu")

        latest_step = _step_of(latest_checkpoint(run_dir))  # raises -> 503, old model kept
        cached_ok = (
            _state is not None
            and _state.run == run_name
            and not reload
            and latest_step <= _state.step
        )
        if cached_ok:
            return _state

        tok_path = _tokenizer_path(root, run_name, run_dir)  # raises -> 503, old model kept

        # Commit under _gen_lock so eviction can NOT null a model an in-flight generation is
        # using: a switch/reload here WAITS for the active stream to finish. Nested inside
        # _state_lock, but stream_generate only ever takes _gen_lock (never _state_lock), so
        # there's no lock-order inversion / deadlock.
        #
        # Residual (NOT closed by this lock, and out of scope): between get_state releasing
        # _gen_lock below and stream_generate re-acquiring it, a second concurrent request
        # could evict — nulling the ServeState this caller already holds -> a mid-stream
        # crash. Fully closing it needs whole-request serialization (the route holding the
        # lock across the streamed body). Unreachable from the serialized UI and the
        # sequential eval harness, so it's a documented limitation, not a live bug.
        with _gen_lock:
            # Evict the resident model first so we never hold two at once.
            if _state is not None:
                _state.model = None
                _state = None
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

            model, step = load_variant_from_run(run_dir, device=device)
            from microlab.tokenizer.fast import FastTokenizer

            serve_cfg = _read_serve_config(run_dir)
            _state = ServeState(model=model, tokenizer=FastTokenizer.load(str(tok_path)),
                                step=step, device=device, run=run_name,
                                mode=serve_cfg["mode"], stop_strings=serve_cfg["stop_strings"])
        return _state


def _stop_scan(text: str, stop_strings: list[str]) -> tuple[int, bool]:
    """How much of ``text`` is safe to emit now, and whether generation should stop.

    Returns (safe_len, hit). If any stop string is present, safe_len is the index of the
    EARLIEST one and hit is True — the caller emits up to (excluding) it and stops. Otherwise
    hit is False and safe_len holds back the longest trailing suffix of ``text`` that is a
    proper prefix of some stop string: emitting it now could leak the start of a stop marker
    (e.g. "###") that only the next token reveals as a stop. That held-back tail is released
    once it's known NOT to be a stop string."""
    earliest: int | None = None
    for s in stop_strings:
        i = text.find(s)
        if i != -1:
            earliest = i if earliest is None else min(earliest, i)
    if earliest is not None:
        return earliest, True
    hold = 0
    for s in stop_strings:
        for k in range(min(len(text), len(s) - 1), 0, -1):
            if text.endswith(s[:k]):
                hold = max(hold, k)
                break
    return len(text) - hold, False


def stream_generate(state: ServeState, prompt: str, max_new_tokens: int = 128,
                    temperature: float = 0.8, top_k: int | None = None,
                    top_p: float | None = None, seed: int | None = None,
                    raw: bool = False) -> Iterator[str]:
    """Yield text DELTAS. Accumulate ids and re-decode the full completion each step so
    byte-level BPE never splits a multi-byte character across chunks.

    Chat vs base is a property of the served run (``state.mode``): a chat run wraps the
    incoming prompt with the SFT template (``format_chat``) and stops when the completion
    contains one of ``state.stop_strings``. ``raw=True`` forces base-style raw completion even
    on a chat model (no template, no stop strings) so the Playground can show a side-by-side.

    Argument limits are validated EAGERLY (at call, not first ``next()``) so the route can
    return a 400 before it commits to a streaming response: the checks run here, then an
    inner generator is returned to do the actual work under the generation lock."""
    if not 0 < max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be in (0, {MAX_NEW_TOKENS}]")
    cfg = state.model.config
    chat = state.mode == "chat" and not raw
    gen_prompt = format_chat(prompt)[0] if chat else prompt
    stop_strings = state.stop_strings if chat else []
    prompt_ids = state.tokenizer.encode(gen_prompt) or [0]
    if len(prompt_ids) + max_new_tokens > cfg.block_size:
        raise ValueError(
            f"prompt ({len(prompt_ids)} tokens) + max_new_tokens ({max_new_tokens}) "
            f"exceeds block_size ({cfg.block_size})")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be None or >= 1 (got {top_k})")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError(f"top_p must be None or in (0, 1] (got {top_p})")
    # The RNG generator must live on the same device as the logits multinomial() samples
    # from — a CPU generator against CUDA logits raises. Only bites GPU serving with a seed.
    gen = None if seed is None else torch.Generator(device=state.device).manual_seed(seed)

    @torch.no_grad()
    def _run() -> Iterator[str]:
        # The single shared generation lock — held for the whole stream so a run
        # switch/reload in get_state can't evict this model mid-generation.
        with _gen_lock:
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
                if stop_strings:
                    # Detect stop markers on the FULL decoded completion, emit only the delta
                    # up to the safe point, and stop at the earliest marker.
                    safe_len, hit = _stop_scan(text, stop_strings)
                    if safe_len > len(emitted):
                        yield text[len(emitted):safe_len]
                        emitted = text[:safe_len]
                    if hit:
                        break
                elif len(text) > len(emitted):
                    yield text[len(emitted):]
                    emitted = text
                if cache.seq_len >= cfg.block_size:
                    break
                logits, _ = state.model(nxt, kv_cache=cache)

    return _run()
