"""Serve the lab's own model from the console: lazy checkpoint load, a single-generation
lock, and a KV-cached streaming generator. The Phase 6 exercise stack (KVCache +
sample_next) IS the serving stack."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

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


def stream_generate(state: ServeState, prompt: str, max_new_tokens: int = 128,
                    temperature: float = 0.8, top_k: int | None = None,
                    top_p: float | None = None, seed: int | None = None) -> Iterator[str]:
    """Yield text DELTAS. Accumulate ids and re-decode the full completion each step so
    byte-level BPE never splits a multi-byte character across chunks.

    Argument limits are validated EAGERLY (at call, not first ``next()``) so the route can
    return a 400 before it commits to a streaming response: the checks run here, then an
    inner generator is returned to do the actual work under the generation lock."""
    if not 0 < max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be in (0, {MAX_NEW_TOKENS}]")
    cfg = state.model.config
    prompt_ids = state.tokenizer.encode(prompt) or [0]
    if len(prompt_ids) + max_new_tokens > cfg.block_size:
        raise ValueError(
            f"prompt ({len(prompt_ids)} tokens) + max_new_tokens ({max_new_tokens}) "
            f"exceeds block_size ({cfg.block_size})")
    gen = None if seed is None else torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def _run() -> Iterator[str]:
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

    return _run()
