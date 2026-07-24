"""Run EleutherAI's lm-evaluation-harness over a microlab VariantGPT run, so our numbers
are apples-to-apples with public models evaluated by the same harness version.

    python scripts/lmeval_microlab.py --run runs/1b \\
        --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai \\
        --out runs/lmeval_1b.json

Implements the harness's `LM` interface (via `TemplateLM`, which supplies the standard
context/continuation string splitting) for likelihood tasks only; `generate_until` raises.

Tokenizer / special-token policy: FastTokenizer is a byte-level BPE whose `encode` never
adds special tokens — its ONLY special token is <|endoftext|>. There is no BOS, matching
how the model was pretrained (documents separated by <|endoftext|> in a flat stream). We
therefore feed task text verbatim, and use <|endoftext|> as the conditioning prefix where
the harness needs one: for empty-context requests and as the first token of rolling
perplexity windows (`prefix_token_id`, same convention as GPT-2/Pythia in the harness).

Context handling: the model's block_size is 1024, so each scored sequence is LEFT-truncated
to the last block_size+1 tokens — continuations keep at most block_size tokens of context.
This matters for lambada_openai rolling perplexity and any long-context doc: models with
2048-token windows (Pythia, TinyLlama) see more context than we can. Truncating the
continuation itself would silently mis-score, so that raises instead.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from lm_eval import simple_evaluate
from lm_eval.api.model import TemplateLM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window, make_table
from tqdm import tqdm

from microlab.model.reference.checkpoint import load_variant_from_run
from microlab.tokenizer.fast import FastTokenizer


class MicrolabLM(TemplateLM):
    """lm-eval adapter for a VariantGPT + FastTokenizer pair (likelihood requests only)."""

    def __init__(self, model, tokenizer: FastTokenizer, device: str = "cuda",
                 batch_size: int = 32) -> None:
        super().__init__()
        self.model = model.eval().to(device)
        self.tok = tokenizer
        self._device = torch.device(device)
        self.batch_size = batch_size

    @property
    def eot_token_id(self) -> int:
        return self.tok.eot_token

    @property
    def max_length(self) -> int:
        return self.model.config.block_size

    def tok_encode(self, string: str, add_special_tokens: bool | None = None,
                   **kwargs) -> list[int]:
        # FastTokenizer.encode never adds special tokens, so True is a request we cannot
        # honor — raise rather than silently return unmarked ids.
        if add_special_tokens:
            raise ValueError("FastTokenizer never adds special tokens on encode")
        return self.tok.encode(string)

    def _model_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass; bf16 autocast on CUDA (weights stay fp32), plain fp32 on CPU
        (unit tests). Returns [B, T, vocab] logits."""
        with torch.no_grad():
            if x.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = self.model(x)
            else:
                logits, _ = self.model(x)
        return logits

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False,
                              **kwargs) -> list[tuple[float, bool]]:
        """Score token-level (cache_key, context_toks, continuation_toks) requests.

        Only continuation positions contribute: for a row (ctx + cont) left-truncated to
        block_size+1 tokens, the model input is that row minus its last token, and we read
        the log-softmax at the positions predicting each continuation token. Requests are
        sorted by length (descending) so batches are length-homogeneous, then right-padded —
        with causal attention, pad tokens after a row's real tokens cannot affect its scores.
        """
        results: list[tuple[float, bool] | None] = [None] * len(requests)
        order = sorted(range(len(requests)),
                       key=lambda i: len(requests[i][1]) + len(requests[i][2]), reverse=True)
        for start in tqdm(range(0, len(order), self.batch_size), desc="loglikelihood batches",
                          disable=disable_tqdm):
            idxs = order[start:start + self.batch_size]
            rows = []
            for i in idxs:
                _, ctx_toks, cont_toks = requests[i]
                if not cont_toks:
                    raise ValueError("empty continuation cannot be scored")
                if len(cont_toks) > self.max_length:
                    raise ValueError(
                        f"continuation of {len(cont_toks)} tokens exceeds block_size "
                        f"{self.max_length}; truncating it would silently mis-score")
                inp = (ctx_toks + cont_toks)[-(self.max_length + 1):][:-1]
                rows.append((i, inp, cont_toks))
            width = max(len(inp) for _, inp, _ in rows)
            x = torch.zeros((len(rows), width), dtype=torch.long, device=self._device)
            for r, (_, inp, _) in enumerate(rows):
                x[r, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            logprobs = F.log_softmax(self._model_logits(x).float(), dim=-1)
            for r, (i, inp, cont_toks) in enumerate(rows):
                sl = logprobs[r, len(inp) - len(cont_toks):len(inp)]  # [contlen, vocab]
                tgt = torch.tensor(cont_toks, device=self._device)
                is_greedy = bool(sl.argmax(dim=-1).eq(tgt).all())
                ll = float(sl.gather(1, tgt.unsqueeze(1)).sum())
                results[i] = (ll, is_greedy)
                if requests[i][0] is not None:  # rolling windows pass cache_key=None
                    self.cache_hook.add_partial("loglikelihood", requests[i][0], (ll, is_greedy))
        return results  # type: ignore[return-value]  # every slot was filled above

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        """Full-string log-likelihood: split each doc into the harness's standard rolling
        windows (each token predicted exactly once, later windows keep block_size-1 tokens
        of real context), score them as continuation-only requests, and sum per doc."""
        windows: list[tuple[None, list[int], list[int]]] = []
        counts: list[int] = []
        for req in requests:
            (string,) = req.args
            toks = self.tok_encode(string)
            if not toks:
                raise ValueError("cannot compute rolling loglikelihood of an empty string")
            doc_windows = [make_disjoint_window(w) for w in get_rolling_token_windows(
                token_list=toks, prefix_token=self.prefix_token_id,
                max_seq_len=self.max_length, context_len=1)]
            windows.extend((None, ctx, pred) for ctx, pred in doc_windows)
            counts.append(len(doc_windows))
        scored = self._loglikelihood_tokens(windows, disable_tqdm=disable_tqdm)
        out, pos = [], 0
        for req, n in zip(requests, counts, strict=True):
            total = sum(ll for ll, _ in scored[pos:pos + n])
            pos += n
            out.append(total)
            self.cache_hook.add_partial("loglikelihood_rolling", (req.args[0],), total)
        return out

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        raise NotImplementedError("MicrolabLM only supports likelihood tasks")


def load_lm(run_dir: str | Path, device: str, batch_size: int) -> tuple[MicrolabLM, int]:
    """Build the adapter from a run directory (latest ckpt_*.pt + tokenizer.json)."""
    run_dir = Path(run_dir)
    model, step = load_variant_from_run(run_dir, device=device)
    tok = FastTokenizer.load(str(run_dir / "tokenizer.json"))
    if tok.vocab_size != model.config.vocab_size:
        raise ValueError(f"tokenizer vocab {tok.vocab_size} != model vocab "
                         f"{model.config.vocab_size} in {run_dir}")
    return MicrolabLM(model=model, tokenizer=tok, device=device, batch_size=batch_size), step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run dir with ckpt_*.pt + tokenizer.json")
    ap.add_argument("--tasks", required=True, help="comma-separated lm-eval task names")
    ap.add_argument("--out", required=True, help="path for the results JSON")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap docs per task (smoke tests only — not comparable numbers)")
    args = ap.parse_args()

    lm, step = load_lm(args.run, device=args.device, batch_size=args.batch_size)
    n_params = lm.model.num_params()
    print(f"loaded {args.run} @ step {step} ({n_params / 1e6:.0f}M params, "
          f"block_size {lm.max_length}) on {args.device}")

    t0 = time.time()
    results = simple_evaluate(model=lm, tasks=args.tasks.split(","), num_fewshot=0,
                              limit=args.limit)
    elapsed = time.time() - t0

    results["microlab"] = {"run": str(args.run), "step": step, "params": n_params,
                           "block_size": lm.max_length, "batch_size": args.batch_size,
                           "limit": args.limit, "seconds": round(elapsed, 1)}
    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(make_table(results))
    print(f"wrote {out} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
