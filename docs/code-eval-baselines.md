# Code + tool-call eval baselines (pre-code-training floor)

Baselines from the new code-and-tool eval harness (`scripts/eval_code.py`,
`scripts/eval_toolcall.py`; executor, prompts and scoring in `src/microlab/evals/code/`),
run 2026-07-29 — before any code-specialist training. These numbers ARE the point: the
floor the coding-specialist program must beat. Machine-readable copy:
`evals/code/baselines.json`; per-item records in `evals/code/*.jsonl`.

## HumanEval (execution-verified, greedy pass@1)

| run | mode | pass@1 | dominant failures |
|---|---|---|---|
| 1b-4k-chat (step 4074) | chat | **0.0** (0/164) | 108 SyntaxError (prose instead of code), 42 AssertionError (runnable but wrong), 8 NameError, 0 timeouts |

The FineWeb-only 1B chat model writes English *about* the function instead of Python
most of the time; when it does emit runnable code (the 42 AssertionErrors — usually an
echo of the prompt signature with a trivial body) it never passes the tests. Exactly the
expected pre-code-training floor.

## Tool-call eval (120 items: 60 pattern / 60 compositional)

| run | tool acc | strict acc | arg F1 (tool correct) | parse failures | pattern / compositional tool acc |
|---|---|---|---|---|---|
| 1b-4k-chat (4074) | 4.2% | 0.8% | 0.20 | 100/120 | 5.0% / 3.3% |
| 350m-sft-mix (14193) | 0.8% | 0.8% | 1.00 | 114/120 | 0.0% / 1.7% |

Both models fail predominantly on *JSON discipline*, not routing: replies imitate the
few-shot shape but drop closing braces, repeat keys (`"tool": "tool":`), or echo the
demo's `"Asia/Tokyo"` arguments. The expected tool name appears anywhere in the raw
reply only 15% (1B) / 9% (350M) of the time, so routing is also near-floor. The
nonzero compositional cells are mostly lucky `clarify` echoes of the few-shot.

## Deferred (GPU busy with ablation arms)

- HumanEval on `runs/1b` in base/completion mode (same ~0 expectation).
- MBPP (sanitized) baselines — runner implemented + tested, just not run.
- Sampled pass@10 (`--n 10 --temperature 0.8`): 10x generation cost for a floor at 0.

## Repro

```
python scripts/eval_code.py --run runs/1b-4k-chat --dataset humaneval \
    --out evals/code/1b-4k-chat-humaneval.jsonl
python scripts/eval_toolcall.py --run runs/1b-4k-chat \
    --out evals/code/1b-4k-chat-toolcall.jsonl
python scripts/eval_toolcall.py --run runs/350m-sft-mix \
    --out evals/code/350m-sft-mix-toolcall.jsonl
```

Both runners are deterministic (greedy, fixed few-shot), write per-item JSONL
progressively, and resume: rerunning with the same `--out` skips finished items (the
config is pinned in the file's header line; a mismatch raises).
