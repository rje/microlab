# coder-1b at step 20,000 — halfway, measured

Written 2026-08-06. Step 20,000 of 40,000; 10.49B of ~21B tokens. All numbers from
`ckpt_20000` evaluated locally on the mix-v2 val set, plus the trainer's own val print
from the shipped episode log.

## Headline: val 1.3734, slope steepening again

| step | tokens | trainer val (block 32k) |
|---|---|---|
| 4,000 | 2.1B | 1.5593 |
| 10,000 | 5.2B | 1.4655 |
| 14,000 | 7.3B | 1.4245 |
| 16,000 | 8.4B | 1.4138 |
| **20,000** | **10.5B** | **1.3734** |

The 14k→16k stretch looked like the start of a plateau (−0.011 over 2k steps); 16k→20k
recovered to −0.040 over 4k steps. No leakage tripwire (1.3) crossed.

## Per-slice val (block 4096): every slice still improving

| slice | 4,000 | 16,000 | 20,000 | Δ 16k→20k |
|---|---|---|---|---|
| code | 1.281 | 1.1451 | 1.1096 | −0.036 |
| web | 3.367 | 3.1146 | 3.0589 | −0.056 |
| math | 2.563 | 2.3558 | 2.3002 | −0.056 |
| markdown | 2.293 | 2.1010 | 2.0364 | −0.065 |
| arxiv | 1.436 | 1.3116 | 1.2601 | −0.052 |
| commits | 1.345 | 1.2034 | 1.1580 | −0.045 |

No slice diverging behind the aggregate.

## FIM: monotone improvement continues

Middle-span loss 0.758 @4k → 0.735 @10k → 0.705 @16k → **0.6965 @20k** (ppl 2.01,
n=63/64 scorable). As of this milestone the suite computes FIM itself (the val-shard
path was cloud-specific before; `MICROLAB_MIX_DIR` now overrides).

## Repetition: the 16k movement did not continue

Greedy loop rate 1.00 @4k → 1.00 @10k → 0.875 @16k → **0.875 @20k** (7/8 prompts loop).
The first movement at 16k has not extended; at 10.5B tokens greedy long-horizon decoding
still falls into attractors. Watch item stands: per the 4k doc, "not falling by ~8–10B
tokens" is now the live condition — one more flat reading at 22k–24k makes this a real
conversation about decoding-side mitigations vs training-side causes.

Syntax parse rate 4/6 (0.83 @10k, 0.50 @16k — n=6, treat as noise band, not trend).

## Qualitative: first fully-correct multi-function completion

From the frozen sweep (`evals/trajectory/coder-1b-trajectory.md`), step-20k column:

- `fn-skeleton` produced `add`/`subtract`/`multiply`/`divide` — correct bodies, correct
  family extension. Best completion of the whole run so far.
- `self-binding` now writes `self.count += 1` (the increment body is right), then clones
  `Counter_1`, `Counter_2` — local semantics arrived, global stopping has not.
- `argparse` invents plausible `add_argument` flags and a correct `parse_args()`/open
  sequence before decaying into a `data.strip()` loop.
- The four long-budget prompts (first columns at this milestone) confirm the attractors
  persist at 256–512 tokens rather than resolving: `binary-search` still wraps itself
  recursively (`binary_search_binary_search`), `class-long` initializes an LRU cache
  plausibly then oscillates between two attribute assignments.

## Code execution evals: first non-zero

| suite | tasks | pass@1 @20k |
|---|---|---|
| HumanEval | 164 | 0.000 (0/164) |
| MBPP | 257 | **0.0078 (2/257)** |

MBPP is the first non-zero execution result of the run — two problems solved end to end
under the sandboxed Python executor, greedy `--mode base`. Expected to stay near the
floor at compute-optimal scale (`syntax_valid` and FIM are the early-moving signals; the
landscape survey found no published ~1B model clearing ~10 HumanEval at 21–26B raw-code
tokens). Recorded per-task in `evals/suite/coder-1b-20000-{humaneval,mbpp}.jsonl`.

Two harness fixes shipped with this milestone: `generate_until` now builds the cache via
`build_cache` (a hand-rolled `KVCache` crashed on the KDA/MLA hybrid with the same
`conv_hist` error the console hit at 4k), and the suite passes `--mode base` explicitly
(a raw pretrain checkpoint has no `serve_config.json` for `--mode auto`). MultiPL-E JS/TS
are dropped from the default set until a node executor exists — they were recording rc=1
noise, not zeros.

## Run economics at the halfway mark

- 29 episodes; preemption exposure now bounded by milestone-first uploads (the 19,700
  preemption cost ~100 steps).
- Banked spend $87.5 at step 20,000 — but banked figures before this milestone recorded
  the bid FLOOR, not the sent bid; real charges run ~15–20% higher (account credit is
  ground truth going forward; the supervisor now prints and banks the actual price).
- Market drift: the Japan SXM floor rose ~25% this week; the supervisor now auto-rents
  on-demand when it beats the planned bid by ≤10% premium (twice this session the
  interruptible price exceeded the machine's own on-demand ask).
- Forecast to 40k at the current $2.00/h and ~5.6 s/step: ~32 h, ~$64.
