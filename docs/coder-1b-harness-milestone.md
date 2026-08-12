# Test-time harness milestone: the guaranteed win banked; no-tests selection gated out

Written 2026-08-12, closing the harness program (spec:
`docs/superpowers/specs/2026-08-11-test-time-harness-design.md`; pre-registered bands:
`docs/coder-1b-harness-prediction.md`). All local, $0, v1 frozen throughout.

## Headline: with tests at inference, best-of-10 MORE THAN DOUBLES delivered MBPP

| metric (v1, measured this run) | greedy | single sample | **best-of-10 + tests** |
|---|---|---|---|
| MBPP delivered | 14.0% | 8.4% | **29.6% (76/257)** |
| HumanEval delivered | 3.0% | 2.2% | **7.3% (12/164)** |

This is the harness's product number, shipped as `scripts/generate_best_of.py` (3a,
tests-required; exit-code honest: 0 = passer, 3 = none passed, 2 = CLI misuse). Smoke:
a live run passed on candidate 3/4; the impossible-asserts case exits 3 with a warning.

## Prediction scorecard (every band, honestly)

| claim | predicted | measured | verdict |
|---|---|---|---|
| v1 HumanEval pass@10 | 6.1% ± 4 | 7.3% | ✓ in band |
| v1 MBPP pass@10 / pass@1 | 2–3× | **3.5×** (29.6/8.4) | ✗ miss, favorable — the reservoir is bigger than predicted |
| test-free MBPP vs standard | 20–60% relatively below | **≈ equal or slightly above** (31.5% vs 29.6% delivered; 10.5% vs 8.4% p@1) | ✗ miss — test-conditioning ≈ zero at 1.2B; the test-free prompt's explicit signature appears to matter more than the asserts |
| Unit-0 probe | wildcard, soft bands | **16% (13/79) → SYNTHESIS DEAD (<24%)** | pre-registered branch executed |
| Unit-2 clustering recovery 10–50% (powered) | — | **not scored**: the powered bank requires synthesized inputs; probe gated them out. Descriptive HumanEval only, underpowered (gap 9 < 15) → counts only, per the power falsifier | falsifier applied as designed |
| text-plurality <10% | — | descriptive counts only: plurality 6 vs first 3 (gap 9) — notably strong, but no % claim (underpowered) | not scored (power rule) |
| cluster_random ≈ cluster_shortest | within noise | 3 vs 4 (descriptive) | ✓ within noise |

No in-code falsifier fired: no method beat its oracle (the leakage check never raised);
the power gate engaged exactly as pre-registered.

## The third incapacity result: input synthesis is dead at 1.2B

Probe: **13/79 tasks (16%)** produced ≥2 discriminating inputs — below the 24% dead-band
line. Failure split: 20/79 produced no parseable calls at all; 46/79 produced calls that
ran but did not separate right from wrong code. Caveat recorded: mutant-based wrong-targets
barely discriminated (1/29) vs bank-sourced (12/50), so 16% may mildly understate — but
even bank-only is 24%, at the dead-band edge. This joins self-repair (1/221) and the
Phase-2 RL bracket as a consistent series: **at 1.2B, the model cannot act on feedback or
spec text to produce discriminating artifacts — capability harvesting must be selection
over its own samples, judged externally.**

Consequences, all pre-registered: CodeT-lite gated out; the powered no-tests comparison
cannot run; **3b (no-tests serving) is cancelled; the harness ships tests-required.**

## What surprised us (the two favorable band misses)

1. **The selection reservoir is bigger than predicted** — MBPP pass@10 is 3.5× pass@1.
   Best-of-k with verification is worth even more here than the theory said.
2. **Gold asserts in the prompt do not help v1** — the leakage-clean bank scored as well
   or better. The C1 contamination control was scientifically necessary (we could not have
   known this without building the clean bank), and the answer is that the feared
   conditioning is ~absent at this scale, while explicit signatures may matter more.

## Descriptive Unit-2 table (HumanEval, counts only — underpowered)

oracle 12 · text_plurality 6 · cluster_shortest 4 (76-task coverage) · cluster_random 3 ·
first 3 · self_tests gated out. Even descriptively, clustering ≤ plurality — consistent
with the pre-registered "execution signal adds nothing at this scale" outcome.

## Housekeeping done alongside (user-approved, strictly non-destructive)

v1's full resumable checkpoint archived to B2 (`coder-1b-instruct/ckpt_1485_full.pt`,
13.3GB, size-verified). Weights-only fp32 copies of the four eval-only runs written to
`runs-lean/` (13.3 → 4.4GB each); originals untouched. **Deferred deletions (user
decision):** full-fat originals of the three failed arms (~40GB), scratchpad `r2-ckpts/`
side copies (~60GB), stale smoke dirs — a ~100GB reclaim awaiting approval. bf16
conversion + parity check deferred to the inference-efficiency pass.

## Status and what's next

The harness program is complete: **the with-tests win is banked and productized; the
no-tests path is closed by measurement, not assumption.** Next levers on the standing map:
the inference-efficiency pass (cross-task batching → bf16 → compile → FP8 — all local),
and the strategic tokens-at-scale decision (unchanged). The learned-verifier research lane
inherits a sharpened brief: at 1.2B, the verifier cannot come from the model itself — it
must be an external artifact (executor today; a trained verifier model tomorrow).
