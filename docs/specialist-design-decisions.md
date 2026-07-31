# Coding-specialist design decisions (running log)

Each entry: the call, the evidence, and what would reopen it. Ablation protocol:
124M twin arms (muon-ab sizing) screen -> ambiguous results escalate to Phase-B 400-500M ->
1B validates the composition. 150M decides only when the effect clears the noise band and
the literature does not flag scale-inversion.

VERDICT-AUDIT PROTOCOL (added after owner review of verdict 1): no verdict is final until
(a) POSITIVE CONTROL - the losing arm reproduces a known-true literature result with our
implementation (validates the instrument, not just the comparison); (b) HP FAIRNESS - a
short LR sweep for the losing arm rules out tuned-for-the-winner hyperparameters;
(c) NOISE BAND - the gap is placed against the multi-seed variance measured by the Peri-LN
calibration lane; (d) IMPLEMENTATION REVIEW vs the source paper. Verdicts published before
their audit carry PROVISIONAL status.

## 1. Position: RoPE (pure NoPE rejected) — 2026-07-29 — CONFIRMED (audit complete)

124M twins, 4500 steps/737M tokens, identical seeds (runs/nope-ab-*; analysis
scripts/analyze_nope_ab.py; evals/length_gen/). NoPE: +0.057 stable train-length loss
penalty; length generalization CLAIM INVERTED (+1.04 loss at 2x, collapse ppl 506 at 4x vs
RoPE raw-extrapolation staying flat); passkey weak even in-window. Side finding: raw RoPE
keeps LOSS flat to 4x while retrieval cliffs to zero beyond the window — loss alone
mis-scores position schemes; retrieval evals are mandatory (replays the 1b extension saga).
Reconciliation with Kimi K3's global-NoPE: their KDA/recurrent layers carry position; NoPE
works only where position comes from elsewhere. REOPENS IF: the hybrid-linear lane wins —
then retest NoPE on global layers inside that composition (Kimi's actual configuration).
AUDIT STATUS: the 4x-length collapse (+2.9) is far outside any HP/seed effect and likely
robust; the in-window +0.057 is the fragile part (single seed, RoPE-tuned HPs, 737M-token
horizon). Audit checks: Haviv-2022 positive controls (near-parity ppl; implicit position
decodable from hidden states), NoPE LR sweep, noise-band placement.

## 2. Normalization: Peri-LN — 2026-07-31 — KEEP, but DEMOTED (effect ~= noise at compute-optimal)

RETESTED at 15000 steps (0.99x Chinchilla): effect shrinks from -0.0152 to -0.0020, BELOW
the 0.0025 paired band. Sign held (no inversion), magnitude did not. Keep it — free, no
risk — but it is not a quality win and must not be cited as one. Original 4500-step result:
paired effect -0.0152, 3/3 seeds, 54/54 eval points, paired t = -10.48. Effect decays monotonically from -0.069 at step 250, so it buys early
convergence rather than a fixed gap. Variance-reduction claim (the reason we adopted it)
measured at a 0.69 sd ratio vs the published ">half" — directionally right, NOT resolvable
at n=3, and explicitly not usable to justify fewer seeds later. docs/periln-verdict.md.

## 3. Attention layout: GDN/KDA 3:1 hybrid — 2026-07-30 — ADOPT

At compute-optimal (15000 steps = 0.99x Chinchilla) the hybrid WINS: 3.0544 vs dense
3.0602, -0.0059 nats = 2.4x the paired band, leading at all 60 eval points with no
crossover. This INVERTS the 4500-step result (+0.0024 behind), which stopped at 0.30x
Chinchilla and had not converged. Hybrid still carries +5% params, so a param-matched
rerun would sharpen the claim. The axis
that decides the lane is memory, now MEASURED: 3.99x cache reduction at long context
(36.0 -> 9.0 KB/token) with a state fixed at 1.89 MB across a 128x context range, via a new
incremental-decode path (HybridCache + gdn_step) whose cached generation is token-identical
to uncached. Latency is flat at 5.5 ms/token for the hybrid vs dense's 4.1 -> 7.7 ms, so
they CROSS at ~100k context: the hybrid costs ~30% decode latency below 64k and wins above
~100k. Scope of the win is therefore long-context specifically. The long-run gate is PASSED.

STRONGEST evidence, and unpredicted: the hybrid LENGTH-GENERALIZES ~10x better than dense
attention. At 4x training length it costs +0.012 nats vs dense RoPE's +0.129, and on the
last-512-token bucket +0.039 vs +0.352 (9x). Nine of twelve layers are recurrent with no
positional table to run off the end of. This compounds with the 4x memory result — both
argue for the hybrid specifically at long context.

Sub-verdict (closes the conditional from verdict 1): NoPE on the GLOBAL layers of a hybrid
costs +0.063 at 4x length vs dense NoPE's +2.887 — a ~46x smaller penalty, purely because
recurrence supplies position. Verdict 1 stands as scoped to a DENSE stack. But
RoPE-on-globals still beats NoPE-on-globals at extrapolation 5x (+0.012 vs +0.063), so keep
RoPE on the globals for a long-context model; Kimi Linear's NoPE choice is not free here.
docs/gdn-hybrid-verdict.md.

## PROTOCOL CHANGE (2026-07-30): ablations must reach >=1x Chinchilla

The 4500-step protocol trains to 0.30x Chinchilla-optimal and DEMONSTRABLY inverts verdicts:
the GDN hybrid lost at 4500 steps and won at 15000, same seed/data/code. Any verdict from a
0.3x-Chinchilla run is UNDER-TRAINED and not adoption-grade.

At 124M with 160 seqs x 1024 tokens/step, >=1x Chinchilla is ~15000 steps, not 4500 (3.3x
the cost per lane — so triage lanes rather than running them all).

RETROACTIVE: verdict 2 (Peri-LN) was decided entirely inside the under-trained regime, and
its advantage was DECAYING monotonically (-0.069 at step 250 -> -0.015 at 4500) — the same
shape that inverted here. Re-test at ~15000 steps before carrying it into a real pretrain.
Verdict 1 (RoPE vs NoPE) is not at risk: its gap is +0.057 in-window and +2.9 at 4x length,
20-1000x the band, not a margin a schedule change plausibly flips.

## Pending lanes
3. MLA vs GQA (oracle implementation next)
4. GDN/KDA long-run + incremental-decoding memory measurement (see verdict 3);
   NoPE-on-globals arm is now a one-line config change
5. MoBA / ASA (conditional on 4)
6. mHC (opportunistic)
Data lanes: mix ablation incl. constant-vs-staged-curriculum arm and general-first-vs-
from-scratch arm (Code Llama finding); FIM rate; repetition; RHO-1-style selective loss.
