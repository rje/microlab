# Coding-specialist design decisions (running log)

Each entry: the call, the evidence, and what would reopen it. Ablation protocol:
124M twin arms (muon-ab sizing) screen -> ambiguous results escalate to Phase-B 400-500M ->
1B validates the composition. 150M decides only when the effect clears the noise band and
the literature does not flag scale-inversion.

## 1. Position: RoPE (pure NoPE rejected) — 2026-07-29

124M twins, 4500 steps/737M tokens, identical seeds (runs/nope-ab-*; analysis
scripts/analyze_nope_ab.py; evals/length_gen/). NoPE: +0.057 stable train-length loss
penalty; length generalization CLAIM INVERTED (+1.04 loss at 2x, collapse ppl 506 at 4x vs
RoPE raw-extrapolation staying flat); passkey weak even in-window. Side finding: raw RoPE
keeps LOSS flat to 4x while retrieval cliffs to zero beyond the window — loss alone
mis-scores position schemes; retrieval evals are mandatory (replays the 1b extension saga).
Reconciliation with Kimi K3's global-NoPE: their KDA/recurrent layers carry position; NoPE
works only where position comes from elsewhere. REOPENS IF: the hybrid-linear lane wins —
then retest NoPE on global layers inside that composition (Kimi's actual configuration).

## Pending lanes
2. Peri-LN vs Pre-LN (arms launched 2026-07-29; also calibrates ladder noise band)
3. MLA vs GQA (oracle implementation next)
4. GDN/KDA hybrid vs full attention (the big one; carries the NoPE-conditional)
5. MoBA / ASA (conditional on 4)
6. mHC (opportunistic)
Data lanes: mix ablation incl. constant-vs-staged-curriculum arm and general-first-vs-
from-scratch arm (Code Llama finding); FIM rate; repetition; RHO-1-style selective loss.
