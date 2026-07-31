# Microlab-Coder-1B: frontier design + validated implementation plan

Written 2026-07-31 after the owner observed — correctly — that a week of ablations had
argued us back to a 2023/24 recipe. This plan targets the 2025/26 frontier design and
structures the work so the expensive commitment (a multi-week pretrain) is the LAST thing
that happens, behind gates that can each kill the plan cheaply.

## Why the previous proposal was too conservative

Three mechanisms, recorded so the plan corrects for them rather than repeating them:

1. **Resolution bias.** Every effect measurable at 124M lands at 0.002–0.006 nats, at or
   below our noise band. "No significant improvement" silently became "keep the incumbent."
   These techniques were validated at 1.3B–48B; a null at 124M is at least as likely to mean
   our test lacks power.
2. **Judging a technique by a weakened implementation.** We built plain GDN with a SCALAR
   per-head gate and then drew conclusions about the KDA lineage, whose gate is
   Diagonal-Plus-Low-Rank (~64x the capacity). Then we used our unfused Python scan being
   slow as an argument against the architecture. Both are category errors: the correct
   response to "our implementation is weak" is to fix the implementation.
3. **A backward-looking parity anchor.** The 1–3B code cohort is Qwen2.5-Coder (Sep 2024),
   StarCoder2 (Feb 2024), DeepSeek-Coder (2023), CodeLlama (2023) — the 2025/26 band in that
   size class is nearly empty. "The cohort ships GQA, not linear attention" really means
   "*2024* shipped GQA," while the frontier moved to linear hybrids, MLA and NoPE.

## The design

| component | choice | rationale |
|---|---|---|
| shape | 24 layers x 1792, 14 heads, head_dim 128 | our 1B's proven shape; head_dim 128 is universal in the cohort |
| token mixing | **3:1 hybrid — 18 KDA-style linear : 6 global** | Kimi Linear's ratio; the only architecture where we lead rather than follow |
| linear gate | **DPLR (diagonal + low-rank)**, not scalar | the actual KDA mechanism; scalar gating is what weakened our NoPE result |
| global attention | **GQA(2)** — MLA DROPPED | at 14 heads with n_kv=2, GQA caches 512 values/token vs MLA's 576, for 49% fewer params. MLA's win is over MHA or high-kv GQA; it solves a problem we do not have at 1B. |
| position | **NoPE on globals** — NO positional encoding anywhere in the model | Kimi Linear ships exactly this at 48B. Our contrary result is VOID: measured at 0.34x Chinchilla AND on the scalar gate whose weakness was the reason to doubt it. RoPE is now the option needing justification. |
| QK-norm | yes, head_dim variant | OLMo 2 / Gemma 3 / Qwen3 standard since late 2024 |
| norm | RMSNorm, Peri-LN | ours; effect ~= noise but free |
| MLP | SwiGLU | consensus |
| embeddings | tied, vocab 49,152 (code-49k) | our fertility measurement; matches StarCoder2 |
| optimizer | Muon | ours |
| context | pretrain **32k** natively, extend to 256k | 32k is where the fused hybrid is already CHEAPER than dense (0.90x measured), so it costs nothing. See the NoPE consequence below. |
| FIM | 0.5 PSM | DeepSeek's published ablation |
| data | 70% code / 20% web / 10% math, ~27B tokens (~1.3x Chinchilla) | cohort mix ratios |

Falls back to **dense + GQA(2)** only if Gate B1 fails.

## The NoPE consequence for 256k

With NoPE on the global layers and recurrent (scale-free decay) linear layers, the model has
**no positional encoding anywhere**. There is no RoPE table to rescale, no theta to raise, no
ABF stage, no YaRN. The entire context-extension apparatus the cohort needs — and which cost
our own 1B real retrieval quality when we retrofitted it — simply does not apply.

That converts 256k from "pretrain short, then a staged extension we have historically been
bad at" into "train at what we can afford and evaluate longer." It is the single strongest
practical argument for the NoPE choice, and it is downstream of the architecture rather than
a separate feature.

What still has to be VERIFIED (not assumed): that the recurrence actually carries position
well enough with the DPLR gate. The measurement is passkey retrieval and length
generalisation beyond the training window — large-effect tests, not marginal loss deltas.

## Plan — every phase ends in a gate that can stop the plan

### Phase A — correctness foundations (no GPU, ~1 day)

| # | work | GATE |
|---|---|---|
| A1 | install `flash-linear-attention` (0.5.2); confirm it ships GDN **and** a DPLR/KDA variant | if no KDA variant, we write the DPLR gate ourselves against our recurrent reference |
| A2 | **equivalence: fla's kernel vs our `gdn_recurrent` at float64** | must match to ~1e-8. If not, one of the two is wrong and nothing proceeds. This is the single most important test in the plan. |
| A3 | extend `gdn_recurrent` to DPLR gating (reference first, as before) | matches a hand-computed 2-step case |
| A4 | ~~MLA~~ **DROPPED 2026-08-01 on arithmetic, before writing any code.** At 14 heads with n_kv=2, GQA caches 512 values/token vs MLA's 576, for 49% fewer params. MLA's advantage is over MHA or high-kv GQA and does not exist at 1B. Removing it also removes the largest untested surface in the plan. | n/a |
| A5 | QK-norm (head_dim variant) | unit test; param tree unchanged apart from the two norms |
| A6 | FIM transform (0.5 PSM) | round-trip test: PSM-encode then decode returns the original document |

### Phase B — performance, the go/no-go for the whole design (GPU, ~half a day)

**B1 is the cheapest decisive test in the plan and it runs first.**

| # | work | GATE |
|---|---|---|
| B1 | re-run the context-scaling benchmark with fused kernels, 1k–32k | **hybrid must be <=1.2x dense at 8k and <=1.0x at 32k.** Today it is 5.8x and 9.3x. If fused kernels do not fix that, the hybrid cannot be trained where its benefits live and we fall back to dense GQA(2) — having spent one day, not three weeks. |
| B2 | KV/state memory at 8k/32k/256k, measured not analytic | matches the analytic table within 5% |
| B3 | update `preflight_lane.py` trainability table with the fused numbers | gate stops hard-failing the design |

### Phase C — architecture validation (GPU, ~2–3 days)

| # | work | GATE |
|---|---|---|
| C1 | **engineering validation**: full stack trains stably at 32k on repo-packed data | no NaN, no divergence, throughput within 20% of the B1 prediction |
| C2 | **passkey retrieval across the 32k window** | THE decisive test. Binary-ish, large-effect. If the recurrence carries position, retrieval works at depth; if it does not, NoPE is wrong for us and that shows up unmissably. |
| C3 | **length generalisation past the training window** (32k -> 64k/128k) | our length-gen work produced a 240x spread between arms (+0.012 vs +2.887 nats) — this is the class of measurement that has actually decided things here |
| C4 | loss vs dense-GQA(2) baseline | SECONDARY ONLY. Our 124M A/Bs have produced -0.0020 and -0.0059 nats, both noise-adjacent, one of which inverted on duration. Recorded, not relied upon. |

**Why C was reframed (2026-08-01):** the original C1 was a marginal val-loss A/B. Owner
observed that our 124M ablations have been "generally resistant to useful measurement," and
the record agrees — the only decisive results we have ever gotten at this scale had large
effects (NoPE +2.887 at 4x length; Muon 1.3-1.45x). So Phase C now validates that the SYSTEM
works and that it can USE its context, rather than trying to prove a 0.008-nat win.

### Phase D — data (GPU, partly running)

| # | work | GATE |
|---|---|---|
| D1 | repetition lane (RUNNING) | tells us whether TypeScript's 7.38B ceiling binds |
| D2 | retokenise web + math to code-49k, build the 70/20/10 mix | mix corpus builds; val disjoint |
| D3 | FIM-rate lane at 0.5 vs 0 | FIM does not degrade left-to-right completion |

### Phase E — pretrain

Only after A–D. Full parity review re-run against the frozen config, `preflight_lane.py`
clean, and a written prediction of the expected result so the outcome is falsifiable.

## Honest risk register

- **Gate B1 is the plan.** If fused kernels do not make the hybrid competitive at 8–32k, the
  frontier design is not reachable for us and we ship dense GQA(2). One day to find out.
- **NoPE is the position choice and it is unverified BY US.** Kimi ships it at 48B; our only contrary data is void (0.34x Chinchilla, scalar gate). The risk is real and it is concentrated in one place — if the recurrence does not carry position, C2 fails loudly and we add RoPE to the 6 global layers, which is a one-line change and no retraining of anything else.
- **We will still be a compute-optimal 1B**, ~27B tokens against a cohort trained on 2–5T.
  On benchmarks we lose to Qwen2.5-Coder-1.5B. The defensible claim is "frontier architecture
  at compute-optimal training," and we should say so before eval, not after.
- **Using `fla` for production kernels** is the same line already drawn for attention (we call
  SDPA, i.e. FlashAttention, rather than writing our own). Our `gdn_recurrent` stays the
  oracle. Writing our own Triton kernel remains available as a later learning exercise, not a
  blocker.
