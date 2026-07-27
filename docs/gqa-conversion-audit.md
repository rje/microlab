# MHA -> GQA conversion audit

Audit of the `scripts/convert_gqa.py` conversion of the 1B (983M) MHA checkpoint
(`runs/1b/ckpt_40000.pt`, 24L x 1792, 14 heads, head_dim 128, RoPE, RMSNorm, SwiGLU,
21B FineWeb tokens) to GQA `n_kv_head=2`, against the research literature and against
cheap inference-only counterfactual experiments (`scripts/analyze_gqa_grouping.py`).
Date: 2026-07. All CE numbers are nats/token on the fixed val batch (8 x 1024,
seed 1337, fineweb-100bt val split, bf16 autocast) that `convert_gqa.py --kl-data-dir`
uses; benchmark numbers are lm-eval-harness (`runs/lmeval_1b.json` vs
`runs/lmeval_1b-gqa-4500.json`).

## Verdict

**Not an implementation error. Primarily recipe naivety, with a real inherent
component.** The converter implements Ainslie et al. (arXiv 2305.13245) faithfully —
mean-pooling the K/V *projection matrices* over adjacent-index groups is exactly what
the paper describes, and the conversion machinery is verified lossless on identity and
identical-head cases. But the 2023 recipe transplanted to a 2024-era RoPE decoder at
7:1 compression is now known (and measured here) to be close to the worst reasonable
recipe:

- The heads only *look* orthogonal. Raw within-group K/V cosine is ~0.000, but after
  function-preserving per-head rotations (orthogonal Procrustes; RoPE-commuting plane
  rotations for K) within-group similarity is **~0.42 (K) / ~0.49 (V)**. Raw weight
  cosine is gauge-confounded — two heads computing similar functions in rotated bases
  score ~0 — which is exactly what arXiv 2412.20677 ("Align Attention Heads Before
  Merging Them") documents on LLaMA-2 ("the vast majority of KV caches are almost
  orthogonal ... this is the reason why directly mean-pooling projection matrices
  results in significant loss").
- Align-then-pool with per-group norm restoration cuts conversion CE at the same 7:1
  ratio from **10.27 to 5.94**, and at 2:1 from **8.24 to 3.06** (original: 2.54) —
  measured here, zero training.
- The uptrained outcome (-4.4 benchmark points mean after 2.3B tokens) is *in line
  with* published naive-mean-pool decoder conversions, not anomalously bad: at >= 4:1
  compression the literature reports -8.4 pts (8:1, LLaMA2-7B, 2412.20677 baseline,
  after recovery training) and -9.4 pts (4:1, 1B-token budget, DHA's GQA baseline,
  arXiv 2406.06567). Nobody has published a naive adjacent mean-pool RoPE-decoder
  conversion at ~7:1 that recovers to parity.

The inherent component: 14 heads only admit `n_kv_head` in {1, 2, 7, 14}, so the
validated-everywhere 4:1 ratio does not exist for this geometry; 7:1 keeps only 2 of 14
K/V subspaces of a small model, and even the best published conversion recipes lose a
few points at 7-8:1 on 7B-class models. A converted-and-uptrained kv2 model was never
going to be free; the naive recipe just paid roughly double the necessary price.

## 1. What was shipped, and what happened (all reproduced here)

- `convert_gqa.py` mean-pools the 14 K/V heads into 2 groups of 7 *adjacent-index*
  heads (Ainslie recipe). Conversion faithfulness was pinned by identity conversions
  (groups of 1, and identical heads within groups: logits match to 1e-4).
- Converted CE **10.27** vs original **2.54** (reproduced exactly: 10.2717 / 2.5397) —
  the unigram floor; KL(orig||pooled) 7.70 nats/token.
- Cause, measured: within-group K/V head cosine ~ 0.000 at every layer, so the mean of
  7 quasi-orthogonal head projections shrinks norms by 1/sqrt(7) = 0.378 (measured
  pooled/orig norm ratio: K 0.379, V 0.379). The muted K collapses attention logits:
  mean attention entropy goes from 2.4-4.1 nats/query (original, by layer) to
  5.75-5.87 ~= the 5.93 of uniform-over-prefix — pooled attention reads the whole
  context near-uniformly, and the model lands at the unigram floor.
- `--scale-correct` (multiply pooled blocks by sqrt(7)) restores norms but not
  attention structure (entropy still 4.6-5.4): CE 9.67 at init, and in the earlier
  30-step probe it recovered *worse* than plain (7.41 vs 7.01) — rescaled noise is
  still noise, and the sharper logits are confidently wrong.
- Uptraining 2.3B tokens (Muon, two anneal cycles, `runs/1b-gqa`): val ppl 14.86 vs
  base 12.19. Benchmarks (`lmeval_1b-gqa-4500` vs `lmeval_1b`): arc_challenge
  31.5 -> 29.5 (-2.0, acc_norm), arc_easy 65.4 -> 62.2 (-3.2), hellaswag 49.0 -> 42.4
  (-6.6, acc_norm), lambada 41.3 -> 33.6 (-7.6), piqa 70.7 -> 68.0 (-2.8), winogrande
  56.4 -> 52.3 (-4.0). Mean **-4.4**.

## 2. Implementation re-review against Ainslie et al. (2305.13245)

Checked `scripts/convert_gqa.py` and the `GQAAttention` path in
`src/microlab/model/reference/variants.py` clause by clause against the paper
(`papers/architecture/2023-ainslie-gqa-*.pdf`):

- **Pooling target — matches.** Paper Fig. 1 / section 2.1: "The projection matrices
  for key and value heads are mean pooled into single projection matrices." T5 pooling
  operated on the projection *weights*, exactly what `pool_heads` does. Head-major row
  layout is pinned by `tests/scripts/test_convert_gqa.py`
  (`test_identical_heads_within_groups_pool_losslessly`).
- **Grouping — matches.** The paper constructs "each group key and value head by
  mean-pooling all the original heads within that group" with contiguous query-head
  groups; adjacent-index grouping is the paper's recipe (it has no similarity search).
- **RoPE-before/after-pooling — mathematically a non-issue.** Pooling weights and then
  applying RoPE to the pooled head is *identical* to averaging the per-position rotated
  keys, because every head receives the same position rotation R(p) and R(p) is
  linear: mean_h(R(p) W_h x) = R(p) (mean_h W_h) x. There is no hidden
  position-dependent error introduced by pooling weights rather than projections.
  (What RoPE does change is the *repair space*: any per-head basis alignment applied
  before pooling must commute with RoPE, i.e. be block-diagonal over the 64 frequency
  planes — see section 4.)
- **Bias handling — correct extension.** T5 attention has no biases so the paper is
  silent; our checkpoint has `c_attn` biases and pooling them with the same group mean
  is the unique choice that makes the pooled head function the mean of the member head
  functions: mean_h(W_h x + b_h) = (mean W_h) x + mean b_h.
- **Norm/scale interactions — no deviation.** T5 has no 1/sqrt(d_k) logit scaling
  (folded into init) while ours uses standard SDPA scaling, but pooling attenuates the
  K side by the same *relative* factor either way; RMSNorm pre-norm placement matches
  T5.1.1. Nothing in our stack changes the pooling calculus vs the paper.
- **GQA forward — standard.** `kv_proj` split + `repeat_interleave(groups)` matches
  reference implementations; verified bit-equivalent to MHA when groups share
  identical heads.

Two things the paper *quietly relies on* that do not transfer, and that the shipped
recipe inherited untested:

1. **T5's evaluation regime hides conversion damage.** Every Ainslie number is after
   uptraining (alpha=0.05 of pretraining, ~50B tokens for XXL) *plus per-task
   fine-tuning to convergence*. The paper's "GQA already achieves reasonable
   performance after conversion" (Fig. 5, alpha=0) is a fine-tuned-task number, not
   zero-shot LM quality. No published number implies the pooled T5 retained zero-shot
   capability either.
2. **T5's relative-position bias is a structural safety net RoPE lacks.** T5 adds a
   learned per-query-head position bias to the attention logits; when pooled-K content
   logits shrink ~2.6x toward zero, T5 attention degrades toward its intact positional
   prior. In a RoPE decoder the position signal lives *inside* the same q.k product
   that pooling attenuates, so attention degrades toward uniform — which is what we
   measure (entropy 5.8 vs uniform 5.93). This is a hypothesis for why the same recipe
   is much more catastrophic at init on RoPE decoders; the literature corroborates the
   collapse (2412.20677, DHA both report pooled decoder inits lose most capability)
   without isolating the mechanism.

**Conclusion: no implementation error.** The recipe was applied correctly; the recipe
itself is the problem.

## 3. Empirical counterfactuals (inference-only, `analyze_gqa_grouping.py`)

### 3.1 Similarity structure (full 14x14 per-layer matrices in `runs/gqa_audit/`)

Mean within-group pairwise similarity, averaged over layers (kv2 = 2 groups of 7,
kv7 = 7 pairs; "best" = exact exhaustive partition search per layer, bitmask DP):

| similarity space        | kv2 adjacent | kv2 best | kv7 adjacent | kv7 best |
|-------------------------|--------------|----------|--------------|----------|
| K weights               | -0.000       | +0.003   | -0.000       | +0.010   |
| V weights               | -0.000       | +0.001   | +0.000       | +0.004   |
| K activations           | +0.001       | +0.035   | +0.005       | +0.108   |
| V activations           | +0.000       | +0.003   | -0.000       | +0.011   |
| K activations, aligned  | +0.420       | +0.443   | +0.422       | +0.491   |
| V activations, aligned  | +0.485       | +0.503   | +0.485       | +0.528   |

- Raw similarity is ~0 in every space, at every layer, under *every possible
  grouping*: the best of all 1716 (kv2) / 135135 (kv7) partitions is still ~0.01-0.11.
  There is no hidden good permutation; adjacent-index grouping was not the mistake.
- Aligned similarity (best function-preserving rotation per head: full orthogonal for
  V, RoPE-commuting per-frequency-plane rotations for K) is **~0.42-0.53 everywhere**,
  and nearly flat across partitions. The heads share substantial common structure;
  it is hidden by per-head basis freedom ("gauge"), exactly as 2412.20677 found for
  LLaMA-2. Pooling *aligned* heads at cos ~0.45 predicts a norm ratio of
  sqrt((1 + 6 x 0.45)/7) ~= 0.73 instead of 0.378.

### 3.2 Conversion CE, all variants (no training; orig CE 2.54)

| variant                    | kv7 (2:1) | kv2 (7:1) | kv1 (14:1) |
|----------------------------|-----------|-----------|------------|
| adjacent mean (shipped)    | 8.24      | **10.27** | 10.18      |
| adjacent mean + sqrt(g)    | 5.97      | 9.67      | 9.79       |
| adjacent select (medoid)   | 7.28      | 9.22      | 10.01      |
| optimal-partition mean     | 7.74      | 9.62      | —          |
| aligned mean               | 4.85      | 9.23      | 9.72       |
| aligned mean + renorm      | **3.19**  | **5.94**  | 6.83       |
| align-opt partition mean   | 4.28      | 9.04      | —          |
| align-opt mean + renorm    | **3.06**  | 6.05      | —          |

("aligned" = generalized Procrustes within groups, rotations folded into W_q/W_k and
c_proj so the rewrite is exactly function-preserving; "renorm" = restore each pooled
block to the mean member Frobenius norm — the correct version of `--scale-correct`,
which over-scales once heads are aligned; "optimal partition" = per-layer exhaustive
best grouping on the relevant similarity.)

Readings:

- **Regrouping alone: worthless** (10.27 -> 9.62). **Selection: slightly better**
  (9.22) because it preserves norms and 2 of 14 query heads keep their exact K/V —
  consistent with Ainslie's own ablation where first-head selection lands within ~0.2
  pts of mean-pool after uptraining.
- **Alignment + renorm is worth 4.3 nats at 7:1** (10.27 -> 5.94) and **5.2 nats at
  2:1** (8.24 -> 3.06, only +0.52 over the unconverted model). The same mean-pool
  arithmetic, done in the right coordinate frames with norms restored, preserves most
  of the model at 2:1 and half the gap at 7:1.
- Even at 2:1 the *naive* recipe collapses (8.24): the failure was never specifically
  the 7:1 ratio at init time; it is mean-pooling unaligned heads at any ratio. (The
  ratio matters for the *ceiling* after uptraining, per the literature.)
- kv1 vs kv2 naive are both at the floor (10.18 vs 10.27) — once attention is uniform,
  more pooling cannot make it much worse; the floor saturates.

## 4. Literature (post-2023) on MHA -> GQA conversion

**(a) Similarity-aware grouping.** AsymGQA (arXiv 2406.14963): activation-informed
grouping beats adjacent grouping by up to +7.5 MMLU pts on LLaMA-2-7B at group size 4
(with per-task fine-tuning as recovery); weight-informed grouping is consistently
worse than activation-informed. QCQA (arXiv 2406.10247): evolutionary grouping with a
weight-sharing-error fitness, +20% accuracy vs vanilla GQA at equal cache without
fine-tuning. But both operate on models whose heads *have* exploitable raw similarity
structure or use asymmetric groups; on our checkpoint the exhaustive search shows raw
similarity is flat ~0, and measured regrouping gains are ~0.6 nats — grouping is not
where our damage is.

**(b) Alignment / selection / weighted pooling.** The key paper is arXiv 2412.20677
(Findings of EMNLP 2025): LLaMA-2's KV heads are "almost orthogonal" (our finding,
their words), fixed by generalized Procrustes alignment fused into the projections —
RoPE-constrained to per-frequency-plane rotations for K — plus similarity-searched
grouping, then mean-pool and a small distillation (93M-279M tokens). At 8:1 on
LLaMA2-7B: naive mean-pool 77.0% vs aligned 81.8% (teacher 85.5%). DHA (arXiv
2406.06567, NeurIPS 2024) replaces pooling with *learned* per-group fusion initialized
from CKA-clustered heads: 97.6% of MHA performance at 75% KV reduction for 0.25% of
pretraining compute, vs mean-pool GQA baseline at -9.4 pts for the same budget; also
finds V heads more redundant than K heads and redundancy concentrated mid-stack.
Ainslie's own ablation (Fig. 4): mean 55.6 > first-head 55.4 > random 55.2 after
uptraining — selection was always within noise of pooling, i.e. the *information
argument* for mean-pooling was never strong. SVD-family conversions (Palu, arXiv
2407.21118; MHA2MLA, arXiv 2502.14837; GQLA, arXiv 2605.15250) factorize instead of
pool and recover with 0.3-0.6% of pretraining data or calibration only.

**(c) Head decorrelation — corroborated.** Near-zero raw pairwise cosine between K/V
head weights is the expected state of trained transformers, not a property of our run:
2412.20677 measures it on LLaMA-2; attention-head redundancy famously exists at the
*behavioral* level (Michel et al., arXiv 1905.10650: 20-40% of heads prunable) without
implying weight-space alignment, because per-head orthogonal basis freedom (any R
applied to W_q,W_k jointly, or to W_v with c_proj absorbing R^T) makes weight cosine
gauge-dependent. Our aligned-similarity measurement (~0.45) plus the 5.2-nat CE
improvement from align-then-pool at 2:1 confirms the gauge explanation on this exact
checkpoint. No published evidence that T5's heads were special; T5's graceful
conversion is attributable to its recovery regime (huge uptrain + per-task fine-tune)
and arguably its additive position biases (section 2).

**(d) Budgets, ratios, and expected recovery.** Ainslie: T5-XXL 64 heads -> 8 (8:1),
alpha=0.05 (~50B tokens), then task fine-tuning: -0.1 pts. Modern converted-decoder
results: 2412.20677 at 8:1 with alignment + ~0.1-0.3B distillation tokens: -3.1 pts
(naive: -8.4); DHA at 4:1 with 5B tokens: ~-2.6 pts zero-shot avg; MHA2MLA (different
target format) near-parity with 0.6-1% of data. Natively-trained production ratios:
Llama-3.x-1B/8B and Mistral-7B use 4:1; Qwen2.5-0.5B is literally our target shape
(14 Q heads, 2 KV heads, 7:1) and Qwen2.5-7B is 7:1 — so the *endpoint* is a sound
architecture; those models were just never converted into it. Scaling analyses
(Cost-Optimal GQA, arXiv 2503.09579) find loss rises steeply as head count drops for
small models — 7:1 at 1B is aggressive, and *no published conversion at >= 7:1
reaches parity on a decoder*, aligned or not. Our -4.4 pts after 2.3B tokens (~11% of
our pretraining, but tiny in absolute terms) sits inside the published band for naive
conversions (-8 to -9 pts) — i.e. the uptrain partly worked; the init was the problem.

## 5. Recommended path to a viable GQA model

Cheapest first; all conversion steps are already implemented in
`scripts/analyze_gqa_grouping.py` (`convert_with_groups` + `compute_rotations`):

1. **Re-init from align-opt + renorm and rerun the same uptrain.** kv2 init CE 5.94
   vs 10.27 shipped. DHA and 2412.20677 both show aligned inits convert uptrain tokens
   to quality several times faster (DHA: 5x faster convergence than pooled init).
   Cost: conversion is minutes; reuse the existing 2.3B-token budget (or less —
   2412.20677 recovered with 0.1-0.3B distillation tokens at 8:1 on a 7B).
   Expected: recovers a large fraction of the -4.4; the literature's aligned results
   at 8:1 suggest landing around -1.5 to -3 rather than -4.4.
2. **Uptrain with KL distillation from the MHA parent instead of plain CE** (Minitron
   best practice, arXiv 2407.14679; also 2412.20677's recovery loss). Self-distillation
   from our own base model, so it builds no external capability in. Combine with (1).
3. **If 2:1 is acceptable, kv7 aligned+renorm starts at CE 3.06** (+0.52 over base)
   and should reach parity with a short uptrain — but only halves the KV cache, and
   14 heads admit no 4:1 middle option. A DHA-style *per-layer mixed allocation*
   (kv7 in the entropy-sensitive early/late layers, kv2 mid-stack where redundancy
   concentrates) is the way to buy a better-than-2:1 average on this geometry; needs
   a small trainer change (per-layer n_kv_head).
4. **Do not** spend more tokens uptraining the existing 10.27-init kv2 model; both the
   measured 30-step probes and the DHA baseline curves say the pooled init converts
   uptrain compute to quality at the worst rate of any option on the table.

## 6. Reproduction

```
python scripts/analyze_gqa_grouping.py similarity runs/1b/ckpt_40000.pt \
    --data-dir data/shards/fineweb-100bt --out runs/gqa_audit --device cuda
python scripts/analyze_gqa_grouping.py eval runs/1b/ckpt_40000.pt \
    --data-dir data/shards/fineweb-100bt --out runs/gqa_audit --device cuda
```

Outputs: `runs/gqa_audit/similarity.pt` (per-layer 14x14 matrices, six spaces),
`similarity_summary.json` (partition scores, norm ratios, per-layer attention
entropies), `eval_variants.json` (the CE table). Pure logic (partition DP, Procrustes
rotations, generalized conversion) is tested in
`tests/scripts/test_analyze_gqa_grouping.py`, including exact function-preservation of
the rotation rewrite and bit-equality with `convert_gqa.convert_state_dict` on
adjacent groups.

## References

- Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models from
  Multi-Head Checkpoints, arXiv 2305.13245 (paper in `papers/architecture/`).
- Align Attention Heads Before Merging Them: An Effective Way for Converting MHA to
  GQA, arXiv 2412.20677.
- DHA: Learning Decoupled-Head Attention from Transformer Checkpoints via Adaptive
  Heads Fusion, arXiv 2406.06567.
- AsymGQA: Optimised Grouped-Query Attention Mechanism for Transformers, arXiv
  2406.14963. QCQA, arXiv 2406.10247.
- Michel, Levy, Neubig: Are Sixteen Heads Really Better than One?, arXiv 1905.10650.
- Minitron: Compact Language Models via Pruning and Knowledge Distillation, arXiv
  2407.14679 / 2408.11796.
- MHA2MLA, arXiv 2502.14837. Palu, arXiv 2407.21118. GQLA, arXiv 2605.15250.
- Cost-Optimal Grouped-Query Attention for Long-Context Modeling, arXiv 2503.09579.
