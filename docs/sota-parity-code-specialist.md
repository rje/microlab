# SOTA parity review: the coding specialist, BEFORE the pretrain commits

Run 2026-07-31, per the standing rule from `sota-parity-1b.md` — table our design against
contemporary same-class releases, and give every divergence an explicit **CHOSEN** (stands,
with a reason) or **CHANGE** (fix before the run). This is the review that did not happen
before the 1B, which is why the 1B shipped MHA at 1024 context.

Cohort specs were read from `config.json` on HuggingFace and from paper PDFs, with parameter
counts reconstructed arithmetically and checked against safetensors metadata. That check
caught **three errors in the published Qwen2.5-Coder paper** (Table 1 gives head size 128 for
the 0.5B where the config implies 64; intermediate 4,864 for the 3B where it is 11,008; vocab
151,646 where it is 151,936). **Do not copy hyperparameters out of that table — use the
configs.**

## Reading the "ours" column

**CORRECTION (2026-07-31, after owner review).** The first draft headed this column "Ours
(proposed)", which was wrong and misleading: it merged three different kinds of thing and
presented all of them as decisions we had made.

- **[DECIDED]** — an actual verdict with evidence behind it (decisions log).
- **[ABLATION]** — a scratch default in `code-base.py` that exists only so ablation arms are
  mutually comparable. **Never proposed for a pretrain.** Listing these as "ours" made it
  look as though we had chosen MHA and 1024 context. We had not; they are unset defaults,
  which is precisely the failure mode this review exists to catch.
- **[OPEN]** — not decided either way. The data mix is the important one: it was ALWAYS a
  pending data lane, and the owner had already raised that code-only pretraining is uncommon
  before this review ran. Presenting it here as a new finding was a mistake.

## The cohort at our size class

| | Ours | Qwen2.5-Coder-1.5B | DeepSeek-Coder-1.3B | StarCoder2-3B |
|---|---|---|---|---|
| position | **[DECIDED]** RoPE on globals (NoPE tested, rejected) | RoPE | RoPE | RoPE |
| attention | **[DECIDED]** GDN 3:1 hybrid | GQA | **MHA** | GQA |
| n_head : n_kv_head | **[ABLATION]** 12 : absent = MHA | 12 : **2** | 16 : 16 | 24 : **2** |
| pretrain ctx (θ) | **[ABLATION]** 1024 (1e4) | 8192 (1e4) | ~4096 (1e4) | 4096 (1e5) |
| shipped ctx (θ) | **[OPEN]** not yet planned | **32768 (1e6)** | 16384 (1e5) | 16384 (1e6) + SWA 4096 |
| vocab | **[DECIDED]** 49,152 | 151,936 | 32,256 | **49,152** |
| norm / act | RMSNorm Peri-LN / SwiGLU | RMSNorm pre / SwiGLU | RMSNorm pre / SwiGLU | LayerNorm pre / **GELU non-gated** |
| QK-norm | **[OPEN]** | no | no | no |
| tied emb | yes | yes | no | yes |
| training tokens | **[OPEN]** (corpus 27.35B) | 5.5T (continued from 18T) | 2T from scratch | 3.3T over **622B unique = 4.98 epochs** |
| data mix | **[OPEN]** — lane pending, not a plan | 70% code / 20% text / 10% math | 87 / 10 / 3 | code + math/wiki (7B only) |
| FIM | **[OPEN]** not implemented | yes, rate unpublished | yes, **rate 0.5 PSM** | ~0.25 effective |

## Verdicts

1. **Global-layer attention: MHA → CHANGE to `n_kv_head=2`.** The single most consistent
   finding in the cohort: **n_kv_head=2 is an absolute budget, not a ratio**, for everything
   under ~3B — Qwen2.5-Coder uses 2 at 0.5B, 1.5B and 3B (ratios 7:1, 6:1, 8:1) and
   StarCoder2-3B uses 2 (12:1); both move to 4 only at 7B. StarCoder2 states the rationale
   outright: *"we keep the number of key-value heads relatively low—2 for the 3B, 4 for the
   7B and 15B—to prevent significantly slowing down inference."* This compounds with our
   measured 4x hybrid KV reduction rather than competing with it: only 3 of 12 layers cache
   at all, so 12:2 on those layers multiplies the saving. **This is the 1B's exact error and
   it is currently live in `code-base.py`.**
   *Counter-signal priced in:* 2025-generation general models went the other way (Qwen3-0.6B
   and 1.7B use 16:8) — so if general reasoning matters alongside code, the KV budget is
   trending back up. For a pure code specialist at ~1B, 2 is the modal and well-supported
   choice.
2. **Context: 1024 → CHANGE.** Not a research question — a staging decision the whole cohort
   makes the same way. **Everyone pretrains short and runs a separate extension stage**, and
   the extension is cheap: StarCoder2 spent 200B tokens (~6% of budget), DeepSeek-Coder spent
   1000 steps. **4k is acceptable as a pretraining stage; it is not acceptable as a shipped
   number.** The floor is 16k and the mode at our size is 32k (all three small Qwen2.5-Coder
   models ship 32768). Plan: pretrain at 4–8k with θ=1e4, then ABF to 16–32k with **θ=1e6**,
   which is the standard pairing (CodeLlama, StarCoder2 post-extension, all Qwen2.5-Coder).
   Above 32k everyone switches from ABF to YaRN.
   *Note:* Qwen2.5-Coder's 128K claim applies **only to the 7B** — the 0.5B/1.5B/3B configs
   carry no `rope_scaling` at all, so YaRN there is opt-in at inference and untrained.
3. **RoPE base 1e4 → CHANGE at the extension stage** (to 1e6), CHOSEN for the pretrain stage.
4. **GDN 3:1 hybrid → CHOSEN, and it is our one genuine architectural divergence.** *None* of
   the cohort uses linear or hybrid attention; StarCoder2's sliding-window 4096 is the only
   sparsity present. The nearest precedent is Qwen3-Coder-Next (~80B, Gated DeltaNet,
   `full_attention_interval: 4` — the same 3:1 ratio), which is 25x our size class. We adopt
   it on our own measurements (wins at compute-optimal, 4x KV, ~10x length-gen), and we
   should be explicit that we are ahead of the code-specialist field here rather than
   following it.
5. **Vocab 49,152 → CHOSEN, and it exactly matches StarCoder2.** Pleasing independent
   convergence: our fertility-driven choice landed on the same number as the one cohort model
   built by a code-focused consortium.
6. **QK-norm → ADOPT (new decision).** Absent from *every* flagship code specialist — but all
   of those are either 2024-vintage (Qwen2.5-Coder, StarCoder2) or Qwen3-derived MoE far
   above our class. It became standard in the 2025/26 general small-model lineage: OLMo 2
   (Nov 2024), Gemma 3, Qwen3 all apply it unconditionally. It is cheap and buys stability at
   high LR. **Use the head_dim variant (Qwen3/Gemma3), not the full-projected-dim variant
   (OLMo 2).** Recorded as a deliberate divergence from the code table that puts us in line
   with the general consensus.
7. **Data mix → the lane was already open; this review only supplies the cohort numbers.**
   Not a new finding: the owner flagged that code-only pretraining is uncommon before this
   review, and a mix ablation has been on the pending-lane list throughout. What is new is
   the cohort's actual ratios. **Every single cohort model mixes in natural language
   and math**: Qwen2.5-Coder 70/20/10, DeepSeek-Coder 87/10/3, CodeLlama 85/8/7,
   DeepSeek-Coder-V2-Lite 60/30/10. Pure-code pretraining is a divergence from the entire
   field and we have no evidence for it. This is the same class of miss as the 1B's
   FineWeb-only mix — and note it needs FineWeb (or math) **retokenised to code-49k**, which
   is currently a blocking dependency for both this and the general-first lane.
8. **FIM: not implemented → CHANGE.** Universal in the cohort. Best-supported default is
   **0.5 PSM**: DeepSeek ran the ablation explicitly (0%, 50%, 100%, 50%-MSP) and found 100%
   maximises HumanEval-FIM but yields "the weakest code completion capability." CodeLlama
   uses 0.9, StarCoder2 ~0.25 effective, Qwen2.5-Coder does not publish its rate.
9. **Token budget → the biggest open risk.** The cohort trains on **2T–5.5T tokens**; our
   corpus is **27.35B**, roughly 100x short, and TypeScript already exhausted the
   permissively-licensed Stack at 7.38B. Two mitigations, both already in motion: **StarCoder2
   trained 3.3T tokens over 622B unique — 4.98 epochs** — which is direct cohort precedent
   that heavy repetition works and is exactly what our running repetition lane is measuring;
   and mixing in NL/math (verdict 7) adds volume from a non-exhausted source.
10. **Peri-LN → CHOSEN (divergence, low stakes).** Nobody in the cohort uses it; our own
    retest put it at −0.0020 nats, below our noise band. Free to keep, but it is not load-
    bearing and should not be defended.
11. **Tied embeddings → CHOSEN.** Matches Qwen2.5-Coder and StarCoder2 at our scale.
12. **Position: RoPE on the global layers → CHOSEN (measured, not inherited).** Omitted from
    the first draft of this review, which was an error — it is a decided divergence question
    and belongs here. We trained the NoPE-on-globals arm (the literal Kimi Linear config) and
    it LOST on length generalisation: +0.063 nats at 4x training length vs RoPE-globals'
    +0.012, a 5x gap, while being a hair better in-window (inside the noise band). The whole
    cohort uses RoPE, so this is convergence rather than divergence — but it was reached by
    our own measurement. See `gdn-hybrid-verdict.md`.

## What this review changes

Four CHANGEs must land before any pretrain config is frozen: **`n_kv_head=2`**, **a staged
context plan (4–8k → 16–32k, θ 1e4 → 1e6)**, **a non-code slice in the mix**, and **FIM at
0.5 PSM**. Two are pure config; two need implementation (a FIM transform, and FineWeb/math
retokenised to code-49k). One new adoption: **QK-norm, head_dim variant**.

None of this invalidates the running data lanes — they hold architecture constant and vary
data, so their *relative* comparison stands. But `code-base.py` must not become the pretrain
config; it carries three of these four gaps and says so at the top.
