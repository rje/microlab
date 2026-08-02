# Should a 124M model retrieve at 32k? What the literature says

Written 2026-08-02, after the C2 passkey grid was retracted for having no positive control.
The question: is 32k retrieval a reasonable thing to ask of `runs/frontier-32k` (124M,
3.93B tokens, 18 KDA : 6 MLA, NoPE, 100% repo-packed code), or was the test never winnable?

**Answer: sub-1B 32k retrieval is achievable, but every demonstration comes from an
architecture built specifically to fix this failure, at 2.7–3.2x our parameters and ~4x our
token budget. Nothing in the literature suggests our configuration should have passed.**

## The general finding: recall is state-bound, and linear models are the weak class

- **Zoology** (Arora et al., [2312.04927](https://arxiv.org/abs/2312.04927)): on associative
  recall, a **70M attention model beats a 1.4B gated-convolution model**. Gated-conv models
  "require model dimensions that **scale with sequence length**" to solve multi-query
  associative recall; attention solves it at constant dimension. 82% of the perplexity gap
  between gated-conv and attention is explained by recall alone.
- **Based** (Arora et al., ICML 2024, [2402.18668](https://arxiv.org/abs/2402.18668)):
  identifies "a key tradeoff between a model's **state size** and recall ability." Smaller
  recurrent state buys throughput and costs precise retrieval.
- **Just Read Twice** (Arora et al., [2407.05483](https://arxiv.org/abs/2407.05483)):
  "due to the limited memory, recurrent LMs **cannot recall and use all the information in
  long contexts**." Their fix reaches 99% of Transformer quality at 360M/30B tokens — by
  reading the input twice, i.e. by working around the state limit rather than removing it.

Our design is 75% linear layers. That is the architecture class these papers identify as
structurally worst at recall.

## The counterexamples — and why they do not cover us

Sub-1B models *have* done 32k+ needle retrieval:

| work | scale | budget | what it took |
|---|---|---|---|
| HOLA ([2607.02303](https://arxiv.org/html/2607.02303v1)) | 340M | 15B SlimPajama | adds an **exact external memory** for "what the recurrent state forgets"; RULER-robust to 32k |
| KLA ([2605.08587](https://arxiv.org/html/2605.08587v1)) | 0.4B | 1B tokens | a **new linear attention**; 100% single-needle, stable to 65k |
| ATMA ([2606.25156](https://arxiv.org/html/2606.25156)) | — | — | adds a **long-term recurrent compression memory**; >90% to 64k |

The pattern is the point: **each paper's contribution IS the mechanism that makes long
retrieval work.** Their baselines — plain linear and hybrid models of the kind we built —
are the thing that fails. Citing them as proof that 124M can retrieve at 32k inverts what
they actually report.

They are also 2.7–3.2x our parameter count, ~4x our token budget, and trained on general
text rather than 100% code.

## Even large models mostly fail this

**RULER** ([OpenReview](https://openreview.net/forum?id=kIoBbc76Sy)): of models *claiming*
32k+ context, only half hold up at 32k, and "none maintains performance above the
Llama2-7B baseline at their claimed length" except Mixtral. Effective context is typically
**50–65% of the marketed number**. Frontier 7B+ models fail at 32k; asking it of a 124M is
not a marginal stretch.

## Our data recipe probably could not teach it either

The long-context training literature is consistent: **"simply joining short texts during
pretraining causes the model to learn short patterns and struggle with long context
dependencies."** Working recipes deliberately manufacture long-range dependencies — Llama 3
upsamples >32K documents 5x; NExtLong and "Untie the Knots" interleave document segments to
force cross-distance attention.

We repo-packed code into 32k windows. That is exactly "joining short texts": it produces
long *sequences* without producing long *dependencies*. Most next-token predictions in a
32k code window are satisfied locally. The model was never given a reason to learn
retrieval at range, and was never trained on a retrieval task at all.

## The state arithmetic, concretely

KDA state per layer is fixed at `n_head x head_dim^2` regardless of context:

| context | values of state per token (124M, 12x64x64 = 49,152) |
|---|---|
| 1,024 | 48 |
| 8,192 | **6** |
| 32,768 | 1.5 |

Observed: perfect at 1k, degradation begins at 8k. That is where per-token state budget
falls to ~6 values, and it matches Zoology's "dimension must scale with sequence length."

Extrapolating to the planned 1B (14 x 128^2 = 229,376): at 32k it would have **7 values per
token** — roughly what the 124M has at 8k, where it *starts failing*.

Caveat, held honestly: the 6 MLA layers are full attention with unbounded KV, so pure state
arithmetic understates what the architecture can do, and Zoology found hybrids recover most
of the recall gap. The arithmetic is a heuristic, not a proof. But it points the same way as
everything else.

## Consequences

1. **C2 as specified was not winnable.** Passkey at 32k on a 124M model with this budget and
   data recipe contradicts the literature in four independent ways. Its failure carries no
   information about NoPE.
2. **The 1B may not clear it either.** The state-per-token extrapolation puts the 1B at 32k
   near where the 124M breaks. Gating the 1B on 32k retrieval risks failing a test the
   design cannot pass for reasons unrelated to the thing being tested.
3. **If we want long-range retrieval, it has to be built and trained for**, not assumed as a
   by-product of a long window: long-dependency data construction, and possibly an explicit
   recall mechanism of the kind every successful sub-1B result added.
4. **The honest framing for the 1B** is a compute-optimal code model with a long *window*,
   with retrieval quality measured and reported rather than claimed.
