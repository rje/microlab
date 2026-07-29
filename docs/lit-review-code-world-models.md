# Literature review: code world models and JEPA-style latent prediction over programs

Deep-research sweep run 2026-07-29, scoping the "trace-JEPA" idea: a small (150M–350M) encoder that
takes `(code, input state)`, predicts the **embedding** of the post-execution program state (not the
literal values), with ground truth manufactured by our sandboxed executor, used as a learned
verifier/ranker for AR-generated candidates and potentially for latent-space edit planning.

**Method caveat, stated up front.** The fan-out (5 search angles, 15 sources fetched, claim
extraction per source) completed; the 3-vote adversarial verification pass and the synthesis agent
did **not** — 30 of 103 agents died on an API usage-credit exhaustion mid-run. 15 claims completed
verification and **all 15 survived (zero refutations)**; the remainder carry extracted verbatim
quotes from primary sources but no adversarial check. The two claims that carry the whole verdict
(LLM-JEPA's paired-view bottleneck; the 2025 disconfirming trace study) were re-checked by hand
against the primary sources and both held. Per the verdict-audit protocol, findings below are marked
**[verified]**, **[quote-checked]** (verbatim quote from primary source, no adversarial pass), or
**[search-snippet]** (identified but not fetched).

---

## Summary

The territory splits cleanly, and the split is favorable but narrower than it first looks. Every
*component* of the proposal is occupied — small learned verifiers (CodeRanker, 125M), executor-
generated ground truth (CodeExecutor, TRACED, CWM), execution-prediction-as-candidate-screen
(CodeExecutor already did exactly this in 2023), and JEPA on structured discrete data (Graph-JEPA,
19M params). What is **not** occupied is their intersection: no published work predicts an
*embedding* of post-execution program state using an executor as the source of paired views. The
closest predecessor, LLM-JEPA (LeCun et al., ICLR 2026), names precisely this as its blocker —
language has no data-augmentation analogue, so JEPA needs naturally paired views — and a sandboxed
executor manufactures such pairs on demand. That is the real idea here, and it is a genuine one.

Two findings should temper enthusiasm. First, a September 2025 systematic study found that
execution-trace information **does not help** SFT or test-time scaling of 7B code LLMs at all,
contradicting the positive claims of NExT/SemCoder/CodeExecutor — though it tested traces in token
space (prompts and SFT targets), leaving representation-space objectives untested. Second,
Discrete-JEPA shows continuous latent rollout compounds error over multi-step horizons, which is bad
news specifically for the *latent planning* half of the idea, less so for single-step verification.

The scale (150M–350M) is not a contribution — it is simply the well-precedented right size, matching
CodeRanker (125M), TRACED (125M), and CodeExecutor (110M). The executor ground truth is not a
contribution either. **The latent-prediction objective with executor-manufactured views is the
contribution**, and it is a small, sharp, testable one.

---

## Occupancy map, most to least occupied

### 1. Learned verifiers/rankers for code — VERY OCCUPIED, including at our exact scale

This is the *deployment role* the trace-JEPA would fill, and it is crowded from 2022 through
late 2025.

- **CodeRanker** (NeurIPS 2022) **[quote-checked]** — a ~125M fine-tuned CodeBERT that predicts
  whether a sampled program is correct **without executing it**, trained fault-aware (predicting the
  exact compile/runtime error class). Labels come from actually executing 100 samples per training
  task. Lifts Codex pass@1 on APPS validation 26% → 39.6%; transfers zero-shot (HumanEval 26% → 32%,
  MBPP 36% → 42%). Gains collapse on the hard APPS test split (3.8% → 4.5%). *This is our proposed
  scale and our proposed role, already published four years ago, with label-space supervision.*
- **LEVER** (ICML 2023) **[quote-checked]** — T5-base/large and RoBERTa-large verifiers (~0.2–0.8B,
  ~0.5% of the Codex generator) rerank by combining verification score with generation probability;
  +4.6–10.9% over code-davinci-002 on Spider/WikiTQ/GSM8k/MBPP. Critically, **LEVER executes each
  candidate** — the execution result is a mandatory verifier input. Its ablation quantifies the gap
  the trace-JEPA would need to close without running code: removing the execution result costs 6.6%
  on WikiTQ, 5.6% on MBPP, 3.0% on Spider, 1.2% on GSM8k.
- **SWE-RM** (Dec 2025, Qwen-affiliated) **[verified]** — 30B-total/3B-active MoE execution-free
  reward model; Qwen3-Coder-Flash 51.6% → 62.0% on SWE-bench Verified via test-time reranking.
  Originating-lab only. Its authors still describe execution-free feedback for realistic SWE agents
  as "underexplored." Also reports a design warning relevant to us: two verifiers with near-identical
  best-of-n performance can behave very differently as RL reward — ranking skill does not imply
  reward-model skill.
- **On LLMs' Internal Representation of Code Correctness** (arXiv 2512.07404, preprint)
  **[quote-checked]** — the sharpest result for our thesis and the sharpest threat to it. A linear
  "correctness direction" found by PCA on correct-vs-incorrect hidden states of 7–8B code LLMs ranks
  candidates better than log-likelihood or verbalized confidence: +21.3% pass@1 on HumanEval vs
  RankEF's +17.7%. It is **fitted, not trained** — 3.75 seconds of PCA versus ~172 GPU+CPU-hours to
  reproduce RankEF. The authors explicitly leave open whether the signal reflects semantic
  understanding or superficial syntactic correlates — which is exactly the question an
  executor-grounded latent objective would answer.

**Implication:** we do not get to claim the verifier role as novel, and any trace-JEPA must beat a
3.75-second PCA probe to justify its existence. That probe is the real baseline, not CodeRanker.

### 2. Token-space trace/execution modeling — VERY OCCUPIED, and recently contested

- **Learning to Execute** (Zaremba & Sutskever 2014) **[verified]** — character-level LSTMs mapping
  short programs to literal outputs. Restricted to programs "evaluable with a single left-to-right
  pass using constant memory"; conventional curriculum learning *failed* and a new curriculum variant
  was required. The origin point, and a standing reminder that literal execution prediction is
  brittle.
- **Scratchpads** (Nye et al. 2021) **[verified]** — the canonical token-space trace formulation:
  alternating executed source lines and JSON-serialized local-variable states. The load-bearing
  number for us: **direct end-state prediction scores 10.3% on MBPP versus 26.6% with a
  step-by-step scratchpad** (20% vs 41.5% on synthetic programs). Predicting the end state in one
  shot is much harder than walking there — *in token space, for exact literal values*.
- **CodeExecutor** (Findings of ACL 2023) **[quote-checked]** — our closest scale-matched baseline: a
  ~110M UniXcoder-based model generating full traces, curriculum-trained on ~15M executor-produced
  examples. On the realistic held-out split it gets 48.06% output accuracy and only **33.38%
  exact-trace accuracy**. It **already did our ranker use-case**: predicting outputs for 200 Codex
  HumanEval samples and ranking by edit similarity lifts pass@1 12.48 → 17.87 and pass@10 45.59 →
  49.69, "without a real-world code executor." It also names two failure modes a fixed-size embedding
  target would not inherit: unfaithfulness on complex logic, and a hard 1024-token trace cap that
  breaks loop-heavy programs.
- **TRACED** (ICSE 2024) **[quote-checked]** — 125M RoBERTa-base encoder from UnixCoder, further
  pretrained on **2× RTX 3090** (our hardware envelope) on 121,319 gdb-derived C traces. Predicts
  program states as classification over 30 quantized value bins; +12.4% path prediction, +25.2%
  variable-value prediction. But the ceiling is low: 71.6% full-path accuracy and **all variable
  values correct in only 49.2% of executions**, even after quantization. Transfers to representation
  tasks (91.2% MAP@R on POJ104 clone retrieval).
- **CWM** (Meta FAIR, Sept 2025) **[verified]** — the 32B incumbent, ~100× our scale. Dense
  decoder-only, 131k context, mid-trained on ~5T tokens of observation-action trajectories from two
  sources: Python interpreter traces (>120M traced functions, locals + next line in tokenized
  JSON-like format) and ~3M agentic Docker trajectories across ~10k images / 3.15k repos. **The
  objective is plain next-token prediction over trace tokens** — `CwmForCausalLM` with a standard LM
  head; no latent objective anywhere. 65.8% SWE-bench Verified with test-time scaling (53.9%
  without), 94.3% CruxEval-Output. All numbers originating-lab-only. Checkpoints released at
  mid-training/SFT/RL stages, so the pure-trace representation is publicly probeable. Meta frames the
  verifier use ("neural debuggers that can jump ahead") as *future work*, not a demonstrated result.
- **Do Code Semantics Help?** (arXiv 2509.11686) **[quote-checked — the disconfirming study]** —
  systematically evaluated trace-based semantic information across five formats (NExT, SemCoder,
  CodeExecutor, Concise, Scratchpad) on DeepSeek-Coder-6.7B / Llama-3.1-8B / Gemma2-9B over
  HumanEval, MBPP, LiveCodeBench, BigCodeBench, CRUXEval. Finding: "semantic information has limited
  usefulness for SFT and test time scaling of Code LLM" — explicitly disagreeing with prior work.
  Test-time scaling itself beat greedy/CoT in 65 of 70 cases, so *compute*, not trace information,
  drove the gains. **Scope limit that matters to us:** traces were serialized into prompts and SFT
  targets. Representation-space objectives were not tested.

### 3. Execution-aware code representations — OCCUPIED, mostly pre-2021

- **Dynamic Neural Program Embeddings** (ICLR 2018) **[quote-checked]** — the founding evidence that
  execution beats syntax in representation space: embeddings from execution traces (sequential tuples
  of live variable values) hit >92% accuracy on error-pattern prediction versus **<27% for
  syntax-based embeddings**; best variant 99.3/98.8/99.2% vs a token baseline at 16.8–21.2%. Tiny
  scale (2-layer GRUs, 200 hidden units, a few thousand student submissions). Used as a learned
  prioritizer for search-based repair: >10× speedup when ≥4 fixes needed. **Traces are model
  *inputs*, not prediction targets** — the program must actually run to get its embedding.
- **ContraCode** and **GraphCodeBERT** (ICLR 2021) **[search-snippet]** — contrastive
  semantic-equivalence pretraining over compiler-transformed variants, and dataflow-graph-aware
  pretraining. Both establish that embedding spaces can capture behavioral/semantic structure, but
  from static signal only. GraphCodeBERT marks the boundary the proposed lane crosses: static
  dataflow versus dynamic executor ground truth.
- **FuzzPretrain, SemCoder, NExT** **[search-snippet]** — the adjacent 2024–2025 family surveyed by
  the disconfirming study above.

### 4. Outcome prediction without execution — OCCUPIED at label level, sparse in latent space

- **Patch-correctness from embeddings** (Tian et al. 2020) **[quote-checked]** — BERT embeddings of
  code changes + logistic regression reach AUC ≈ 0.8 on 1000 labeled patches, "comparable to
  PATCH-SIM, which relies on dynamic information." Static embeddings substituting for execution.
- **Predictive Test Selection** (Meta, ICSE-SEIP 2019) **[search-snippet]** — production-deployed
  gradient-boosted trees predict which tests a change will fail, halving CI cost while catching
  >99.9% of faulty changes. The strongest real-world proof that outcomes are predictable from diffs
  without running anything — but hand-engineered features, no learned representation. It bounds the
  problem's feasibility, not its method space.
- **Neural Abstract Interpretation** (VerifAI workshop @ ICLR 2025) **[search-snippet]** — learns
  neural abstract transformers propagating abstract state through a learned function. Conceptually
  the closest formal-methods analogue to the proposal: **abstract interpretation *is* state
  prediction in a compressed domain**, and a trace-JEPA is learned abstract interpretation with a
  *learned* abstract domain. Toy scale, workshop tier — a framing gift more than an evidence base.

### 5. JEPA on discrete/structured domains — SPARSE, and instructive

- **LLM-JEPA** (Huang, LeCun, Balestriero; ICLR 2026) **[quote-checked by hand]** — the direct
  predecessor. Loss: `L_LLM + λ·d(Pred(Enc(Text)), Enc(Code))`, predictor implemented via appended
  predictor tokens, cosine distance, next-token loss retained (JEPA term is an auxiliary
  regularizer, not a replacement). Gains at our scale class: Llama-3.2-1B on NL-RX-SYNTH 57.29% →
  71.46%, Spider 47.52% → 50.55%, GSM8K 32.36% → 36.36%, all p<0.05. **The views are static
  text↔code pairs. No execution, no traces, no program state anywhere in the paper.** Two stated
  limitations, and both are load-bearing for us: "The lack of JEPA-style LLM is a testimony of the
  challenge in designing such objectives for language"; and "Being able to obtain non-trivial
  views... is crucial to the success of JEPA objectives. While we restrict ourselves to datasets
  offering those non-trivial views..." plus a 2× training compute cost.
- **Graph-JEPA** (TMLR) **[quote-checked]** — first JEPA for graphs: predict latent representations
  of masked target subgraphs from a context subgraph. ~19M params, SOTA-as-backbone on 5 of 8
  datasets, 2.5× faster than GraphMAE. Evaluated **only** on molecules and social networks — zero
  code, ASTs, or program graphs. Notably, vanilla latent regression did not work: the authors had to
  predict subgraph coordinates on a unit hyperbola with smooth-L1, plus the usual stop-gradient +
  EMA target encoder anti-collapse machinery. **Latent prediction on discrete structured data
  required objective redesign.**
- **Discrete-JEPA** (ICML 2025 Tokenization workshop) **[quote-checked]** — extends JEPA to discrete
  symbolic tokens via semantic vector quantization. The finding that matters to us: **continuous
  latent rollout compounds error over long horizons while a discrete latent space does not** —
  Discrete-JEPA holds 1.0 accuracy across 200 rollout steps where I-JEPA degrades; ~6× better LPIPS
  and 5× better MSE at 1000 steps, though I-JEPA is better at short horizons (10–50 steps). Synthetic
  vision only, single group, workshop tier.
- **DLLM-JEPA** (2026) and the next-token-alternatives survey **[search-snippet]** — text-JEPA
  variants; none use execution ground truth.

### 6. JEPA-for-code with executor ground truth — UNOCCUPIED

Nothing found. Five search angles, fifteen fetched primary sources, and the nearest neighbors in
every direction (LLM-JEPA on static pairs; Graph-JEPA on molecules; CWM on trace tokens; TRACED on
quantized value classes) all stop short of predicting an embedding of post-execution state.

---

## Novelty assessment

Three axes, and only one survives:

| Axis | Novel? | Why |
|---|---|---|
| Scale (150M–350M) | **No** | CodeRanker 125M, TRACED 125M, CodeExecutor 110M, LEVER T5-base. Well-precedented — this is the *right* size, not a contribution. |
| Executor-generated ground truth | **No** | CodeExecutor (15M sandbox-run examples), TRACED (gdb), CWM (120M traced functions), CodeRanker and LEVER (execution-derived labels) all do this. |
| **Latent prediction of post-execution state, with the executor manufacturing paired views** | **Yes** | No published instance found. And it dissolves the specific blocker LLM-JEPA names: language has no augmentation analogue, so JEPA needs naturally paired views — an executor *manufactures* them on demand, unboundedly, with a ground-truth relation between the views. |

The sharpest one-sentence framing: **`(code, input)` and `post-execution state` are two views of the
same semantic object, and unlike vision — where views come from augmentation — or language — where
they must be found in naturally paired data — an interpreter generates them at will.** That is the
thesis, and it is defensible.

The honest counter-case, which must be answered before any GPU time: a 3.75-second PCA probe on an
existing 7B model's hidden states already beats a 172-GPU-hour trained execution-feedback ranker
(2512.07404). If execution-outcome information is *already* linearly decodable from an ordinary
AR model, the marginal value of training a dedicated latent world model to put it there is unclear.
The trace-JEPA's case rests on the claim that a *purpose-built, executor-grounded* latent is
qualitatively better than an incidental one — and that claim is exactly what 2512.07404's authors
say they could not settle ("we showed that this signal exists... but not why or how").

---

## Is latent-space state prediction harder or easier than token-space trace prediction?

Evidence cuts both ways and no source tests the comparison directly.

**Arguments that latent is easier:**
- Token-space exact-state prediction has a low measured ceiling. TRACED gets *all* variable values
  right in only 49.2% of executions even after quantizing to 30 bins; CodeExecutor achieves 33.38%
  exact-trace accuracy on realistic held-out programs. An embedding target does not require exact
  values — it requires that semantically-equal states land nearby, a strictly weaker demand.
- Token-space trace generation pays a length cost that embeddings do not: CodeExecutor's hard
  1024-token cap "can be a limitation for programs with long execution traces, particularly those
  with loops." A fixed-size post-state embedding is O(1) in trace length.
- Latent probes already recover outcome information cheaply and transfer out-of-distribution
  (2512.07404), suggesting the information is well-conditioned in representation space.
- Executor ground truth is exact and discrete, unlike the approximate continuous physics visual world
  models must fit — the one structural advantage code has over the domain JEPA was invented for.

**Arguments that latent is harder:**
- Scratchpads' central result is that direct end-state prediction badly underperforms step-by-step
  trace generation (10.3% vs 26.6% on MBPP). A single-shot post-state predictor is structurally the
  "direct" formulation. The penalty *may* be a literal-value artifact — but it may not be.
- Graph-JEPA needed a redesigned objective (hyperbolic coordinate targets) beyond vanilla latent
  regression to work on discrete structured data; representation collapse is a live failure mode
  requiring EMA + stop-gradient machinery we have never built.
- Discrete-JEPA shows continuous latent rollout compounds error over multi-step horizons. This
  specifically threatens the **latent edit-planning** half of the proposal; single-step verification
  is largely unaffected.
- LLM-JEPA's 2× training compute cost applies to us too.
- And the field-level risk: if 2509.11686 is right that trace information simply doesn't help code
  LLMs, repackaging it in latent space may not rescue it.

---

## Five papers to read first

1. **Do Code Semantics Help? A Comprehensive Study on Execution Trace-Based Information for Code
   LLMs** (arXiv 2509.11686) — read this *first*, before the positive results. It is the recipe-
   vintage check on the entire premise, and its scope limit (token space only) is the gap the whole
   proposal lives in. If we cannot articulate why representation-space objectives escape its
   conclusion, there is no lane.
2. **LLM-JEPA** (arXiv 2509.14252, ICLR 2026) — the direct predecessor and the source of the novelty
   claim. Read for the loss form, the predictor-token trick, and the paired-views bottleneck we would
   dissolve.
3. **CodeExecutor** (arXiv 2305.05383, Findings of ACL 2023) — the scale-matched baseline that
   already did our ranker use-case. Its HumanEval pass@1 12.48 → 17.87 is the number to beat, and its
   33.38% exact-trace accuracy is the token-space ceiling to compare against.
4. **TRACED** (arXiv 2306.07487, ICSE 2024) — 125M on 2× RTX 3090, i.e. our exact envelope. Read the
   value-quantization design and the 49.2% all-variables-correct ceiling.
5. **On LLMs' Internal Representation of Code Correctness** (arXiv 2512.07404) — the cheapest
   possible baseline (3.75s PCA) and the strongest threat to the proposal's marginal value.

Runners-up: **CWM** (arXiv 2510.02387) for the trace data recipe (its format is directly reusable and
its mid-training checkpoint is public), **Graph-JEPA** (arXiv 2309.16014) for anti-collapse machinery
on structured data, **Discrete-JEPA** (arXiv 2506.14373) for the rollout-stability finding,
**CodeRanker** (arXiv 2206.03865) for the 125M verifier recipe.

---

## Fit against existing lab assets

- `src/microlab/evals/code/executor.py` — the sandbox (process-group SIGKILL, RLIMIT_AS/CPU/FSIZE,
  netns isolation) is already the ground-truth oracle this lane needs. Built for GRPO reward; the
  view-generator role is a second use with no new infrastructure.
- The 150M ablation harness and Muon-validated training loop give us the sieve scale that CodeRanker/
  TRACED/CodeExecutor all validate as sufficient for this class of result.
- The code corpus pipeline (~9B tokens/hr) and code-49k tokenizer supply the program source; the
  missing piece is a tracer (CWM's format is the reference; CPython `sys.settrace` is the local path).
- The verdict-audit protocol applies directly: a positive control (can the encoder recover *any*
  execution-dependent property?), an implementation review, and a noise band would all be mandatory
  before believing a latent-verifier win.

## Suggested next step (not yet committed)

The cheapest decisive experiment is a **probe, not a training run**: take our existing 1B, run
programs through the sandbox, and measure whether a linear probe on its hidden states predicts
post-execution properties above chance — and whether a trained fixed-size post-state embedding target
is learnable at all at 150M. That answers "is this well-conditioned" for ~a day of GPU, before
committing to a world-model training lane. It also directly replicates 2512.07404's probe on our own
model, which we would want as the baseline regardless.
