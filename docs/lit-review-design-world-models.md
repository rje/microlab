# Literature review: world models over software design

Companion to `lit-review-code-world-models.md`, which reviewed the wrong level of abstraction. That
review covered *runtime* semantics — predict post-execution program state. This one covers the
question actually asked: **does software architecture, and the ramifications of a design decision,
lend itself to a JEPA-style world model used for planning/refactoring before any code is written?**

Three targeted agent sweeps run 2026-07-29: repository/architecture-level representation learning;
change-impact and refactoring-outcome prediction; world models and latent planning for design.

**Provenance and caveats.** Sources are primary (arXiv/venue) except where marked. One sweep flagged
that a fetch returned a confidently-worded "verbatim quote" from StarCoder 2 with the two sentinel
tokens' roles reversed — fabricated, caught by cross-checking two independent sources, and that item
is reported unquoted. Publisher sites (ACM DL, Springer, IEEE Xplore, OpenReview) bot-block fetching,
so a handful of items are confirmed by metadata rather than full text and are flagged as such below.
The same sweep self-corrected a headline claim mid-report (see CGM, §2) — the corrected version is
what appears here.

---

## Verdict

**The literal proposal — a JEPA-style latent world model over design state, rolled forward to
evaluate refactors — is not supported, and three independent lines of evidence converge against it.
A reframed version is supported, genuinely open, and sits directly on machinery this lab already
built.**

The reframe in one line: **a static analyzer computes the post-change structure exactly, so put no
learning there; the *value* of a design is what nothing computes, so put all the learning there.**

That is MuZero's actual lesson. MuZero matched AlphaZero on Go/chess/shogi without being given the
rules — but its model is trained to be *value*-equivalent, not *state*-equivalent. It never predicts
the board. Mapped onto software design: the transition is exactly computable and is not the hard
part; the hard part is whether a design is any good, and no analyzer computes that.

---

## Why the world-model framing fails

### 1. Latent rollout is far shallower than its reputation

**V-JEPA 2-AC** (arXiv 2506.09985, Meta) — the flagship JEPA planning result — plans at **horizon
T=1**: optimize, execute one action, re-plan under receding-horizon MPC. Multi-step tasks require
hand-specified subgoals on a *hardcoded schedule* (4 steps on subgoal 1, 10 on subgoal 2, 4 on
subgoal 3 for pick-and-place). The paper states directly that representation-space prediction
accuracy decreases with longer autoregressive rollouts. Grasp-box succeeds 25% of the time; the
predictor is ~300M params.

Corroborating the depth ceiling from the model-based RL lineage: **MBPO** (NeurIPS 2019) replaces
long rollouts with many short (k≈1–5) branched rollouts *from real replay-buffer states*, explicitly
to bound compounding error. DreamerV3's imagination horizon of 15 is for policy-gradient signal, not
for a decision you act on. The *Critique of World Model* position paper (arXiv 2507.05169) puts the
practical MPC ceiling at 10–20 steps before action-sequence explosion and error accumulation.

**Anyone citing V-JEPA-2 as evidence that latent imagination supports deep planning is misreading
it.** Reliable open-loop latent rollout for decision-making is roughly 1–5 steps everywhere it has
been measured.

### 2. The one team who built a code world model refused to learn the deterministic part — and
measured what it costs when you don't

**SWE-World** (arXiv 2602.03419, Feb 2026) runs SWE agents Docker-free by *splitting* the state.
File navigation and edit commands (`ls`, `grep`, `view`, `str_replace`) execute deterministically in
a real lightweight sandbox that directly updates repo state; only execution commands (`pytest`) go to
a learned transition model. Their stated reason, verbatim: *"file operations are deterministic; LLM
simulation may hallucinate files or contents and catastrophically mislead the agent."*

They quantified the surrogate's cost with the agent held fixed: **60.2% resolve rate with the learned
72B world model vs 68.4% with real Docker.** Eight points, paid to simulate what could be computed.

Their RL run shows the second-order failure: a low-precision learned reward model got hacked within
~20 steps, the policy collapsing to short invalid submissions as trajectory length fell off. Same
KL-up/reward-flat signature we caught on the 1B GRPO run, in a far more expensive setting.

Supporting: **CodePlan** (FSE 2024, Microsoft Research) plans multi-step repo edit chains and its
transition model is *deliberately not learned* — incremental dependency analysis plus change-impact
analysis. Highest-quality venue evidence that repo-state planning works, and it uses an exact
analyzer as the model. **WorldCoder** (NeurIPS 2024) makes the same choice in miniature: in symbolic
domains the "learned world model" that wins is an exact executable program, learned as code, not a
latent vector. And the DEVS paper (arXiv 2603.03784) notes that in discrete process-driven domains
model drift produces *categorically invalid* states — impossible event orders, violated constraints —
not gracefully-degraded ones, so the usual "approximate but useful" defense of learned dynamics does
not apply.

Theoretical backstop: *Critique of World Model* Proposition 1 shows a latent-reconstruction objective
admits a **collapse solution** — all observations map to a constant vector while the transition model
learns trivially invariant dynamics, scoring perfectly while learning nothing about dynamics.

### 3. Design search is off-distribution by construction, which is exactly where learned models break

*What model does MuZero learn?* (de Vries et al., ECAI 2024) finds MuZero's learned model becomes
increasingly inaccurate specifically when evaluating policies that differ from the data-collection
policy. A design-search agent's entire purpose is to score *novel* designs. This is the most specific
transfer risk in the whole review.

Related: Hamrick et al. (**ICLR 2021**) find planning is most useful *in the learning process* —
shaping policy updates and the training data distribution — with shallow trees matching complex
search except on the hardest tasks, and planning with the true simulator still beating planning with
the learned model. If that transfers, a design world model earns its keep as a **training-data
generator**, not a deliberation engine. Consistent with SWE-World's own result: its learned-surrogate
trajectories produced 52.2% SFT vs 51.4% from real Docker trajectories.

### 4. Learned repo-structural representations have no refereed win over a static index

This is the finding that most surprised me. **RepoGraph** (ICLR 2025) trains *nothing* — tree-sitter
plus ego-graphs plus a prompt template — holds open-source SWE-bench SOTA, and has the best evidence
hygiene in the set, having been plugged into four external systems to isolate the graph's
contribution. **GraphCoder** (ASE 2024) is the cleanest non-learned index: tree-sitter CFG+DDG+CDG,
coarse retrieval by Jaccard over bag-of-words, and an external LLM used as a black box. Nothing
trained. **CodexGraph** (NAACL 2025): static graph DB plus a frozen LLM writing Cypher.

Four independent results say the learned graph machinery is not clearly earning its keep: RepoGraph,
arXiv 2510.13697, ClassLAR (frozen class-name embeddings beat graph SOTA), and an NMI finding that
node2vec ≈ GAT+PPR. **Any structural model we build must beat a tree-sitter index and a bag of
identifier names — not just beat other GNNs.**

---

## Occupancy map

### Repository structure as a trained signal — exists exactly once

**CGM** (Code Graph Model, arXiv 2505.16901, **NeurIPS 2025**, Ant Group/ShanghaiTech). *Citation
hazard: the arXiv comments field names no venue — cite it as a preprint and you'll be wrong.*
Qwen2.5-72B via 4-bit QLoRA on 64×A100, CodeT5+ node encoder (dim 256) with a trained 256→8192
adapter; graph injected into the decoder's attention mask. No message-passing GNN. Data: *"500k
Python and 360k Java subgraphs (with the maximum length of 8k tokens) from a total of 20k high-star
Github repositories"*, plus 200k issue-patch pairs. Objective, verbatim: *"Subgraph Reconstruction
Pre-training… a novel pre-training task that requires the model to reconstruct code content from its
corresponding code graph, a process we refer to as Graph-to-Code."* Graph spans *"up to seven types
of nodes and five types of edges… ranging from the repository level (REPO) to fine-grained
attributes."* 43.00% SWE-bench Lite, weights released.

**This is input-space reconstruction of the *current* state.** No latent prediction, no temporal
component. It doesn't occupy the gap — but it establishes that the data pipeline and graph
conditioning work at 20k-repo scale, which is a precedent to build on rather than an obstacle.

Two framings to stop repeating: **LongCoder** (ICML 2023) is file-level, not repo-level — its LCC
benchmark filters individual GitHub files and its objective is plain next-token; structure lives in
the attention pattern, not the loss. **GraphCodeBERT** (ICLR 2021) is intra-function, but its edge-
prediction and node-alignment objectives are the real existence proof for structural pretraining.
**Five years passed between GraphCodeBERT and CGM with nobody scaling structural pretraining
objectives from function to repository.**

What "repo-level" means at web scale is less than it sounds: **DeepSeek-Coder** parses file
dependencies with *regular expressions* to topologically order training samples offline, then feeds a
flat token stream — no trained structural component. **StarCoder 2** uses a data-packing convention
(`<repo_name>…<file_sep>…`) with same-repo files grouped and randomly ordered. The only web-scale
entries do the *least* structural modeling.

Data scale is the real gap: RepoFusion 100 repos (preprint only, no venue in three years); Repo2Vec
(ICSME 2021) 1,013 repos and composed from off-the-shelf embedders; CGM 20k repos; DeepSeek/StarCoder
web-scale with minimal structure. **Nobody has done repository-structural pretraining at pretraining
scale.**

### Predicting design consequences before the change — nearly empty

A 2024 survey of deep-learning-based refactoring (arXiv 2404.19226) breaks the field into 56% smell
detection, 33% recommendation, 6% end-to-end transformation — and states there is **no literature on
quality assurance for refactoring**. Nobody predicts whether a refactoring will help.

The recommendation literature measures the wrong thing. Aniche et al. (**TSE 2020**, >2M refactorings
across 11,149 projects, >90% accuracy, hand-engineered code + process + ownership metrics) predicts
*that a developer will refactor something* — conflating "should be refactored" with "happened to get
touched." No outcome variable.

The single paper that does what was described: **Higo et al., ASWEC 2008** computes complexity metrics
on the hypothetical revised program *without performing the refactoring*. Analytic, not learned —
it knows the refactoring's semantics and simulates the metric delta. Minor venue, no successor line
in eighteen years. Search-based refactoring (MORE, MIRROR, MORCoRA) does the same by *applying* the
transformation and recomputing.

The only sustained line predicting future *structure*: Díaz-Pace et al., link prediction over module
networks to anticipate cyclic and hub-like dependency smells (**SCAM 2018** → ***EAAI* 2022**, 6 OSS
projects, hand-engineered similarity indices, precision improved up to 3× by a feedback mechanism).

Change impact with learned representations is active but far from solved: **ATHENA** (FSE 2024,
transformer embeddings fused with program-dependence structure) reaches mRR 60.32%, **mAP 35.19%**,
HIT@10 81.48% on a manually-verified 25-project benchmark, with gains concentrated in impacted
methods *outside* the query method's class. **RIPPLE** (ICSE 2026) adds an LLM planner conditioned on
change intent — the closest published work to "given a design intent I haven't implemented, what will
need to change." A GPT-5 study (arXiv 2512.19481) reports naive frontier-LLM prompting performs
poorly at change impact, with no static baseline.

### Learning whether decisions turned out well — occupied, and the verdict is discouraging

**"Deep Just-in-Time Defect Prediction: How Far Are We?"** (ISSTA 2021, artifact-evaluated, 310,370
changes): CC2Vec doesn't consistently beat DeepJIT, neither consistently beats traditional
feature-based prediction, and **a logistic regression on the single added-line-number feature
outperforms both** at ~100,000× faster training. This is the field's own "measure, don't guess"
result, aimed at exactly the class of learned change representation we'd be proposing.

The pattern across everything deployed: learned representations decisively win in one setting only —
enormous proprietary data plus a hard operational label. Meta's **Diff Risk Score** (fine-tuned
Llama3-70B, 535K+ reviewed diffs, predicting real production incidents; revert rate 1/3, incident
rate 1/50, validated difference-in-differences) earns its keep. Meta's **Predictive Test Selection**
(ICSE-SEIP 2019) is gradient-boosted trees on hand-engineered features — 2× infra cut, >99.9% of
faulty changes still caught. A learning-to-rank random forest beat both FCP2Vec and StarCoder 2 on
co-change identification (150 projects, 634k PRs) — and needs bimonthly retraining for concept drift.
J.P. Morgan's revert predictor is a GNN over the import graph (workshop-grade reporting).

**Every deployed production system predicts a cheaply and objectively labeled *operational* outcome —
test failed, build failed, diff reverted, incident filed. None predicts a design-quality outcome. The
constraint is label supply, not modeling.**

And the design-quality target is itself contested: one line finds >94% of applied refactorings degrade
at least one internal quality attribute; another (3,795 projects, 1,245 developer-declared
quality-improvement commits) finds design metrics *sometimes do not capture* the improvement
developers report. This is exactly why industry predicts SEVs instead.

### Design-level planning by LLM agents — all prompting, one trained scorer

MetaGPT (ICLR 2024), ChatDev (ACL 2024), and the LLM-architecture literature (design-rationale
generation, ATAM scenario analysis, ADR generation) are prompted roles emitting design documents.
None learns a consequence model. The recurring limitation across that literature: outputs require
expert review, evaluation is small or artifact-specific, and *most studies do not validate the
generated architectural designs at all*.

**SmellBench** (arXiv 2605.07001, May 2026) is the motivating failure: 65 hard-severity architectural
smells in scikit-learn across 11 agent configs, best resolution rate **47.7%**, 63.1% of detected
smells false positives, and **the most aggressive agent introduced 140 new smells**, with repair
aggressiveness inversely correlated with net codebase quality. Current agents demonstrably cannot
anticipate the architectural consequences of their own edits.

### The precise gap

1. Repository structure as a **trained** signal: exists once (CGM), input-space reconstruction of
   current state, NeurIPS 2025.
2. Repository structure in **latent** space: nobody.
3. Repository structure **over time**: one seed-conditioned temporal graph network (Germanos et al.,
   *IST* 2024 — metadata-confirmed only) and one 2015 non-neural growth model.
4. The **intersection** — latent × structural × temporal: **empty**, and now empty for a documented
   reason rather than an unsearched one.

No action-conditioned JEPA-style latent world model over a discrete symbolic state space exists in
any domain, and none over code.

---

## The version that survives

Put the learning in the evaluator, not the transition:

- **Transition**: static analysis. Dependency graph, coupling/cohesion, cycles, fan-in/out, boundary
  violations — computable exactly, labels free and uncontested. Applies equally to a hypothetical
  refactor, since mechanical refactorings (extract, move, invert, split) can be applied and measured.
- **Value**: learned. Which of these candidate designs is better? Nothing computes this.
- **Training signal**: maintainer preference between competing proposals, not a quality metric. This
  sidesteps the contested-target problem entirely — you're not predicting "is this good
  architecture," you're predicting "which proposal would a maintainer pick," which has real ground
  truth in PR and issue discussions and is the same preference-pair shape as our RLHF data.

The small-scale evidence here is encouraging where the world-model evidence is discouraging:

- **SWE-Manager** (arXiv 2601.22956, Jan 2026) — an **8B** model, RL-trained, selecting among
  competing natural-language proposals for fixing an issue **with no code execution and no tests**,
  and synthesizing a merged "golden proposal." 53.21% selection accuracy, 57.75% earn rate on
  SWE-Lancer Manager. The most encouraging datapoint in this review for a small-lab replication.
- **Web-Shepherd** (**NeurIPS 2025 spotlight**) — a *trained* checklist-structured process reward
  model beats a prompted LLM judge by **+10.9 points at 10× lower cost** (WebArena-lite 34.55% vs
  23.64%). Strongest evidence that a trained plan-structured verifier beats prompting.
- **PlanSearch** (ICLR 2025) — searching over natural-language *plans* rather than code: Claude 3.5
  Sonnet pass@200 77.0% vs 60.6% repeated sampling vs 41.4% pass@1, with search gain a predictable
  function of idea diversity. The best published argument for branching at the design level.

This is machinery we have already built and validated: reward model, best-of-n, GRPO (see
`canonical-rlhf-vs-direct-po`). Pointing it at designs rather than chat responses is a scope change,
not a capability change.

---

## Baselines any entry here must beat

Non-negotiable, given §4 above and the ISSTA 2021 result:

1. A **tree-sitter static index** (RepoGraph-style, trains nothing).
2. A **bag of identifier names** (ClassLAR beat graph SOTA with frozen class-name embeddings).
3. A **single hand-engineered feature** in a logistic regression (the JIT-defect lesson).
4. A **prompted frontier LLM** as judge (Web-Shepherd's baseline — beatable, but only by a trained
   structured verifier).

If a learned design evaluator can't clear all four, it's the same story the field has already told
four times.

---

## Reading list

1. **SWE-World** (arXiv 2602.03419) — the split-state architecture and the measured 60.2 vs 68.4 cost
   of learning what you could compute. Read first.
2. **SWE-Manager** (arXiv 2601.22956) — 8B, no execution, design-level selection. The template for
   the version that survives.
3. **V-JEPA 2** (arXiv 2506.09985) — read the planning section specifically, for T=1 and the
   hand-scheduled subgoals.
4. **CGM** (arXiv 2505.16901, NeurIPS 2025) — the only repo-structural pretraining objective; the
   data pipeline is the reusable part.
5. **"Deep JIT Defect Prediction: How Far Are We?"** (ISSTA 2021) — the field's own negative result
   on learned change representations.

Runners-up: RepoGraph (ICLR 2025) as the baseline to beat; Web-Shepherd (NeurIPS 2025) for trained
verifier design; *Critique of World Model* (arXiv 2507.05169) for the collapse proposition;
Hamrick et al. (ICLR 2021) for planning-helps-training-not-inference; Díaz-Pace (*EAAI* 2022) for the
only future-structure prediction line; SmellBench (arXiv 2605.07001) for the motivating failure.

---

## Fit against lab assets

- Reward model + best-of-n + GRPO (`src/microlab/train/{reward,grpo}.py`) — directly reusable; a
  design evaluator is an RM with a different input distribution.
- The pairwise-eval harness and position-swapped judging protocol transfer unchanged.
- `src/microlab/evals/code/executor.py` — still the ground-truth oracle, but for *validating* that a
  chosen design's implementation passes, not for generating world-model targets.
- Missing: a repo-graph extractor (tree-sitter; CGM's node/edge taxonomy is the reference), and repo
  *histories* rather than the Stack's file-level snapshots.
- The verdict-audit protocol applies with force here: given four independent results where learned
  graph machinery failed to beat trivial baselines, a positive control and an explicit
  beat-the-tree-sitter-index gate are mandatory before believing any win.

## If we pursue this

The cheap decisive experiment is a **baseline bake-off, not a training run**: build the tree-sitter
repo index, extract maintainer-preference pairs from PR discussions on a handful of repos, and check
whether *any* learned scorer beats the bag-of-identifiers and prompted-LLM baselines on held-out
proposal selection. That is a few days of work and it answers the only question that matters before
committing GPU — because the field's own record says the trivial baseline usually wins.
