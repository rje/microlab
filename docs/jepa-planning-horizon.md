# The JEPA planning-horizon problem and what is being done about it

Sweep run 2026-07-29. Question: V-JEPA 2-AC plans at horizon 1 and needs hand-specified subgoals on a
hardcoded schedule — what research is aimed at fixing that?

Companion to `lit-review-design-world-models.md`, which used the shallow-horizon finding as evidence
against a design-level world model. This records the primary sources behind that claim and the state
of the fix.

*Evidence note:* almost everything here is preprint. **Exactly one peer-reviewed primary source
exists in this literature** (Terver et al., TMLR) — and it is the deflationary one. Weight
accordingly. A paper titled "Scaling Laws and Architectural Advances of Hierarchical JEPA" ranks high
on H-JEPA searches and is a predatory-journal student exercise whose own abstract disclaims SOTA
intent — do not cite it.

---

## Answer: hierarchy was built, it works, and it does not do what it sounds like

**H-JEPA is no longer a proposal.** LeCun's own group shipped one in April 2026: *Hierarchical
Planning with Latent World Models* (HWM, arXiv 2604.03208 — NYU / Meta FAIR / Mila / Brown; preprint,
no venue; code public). Two levels sharing **one latent space**: a high-level planner optimizes
macro-actions over a coarse-stride world model, and its first predicted latent becomes the subgoal
for a low-level MPC planner. No inverse model, no skill library, no goal-conditioned policy — and
critically, **it removes V-JEPA 2-AC's hand-specified subgoal schedule**, which was the defect.

Results are large and include real hardware:

| | HWM | Flat baseline |
|---|---|---|
| Franka pick-and-place, single goal image | 70% cup / 60% box | **0%** (V-JEPA 2-AC) |
| Franka drawer | 70% | 30% |
| Push-T @ goal distance 75 | 61% | 17% (DINO-WM) |
| Diverse Maze D∈[13,16] | 83% | 44% (PLDM) |

Plus 3–4× lower planning compute. Scale is modest: Franka uses a frozen ViT-g/16 with a ~300M
predictor; Push-T uses DINOv2 ViT-S (25M) + 75M high-level; the maze models are ~53k CNNs.

**But nobody extended the depth of accurate latent rollout.** Across every verified paper, the
deepest rollout inside a working planner is 5–25 steps: V-JEPA 2-AC plans at 1, V-JEPA 2.1 at 8,
DINO-WM at 5, HWM's low level at 2–15, TD-MPC2 at 3. HWM buys horizon by planning 25–47 macro-actions
at coarse stride while **replanning every 1–5 primitive steps** — the fine-grained predictor is never
trusted past ~15. That is a legitimate engineering answer. It is not "the model learned to predict
further ahead."

Meta's own admission, in V-JEPA 2.1 (arXiv 2603.14482): *"we actually observe a degradation in the
success rate of VJEPA 2 when planning over longer horizons."* The deepest horizon Meta reports
anywhere is 8, at n=10 trials — inside the noise floor.

**Caveat worth weighting heavily:** HWM's Diverse Maze high-level model is trained with T=6 rollouts
but planned to **H=47** — roughly 8× extrapolation past training depth, unflagged by the paper. That
is the benchmark producing the 83%-vs-44% headline.

## The bottleneck may not be the world model at all

Three independent results say improving the predictor is treating the wrong thing.

1. **Terver et al., "What Drives Success in Physical Planning with JEPA World Models?"** (arXiv
   2512.24497, **TMLR** — the only peer-reviewed source here). Verbatim: *"even with models which are
   able to faithfully unroll a large number of actions, success at the planning task is not an
   immediate consequence."* Optimal rollout depth is **2 in sim, 6 on DROID**. **Scaling the encoder
   and predictor gave no benefit in sim.** Gradient-based planners fail on multimodal costs; CEM
   wins.
2. **IMWM** (arXiv 2606.01626) — the sharpest counter-thesis. Replacing the learned predictor with an
   **oracle rollout of the true dynamics still fails**: Two-Room 85.5% vs IMWM's 99.2%. Their
   diagnostic: **98.9% of failed episodes contained zero goal-reaching candidates in the CEM
   population.** The bottleneck is the *search*, not the model.
3. **TRM** (arXiv 2605.22164) — why latent distance is a bad planning cost: XY position is linearly
   decodable from the latent at R²=0.998, yet the XY-probe rowspace accounts for **<1%** of
   terminal-goal latent MSE. The information is present and the cost function ignores it.

Add **MoP-JEPA** (arXiv 2607.05238): deterministic predictors return the conditional mean under
stochastic transitions, and standard JEPA predictors succeed on **0.02–0.09** of OGBench graph-search
queries. And **stable-worldmodel** (Maes, Le Lidec, Scieur, LeCun, Balestriero): DINO-WM falls from
94% to **4–20%** under controlled factor variation, and to 12% when trained on random-policy rather
than expert trajectories.

## The obvious fix has a bad track record

Training the predictor on its own multi-step rollouts is the intuitive repair. Its canonical
precedent is PlaNet's latent overshooting — and **PlaNet's own final agent does not use it** (*"our
final agent using the RSSM model does not require it"*), with Dreamer later stating it *"did not find
latent overshooting necessary."* The precedent was ablated out by its own authors. The paper never
gives a numeric depth. The closer conceptual ancestor is **Data as Demonstrator** (Venkatraman,
Hebert, Bagnell, AAAI 2015 — no arXiv version).

Direction of travel in the strongest non-JEPA line: **TD-MPC trains with a 5-step unroll; TD-MPC2
shortened it to 3 while scaling 300× to 317M params.** Bigger went shallower.

Mixed supporting evidence: **AR Forcing** (arXiv 2605.31314) gives the cleanest "deeper training
unroll buys horizon" ablation anywhere — `len_traj_pred` 4→8→16, monotonic improvement — but on pixel
and perceptual metrics, not latent planning. **EB-JEPA** (arXiv 2602.03604, ICLR 2026 World Models
Workshop) ablates recursive rollout k=1…4 on Moving MNIST, Pareto optimum ≈4, framed as exposure-bias
reduction. **Somalwar et al.** (arXiv 2504.01766, UPenn) give theory for when it should work at all:
single-step wins under a well-specified model class, multi-step wins under misspecification from
partial observability — which predicts the sim-vs-real split exactly.

Competing diagnoses: **GRWM** (arXiv 2510.26782) argues the bottleneck is **latent geometry, not the
dynamics model**, shown by ablating the encoder with dynamics held fixed (CVPR 2026 acceptance
unconfirmed). *"Imagined Rollouts are Kinematic, Not Dynamic"* (arXiv 2607.05966, RSS 2026 workshop)
argues the failure isn't generic compounding error — latent imagination degenerates to kinematics and
drops dynamics.

## Horizon inflation — how to read these papers

Every paper can claim "long-horizon" while replanning constantly. Two clear instances:

- **FF-JEPA** (arXiv 2606.09311, KU Leuven — valuable as *independent* corroboration of hierarchy):
  flat LeWM scores 3.52% at long horizon, FF-JEPA 91.80%. Real delta. But the low level plans H=25
  replanned every 25 steps, the "300-step" column is 12 successive replans, and **the world model is
  a frozen, single-step-trained predictor** — only the subgoal planner is trained.
- **SkyJEPA** (arXiv 2606.23444, ~9K params, quadrotor proprioceptive state) trains with recursive
  unrolling at T=20 and measures compounding ratio out to k=60, but the real flight demo reverts to
  MPPI horizon 15 replanned at ≥100Hz.

Always separate **open-loop rollout depth** from **receding-horizon replanning**.

## What PLDM actually shows (it is not a horizon result)

*Learning from Reward-Free Offline Data* (arXiv 2502.14819, Sobal, Zhang, Cho, Balestriero, Rudner,
LeCun; preprint). ~2.2M params, MPPI **replanning every step**, ~4× slower than model-free baselines.
Findings: best generalization to unseen layouts (98.7±2.8%), most data-efficient (~80% success from a
few thousand transitions), most robust to suboptimal data. **Model-free RL wins when you have large
amounts of high-quality data.** It never measures rollout degradation and uses no hierarchy. Cite it
for latent planning's *generalization*, not its depth.

## Genuinely empty threads

- No failed reproduction of any JEPA planning headline (nobody has tried).
- No head-to-head JEPA vs DreamerV3/TD-MPC2 on long-horizon tasks.
- **No usable-rollout-horizon benchmark exists** — which is why every paper can claim long-horizon.
- **No JEPA paper ablates a rollout loss against measured rollout depth** — they ablate against task
  success, conflating model and planner.
- Discrete/quantized latents for rollout stability: Discrete-JEPA (arXiv 2506.14373) is a
  non-archival workshop poster on 64×64 toys **with no planner at all**, and has no planning
  successors. The earlier read of it as evidence for discretization stabilizing planning was too
  generous.

## Lesson to carry

The horizon problem was not solved by making the model predict further; it was routed around with
temporal abstraction and constant re-grounding. And the strongest evidence in the literature —
including the only peer-reviewed entry — says the model is not the binding constraint: an oracle
world model still fails, faithful unrolling doesn't imply planning success, and scaling the predictor
bought nothing.

That is the same shape as the conclusions in `lit-review-design-world-models.md`: put the learning
where the uncertainty actually lives, and it is not in the dynamics.
