# Curriculum audit — 2026-07

Audit of the education plan (`docs/curriculum.md`, `site/content/phases.json`,
`docs/hand-write/*.md`, `src/microlab/exercises/`, `site/content/synopses/`) against what
the lab has actually done since the 1B landed. Run 2026-07-29. The prompt for this audit:
the research track sprinted ahead of the curriculum (post-training arc, retrofit saga,
ablation ladder, code-specialist program), and the plan needed re-centering on the
educational goals — absorbing the new work as *pedagogy*, not as a changelog.

Method: the coverage yardstick is the delivered work (`docs/sota-parity-1b.md`,
`docs/gqa-conversion-audit.md`, `docs/arch-review-2026.md`,
`docs/specialist-design-decisions.md`, `docs/benchmarks-1b.md`,
`docs/tokenizer-fertility.md`, `docs/code-corpus-*.md`, `docs/code-eval-baselines.md`,
git history since ~2026-07-20) tabled against every surface the curriculum teaches through
(phase rows, note paragraphs, hand-write guides, graded exercise stubs, reading lists).
`src/microlab/exercises/` is ground truth for what is actually graded.

## 1. Coverage

| Arc / skill | Where it lived before this audit | Verdict | Fix applied |
|---|---|---|---|
| 1B pretrain | Phase 2/4/7 rows; scale path | covered, **stale** (venue still "decided by vendor spike") | scale-path step 4 and four-layers §4 updated: 1B trained locally; cloud drills remain |
| Benchmark placement (lm-eval vs public cohort) | nowhere | **GAP** | Phase 0 real-scale cell; measurement note item 6; capability note upgraded from prediction to measurement |
| Eval methodology: likelihood-MC vs generative, PMI, copy-traps, chance baselines | nowhere (Phase 0 teaches pass@k/ECE only) | **GAP** | measurement note item 1; recommendation R3 for graded stubs |
| Positive controls / verdict-audit protocol | `specialist-design-decisions.md` only | **GAP** | measurement note item 4; Phase 16 real-scale cell |
| Ablation-ladder design + noise calibration (multi-seed band, HP fairness) | nowhere | **GAP** | measurement note item 3; recommendation R9 |
| SFT (mix, loss masking) | Phase 9 hand-write + guide | covered; real-scale run absent from table | Phase 9 real-scale cell |
| Multi-turn chat SFT + serving integration (threaded Playground) | nowhere | **GAP** | post-training note; Phase 9 + Phase 6 real-scale cells; recommendation R5 |
| DPO/IPO, length bias, SimPO | Phase 12 hand-write (DPO); IPO/SimPO/length-bias papers ingested but **orphaned from all reading lists** | partial | phases.json: 3 papers added to phase-12; post-training note |
| RLAIF / on-policy preference data | RLAIF paper ingested, orphaned | **GAP** | phases.json: RLAIF added to phase-11; post-training note (with the judge-not-teacher house rule) |
| Reward model (BT on the 1B) | Phase 11 hand-write + guide | covered; real run absent | Phase 11 real-scale cell |
| Best-of-n as RM validation / cheapest RM deployment | nowhere | **GAP** | Phase 11 real-scale cell; post-training note; recommendation R4 |
| GRPO + verifiable rewards | Phase 13 hand-writes (verifiable_reward, GRPO advantage, PPO clip) | covered; real run absent | Phase 13 real-scale cell |
| KL control (k3 penalty to frozen reference) | nowhere (the real GRPO uses it; no stub, no guide mention) | **GAP** | post-training note names it; recommendation R2 for a graded stub |
| Reward hacking as observed phenomenon | phases.json phase-13 summary mentions it as a topic | partial (hypothetical only) | post-training note: in-house observation, judge-graded RL runs |
| GQA retrofit (uptrain + conversion audit) | Phase 3 real-scale row (added by parity review); audit doc referenced in succession note | covered | none needed |
| Context extension + retrieval-decay finding | Phase 3 row + succession note (extension saga) | partial (retrieval-vs-loss lesson implicit) | measurement note item 2 makes it explicit |
| Muon adoption | Muon note + Phase 4 row | covered, **inconsistent** (claimed graded stub does not exist; A/B said 150M, configs are 124M) | Muon note corrected; table marks stub pending; recommendation R1 |
| NoPE/Peri-LN ablation ladder | succession note said NoPE A/B "queued (being built now)" | **stale + contradictory** (A/B ran; claim inverted; verdict PROVISIONAL) | succession note updated with the result, pointing at the design log as live source of truth |
| Code tokenizer / fertility | Phase 1 row promised the study | covered; now delivered | Phase 1 cell cites `docs/tokenizer-fertility.md` |
| Corpus licensing / attribution | `docs/datasets.md` (license column); no curriculum treatment of the attribution pipeline | partial | Phase 1 real-scale cell cites `docs/code-corpus-pipeline.md` (permissive allowlist + attribution manifest) |
| Data-efficiency levers (mix, repetition, selective loss, FIM) | mixture: parity note (Phase 4); repetition: `scaling-data-constrained` in phase-4 readings; selective-loss/FIM: design-log lanes only | partial | left in the design log deliberately (undemonstrated at our scale; see §4) |
| Code + tool eval floors | nowhere | **GAP** | Phase 0 + Phase 15 real-scale cells cite `docs/code-eval-baselines.md` |
| Specialist-design / parity-review process as capstone methodology | parity note (process rules only) | **GAP** as pedagogy | Phase 16 real-scale cell + measurement note items 4–6 |

## 2. Ordering and consistency findings

1. **Curriculum claimed a graded stub that does not exist.** The Muon note and the Phase 4
   table row said the Newton–Schulz step "is an oracle-graded Phase 4 hand-write";
   `exercises/phase04_scaling.py` ends at the muP table and `tests/exercises/` has no such
   test. Fixed: note and row now say the stub is pending (reference: `microlab.train.muon`);
   cutting it is R1 below.
2. **Stale experiment status contradicting a landed verdict.** The succession note said the
   NoPE-vs-RoPE ablation was "queued (being built now)" while
   `docs/specialist-design-decisions.md` records verdict 1 (RoPE stays, pure NoPE rejected,
   PROVISIONAL pending audit) — and the note's framing ("length-generalize *better* with no
   positional encoding") taught the exact claim our own A/B inverted at our scale. Fixed:
   the note now reports the result and the reopen condition, and defers to the design log
   for the verdict's audit status (a verdict audit was in flight during this audit).
3. **Sizing drift.** Muon note said "150M Muon-vs-AdamW A/B"; the ladder standardized on
   124M twins (`configs/muon-ab-*.py`, `muon-ab sizing` in the design log). Fixed.
4. **Resolved decisions still framed as open.** Scale path said the 1B's venue would be
   "decided by the Phase 7 vendor spike"; the 1B has trained locally (22.6 → ~13.5 days
   after the perf work). Fixed in curriculum.md. The same staleness remains in the
   phases.json phase-7 *summary*, which this audit could not touch (edits restricted to
   reading lists) — see R8.
5. **Prediction vs measurement.** The capability note predicted "GPT-2-XL / Pythia-1B
   class"; `docs/benchmarks-1b.md` has since measured it. Fixed: the note now carries the
   measured placement (and the placement *practice* joins Phase 0).
6. **Five ingested papers were orphaned from every reading list** (IPO, SimPO,
   DPO-length-bias → phase-12; RLAIF → phase-11; MT-Bench/LLM-as-judge → phase-0). They
   were ingested by the RLHF-papers commit but never linked. Fixed in phases.json; the
   console's `validate_state` confirms all ids resolve.
7. **Real-scale column had gone dark after Phase 8.** Rows 9–16 said "(uses scale)" or "—"
   while the lab ran the whole post-training arc for real. Fixed: rows 0, 1, 6, 9, 11, 12,
   13, 15, 16 now name the delivered real-scale work, one line each.
8. **Phase-3 dumping-ground watch.** Phase 3 now carries the ablation hand-writes, two
   retrofit real-scale items, and the longest note (attention/position succession). Verdict:
   acceptable — everything in it is genuinely attention/position — but it was also
   accreting *methodology* (how to trust an ablation), which now lives in the measurement
   note instead. No renumbering; watch it.
9. **Ordering overall: sound.** 0 (measure) → 1–2 (data, model) → 3–4 (architecture,
   scale) → 5–7 (open it up, serve it, distribute it) → 8–10 (adapt) → 11–13 (align) →
   14–15 (reason, act) → 16 (report) still tells the right story; no phase moves. The
   RM-before-DPO order (11 → 12) is deliberate: Bradley-Terry first makes DPO's
   implicit-reward derivation land.
10. **phases.json statuses track the learner, not the lab.** phase-0 is "current" and
    everything else "planned" while the research track is deep in phase-13 territory. That
    is *correct* (the statuses are the student's climb through the graded surface) but
    nowhere stated; without a legend it reads as neglect. See R8.
11. **Hand-write guide drift** (guides are ground-truthed against stubs in §5; the notable
    mismatches): phase5-interp.md says "two hand-writes" but four stubs are graded
    (`collect_residual_stream` and `attention_patterns` are also `NotImplementedError`);
    phase0/phase1 guides carry a "You're on branch the exercises folder" refactor typo;
    phase4-scaling.md has no Newton–Schulz section; phase13-rl.md never mentions the KL
    term the real GRPO run uses.

## 3. Drift assessment

The lab's identity expanded — research receipts (parity review, conversion audit, verified
2026 literature sweep, verdict-audit protocol), an affordability instinct (local 1B, ladder
screens at 124M), and release ambitions (the coding-specialist program). The curriculum's
job is to absorb that as *transferable method*, and the shape it wanted was not a new phase:
every new practice already had a natural phase home. So the absorption is:

- **A cross-cutting methodology strand** — the new "Measurement & methodology note" in
  curriculum.md: honest scoring (likelihood/PMI/copy-traps/chance), loss-is-not-capability
  (retrieval gates), ablation discipline (twins, sizing, ladder, noise band), verdict
  audits (positive control / HP fairness / noise band / implementation review,
  PROVISIONAL-until-audited), reviews at the right vintage (parity table + dated literature
  sweep, adopt-at-demonstrated-scale), and benchmark placement hygiene. Each item names its
  phase; Phase 16 is named as where the strand culminates ("the capstone is not the model
  but the audited case for every claim about it").
- **A post-training note** — the canonical RLHF arc (SFT mix → multi-turn chat → preference
  data from public + RLAIF sources → DPO/IPO with the length-bias literature → BT reward
  model → best-of-n validation → KL-leashed GRPO → reward over-optimization as an in-house
  observation), each stage tied to its phase, one clause of receipts each.
- **Real-scale cells as receipts, not diary** — one line per phase naming what ran, with at
  most one doc pointer.

What was deliberately *not* done: no new phase and no renumbering (17 phases still tell the
story); no per-run numbers in curriculum.md beyond the few that carry a lesson (the 73%
best-of-n win rate, the 5-of-6 vs Pythia); no absorption of undecided design-log lanes
(MLA/GDN/MoBA/FIM/selective-loss stay in `specialist-design-decisions.md` until they
produce a verdict — the curriculum teaches settled method, the log holds live bets); no
phases.json edits beyond reading links (constraint of this audit); no exercise stubs or
hand-write edits (code and guide changes are recommendations below, so the graded surface
never silently diverges from its tests).

## 4. Recommendations (not implemented here)

Exercise stubs are code and out of scope for this audit; guides move with their stubs.
In priority order:

- **R1 — Phase 4: cut the `newton_schulz_step` stub** in `exercises/phase04_scaling.py`
  (graded vs `microlab.train.muon`'s NS iteration) and add the matching section to
  `docs/hand-write/phase4-scaling.md`. This closes the one place the curriculum
  over-claimed its graded surface.
- **R2 — Phase 13: add a `kl_penalty` (k3 estimator) stub** graded vs the GRPO
  implementation's KL term, and a guide paragraph on why the leash exists (the observed
  RM-score inflation). KL control is the only canonical-RLHF component with no graded
  treatment.
- **R3 — Phase 0: add likelihood-MC scoring stubs** — per-choice log-likelihood and PMI
  scoring graded vs `scripts/track_probes.py`'s scorer — plus a guide section on
  copy-traps, balanced positions, and chance baselines (the below-chance reasoning-set
  story is the motivating example).
- **R4 — Phase 11: add a `best_of_n` stub** (argmax-by-RM over K candidates + RM/judge
  agreement rate), with `runs/bestofn_eval.json` as the run-for-real shape.
- **R5 — Phase 9: add a multi-turn `build_chat_example` stub** (per-turn loss masking over
  a whole conversation path) graded vs the chat-mix builder's masking; the single-turn
  version stays as the on-ramp.
- **R6 — phase5-interp.md: correct "two hand-writes" to four** (all four stubs are graded).
- **R7 — fix the "You're on branch the exercises folder" typos** in phase0/phase1 guides.
- **R8 — phases.json (beyond this audit's edit scope):** refresh the phase-7 summary
  (venue resolved: 1B trained locally; cloud drills remain), add a one-line status legend
  ("statuses track the learner's climb, not the lab's research runs"), and start populating
  `tasks[]` for phases whose real-scale work has shipped so the console shows the same
  receipts the table now does.
- **R9 — Phase 4 stretch: a noise-band placement hand-write** (e.g. bootstrap CI on a
  val-loss delta vs the multi-seed band) so the ablation-discipline strand gets a graded
  anchor once the Peri-LN calibration numbers exist.
- **R10 — a short retrofit run-for-real guide** (`docs/hand-write/` addition) walking the
  MHA→GQA uptrain and staged context extension as exercises against a *student-trained*
  checkpoint, with `docs/gqa-conversion-audit.md` as the worked "audit your conversion"
  example.
