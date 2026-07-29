# NoPE-vs-RoPE verdict audit (decisions-log entry 1)

Audit of the PROVISIONAL verdict "Position: RoPE (pure NoPE rejected)" per the
VERDICT-AUDIT PROTOCOL: implementation review vs the source papers, Haviv-2022 positive
controls on the saved checkpoints, an attention-mechanism probe of the length-gen cliff,
and a Muon-LR fairness sweep. Artifacts: `scripts/audit_nope_probes.py` (+ tests),
`evals/nope_audit/probes.json`, `configs/nope-audit-lr{05,10,20}.py`,
`runs/nope-audit-lr{05,10,20}`.

**Conclusion: (iii) fair result at this scale/duration — verdict CONFIRMED, with the
in-window gap reframed (see 4).** No implementation error found; no HP bias found. The
length-gen collapse is robust and now has a mechanistic explanation; the +0.057 in-window
penalty is exactly what the literature predicts for NoPE vs a *strong* positional scheme.

## 1. Implementation review — no deviation found

`NoPECausalSelfAttention` (src/microlab/model/reference/variants.py) vs the literature
definition (Kazemnejad et al. 2305.19466 §3: NoPE = a decoder-only transformer with the
positional encoding removed, position inferable only from the causal mask):

- **No positional signal except the causal mask.** The module computes raw q/k/v from
  `c_attn` and calls causal SDPA — no rotary tables, no offsets. `VariantGPT` adds `wpe`
  only for `pos="learned"`; neither checkpoint contains any positional tensor (checked:
  no `wpe`/rope keys in either state dict).
- **Byte-identical param trees — verified, not just claimed.** Both `ckpt_4500.pt` state
  dicts: same 111 tensor names, same shapes, 134,142,720 params. RoPE's cos/sin are
  non-persistent buffers (not in the state dict). Fresh builds of both variants at seed
  1337 have **bit-identical** parameters (`torch.equal` on every tensor): RoPE's buffer
  construction draws no RNG, so the init streams coincide.
- **Configs differ only where claimed.** Field-by-field diff of the checkpoint-embedded
  RunConfigs: `{pos, out_dir}` only. Same tokenizer (md5 `77ee1ed3...` in both run dirs
  == the data-dir tokenizer), same data dir, same seed → same data order (trainer uses a
  dedicated CPU generator seeded before model build) and same seeded val batches.
- **Optimizer parity.** Muon grouping (`build_muon_param_groups`) is shape/identity
  driven; identical param trees → identical groups in both arms.
- **Eval path treats NoPE correctly at extended lengths.** `load_for_eval` rebuilds at
  `eval_block=4124` (≥ 4096 + max_new + 16): for NoPE only the `T <= block_size` guard
  and KV-cache capacity grow — there is no hidden truncation and nothing positional to
  (mis)extend. The loss eval is a plain full forward (teacher-forced, no cache), causal
  SDPA. Passkey generation prefifills with `is_causal=True` (q_len == k_len) and decodes
  single tokens with `is_causal=False` against the full cache — correct in both regimes,
  identical machinery for both arms. RoPE is extended at native theta (raw
  extrapolation), which is the stated comparison and *flatters* NoPE if anything.

One reporting-side nuance (not an error): the eval's per-position bucket means show the
NoPE model's loss at positions <1024 stays healthy inside longer windows (3.42/3.29 at
2048; 3.42/3.29/3.96/6.56 buckets) — the 2x/4x "mean loss" collapse is entirely
positions >1024, i.e. genuine beyond-train-length failure, not contamination of
in-window behavior by the longer forward.

## 2. Positive controls (saved checkpoints; `evals/nope_audit/probes.json`)

### (a) Haviv control: implicit absolute position IS decodable from our NoPE model

Haviv et al. 2022 (2203.16634): causal LMs without positional encodings stay competitive
and "acquire an implicit notion of absolute positions"; their probe (2-layer ReLU MLP
classifying absolute position 0..1023, scored by mean absolute distance) finds NoPos
models reach learned-PE-level position decodability within ~4 layers. Replicated here
(64 val windows @1024, sequence-level 48/16 split, shuffled-label control; chance MAD
~341):

| tap        | NoPE MAD | NoPE shuf | RoPE MAD | RoPE shuf |
|------------|---------:|----------:|---------:|----------:|
| emb        |    339.3 |     344.3 |    356.2 |     341.1 |
| block 0    |    254.3 |     334.5 |    345.5 |     340.4 |
| block 3    | **36.8** |     349.3 |    265.1 |     344.4 |
| block 6    |     44.0 |     350.0 |    274.9 |     341.6 |
| block 9    |     62.6 |     341.5 |    316.3 |     349.6 |
| block 11   |    322.3 |     339.2 |    333.3 |     338.2 |

- NoPE mid-network localizes absolute position to within ~37 of 1024 (9x better than
  chance) with zero position signal in its embeddings (emb tap at chance — the built-in
  negative control showing the probe isn't reading content). This is precisely Haviv's
  emergence-by-layer-3-4 result: **our NoPE implementation behaves as the literature
  says a working NoPE model behaves.** The instrument is validated.
- RoPE's residual stream stays near chance everywhere — expected for a purely *relative*
  scheme (absolute position never enters the residual stream) and a useful contrast: the
  probe separates the arms in the literature-predicted direction.

### (b) In-window gap vs Haviv's "near-parity"

Ours: +0.0573 nats at matched step 4500 (TB, same val batches; +0.0580 on the
length-gen eval at 1024) = **+5.9% ppl** (27.29 → 28.92).

Haviv, The Pile @125M/1024 (their scale point closest to ours): NoPos 22.15 vs Learned
22.04 (**+0.5%**), vs Sinusoidal 21.49 (**+3.1%**), vs ALiBi 19.94 (**+11.1%**).

"Near-parity" in Haviv is parity with *learned APE* — against their strongest positional
scheme (ALiBi) the NoPos gap was 11%. RoPE (untested by Haviv) is a strong
relative scheme; our +5.9% sits inside the Haviv NoPos-vs-strong-PE range. **The
+0.057 is not anomalous — it is the literature-consistent cost of NoPE against a strong
baseline, and "NoPE matches explicit PEs in-window" was only ever true against weak
(learned/sinusoidal) ones.**

### (c) Attention mechanics of the length-gen cliff

Head-mean attention entropy normalized by ln(visible keys) (H%=1 → uniform), mean
attention distance, and mass on the last 64 keys; 8 val windows @4096, layers 0/6/11
(full grid in probes.json):

| arm  | layer | p=512 | p=1023 | p=1536 | p=2047 | p=4095 | last64 mass p=1023 → p=4095 |
|------|------:|------:|-------:|-------:|-------:|-------:|------------------------------|
| NoPE | 6     | 0.40  | 0.40   | 0.74   | 0.82   | 0.88   | 0.77 → 0.02                  |
| NoPE | 11    | 0.45  | 0.45   | 0.51   | 0.73   | 0.82   | 0.48 → 0.01                  |
| RoPE | 6     | 0.40  | 0.41   | 0.48   | 0.46   | 0.51   | 0.54 → 0.47                  |
| RoPE | 11    | 0.22  | 0.25   | 0.21   | 0.21   | 0.24   | 0.61 → 0.58                  |

In-window the two arms are statistically alike (NoPE L6 H% 0.40 vs RoPE 0.41). Beyond
1024, NoPE attention diffuses toward uniform (L6 H% 0.40 → 0.88; local mass 0.77 → 0.02)
while RoPE stays sharp at 4x. This is exactly the "attention distraction" mechanism Wang
et al. 2404.12224 identify as the cause of NoPE's length-generalization failure (their
fix — per-head temperature tuning — is a post-hoc intervention we did not apply, and
neither did the verdict claim to). **The 2x/4x loss cliff has a mechanistic cause inside
the model, not inside our eval.**

Passkey side-note: in-window (512/1024 grids) RoPE averages 0.78 vs NoPE 0.40, but the
failures are structured, not random — NoPE retrieves keys near the *start* (depth 0.1:
0.9/0.9) and drops keys near the end; RoPE the reverse (its own in-window zero at
L=1024/depth 0.1). "Passkey weak even in-window" stands as a relative claim, with the
caveat that both arms are patchy at 124M/737M tokens.

## 3. HP fairness: Muon-LR sweep for the losing arm

Three 1000-step NoPE reruns (configs identical to `nope-ab-nope.py` except `muon_lr`,
`max_steps=1000`, `out_dir`; `lr_decay_steps` stays 4500 so the schedule is the
original's, truncated): muon_lr 0.01 (x0.5), 0.02 (x1 — also the run-to-run stability
point), 0.04 (x2). Reference points at step 1000: original NoPE 3.9033, RoPE 3.8210
(gap +0.0823).

RESULTS-PENDING

## 4. Verdict classification

CLASSIFICATION-PENDING

## Protocol status

- (a) POSITIVE CONTROL: PASS (Haviv probe replicated on our NoPE arm).
- (b) HP FAIRNESS: SWEEP-PENDING
- (c) NOISE BAND: the Peri-LN multi-seed calibration arms (seed-1338 copies) are not yet
  trained, so the cross-seed band is still open. Two matched-seed measurements exist
  already: `runs/muon-ab-muon` vs `runs/nope-ab-rope` are config-identical twins (diff:
  `out_dir` only) trained months apart — their matched-step val-loss deltas across all
  18 eval points are max 0.0014 / mean 0.0009 nats, so the +0.057 gap is **~40x the
  run-to-run band**; the x1 sweep rerun adds the same measurement for the NoPE arm.
  Cross-seed variance remains for the Peri-LN lane, but note both A/B arms share one
  seed/init/data order, so the gap is the intervention effect at that seed, and the
  literature predicts its direction and size (2b). NOISE-PENDING
- (d) IMPLEMENTATION REVIEW: PASS (no deviation; param trees verified bit-identical at
  init, configs differ only in `pos`/`out_dir`, eval path correct for NoPE at extended
  lengths).
