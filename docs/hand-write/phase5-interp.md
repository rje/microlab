> **Exercise — on `main`, no branch switching.** Implement the stub in
> `src/microlab/exercises/phase05_interp.py`, then run
> `pytest tests/exercises/test_phase05_interp.py -m exercise` to grade it.

# START HERE — look inside the model you trained (Phase 5)

You own every weight of the 150M TinyStories model. This phase is about seeing what those
weights learned. Two hand-writes, both graded against `microlab.interp.reference.lens`:

1. **`logit_lens`** — decode EVERY layer's residual stream through the model's own final
   norm + unembedding. Early layers predict garbage; late layers converge on the real
   output. Where does the answer "appear"? (Tuned Lens is the paper-grade version.)
2. **`induction_score`** — feed a repeated random sequence [A B C … A B C …]; an induction
   head at position i attends to i−period+1 (the token AFTER the previous occurrence of
   the current token) — that's copying, the mechanism behind in-context learning. Score =
   mean attention mass on that diagonal.

## See it on the real model

```bash
python scripts/interp_report.py runs/150m
```

## Stretch (uses a decision made during training)

Checkpoint pruning is off, so `runs/150m/ckpt_200.pt, ckpt_400.pt, …` all exist. Run the
induction scoring across checkpoints and plot max-score vs step: induction heads form in a
phase change (Anthropic, "In-context Learning and Induction Heads" —
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/).

## Readings

Tuned Lens; ROME (locating factual associations). Both in the console reading workspace.
