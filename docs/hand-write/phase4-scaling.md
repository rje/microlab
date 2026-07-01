> **Exercise — on `main`, no branch switching.** Implement the stub in `src/microlab/exercises/phase04_scaling.py`, then run `pytest -m exercise -k phase04_scaling` to grade it against the reference oracle. Your solution is tracked in git; commit it when it passes.

# START HERE — hand-write the scaling tools (Phase 4)

You're on the exercises folder on `main`. You implement three closed-form functions in
`src/microlab/exercises/phase04_scaling.py`; the reference tools, the model-family generator,
and the GPU sweep runner are already on `main`. The nice part: your `count_params` is
graded against the **real model** — derive the formula, and `GPT(config).num_params()`
tells you if you got it.

## 1. See the sweep run on the GPU first (1 min)

```python
import torch
from microlab.data.reference.loaders import load_sample
from microlab.data.reference.pipeline import clean_text
from microlab.tokenizer.reference.bpe import BPETokenizer
from microlab.model.reference.scaling import run_scaling_sweep
from microlab.model.reference.train import TrainConfig

text = clean_text(load_sample()) * 80
tok = BPETokenizer(); tok.train(text, 512)
data = torch.tensor(tok.encode(text), dtype=torch.long)
print(run_scaling_sweep(data, [64, 128, 256], TrainConfig(steps=120, batch_size=24, block_size=64, device="cuda")))
```

Observed: loss fell 4.69 → 3.08 → 0.40 as `n_embd` went 64 → 128 → 256, with a fitted
`L = A·N^(-alpha)`. Bigger model, lower loss — the scaling phenomenon, on your GPU.

## 2. What you implement

```bash
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/exercises/test_phase04_scaling.py -v
```

1. **`count_params(config)`** — total params from the config alone (no instantiating the
   model), must equal `GPT(config).num_params()`. Work through the architecture: tied token
   embedding `V·C` (counted once) + positional `block·C`; per block — `c_attn` (3C×C),
   attn `c_proj` (C×C), `c_fc` (4C×C), mlp `c_proj` (4C×C), two LayerNorms (weight+bias);
   a final LayerNorm. Linear **biases only when `config.bias`**; LayerNorms always weight+bias.
2. **`training_flops_per_token(config)`** — `6 × (non-embedding params)`. The Chinchilla
   rule of thumb: ~2N flops/token forward, ~4N backward. (Why is embedding lookup ~free?)
3. **`fit_scaling_law(params, losses)`** — fit `L = A·N^(-alpha)`. Take logs: `log L` is
   linear in `log N` (slope `-alpha`, intercept `log A`); least-squares the line
   (`numpy.polyfit` is fine).

## 3. Why this matters (the papers)

- **Kaplan scaling laws / Chinchilla** — loss follows smooth power laws in params, data, and
  compute, which is *why* labs can predict a big model's loss from small runs. The Chinchilla
  result: for a fixed compute budget, params and tokens should scale together (~20 tokens per
  param) — most pre-Chinchilla models were over-parameterized and under-trained.
- The `6N` estimate is how you convert a model+token budget into a FLOPs number and place a
  run on the compute axis.

## 4. Honest caveat about the toy curve

On the tiny bundled corpus the models partly memorize, so the fitted `alpha` is steep and not
a real Chinchilla number. A *meaningful* curve needs a larger corpus and **validation** loss
(not train loss) — pull TinyStories/WikiText via `microlab.data.reference.loaders` and sweep
on held-out loss. The machinery is identical; only the data changes.

## 5. When you're done

`pytest tests/exercises/test_phase04_scaling.py` green → ping me for the Socratic review, then run
a real sweep on TinyStories and fit the curve on validation loss.

## Optional stretch

- Add a data axis: fix model size, vary token count, fit `L(D)`.
- Estimate the compute-optimal model size for a fixed FLOPs budget from your fitted exponents
  and check it against an actual run.
