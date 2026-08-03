# Written BEFORE the run: what coder-1b's loss curve should look like

Committed 2026-08-04, before the first paid step of the capstone. The point is to make
"is this working?" falsifiable at hour two instead of a judgement call at day three.

## The two anchors, both ours

| run | params | context | data | tokenizer |
|---|---|---|---|---|
| `runs/frontier-32k` | 124M | 32k | 100% code | 49k |
| `runs/1b` | 983M | 1,024 | FineWeb (prose) | 32k |

| tokens | frontier-32k (code) | 1b (prose) |
|---|---|---|
| 0.25B | 2.265 | 4.661 |
| 0.50B | 1.516 | 3.703 |
| 1.00B | 1.325 | 3.261 |
| 2.00B | 1.172 | 3.005 |
| 4.00B | 1.061 | 2.839 |
| 21.0B | — | 2.501 |

`coder-1b` sits between them: 1.01B params like the second, 32k context and a **67.5%
code-like** mix like the first.

## The prediction

Mix the two anchors by corpus composition (66.3% code + 1.2% commits = code-like; 15% web,
10% math, 5% markdown, 2.5% arXiv = prose-like), and allow 0-15% improvement from the 8.2x
parameter increase. The band is deliberately not tighter — at these token counts the model
is deeply UNDER-trained (D/N ≈ 1-4 against Chinchilla's 20), so extra parameters buy much
less than they will later.

| tokens | step | **predicted mix val loss** |
|---|---|---|
| 0.25B | 476 | 2.59 – 3.04 |
| 0.50B | 953 | 1.89 – 2.23 |
| **1.05B** | **2,000** | **1.65 – 1.94** |
| **2.10B** | **4,000** | **1.49 – 1.76** |
| 4.00B | 7,629 | 1.39 – 1.64 |

The two bold rows are the milestones a $20 run reaches.

## What would falsify it

- **Above 2.2 at step 2,000** — materially worse than a 124M model reached on code alone.
  Something is wrong: the architecture, the LR, or the data.
- **Below 1.3 at step 2,000** — too good, which is a *warning*, not a win. The most likely
  cause is train/val leakage in the mix builder, and it should be investigated before being
  believed.
- **Non-monotone val loss** after warmup (700 steps) — instability.
- **Per-slice divergence**: code improving while math or web get worse. The eval suite
  reports per-slice val precisely because one aggregate number hides this.

## Caveats stated up front

- **The tokenizers differ** (32k vs 49k+FIM). Larger vocabularies raise per-token loss
  mechanically, so the prose anchor is, if anything, optimistic — its absolute numbers are
  not directly comparable to ours.
- **The prose anchor was trained at 1,024 context**, not 32k. Longer context usually helps
  slightly at equal tokens.
- **FIM is in our mix and in neither anchor.** Half the code documents are reordered into
  prefix/suffix/middle form, which is *harder* to predict than left-to-right. That pushes
  our loss up relative to the code anchor, and is the most likely reason to land in the
  upper half of the band.
- The band is wide on purpose. A prediction narrow enough to always be wrong tells you
  nothing; this one is meant to catch the failure modes above, not to score points for
  precision.

## Independent checks at each milestone

Beyond val loss, `scripts/eval_suite.py` gives four things the loss curve cannot:

1. **per-slice val loss** — is any slice being ignored?
2. **FIM middle-span loss** — is infilling being learned at all, or is half the code budget
   buying nothing?
3. **syntax validity** — the early-signal code metric; it moves long before HumanEval
   leaves zero
4. **probe battery** — code and math categories, which were added for exactly this run
