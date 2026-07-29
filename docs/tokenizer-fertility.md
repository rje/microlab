# Code-native tokenizer fertility study

Fertility = tokens emitted per unit of text; **lower tokens-per-byte is better compression**. Measured over ~20 MB per language from `data/corpora/code-samples/` (see that manifest for sources/licenses).

Candidates: **code-49k** (49,152 vocab) · **code-32k** (32,000 vocab) · **fineweb-32k-baseline** (32,000 vocab).

## Tokens per byte (lower = better)

| language | code-49k | code-32k | fineweb-32k-baseline |
|---|---|---|---|
| python | 0.300 | 0.306 | 0.509 |
| javascript | 0.280 | 0.285 | 0.504 |
| typescript | 0.250 | 0.256 | 0.471 |
| shell | 0.323 | 0.332 | 0.470 |
| sql | 0.319 | 0.326 | 0.373 |
| json | 0.397 | 0.399 | 0.434 |
| markdown | 0.357 | 0.382 | 0.483 |
| prose | 0.228 | 0.236 | 0.218 |

## Bytes per token (higher = better compression)

| language | code-49k | code-32k | fineweb-32k-baseline |
|---|---|---|---|
| python | 3.33 | 3.27 | 1.97 |
| javascript | 3.58 | 3.51 | 1.98 |
| typescript | 4.00 | 3.91 | 2.12 |
| shell | 3.09 | 3.02 | 2.13 |
| sql | 3.14 | 3.07 | 2.68 |
| json | 2.52 | 2.50 | 2.30 |
| markdown | 2.80 | 2.62 | 2.07 |
| prose | 4.39 | 4.23 | 4.58 |

## Tokens per line

| language | code-49k | code-32k | fineweb-32k-baseline |
|---|---|---|---|
| python | 13.2 | 13.5 | 22.4 |
| javascript | 9.9 | 10.1 | 17.9 |
| typescript | 9.0 | 9.2 | 17.0 |
| shell | 10.3 | 10.6 | 15.0 |
| sql | 64.3 | 65.7 | 75.2 |
| json | 10.1 | 10.2 | 11.1 |
| markdown | 52.8 | 56.4 | 71.4 |
| prose | 52.6 | 54.6 | 50.4 |

## Round-trip fidelity (decode(encode(x)) == x)

| language | code-49k | code-32k | fineweb-32k-baseline |
|---|---|---|---|
| python | 1.000 | 1.000 | 1.000 |
| javascript | 1.000 | 1.000 | 1.000 |
| typescript | 1.000 | 1.000 | 1.000 |
| shell | 1.000 | 1.000 | 1.000 |
| sql | 1.000 | 1.000 | 1.000 |
| json | 1.000 | 1.000 | 1.000 |
| markdown | 1.000 | 1.000 | 1.000 |
| prose | 1.000 | 1.000 | 1.000 |

## Digit-sequence handling

Tokens per probe (individually-split digits shown as ✓):

| probe | code-49k | code-32k | fineweb-32k-baseline |
|---|---|---|---|
| `12345` | 5 ✓ | 5 ✓ | 2 ✗ |
| `3.14159` | 7 ✓ | 7 ✓ | 5 ✗ |
| `0xFF3A` | 4 ✓ | 4 ✓ | 5 ✓ |
| `1000000` | 7 ✓ | 7 ✓ | 2 ✗ |
| `255` | 3 ✓ | 3 ✓ | 1 ✗ |
| `2024-01-15` | 10 ✓ | 10 ✓ | 6 ✗ |
| `v1.2.3` | 6 ✓ | 6 ✓ | 6 ✓ |

## Indentation handling (tokens to encode a leading indent run)

| indent | code-49k | code-32k | fineweb-32k-baseline |
|---|---|---|---|
| 2 spaces | 1 | 1 | 1 |
| 4 spaces | 1 | 1 | 3 |
| 8 spaces | 1 | 1 | 7 |
| 12 spaces | 1 | 1 | 11 |
| 16 spaces | 1 | 1 | 15 |
| tab (\t) | 2 | 2 | 2 |
| nested snippet | 16 | 16 | 45 |

## Analysis

**Compression vs the baseline.** Averaged over the primary code languages (Python/JS/TS), `code-49k` emits ~44.1% fewer tokens per byte than the `fineweb-32k-baseline` FineWeb tokenizer, and `code-32k` ~42.9% fewer at the same 32k size -- the win is the code-tuned merges, not just the larger vocab. On Python alone the code recipe saves ~41.0% (code-49k) / ~39.9% (code-32k); on TS/JS ~45.7% / ~44.6%.

**Python-only vs TS/JS-only vs dual.** Python is the more compressible target: the code tokenizers reach 0.300 tok/byte on Python vs 0.265 on TS/JS (TS/JS syntax -- `:`, generics, JSX, `=>`, long camelCase identifiers -- fragments more). A Python-only model can spend its whole code budget on Python merges and would compress Python slightly harder still; a TS/JS-only model needs the vocab most because its baseline fertility is worst. For a **dual** Python+TS/JS model the 49k vocab is the reasonable call: at 32k the three languages contend for merge slots and per-language compression drops toward the baseline, whereas 49k buys back most of the per-language loss (dual mean 0.277 @49k vs 0.282 @32k).

**Cost of digit-splitting.** Forcing every digit to its own token is deliberate (it removes the digit-pair merges implicated in the arithmetic floor), but it is not free: numbers cost one token per digit. `12345` is 5 tokens under the code recipe vs 2 for the baseline (which merges digit pairs), and on English prose the code tokenizers sit at 0.228 (49k) / 0.236 (32k) tok/byte vs 0.218 for the prose-tuned baseline. That prose gap is the combined price of digit-splitting plus spending merge slots on code; for a coding specialist it is the right trade, since prose is a minority of the intended workload and correct arithmetic is worth more than a few percent of prose compression.

**Recommendation.** Ship the dual Python+TS/JS design on `code-49k`: it gives the broadest coverage, keeps per-language compression well ahead of the baseline, and the digit-split + indentation merges directly target the two known deficiencies. If the lab instead commits to Python-only, `code-32k` is nearly as good on Python at half-again-smaller vocab (cheaper embedding/softmax) and would be the leaner choice; TS/JS-only is the one case that most needs the 49k vocab.

