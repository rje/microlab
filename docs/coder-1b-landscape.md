# The small-model landscape: public 1B–4B models on reasoning & coding evals

Compiled 2026-08-06 from three parallel research sweeps (primary sources: papers, model
cards, EvalPlus/BigCode/Open-LLM leaderboards). Full tables below; every number is
traceable, self-reported vs independent is marked, and N/A means no primary source —
several circulating figures (e.g. "Gemma 3 1B MMLU 59.6") turned out to be column-paste
errors from bigger siblings.

## Why this exists: placing coder-1b honestly

coder-1b is 1.2B params trained from scratch on **21B tokens** (66% code, chunk-FIM).
The headline rows below sit **21×–262× to the right of us in tokens**, and most of the
modern general-purpose cohort is additionally **pruned/distilled from larger parents**
(off-menu here by house rule). The fair published comparisons at our token budget:

- **phi-1's ablation grid**: raw-Stack data at ~26B tokens → ~8 HumanEval (350M);
  curated "textbook" data at the same budget → ~17–20. Data quality ≈ 2–2.5× at our scale.
- **SantaCoder's 118B-token ablations** and CodeGen's staged 350M/2.7B curves.
- **No published ~1B model exceeds ~10 HumanEval by ~21–26B raw-code tokens.** Our
  realistic end-of-run target on this axis is single digits; anything more means the mix
  or FIM objective is buying something unusual.

Two cross-cutting facts from the sweeps that shape the 3B plan:

1. **Token counts left Chinchilla behind**: the 2024–26 leaders train 1–4B models on
   9–36T tokens (150–450× "optimal"). Capability at small scale is mostly a token/data
   story, not an architecture story.
2. **Distillation is the majority recipe** (Llama 3.2, Gemma 2/3, Falcon 3, Ministral 3,
   Minitron). The honest from-scratch cohort — OLMo, SmolLM2/3, TinyLlama, Pythia, Qwen —
   is our real reference class, and the data-quality lever (phi family,
   Arctic-SnowCoder's 555B ≈ StarCoder2-3B's 3.3T) is the documented way to punch up
   without a token mountain.

---



# SECTION: 1B-class general models

Research complete. Below is the full deliverable.

# Public ~1B-Class LLMs (0.5B–2B): Reasoning & Coding Benchmark Survey

**Legend:** `S` = self-reported by model developer (card/paper/blog); `I` = independent measurement (named source); shot config in parens where stated. N/A = not reported by any source I could verify. Scores are for the **base** model unless the row says Instruct. Cross-model comparison is indicative only — harnesses, shot counts, and metrics differ per row.

## Core models

| Model (variant) | Params | Train tokens | Released | Weights / license | MMLU | GSM8K | ARC-C | HellaSwag | HumanEval | MBPP |
|---|---|---|---|---|---|---|---|---|---|---|
| **Llama 3.2 1B** (base) | 1.23B | "up to 9T" | Sep 2024 | Open, Llama 3.2 Community License | 32.2 (5s, S); 31.1 (5s, I-Falcon) | 6.6 (5s, I-Falcon) | 32.8 (25s, S); 40.2 (25s, I-Falcon) | 61.2 (I-SmolLM2 paper) | 18.9 (I-SmolLM2 paper) | N/A |
| **Llama 3.2 1B Instruct** | 1.23B | — | Sep 2024 | same | 49.3 (5s, S) | 44.4 (8s CoT, S) | 59.4 (0s, S) | 41.2 (0s, S) | N/A | N/A |
| **Gemma 3 1B** (PT) | 1.0B | 2T | Mar 2025 | Open, Gemma license | N/A (not reported for 1B) | N/A | 38.4 (25s, S) | 62.3 (10s, S) | N/A | N/A |
| **Gemma 3 1B** (IT) | 1.0B | — | Mar 2025 | same | N/A (MMLU-Pro 14.7, 0s, S) | 62.8 (0s, S) | N/A | N/A | 41.5 (0s, S) | 35.2 (3s, S) |
| **Qwen2.5-1.5B** (base) | 1.54B | 18T | Sep 2024 | Open, Apache 2.0 | 60.9 (5s, S); 60.9–61.0 (I-Hymba, I-Falcon) | 68.5 (4s, S); 62.2 (5s, I-Falcon) | 54.7 (25s, S); 54.8 (25s, I-Falcon) | 67.9 (10s, S); 66.4 (I-SmolLM2) | 37.2 (0s, S) | 60.2 (0s, S) |
| **Qwen2.5-1.5B-Instruct** | 1.54B | — | Sep 2024 | same | MMLU-redux 50.7 (S) | 73.2 (S) | N/A | N/A | 61.6 (S) | 63.2 (S) |
| **Qwen3-1.7B** (base) | 1.7B (1.4B non-emb) | 36T | Apr 2025 | Open, Apache 2.0 | 62.63 (S, tech report) | 75.44 (S) | N/A | N/A | 52.7 (EvalPlus, S) | 55.4 (EvalPlus, S) |
| **TinyLlama 1.1B** (base, 3T ckpt) | 1.1B | 3T | Dec 2023–Jan 2024 | Open, Apache 2.0 | 26.04 (5s, S) | 1.44 (5s, S) | 33.87 (25s, S) | 60.31 (10s, S) | 9.15 (0s, S paper) | N/A |
| **OLMo 1B** | ~1.2B | 3T (Dolma; per card) | Feb 2024 | Open, Apache 2.0 | N/A (not reported for 1B) | N/A | 34.45 (0s, S) | 62.5 (0s, S) | N/A | N/A |
| **OLMo 2 1B** (0425, base) | ~1B (HF lists "1B") | 4T | Apr 2025 | Open, Apache 2.0 | N/A on card (see note) | N/A | N/A | N/A | N/A | N/A |
| **OLMo 2 1B Instruct** | ~1B | — | Apr 2025 | same | 40.0 (S) | 68.3 (S) | N/A | N/A | N/A | N/A |
| **Pythia-1B** | 1.0B | 300B (Pile) | Feb–Apr 2023 | Open, Apache 2.0 | 25.70 (5s, I-TinyLlama paper) | N/A (~0) | 27.05 (0s, I) | 47.16 (0s, I) | 1.83 (0s, I) | N/A |
| **Pythia-1.4B** | 1.4B | 300B (Pile) | 2023 | Open, Apache 2.0 | 25.41 (5s, I) | N/A | 28.50 (0s, I) | 52.01 (0s, I) | 4.27 (0s, I) | N/A |
| **phi-1.5** | 1.3B | 150B seen (30B-token dataset) | Sep 2023 | Open, MIT (relicensed) | 37.6 (2s, S paper) | 40.2 (0s pass@1, S) | 44.4 (0s, S) | 47.6 (0s, S) | 34.1 (0s pass@1, S) | 37.7 (0s pass@1, S) |
| **SmolLM2-1.7B** (base) | 1.7B | 11T | Nov 2024 | Open, Apache 2.0 | 50.1–50.3 (5s, I-Falcon/Hymba) | 31.0 (5s, S) | 54.1 (25s, I-Falcon); ARC-avg 60.5 (S) | 68.7 (S) | 22.6 (S paper) | N/A |
| **SmolLM2-1.7B-Instruct** | 1.7B | — | Nov 2024 | same | N/A | 48.2 (5s, S) | N/A | N/A | N/A | N/A |
| **StableLM 2 1.6B** (base) | 1.64B | 2T-token dataset, 2 epochs | Jan 2024 | Open, Stability AI Community License | 38.9 (5s, S) | 17.4 (5s, S) | 43.3 (25s, S) | 70.5 (10s, S) | N/A | N/A |
| **Falcon3-1B-Base** | 1.67B | 80GT (pruned + distilled) | Dec 2024 | Open, TII Falcon-LLM License 2.0 | 42.5 (5s, S) | 34.3 (5s, S) | 48.1 (25s, S) | N/A (card omits) | N/A | N/A |
| **MobileLLM-1B** | 1.01B | 1T | 2024 (paper Feb; weights later) | Weights on HF, FAIR **non-commercial** | N/A | N/A | 39.0 (0s, S) | 61.4 (0s, S) | N/A | N/A |
| **MobileLLM-1.5B** | 1.51B | 1T | 2024 | same | N/A | N/A | 40.9 (0s, S) | 64.5 (0s, S) | N/A | N/A |
| **MobileLLM-R1-950M** | 949M | ~2T high-quality (<5T total) | Sep 2025 | Weights on HF, FAIR **non-commercial** | 47.4 (5s, S, base) | 61.6 base / 67.5 post-trained (0s, S) | N/A | N/A | 46.3 (S, base) | 39.2 (S, base) |

## Notable additions (2024–2026, same class)

| Model | Params | Train tokens | Released | License | MMLU | GSM8K | ARC-C | HellaSwag | HumanEval | MBPP |
|---|---|---|---|---|---|---|---|---|---|---|
| **Gemma 2 2B** (base) | 2.6B | 2T | Jul 2024 | Gemma license | 51.3 (5s, S) | 23.9 (5s, S) | 55.4 (25s, S) | 73.0 (10s, S) | 17.7 (pass@1, S) | 29.6 (3s, S) |
| **Qwen2.5-0.5B** (base) | 0.49B | 18T | Sep 2024 | Apache 2.0 | 47.5 (5s, S) | 41.6 (4s, S) | 35.6 (25s, S) | 52.1 (10s, S) | 30.5 (S) | 39.3 (S) |
| **DeepSeek-R1-Distill-Qwen-1.5B** | ~1.78B | SFT distill (800K R1 samples) | Jan 2025 | MIT | N/A | N/A | N/A | N/A | N/A | N/A |
| **MiniCPM-2B** (SFT) | 2.4B non-emb | ~1.1T | Feb 2024 | Custom GML (research free, commercial by permission) | 53.46 (S) | 53.83 (S) | 68.00 (S, own harness) | 68.25 (S) | 50.00 (S) | 47.31 (S) |
| **BitNet b1.58 2B4T** | 2B | 4T | Apr 2025 | MIT | 53.17 (S) | 58.38 (S) | 49.91 (S) | 68.44 (S) | HumanEval+ 38.4 (S) | N/A |
| **LFM2-1.2B** (Liquid AI) | 1.17B | 10T | Jul 2025 | LFM Open License v1.0 | 55.23 (S) | 58.3 (S) | N/A | N/A | N/A | N/A |
| **Granite 4.0 H 1B Instruct** (IBM) | ~1.5B | undisclosed | Oct 2025 | Apache 2.0 | 59.74 (5s, S) | 69.83 (8s, S) | N/A | N/A | 73 (pass@1, S) | 69 (pass@1, S) |
| **OpenELM-1.1B** (Apple) | 1.1B | ~1.8T | Apr 2024 | Apple sample-code license | 27.05 (5s, S) | N/A | 32.34 (0s, S) | 64.81 (0s, S) | N/A | N/A |
| **Hymba-1.5B-Base** (NVIDIA) | 1.5B | 1.5T | Nov 2024 | NVIDIA Open Model License | 51.19 (5s, S) | N/A base (Instruct 58.76, S) | 45.90 (S) | 53.55 (S; likely acc not acc_norm) | N/A | N/A |

## Per-model notes

- **Llama 3.2 1B** — not trained from scratch: structurally pruned from Llama 3.1 8B in one shot, with logits from 3.1 8B/70B used as distillation targets during pretraining (Meta blog). Meta's 0-shot HellaSwag 41.2 for Instruct is a harness artifact; independent base measurement is ~61.
- **Gemma 3 1B** — 2T tokens, trained with distillation from a larger teacher (Gemma 3 report); text-only, 32K context. Google reports no MMLU/GSM8K/code for the 1B *PT* model (STEM/code PT table starts at 4B) — beware third-party pages that paste the 4B column onto the 1B.
- **Qwen2.5-1.5B** — 18T tokens; the standout claim is base GSM8K 68.5 and MBPP 60.2, far above same-size peers; independent replications (Falcon card GSM8K 62.2, Hymba MMLU 60.9) broadly confirm.
- **Qwen3-1.7B** — 36T tokens (largest disclosed corpus in class); hybrid thinking/non-thinking modes. Instruct evals are only reported on newer suites (GPQA/AIME/LiveCodeBench etc.) in the tech report; classic-benchmark instruct numbers are not published. HumanEval/MBPP figures are EvalPlus variants.
- **TinyLlama 1.1B** — 3T tokens on a 1.1B model (extreme tokens-per-param for its time); MMLU/GSM8K essentially at chance.
- **OLMo 1B / OLMo 2 1B** — fully open (data, code, checkpoints). OLMo 2 1B (4T tokens) publishes only post-trained (Tülu-3-style SFT+DPO+RLVR) numbers on its card; GSM8K 68.3 for a 1B instruct is the headline. I could not verify base classic-benchmark scores from a primary source, so they're N/A.
- **Pythia** — research suite for interpretability (154 checkpoints, exact data ordering), not capability; ~0.3T tokens. All scores above are independent (TinyLlama paper, lm-eval).
- **phi-1.5** — "Textbooks Are All You Need II": mostly synthetic textbook-style data, only 150B tokens seen, yet HumanEval 34.1/GSM8K 40.2 — the original proof that data curation beats token count at this scale. No instruct tuning; benchmark-contamination debates surrounded this family.
- **SmolLM2-1.7B** — 11T tokens, fully documented data recipe (FineWeb-Edu, DCLM, Stack); strongest open-data model of late 2024 on commonsense, but math/code lag Qwen2.5.
- **SmolLM3** — **no 0.5–2B variant exists**; the family is 3B only (11T tokens, Jul 2025), so it's out of scope here.
- **StableLM 2 1.6B** — multilingual (7 languages); scores are Open-LLM-Leaderboard-style configs from its own tech report. Zephyr-tuned instruct: MMLU 41.8, GSM8K 34.8.
- **Falcon3-1B** — pruned + knowledge-distilled from a larger Falcon 3 with only 80GT of curated data; card is unusually honest, showing Qwen2.5-1.5B winning most rows.
- **MobileLLM** — deep-and-thin architecture study (weight sharing, grouped attention) at only 1T tokens; reports zero-shot commonsense only, no MMLU/math/code. Non-commercial license.
- **MobileLLM-R1-950M** — Sep 2025 reasoning-specialized edge model; ~2T high-quality tokens (<5T total) yet post-trained MATH500 74.0; Meta claims parity with Qwen3-0.6B at ~1/10th the tokens. Non-commercial license; benchmarks are the newer reasoning suites, classic ARC/HellaSwag not reported.
- **Gemma 2 2B** — 2.6B params, trained via knowledge distillation from a larger teacher; long the strongest sub-3B chat model (LMSYS Arena) despite weak GSM8K 23.9.
- **DeepSeek-R1-Distill-Qwen-1.5B** — SFT on 800K R1-generated reasoning samples over Qwen2.5-Math-1.5B; MATH-500 83.9 / AIME'24 28.9 (pass@1) — but no standard MMLU/GSM8K/HumanEval reported at all; it's a math/reasoning specialist.
- **MiniCPM-2B** — 2.4B non-embedding params, ~1.1T tokens, WSD LR schedule; self-reported HumanEval 50.0 and ARC-C 68.0 come from OpenBMB's own harness with per-model prompt templates — treat with caution; no independent replication found at these levels.
- **BitNet b1.58 2B4T** — native 1.58-bit (ternary) weights, 4T tokens; roughly Qwen2.5-1.5B-class scores at ~0.4GB memory. Unique architecture, MIT license.
- **LFM2-1.2B** — Liquid AI hybrid conv/attention edge model, 10T tokens; only aggregate-style self-reported numbers, no ARC/HellaSwag/code published.
- **Granite 4.0 H 1B** — hybrid Mamba2/attention (~1.5B, Oct 2025); self-reported HumanEval 73–74 pass@1 would be the best code score in the class by a wide margin — no independent verification found; training tokens undisclosed.
- **OpenELM-1.1B** — layer-wise scaling architecture; MMLU at chance (27) despite 1.8T tokens; released with full training framework.
- **Hymba-1.5B** — parallel attention+Mamba heads with learned meta-tokens; beat SmolLM2-1.7B on average with 7x fewer tokens (self-reported).
- **Also in this class (verified to exist; no scores pulled here):** Qwen3-0.6B, AMD-OLMo-1B, Zamba2-1.2B (Zyphra), H2O-Danube3, Fox-1-1.6B, Helium-1 2B (Kyutai), Index-1.9B, Salamandra-2B, EuroLLM-1.7B, Gemma 3n E2B, Granite 4.0 (non-H) 1B/350M.

## Cross-cutting observations

1. **Token counts exploded 100x in two years:** Pythia 0.3T (2023) → TinyLlama 3T (2024) → SmolLM2 11T → Qwen2.5 18T → Qwen3 36T (2025). Every 2024+ leader is trained far beyond Chinchilla-optimal.
2. **Distillation/pruning from bigger models is now the norm, not the exception:** Llama 3.2 1B, Gemma 2 2B, Gemma 3 1B, Falcon3-1B, DeepSeek-R1-Distill all derive from larger parents; the fully from-scratch open-data holdouts are OLMo, SmolLM2, Pythia, TinyLlama.
3. **GSM8K is the most gameable/divergent metric** in this class (1.4 → 76 across models of similar size), driven by math-heavy synthetic data and post-training; MMLU saturates around 60-63 at 1.5-2B as of 2025.
4. **Coding numbers are the least independently verified** — HumanEval/MBPP are almost always self-reported, and the two most impressive claims (Granite 1B HumanEval ~73, MiniCPM-2B 50) have no independent replication I could find.

## Sources

- https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct • https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
- https://ai.google.dev/gemma/docs/core/model_card_3 • https://huggingface.co/google/gemma-3-1b-it • https://huggingface.co/google/gemma-2-2b
- https://qwenlm.github.io/blog/qwen2.5-llm/ • https://qwenlm.github.io/blog/qwen3/ • https://arxiv.org/html/2505.09388v1 • https://huggingface.co/Qwen/Qwen2.5-1.5B • https://huggingface.co/Qwen/Qwen3-1.7B
- https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T • https://ar5iv.labs.arxiv.org/html/2401.02385
- https://huggingface.co/allenai/OLMo-1B • https://huggingface.co/allenai/OLMo-1B-hf • https://huggingface.co/allenai/OLMo-2-0425-1B
- https://huggingface.co/microsoft/phi-1_5 • https://ar5iv.labs.arxiv.org/html/2309.05463
- https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B • https://arxiv.org/html/2502.02737v1 • https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- https://huggingface.co/stabilityai/stablelm-2-1_6b • https://ar5iv.labs.arxiv.org/html/2402.17834
- https://huggingface.co/tiiuae/Falcon3-1B-Base • https://huggingface.co/blog/falcon3
- https://huggingface.co/facebook/MobileLLM-1B • https://huggingface.co/facebook/MobileLLM-R1-950M
- https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- https://arxiv.org/html/2404.06395v2 (MiniCPM) • https://huggingface.co/microsoft/bitnet-b1.58-2B-4T • https://huggingface.co/LiquidAI/LFM2-1.2B
- https://huggingface.co/ibm-granite/granite-4.0-h-1b • https://huggingface.co/blog/ibm-granite/granite-4-nano
- https://huggingface.co/apple/OpenELM-1_1B • https://arxiv.org/html/2411.13676v1 (Hymba)

**Verification caveats I hit:** OLMo 2 1B base-model classic-benchmark scores are not on the HF card or in reachable versions of the OLMo 2 paper (a search snippet suggested MMLU 44.3 / ARC-C 51.3 / HellaSwag 69.5 / GSM8K 43.8 but I could not trace it to a primary source, so I left them N/A). Granite 4.0 Nano base scores (MMLU 59.39, GSM8K 76.35, HumanEval 74) came via search summary of the HF blog, with the instruct card independently confirming the same range. A widely circulated "Gemma 3 1B MMLU 59.6 / HumanEval 36.0" is actually the 4B column — Google reports no such numbers for the 1B PT model.


# SECTION: 3B-class general models

All research complete. Here is the full deliverable.

---

# Survey: Public ~3B-Class Language Models (2B–4B), Reasoning & Coding Benchmarks

**Reading guide.** Scores are **self-reported by the model developer** unless marked *(indep)* = independent harness/leaderboard measurement. Shot configs in parentheses when disclosed. Configs vary wildly across vendors (0-shot vs 25-shot ARC-C, CoT vs non-CoT GSM8K, EvalPlus vs original HumanEval/MBPP) — **numbers in a column are not directly comparable across rows**. N/A = not reported in primary sources I could verify.

## Main table

| Model (variant scored) | Params | Train tokens | Released | License / weights | MMLU | GSM8K | ARC-C | HellaSwag | MATH | HumanEval | MBPP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Llama 3.2 3B** (instruct) | 3.21B | ~9T | Sep 2024 | Llama 3.2 Community (open weights) | 63.4 (5s); base 58.0 | 77.7 (8s CoT) | 78.6 (0s) | 69.8 (0s) | 48.0 (0s CoT) | N/A | N/A |
| **Qwen2.5-3B** (base) | 3.09B | up to 18T | Sep 2024 | Qwen Research License (open weights, non-commercial) | 65.6 (5s) | 79.1 (4s) | 56.5 (25s) | 74.6 (10s) | 42.6 (4s) | 42.1 (0s) | 57.1 (0s) |
| **Qwen2.5-3B-Instruct** | 3.09B | 〃 | Sep 2024 | 〃 | N/A | 86.7 | N/A | N/A | 65.9 | 74.4 (0s) | 72.7 (0s) |
| **Gemma 2 2B** (base) | 2.6B | 2T | Jul 2024 | Gemma terms (open weights) | 51.3 (5s) | 23.9 (5s) | 55.4 (25s) | 73.0 (10s) | 15.0 (4s) | 17.7 (p@1) | 29.6 (3s) |
| **Gemma 3 4B** (base/PT) | 4B | 4T | Mar 2025 | Gemma terms (open weights) | 59.6 (5s) | 38.4 (8s) | 56.2 (25s) | 77.2 (10s) | 24.2 (4s) | 36.0 (0s) | 46.0 (3s) |
| **Gemma 3 4B** (IT) | 4B | 4T | Mar 2025 | 〃 | 58.1 (0s) | 89.2 | N/A | N/A | 75.6 | 71.3 | 63.2 |
| **Phi-2** (base) | 2.7B | 1.4T | Dec 2023 | MIT | 56.7 (5s); 58.1 (5s, *indep*) | 61.1 (8s); 55.0 (5s, *indep*) | 61.0 (25s, *indep*) | 74.9 (10s, *indep*) | N/A | 53.7 (0s) | 59.1 (3s) |
| **Phi-3-mini-4k** (instruct) | 3.8B | 3.3T (orig) / 4.9T (Jun-24 refresh) | Apr 2024 | MIT | 70.9 (5s) | 85.7 (8s CoT) | 86.3 (10s) | 75.3 (5s) | N/A | 57.3 (0s) | 69.8 (3s) |
| **Phi-4-mini** (instruct) | 3.8B | 5T | Feb 2025 | MIT | 67.3 (5s) | 88.6 (8s CoT) | 83.7 (10s) | 69.1 (5s) | 64.0 (0s CoT) | N/A | N/A |
| **StableLM-3B-4E1T** (base) | 2.8B | 1T × 4 epochs (~4T exposures) | Oct 2023 | CC BY-SA 4.0 | 45.2 (5s, *indep*) | 3.3 (5s, *indep*) | 46.6 (25s, *indep*) | 75.9 (10s, *indep*) | N/A | N/A | N/A |
| **OpenELM-3B** (base) | 3.04B | ~1.8T | Apr 2024 | Apple AMLR (open weights + full recipe) | 26.8 (0s) | N/A | 42.2 (0s) | 73.3 (0s) | N/A | N/A | N/A |
| **Ministral 3B** (2024, instruct) | ~3B | undisclosed | Oct 2024 | Mistral Commercial — **no public weights** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **Ministral 3 3B** (2025; base/instruct/reasoning) | 3.4B LM + 0.4B vision | undisclosed | Dec 2025 | Apache 2.0 (open weights) | base 70.7 (5s) | N/A | N/A | N/A | base 60.1 (2s CoT); instr. 83.0 maj@1 | N/A | N/A |
| **Falcon 3 3B** (base) | 3B | ~0.1T (distilled) | Dec 2024 | TII Falcon-LLM 2.0 (open weights) | 55.5 (5s) | 63.9 (5s) | 54.9 (25s) | N/A | 9.4 (MATH-Lvl5, 4s) | N/A | N/A |
| **SmolLM3-3B** (base) | 3.08B | 11.2T | Jul 2025 | Apache 2.0 (open weights + full recipe) | 44.1 (MMLU-CF) | 67.6 (5s) | 65.6 | 76.2 | 46.1 (4s) | 30.5 (HumanEval+) | 52.9 (MBPP+) |
| **Qwen3-4B** (base) | 4.0B | ~36T (family) | Apr 2025 | Apache 2.0 (open weights) | 72.99 (5s) | 87.79 (4s CoT) | N/A | N/A | 54.1 (4s CoT) | 63.5 (EvalPlus avg) | 67.0 (0s) |
| **MiniCPM3-4B** (instruct) | 4B | undisclosed | Sep 2024 | Apache-2.0 repo / MiniCPM model terms | 67.2 | 81.1 | N/A | N/A | 46.6 | 68.3 (HumanEval+) | 63.2 (MBPP+) |
| **EXAONE 3.5 2.4B** (instruct) | 2.4B (2.14B non-emb) | 6.5T | Dec 2024 | EXAONE 1.1 NC (open weights, non-commercial) | 60.4 (CoT) | 82.5 (CoT) | 79.2 | N/A | 60.2 (CoT) | 76.2 (EvalPlus base) | 74.3 (EvalPlus base) |
| **Granite 4.0 Micro** (instruct) | 3B | undisclosed | Oct 2025 | Apache 2.0 (open weights) | 66.0 (5s) | 85.5 (8s) | N/A | N/A | N/A | 80 (p@1) | 72 (p@1) |
| **Minitron-4B-Base** (NVIDIA) | 4B | 94B (post-prune) | Jul 2024 | NVIDIA Open Model License | 58.6 (5s) | 24.1 (0s) | 50.9 (0s) | 75.0 (0s) | N/A | 23.3 (0s) | N/A |
| **Zamba2-2.7B** (base) | 2.7B | 3T + 0.1T anneal | 2024 (suite report Nov 2024) | Apache 2.0 (open weights) | 56.0 (5s) | N/A | 60.0 (25s) | 76.4 (10s) | N/A | N/A | N/A |
| **Hunyuan-4B** (pretrain/instruct) | 4B | undisclosed | Jul 2025 | Tencent Hunyuan Community | 74.0 | 87.5 | N/A | N/A | 72.3 | N/A | 76.5 |
| **LFM2-2.6B** | 2.57B | 10T | 2025 (report Nov 2025) | LFM Open License v1.0 | 64.4 | 82.4 | N/A | N/A | N/A | N/A | N/A |
| **Gemma 3n E4B** (IT) | 4B effective (8B raw) | ~11T | Jun 2025 | Gemma terms (open weights) | 64.9 (0s) | N/A | 61.6 (25s) | 78.6 (10s) | N/A | 75.0 (0s) | 63.6 (3s) |
| *MobileLLM-1.5B* (below range, for context) | 1.5B | 1T | 2024 (ICML) | FAIR NC research | N/A | N/A | 40.9 (0s) | 64.5 (0s) | N/A | N/A | N/A |

## Per-model notes (unusual things, flags)

- **Llama 3.2 3B** — pruned from Llama 3.1 8B and trained with **logit-level knowledge distillation** from 8B/70B teachers; ~9T tokens (~150× Chinchilla). Meta's ARC-C 78.6 is a bespoke 0-shot config — not comparable to 25-shot harness ARC. No official HumanEval/MBPP for the small 3.2 models.
- **Qwen2.5-3B** — 18T tokens (~290× Chinchilla). **Oddity: the 3B is the one Qwen2.5 size under a research-only (non-commercial) license** while siblings are Apache 2.0. Huge base→instruct jump on code (42.1→74.4 HumanEval). Qwen2.5-Coder-3B exists for code-specialist use.
- **Gemma 2 2B** — trained by **distillation from a larger Gemma teacher**; only 2T tokens; weakest math/code of the modern cohort (GSM8K 23.9). Actual param count ~2.6B.
- **Gemma 3 4B** — report states "**all Gemma 3 models are trained with knowledge distillation**" (256 sampled logits/token); note the enormous PT→IT delta (GSM8K 38.4→89.2, MATH 24.2→75.6) — post-training does most of the visible work. Multimodal (vision) included.
- **Phi-2** — "textbook-quality" synthetic + curated web data, base-only, no alignment. **Discrepancy flag:** self-reported GSM8K 61.1 (8-shot) vs Open LLM Leaderboard 55.0 (5-shot); MMLU independent 58.1 slightly *above* self-reported 56.7. The phi series has faced persistent community skepticism about benchmark-adjacent synthetic curricula; independent numbers here broadly hold up, with config-driven gaps.
- **Phi-3-mini (3.8B)** — instruct-only release, MIT. Very strong self-reported scores on bespoke configs (ARC-C 86.3 @ 10-shot); the June-2024 weight refresh changed the model under the same name (3.3T→4.9T tokens) — dated citations matter. Phi-3.5-mini (Aug 2024, 3.8B) is the successor in the same line.
- **Phi-4-mini** — 5T tokens, heavy synthetic-data recipe, 200K vocab. HumanEval/MBPP exist in the tech report but weren't on the model card I verified — left N/A rather than guessed.
- **StableLM-3B-4E1T** — deliberately trained **4 epochs over the same 1T tokens** (the "4E1T" name). Scores above are Open LLM Leaderboard (independent). GSM8K 3.3 — effectively no math. Base only; Stable LM Zephyr 3B is the DPO-tuned derivative.
- **OpenELM-3B** — fully open recipe (data, code, logs), but **MMLU 26.8 ≈ random chance** even self-reported (0-shot, lm-eval-harness); by far the weakest 3B-class model on knowledge/reasoning. A cautionary data point, not a competitive model.
- **Ministral 3B (Oct 2024)** — **weights were never released** (commercial API only; only the 8B sibling got research weights). Launch post claims it beats Gemma 2 2B / Llama 3.2 3B via bar charts, but publishes **no per-benchmark table** — honest N/A across the board.
- **Ministral 3 3B (Dec 2025)** — distinct model, part of the Mistral 3 family; Apache 2.0 with base/instruct/**reasoning** variants, all with vision (3.4B LM + 0.4B encoder — effectively ~4B total). Built via "Cascade Distillation" (iterative pruning + distillation). Reasoning variant: AIME24 77.5, AIME25 72.1, LiveCodeBench 54.8 (self-reported). Artificial Analysis independently rates it above the median for open non-reasoning models of its size (Intelligence Index v4.1 = 7 vs median 3).
- **Falcon 3 3B** — pruned + distilled from Falcon3-7B on **only ~100B tokens** (the 7B parent saw 14T) — an extreme low-token outlier that still posts GSM8K 63.9. No HellaSwag/HumanEval/MBPP on the card.
- **SmolLM3-3B** — most transparent modern 3B: full training configs + data public; 11.2T tokens (~180× Chinchilla); dual think/no-think modes. **Flag:** uses MMLU-CF (contamination-free variant), so its 44.1 cannot be compared to others' plain MMLU; instruct+thinking reaches AIME25 36.7.
- **OLMo 2 (Ai2)** — **no variant in the 2–4B window**: family is 1B / 7B / 13B / 32B (OLMo-2-0425-1B is ~1.5B, below range). Fully open data (Dolma/Tulu) makes the 1B the nearest fully-documented neighbor; OLMo 3 (Nov 2025) is 7B/32B, also out of range.
- **MobileLLM (Meta)** — largest variant is **1.5B — the line never reached 2B**; zero-shot commonsense only (no MMLU/GSM8K reported), FAIR non-commercial. Successor MobileLLM-R1-950M (Sep 2025, ~2T tokens) is the reasoning-focused follow-up, still sub-1B.
- **Qwen3-4B** — 36T family tokens (~450× Chinchilla for a 4B). Base numbers are the strongest in this survey (MMLU 72.99). The **Qwen3-4B-Instruct-2507** refresh (non-thinking) self-reports MMLU-Redux 84.2, AIME25 47.4, MultiPL-E 76.8, LiveCodeBench v6 35.1; Qwen claims it "rivals Qwen2.5-72B-Instruct." Same-name-different-weights caveat applies (April vs July 2025 checkpoints).
- **MiniCPM3-4B** — claims to surpass GPT-3.5-Turbo-0125; code scores are EvalPlus (+) variants, not directly comparable to vanilla HumanEval/MBPP; training tokens undisclosed.
- **EXAONE 3.5 2.4B** — 6.5T tokens; excellent instruct math/code for its size (HumanEval 76.2), but **non-commercial license**; strong Korean/English bilingual focus.
- **Granite 4.0 Micro** — IBM enterprise line, Apache 2.0, 128K context; HumanEval 80 self-reported is among the best in class — no independent confirmation found; training token count not disclosed at the model level.
- **Minitron-4B** — NVIDIA's pruning+distillation study artifact (from Nemotron-4 15B, only 94B post-prune tokens, "40× fewer tokens"); Nemotron-Mini-4B-Instruct is its aligned derivative.
- **Zamba2-2.7B** — hybrid **Mamba2/shared-attention** architecture (non-standard transformer); competitive commonsense/MMLU with strong latency claims; no math/code numbers published.
- **Hunyuan-4B** — hybrid fast/slow thinking, 256K context. **Flag:** the model card's table labels MMLU 74.0 / GSM8K 87.5 / MATH 72.3 as "pretrain" scores, which would be extraordinary for a 4B base — likely CoT/instruct-style eval; treat with caution. AIME24 78.3 is with thinking mode.
- **LFM2-2.6B** — Liquid AI hybrid (convolution + attention) edge model, 10T tokens, "dynamic hybrid reasoning"; custom LFM Open License; no ARC/HellaSwag/code numbers published.
- **Gemma 3n E4B** — "effective 4B" via selective parameter activation over 8B raw weights — param count is not apples-to-apples with dense 4B models; multimodal (text/image/audio/video).
- **Honorable mentions** (in-range, not tabled): Phi-3.5-mini (3.8B, Aug 2024), Qwen2.5-Coder-3B, StableLM Zephyr 3B, H2O Danube3-4B (Jul 2024), Salamandra-2B (BSC, fully open EU model), Microsoft **BitNet b1.58 2B4T** (Apr 2025 — native 1.58-bit 2B trained on 4T tokens, MIT; notable architecture first), Nemotron-Mini-4B-Instruct.

## Cross-cutting observations

1. **Token counts have left Chinchilla far behind**: the 2024–2025 cohort trains 3B models on 9–36T tokens (150–450× Chinchilla-optimal); the correlation with benchmark position is strong (Qwen3-4B and SmolLM3 vs StableLM/OpenELM at ≤2T-effective).
2. **Distillation is now the norm, not the exception**: Llama 3.2, Gemma 2/3, Falcon 3 3B, Minitron, and Ministral 3 are all explicitly pruned/distilled from larger parents — of the modern cohort, only Qwen, SmolLM3, and the fully-open research models are clearly trained from scratch at size.
3. **Self-reported configs are the main comparability hazard**: ARC-C spans 0-shot to 25-shot (a ~20-point swing), GSM8K 4/5/8-shot ± CoT, and EvalPlus vs original HumanEval/MBPP. The only truly independent numbers in this survey are the Open LLM Leaderboard rows (Phi-2, StableLM) and Artificial Analysis's index for Ministral 3 3B.

## Sources

- https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- https://qwenlm.github.io/blog/qwen2.5-llm/
- https://huggingface.co/google/gemma-2-2b
- https://huggingface.co/google/gemma-3-4b-it and https://arxiv.org/html/2503.19786 (Gemma 3 report)
- https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/
- https://huggingface.co/datasets/open-llm-leaderboard-old/details_microsoft__phi-2 (independent Phi-2)
- https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- https://huggingface.co/microsoft/Phi-4-mini-instruct
- https://huggingface.co/stabilityai/stablelm-3b-4e1t
- https://huggingface.co/apple/OpenELM-3B
- https://mistral.ai/news/ministraux (Ministral 3B, 2024)
- https://mistral.ai/news/mistral-3/ ; https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512 ; https://arxiv.org/abs/2601.08584 (Ministral 3)
- https://artificialanalysis.ai/models/ministral-3-3b (independent)
- https://huggingface.co/tiiuae/Falcon3-3B-Base ; https://huggingface.co/blog/falcon3
- https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- https://qwenlm.github.io/blog/qwen3/ ; https://arxiv.org/html/2505.09388 (Qwen3 report) ; https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- https://huggingface.co/openbmb/MiniCPM3-4B
- https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct ; https://arxiv.org/html/2412.04862v1 (EXAONE 3.5 report)
- https://huggingface.co/ibm-granite/granite-4.0-micro
- https://huggingface.co/nvidia/Minitron-4B-Base
- https://huggingface.co/Zyphra/Zamba2-2.7B ; https://www.zyphra.com/post/zamba2-small ; https://arxiv.org/pdf/2411.15242
- https://huggingface.co/tencent/Hunyuan-4B-Instruct
- https://huggingface.co/LiquidAI/LFM2-2.6B
- https://huggingface.co/facebook/MobileLLM-1.5B
- https://huggingface.co/google/gemma-3n-E4B-it
- https://allenai.org/olmo2 (OLMo 2 size lineup)


# SECTION: Code-specialist small models

All five research agents have reported. Here is the complete deliverable, synthesized and cross-checked across agents (all figures traced to primary sources; harness caveats and N/As preserved).

---

# Survey: Public Small (0.5B–4B) Code-Specialist LMs
### Comparison class for a from-scratch 1.2B model, 21B tokens, 66% code, FIM objective

**Reading guide.** "HE" = HumanEval pass@1. Scores are **self-reported** from the model's paper/card unless marked **[EP]** = EvalPlus leaderboard (independent, greedy, HumanEval/HumanEval+) or **[BC]** = BigCode Models Leaderboard (independent). Self-reported and EvalPlus numbers routinely differ by 2–6 pts at this scale (e.g., DeepSeek-1.3B-base 34.8 self vs 28.7 EP) — never mix regimes in one comparison. MBPP variants also differ per paper (full few-shot vs sanitized vs EvalPlus harness); the notes say which.

## Master table

| Model | Params | Released | License | Train tokens | Code frac | FIM | HE pass@1 (base) | HE (instruct) | MBPP | Infilling |
|---|---|---|---|---|---|---|---|---|---|---|
| InCoder-1.3B | 1.3B | Apr 2022 | CC-BY-NC-4.0 | 52B (1 epoch) | ~100% code+StackOverflow | Causal-masking (FIM precursor) | 8–8.9; 12.2/11.0 [EP] | — | 10.9 | N/A at 1.3B (6.7B: 56.3 EM single-line) |
| SantaCoder | 1.1B | Dec 2022 | BigCode OpenRAIL-M | 236B | ~100% code (Py/Java/JS) | Yes, rate 0.5, PSM+SPM | 18 (MultiPL-E harness); 14.6/14.0 [EP] | — | 35 (MultiPL-E MBPP, Py) | Single-line EM: Java .62 / JS .60 / Py .44 |
| Replit-code-v1-3b | 2.7B | Apr 2023 | CC BY-SA-4.0 | 525B (175B×3) | ~100% code/dev | **No** | 21.9; 20.1 [BC] | — | N/A | N/A |
| CodeT5+ 770M / 2B | 0.77B / ~2B | May 2023 | BSD-3-Clause | 51.5B (stage 1) | ~100% code | Span denoising (enc-dec), not FIM | 15.5 / 24.2 (Py-adapted); 2B: 25.0/22.0 [EP] | — | 2B: 48.4/38.1 [EP] | N/A |
| phi-1 | 1.3B | Jun 2023 (weights Sep) | MIT | ~7B unique, ~54B seen | 100% code/code-teaching (Python) | No | **50.6** | — | 55.5 | N/A |
| StarCoderBase-1B | 1.14B | Jul 2023 | BigCode OpenRAIL-M | 1T | ~100% code+GH | Yes, 0.5, PSM+SPMv2 | 15.17; 14.6/12.2 [EP] | — | N/A | N/A |
| StarCoderBase-3B | ~3.0B | Jul 2023 | BigCode OpenRAIL-M | 1T | ~100% code+GH | Yes, 0.5 | 21.46 (card) / 21.3, HE+ 17.1 (SC2 paper); 17.7/15.9 [EP] | — | 42.6 / MBPP+ 35.8 | RepoBench Py EM 30.0; CrossCodeEval Py ES 69.5 |
| DeciCoder-1B | 1.11B | Aug 2023 | Apache-2.0 | 446B | 100% code (3 langs) | Yes (rate n/d) | 19.1 | — | N/A | N/A (no published) |
| Refact-1.6B | 1.59B | Sep 2023 | BigScience OpenRAIL-M | 1.2T pretrain (50:50 text:code) + 40B FT | ~50% (pretrain) | Yes (rate n/d) | 32.0 (38.4 chat fmt); 31.1 [BC] | — | N/A | N/A (no published) |
| Replit-code-v1.5-3b | 3.3B | Oct 2023 | Apache-2.0 | 1T (~200B×5) | mostly code + MD/SE | **No** | 27.4; 23.0 (indep, Stability card) | — | N/A | N/A |
| DeepSeek-Coder-1.3B | 1.3B | Nov 2023 | DeepSeek License v1.0 | **2T** | **87% code / 10% code-NL / 3% zh** | Yes, 0.5, PSM | 34.8; 28.7/25.6 [EP] | 65.2; 65.9/60.4 [EP] | 46.2 (500-prob few-shot); 56.9/47.9 [EP] | Single-line EM mean **70.4** (Py 57.4 / Java 82.2 / JS 71.7) |
| Stable Code 3B | 2.8B | Jan 2024 | Stability Community License | 1.3T (on 4T-NL base) | ~80% of the 1.3T | Yes, 0.5, SPM/PSM | 32.4 (Py, MultiPL-E); 29.3/25.6 [EP] | instruct variant Mar 2024 | 54.8/45.8 [EP] | FIM eval: Py 59.1 / JS 73.4 / Java 64.1 |
| StarCoder2-3B | 3.03B | Feb 2024 | BigCode OpenRAIL-M | **3.1T** (4.98 ep × 622B unique) | ~94% code-ish | Yes (repo-context FIM, 50%×50%) | **31.7 / HE+ 27.4** (= [EP]) | — | 57.4 / MBPP+ 47.4 | Single-line EM: Java 75.0 / JS 73.0 / Py 59.1; RepoBench Py EM 32.5 |
| CodeGemma-2B (v1.0/v1.1) | ~2B (exact n/d) | Apr / May 2024 | Gemma ToU | Gemma base + 500B (v1.1: 1T) | 100% (of added tokens) | Yes, **80% rate** (90% v1.1), 50/50 PSM/SPM | 31.1 (v1.1 37.8); 26.8/20.7 [EP] | no 2B instruct | 43.6 (v1.1 49.2); 55.6/46.6 [EP] | HE-Infill single-line **78.4**, multi-line 51.4 |
| Granite-3b-code | 3.48B | May 2024 | Apache-2.0 | 4T code + 500B (80/20) | ~100% then 80% | Yes, α=0.5 | 36.6 (HumanEvalSynthesize-Py) | 51.2 | 36.0 / MBPP+ 45.1 (as published) | SantaCoder-FIM EM: Java 79.7 / JS 71.6 / Py 61.8 |
| Yi-Coder-1.5B | 1.48B | Sep 2024 | Apache-2.0 | 2.4T (continued) | n/d | n/d | 41.5 (Py; 9-lang avg 33.6) | 67.7 chat (Py) | N/A (9B only) | N/A |
| Arctic-SnowCoder-1.3B | 1.3B | Sep 2024 (**paper only — no public weights found**) | N/A | **555B** | ~100% code | n/d | HE+ 28.0 (raw HE n/d) | — | MBPP+ 42.9 | N/A |
| Qwen2.5-Coder-0.5B | 0.49B | Nov 2024 | Apache-2.0 | 5.5T (70:20:10 code:text:math) | ~70% | Yes (file+repo level; rate n/d) | 28.0 / HE+ 23.8 | 61.6 / 57.3 | 52.9 / MBPP+ 47.1 | HE-FIM avg 77.7; CCEval EM 24.6 |
| Qwen2.5-Coder-1.5B | 1.54B | Sep 2024 | Apache-2.0 | 5.5T | ~70% | Yes | **43.9 / HE+ 36.6** | 70.7 / 66.5 | 69.2 / 58.6 | HE-FIM avg 83.5; CCEval EM 40.2; RepoEval avg 40.5 |
| Qwen2.5-Coder-3B | 3.09B | Nov 2024 | **Qwen Research License** | 5.5T | ~70% | Yes | **52.4 / HE+ 42.7** | **84.1 / 80.5** | 72.2 / 61.4 | HE-FIM avg **85.7**; CCEval EM 44.9; RepoEval avg 44.0 |
| OpenCoder-1.5B | 1.9B actual | Nov 2024 | INF license (custom) | 2T + 100B anneal (RefineCode ~960B unique) | ~90% | **No** (dev-confirmed) | 54.3 / HE+ 49.4 | 72.5 / 67.7 | 70.6 / 58.7 | N/A (no FIM) |
| Mellum-4B (JetBrains) | 4.02B | Apr 2025 | Apache-2.0 | ~4.2T | overwhelmingly code | Yes, 50% of files, SPM order | N/A (completion-only model) | — | N/A | **SAFIM 38.1** (card) / 52.2 (paper cfg); HE-Infill single-line 66.2; RepoBench Py EM 28.0 |

**Qwen3-Coder: no ≤4B variant exists** (verified against the QwenLM repo). Smallest is Qwen3-Coder-30B-A3B ("Flash" is this model's marketing name), a 30.5B-total MoE with ~3.3B active. Same borderline MoE category: DeepSeek-Coder-V2-Lite (16B/2.4B active), Ling-Coder-Lite (16.8B/2.75B active). The 2025–26 trend moved small-code to MoE-with-small-active rather than small-dense; Mellum-4B is the notable dense exception.

## Per-model notes (key details and gotchas)

- **DeepSeek-Coder-1.3B** (arXiv 2401.14196): the exact corpus disclosure is 87% source code / 10% English code-related NL (markdown, StackExchange) / 3% Chinese NL. FIM ablated explicitly; they chose 50% PSM. Instruct = base + 2B instruction tokens. Caveat: the paper's Table 6 infilling baselines contain a transcription inconsistency (its SantaCoder row's mean doesn't match its cells or SantaCoder's own paper) — trust each paper's own infilling numbers.
- **phi-1** (2306.11644): the outlier data-efficiency point — 50.6 HE from ~7B unique curated/synthetic tokens (≈8 epochs, ~54B seen) + a small CodeExercises finetune. phi-1-base (no finetune) = 29. Python-only, no FIM, no MultiPL-E, not on EvalPlus. The paper includes its own contamination-pruning study (scores drop but stay above StarCoder).
- **SantaCoder** (2301.03988): closest historical analog to a small-budget from-scratch run (236B tokens, 1.1B params, FIM 0.5). Its self-reported "18" is MultiPL-E-harness; EvalPlus says 14.6. Its final 236B run did not improve FIM metrics over 118B-token ablations.
- **StarCoderBase-1B/3B**: fixed 1T-token size ladder (1B=15.17, 3B=21.46, 7B=28.37, 15B=30.4) — useful as a params-axis at constant tokens. Dataset ~250B unique → ~4 epochs. MBPP for the 1B was never published.
- **StarCoder2-3B** (2402.19173): the best-documented 3B — exact token accounting (4.98 epochs × 622.09B unique = 3.1T; 3B variant excludes Arxiv/Wikipedia/OpenWebMath), per-language single-line FIM, RepoBench, CrossCodeEval all in the paper, and its paper numbers match EvalPlus exactly. Paper notes its 3B matches StarCoderBase-15B on FIM.
- **CodeGemma-2B**: pure completion specialist (no 2B instruct). Highest disclosed FIM rate of the class (80–90%). Its 78.4 single-line infill was SOTA-for-size at release. BabelCode multilingual numbers exist in the paper but my agent flagged its extraction of that one table as lower-confidence.
- **Qwen2.5-Coder small trio** (2409.12186): strongest self-reported scores in the class at every size; 5.2T file-level (70:20:10) + ~300B repo-level. FIM rate undisclosed. None of the ≤3B variants appear on EvalPlus — all numbers are self-reported. 3B's license (Qwen Research) is materially more restrictive than its Apache-2.0 siblings. SAFIM appears only as a figure (no per-model table).
- **OpenCoder-1.5B**: strong scores, ~90%-code corpus, but confirmed no FIM capability — not a completion-market model. "1.5B" is actually 1.9B with embeddings.
- **Granite-3b-code**: multilingual numbers are HumanEvalSynthesize, not MultiPL-E. Published MBPP+ (45.1) > MBPP (36.0) is as-published (different task counts in early MBPP+).
- **Replit v1/v1.5**: no FIM in either; v1.5's numbers exist only as blog-post images, and the only independent measurement (Stability's card) reads ~4 pts lower than self-reported.
- **Arctic-SnowCoder-1.3B**: high-value paper (quality-over-quantity at 1.3B) but apparently weights were never released — if your criterion is "public model," it belongs in the learning-curve section, not the model table.
- **Mellum-4B**: the only 2025 dense small code model of note; FIM/completion-only, so it's the best SAFIM/RepoBench comparison target but offers no HumanEval. Its paper-vs-card scores differ substantially by eval config (SAFIM 38.1 card vs 52.2 paper) — cite the config. Successor (Mellum2, Jun 2026) is a 12B MoE, out of range.
- **Also noted**: NT-Java-1.1B (Jul 2024, StarCoderBase-1B + 22B Java tokens — a continued-pretrain, and incidentally an existence proof that a ~22B-token code budget on a 1B model is a published recipe); StableCode-Completion-Alpha-3B (Aug 2023, 300B+200B tokens, HE 20.2, Apache-2.0) is a distinct lineage from Stable Code 3B. Excluded as out-of-range or fine-tunes: Seed-Coder-8B, aiXcoder-7B, CodeGeeX4-9B, Codestral Mamba 7.3B, Zed Zeta (7B fine-tune), DeepCoder-1.5B (RL fine-tune of a distill).

## Token-normalized context: HumanEval vs training tokens

This is the critical frame for a 21B-token model: the table above spans **446B–5.5T tokens, i.e., 21x–262x your budget**. Published learning-curve points:

**phi-1 paper (2306.11644) — the only explicit (tokens × data-quality × params) grid at this scale.** Figure 2.1, approximate (read from figure, unlabeled bars):
| Params | Data | Tokens seen | HE (approx) |
|---|---|---|---|
| 350M | The Stack+ (raw) | 26B | ~8% |
| 350M | The Stack+ (raw) | 76B | ~12% |
| 350M | CodeTextbook (curated) | 26B | ~17% |
| 350M | CodeTextbook + Exercises | 26B | ~20% |
| 1.3B | CodeTextbook | 51B | ~29% (phi-1-base, exact) |
| 1.3B | + CodeExercises FT | ~54B | 50.6% (exact) |

**Arctic-SnowCoder-1.3B (2409.02326) — exact phase-wise points (HumanEval+, not raw HE):** 500B raw code → HE+ 14.0; +50B model-filtered HQ (12.5B unique ×4) → 21.3; +5B synthetic → 28.0. Also: 4 epochs over 12.5B unique tokens was the optimal repetition count. Headline: 555B tokens matched StarCoder2-3B's 3.3T on HE+.

**CodeGen (2203.13474) — staged progression:** 350M model: HE 2.12 (Pile) → 6.67 (+multi-lang code) → 12.76 (+Python mono, ~570B cumulative); 2.7B: 6.70 → 14.51 → 23.70.

**Other exact anchors:** phi-1.5 1.3B @ 150B tokens (synthetic-heavy): HE 34.1. StarCoderBase ladder @ 1T: 15.17 (1B) / 21.46 (3B). DeepSeek-1.3B @ 2T: 34.8. SmolLM2-1.7B: code composite 8.87 @ 6T → 23.21 @ 11T, with most gain from late data-mix changes (Stack-Edu), not raw tokens. CrystalCoder-7B: HE 23.9 after 1.27T (phases rising smoothly; 143 public checkpoints with per-checkpoint HE in LLM360's Analysis360 if you want raw curves).

**Figure-only (honest N/A for extractable numbers):** StarCoder Fig 2 (pass@1 every 200B to 1T — high-resource langs still improving, ≲1GB langs plateau/decline); DeepSeek-Coder Fig 7 ("benchmark curves during training" — exactly the desired data, unreadable in HTML); OpenCoder Fig 1 (RefineCode vs Stack-v2 at 1.5B to 600B tokens); SantaCoder tracked only pass@100 during training. "Scaling Data-Constrained LMs" has no code benchmarks at all.

**Implication for the 1.2B/21B run:** on raw-code data, no published ~1B model reaches >10% HumanEval by ~21–26B tokens (phi-1's Stack-ablation ~8% @ 26B/350M; CodeGen-350M needed ~570B cumulative for 12.8). The only published curves above that at your token budget are curated/synthetic-data ones (~17–20% @ 26B). The fair published comparisons at your budget are the phi-1 26B-token ablation bars and SantaCoder's 118B ablations — not the headline rows of the master table, which sit 1–2 orders of magnitude to the right in tokens. Repetition to ~4 epochs is consistently reported as near-free (Arctic-SnowCoder, OpenCoder, phi-1 ~8, Replit 3).

## Sources

Papers: [DeepSeek-Coder 2401.14196](https://arxiv.org/abs/2401.14196) · [phi-1 2306.11644](https://arxiv.org/abs/2306.11644) · [phi-1.5 2309.05463](https://arxiv.org/abs/2309.05463) · [SantaCoder 2301.03988](https://arxiv.org/abs/2301.03988) · [InCoder 2204.05999](https://arxiv.org/abs/2204.05999) · [CodeT5+ 2305.07922](https://arxiv.org/abs/2305.07922) · [StarCoder 2305.06161](https://arxiv.org/abs/2305.06161) · [StarCoder2 2402.19173](https://arxiv.org/abs/2402.19173) · [CodeGemma 2406.11409](https://arxiv.org/abs/2406.11409) · [Qwen2.5-Coder 2409.12186](https://arxiv.org/abs/2409.12186) · [Stable Code 2404.01226](https://arxiv.org/abs/2404.01226) · [OpenCoder 2411.04905](https://arxiv.org/abs/2411.04905) · [Granite Code 2405.04324](https://arxiv.org/abs/2405.04324) · [Arctic-SnowCoder 2409.02326](https://arxiv.org/abs/2409.02326) · [Mellum 2510.05788](https://arxiv.org/abs/2510.05788) · [CodeGen 2203.13474](https://arxiv.org/abs/2203.13474) · [SAFIM 2403.04814](https://arxiv.org/abs/2403.04814) · [SmolLM2 2502.02737](https://arxiv.org/abs/2502.02737) · [Crystal 2411.04156](https://arxiv.org/abs/2411.04156)

Model cards / blogs: [deepseek-coder-1.3b-base](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) · [microsoft/phi-1](https://huggingface.co/microsoft/phi-1) · [bigcode/santacoder](https://huggingface.co/bigcode/santacoder) · [starcoderbase-1b](https://huggingface.co/bigcode/starcoderbase-1b) / [-3b](https://huggingface.co/bigcode/starcoderbase-3b) · [starcoder2-3b](https://huggingface.co/bigcode/starcoder2-3b) · [google/codegemma-2b](https://huggingface.co/google/codegemma-2b) / [1.1-2b](https://huggingface.co/google/codegemma-1.1-2b) · [Qwen2.5-Coder family blog](https://qwenlm.github.io/blog/qwen2.5-coder-family/) + HF cards ([0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B)/[1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B)/[3B](https://huggingface.co/Qwen/Qwen2.5-Coder-3B)) · [Qwen3-Coder repo](https://github.com/QwenLM/Qwen3-Coder) · [stable-code-3b](https://huggingface.co/stabilityai/stable-code-3b) · [OpenCoder-1.5B-Base](https://huggingface.co/infly/OpenCoder-1.5B-Base) · [Yi-Coder](https://github.com/01-ai/Yi-Coder) · [granite-3b-code-base](https://huggingface.co/ibm-granite/granite-3b-code-base-2k) · [Mellum-4b-base](https://huggingface.co/JetBrains/Mellum-4b-base) + [JetBrains blog](https://blog.jetbrains.com/ai/2025/04/mellum-goes-open-source-a-purpose-built-llm-for-developers-now-on-hugging-face/) · [replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b) / [v1_5-3b](https://huggingface.co/replit/replit-code-v1_5-3b) · [Refact-1_6B-fim](https://huggingface.co/smallcloudai/Refact-1_6B-fim) · [DeciCoder-1b](https://huggingface.co/Deci/DeciCoder-1b) · [codet5p-2b](https://huggingface.co/Salesforce/codet5p-2b)

Independent leaderboards: [EvalPlus](https://evalplus.github.io/leaderboard.html) ([raw data](https://raw.githubusercontent.com/evalplus/evalplus.github.io/main/results.json)) · [BigCode Models Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard) ([CSV](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard/raw/main/data/code_eval_board.csv))
