# Corpus v3 candidate sources (for the 3B and beyond)

Compiled 2026-08-06 from two research sweeps (code; general/math/reasoning), each
constraint-aware: every candidate scored on license/attribution mechanics (the Stack-v1
manifest requirement generalizes) and on provenance under the house no-distillation rule
(LLM-generated text flagged rule-conflicted; LLM-*judged* natural text is compatible).

## The combined shortlist

Code: Stack-Edu (125B, best ablation evidence, SWHID manifest-compatible) · Stack v3
(4.9T, inline content, ODC-By, needs our own ablation) · CommitPack full (~1T est.,
solves commit exhaustion) · Common Pile stackv2_edu (bulletproof licensing) ·
starcoder2data-extras (issues/kaggle/docs/SO).

General/math: FineWeb-Edu slice (replaces generic FineWeb) · FineMath-3+/4+ (strict
upgrade over open-web-math) · DCLM top slice (blend with Edu; the mix beats either) ·
StackExchange 27.8B via TxT360 (pre-2024 licensing) · peS2o v2 + Dolma 3 science PDFs
(retire the homegrown arXiv pipeline).

Local curation stack (all weights downloadable, no APIs): DCLM/Ultra-FineWeb fastText
first pass → FineWeb-Edu + FineMath classifiers second pass → EAI-taxonomy labels for
facet control. This is the phi-style lever, run on our own terms.

Rule-conflicted (flagged, not adopted): SwallowCode v1/v2, Nemotron synthetic partitions,
MegaMath-Synth/Web-Pro, Cosmopedia, UltraData-Math L3. Gray: Nemotron-CC-Math
(natural content, LLM-normalized formatting) — decision deferred.

Operational notes: Stack v2 terms require tracking opt-out updates (frozen snapshots
drift out of compliance; v3 patch-releases instead). StackExchange post-Jul-2024 dumps
carry a no-LLM rider — use pre-change archives or TxT360/Common Pile packagings.

---



# SECTION: Code data sources

Research complete. Here is the full survey.

---

# Code pretraining data survey for the microlab (21B → 60B+ tokens, 3B model)

Context recap for framing: current corpus = The Stack v1 dedup + CommitPackFT; requirements = per-file attribution manifest feasibility, and no LLM-synthetic capability sources (must be flagged, not silently recommended). "Tokens (est.)" marks my estimate at ~4 bytes/token; everything else is a published figure.

## 1. The Stack family (base corpora)

**The Stack v2** — https://huggingface.co/datasets/bigcode/the-stack-v2
- Sizes: full 67.5 TB / 3.28B files / 658 languages; dedup 32.1 TB (~40% of permissive files were near-dupes); `train-smol-ids` ~525B unique tokens (17 languages, heuristic-filtered); `train-full-ids` ~775B+ unique tokens (600+ languages). The often-quoted "~900B" includes the wider StarCoder2 training mix. Sources conflict slightly between the card and paper (https://arxiv.org/abs/2402.19173); the per-variant numbers above are the paper's.
- vs v1: built by traversing the Software Heritage 2023-09-06 graph (104.2M GitHub repos) instead of a GitHub crawl; repo-level license detection with only Blue-Oak-Council-permissive licenses retained; ~10x larger; train variants ship **SWHIDs + metadata only, no content**.
- Access mechanics: content is fetched per-blob from the public Software Heritage S3 bucket (AWS credentials + `smart_open[s3]`); bulk download formally requires an agreement with Software Heritage/INRIA.
- Attribution/terms: must abide by original per-file licenses including attribution clauses (provenance fields shipped per file — exactly what the lab's manifest needs), acknowledge SWH+INRIA, follow SWH's "principles for LLM training," and **keep your copy updated to the latest version to honor opt-outs**. That last clause is an operational obligation the lab should note: a frozen snapshot technically drifts out of compliance as removals land (updates were still being enacted as of July 2026).
- Opt-out: active, via https://github.com/bigcode-project/opt-out-v2 and the "Am I in The Stack" space (https://huggingface.co/spaces/HuggingFaceCode/in-the-stack).
- Provenance: natural.

**The Stack v3 (Aug 2025)** — https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train
- Published by the Hugging Face code research team (HuggingFaceCode org; same lineage and opt-out infrastructure as BigCode — the in-the-stack space now lives under this org). Announced by Leandro von Werra (BigCode co-lead).
- Sizes: `stack-v3-train` 15.9 TB, ~4.9T tokens, 713 languages, 173M repos — deduplicated, quality-filtered, PII-redacted, grouped by repository; `stack-v3-full` 113.7 TB / 770 languages / 224M repos (HF storage bucket).
- Key differences vs v2: direct GitHub crawl at default-branch HEAD (Aug 7 2025 — ~2 years fresher), **contents inline** (no SWH S3 step), language-agnostic MinHash-LSH dedup, non-permissive licenses excluded.
- License/attribution: ODC-By 1.0 on the compilation; per-file `repo_path`, `commit_id`, `detected_licenses` — manifest-ready. Opt-out honored via patch releases. Ungated download.
- Provenance: natural. Caveat: months old, so essentially no published third-party ablations yet — quality story is the pipeline description, not external replication.

## 2. Commit / diff / issue / notebook data

**CommitPack (full)** — https://huggingface.co/datasets/bigcode/commitpack
- 4 TB of GitHub commits (diff + message), 350 languages, permissively-licensed repos only. Tokens not published; est. very roughly 0.7–1T. CommitPackFT is the 2 GB filtered slice the lab already exhausted, so the full set is ~2000x more commit data.
- License: MIT on the packaging; **per-sample `license` field** from the source repo — identical attribution mechanics to CommitPackFT, so the lab's existing manifest pipeline applies directly.
- Quality evidence: OctoPack paper (https://github.com/bigcode-project/octopack); note the FT filtering existed because raw commit messages are noisy — expect to build an intermediate filter (e.g. keep multi-file diffs, drop trivial bumps) rather than pour all 4 TB in.
- Provenance: natural.

**GitHub issues** — https://huggingface.co/datasets/bigcode/the-stack-github-issues (54 GB, ~11–13B tokens est.) and the reprocessed `issues` subset (15.5M rows) inside https://huggingface.co/datasets/bigcode/starcoder2data-extras. Bot-comment removal, length filters, multi-user threads only, StarPII anonymization. Attribution caveat: issue text is user content under GitHub ToS, not per-repo code licenses — the manifest story here is "source URL + ToS," materially weaker than Stack-derived code. Natural provenance. GH Archive itself (gharchive.org) is the raw upstream if the lab wants PR/issue events at scale, with the same weak licensing story.

**Pull-request corpora**: honest gap. StarCoder2 trained on a PR dataset built from GHArchive, but I could not find it among the released extras subsets — as far as I can tell **no large open PR corpus with a clean licensing story exists**. SWE-bench-style sets are eval-scale (thousands of instances), and most 2025 successors (SWE-smith etc.) are LLM-synthetic (rule-conflicted).

**Jupyter/Kaggle notebooks**:
- Meta Kaggle Code — https://www.kaggle.com/datasets/kaggle/meta-kaggle-code — hundreds of thousands of notebooks, **Apache-2.0 by platform default** (clean attribution: notebook URL + author). Natural.
- Processed versions: `kaggle` subset (580K rows, converted to scripts) in starcoder2data-extras; https://huggingface.co/datasets/HuggingFaceTB/issues-kaggle-notebooks (SmolLM2's version).
- Stack-v1-derived: bigcode/jupyter-structured-clean-dedup, jupyter-code-text-pairs — same attribution mechanics as the lab's existing v1 base.

## 3. Execution-adjacent data

**Project CodeNet (IBM)** — https://github.com/IBM/Project_CodeNet
- 13.9M submissions to 4,053 competitive-programming problems, 55+ languages, ~500M LOC (~5B tokens est.). Each sample annotated with acceptance status, runtime, memory — genuinely execution-labeled natural code (53.6% accepted, rest labeled by failure mode).
- License: CDLA-Permissive-2.0 (IBM relicensed it explicitly for ML use; no downstream restrictions on results). Clean, simple attribution. Hosted on IBM DAX; unofficial HF mirrors exist.
- Provenance: natural (human submissions from AIZU/AtCoder).

**CodeContests (DeepMind)** — https://huggingface.co/datasets/deepmind/code_contests — ~13.5K problems with millions of human solutions + generated test cases; CC BY 4.0; low-single-digit B tokens. Natural solutions. Caveat: problem statements scraped from Codeforces et al.; DeepMind's CC BY claim over upstream statements is conventional but not airtight.

**TACO (BAAI)** — https://huggingface.co/datasets/BAAI/TACO — 25.4K problems, 1.55M solutions, fine-grained algorithm/difficulty labels; Apache-2.0 claimed with CC BY 4.0 web-crawled portions; same upstream-rights caveat as CodeContests, somewhat murkier. Natural.

**Unit tests / code-with-tests**: CAT-LM's aligned code+test corpus (~15M Python/Java files from ~200K repos, https://arxiv.org/abs/2310.01602) — the *model* is on HF (nikitharao/catlm) and code on GitHub, but I could not confirm the corpus itself is downloadable; the alignment pipeline is reproducible over the lab's own Stack data, which may be the real takeaway. Methods2Test (Microsoft) — 780K JUnit-test/focal-method pairs from 91K redistributably-licensed Java repos — is released and small but clean. No large general "code with passing test suites" pretraining corpus exists openly; labs build these internally.

## 4. Repo-level / curated code corpora

**Stack-Edu** — https://huggingface.co/datasets/HuggingFaceTB/stack-edu
- 125B tokens of educational-quality code filtered from StarCoder2Data (Stack v2) across 15 languages (Python 21.8B, Java 42.1B, C++ 16.0B, ...). Ships **SWHIDs only** (17.5 GB manifest, 167M rows); content via the same SWH S3 path the lab presumably already scripted for Stack data — attribution manifest mechanics identical to what it ships today.
- Filtering: StarEncoder classifiers trained on Llama3.1-70B educational-quality annotations, threshold 3/5. Content is 100% natural code; the LLM's role is judging — consistent with the lab's "external models OK for judging/eval" carve-out, but worth a note in the manifest that the *filter* is LLM-derived.
- Quality evidence: the strongest of any candidate — used in SmolLM2/SmolLM3 and OLMo 3 (https://arxiv.org/pdf/2512.13961), and it is the standard baseline other curated sets (SwallowCode) measure against.

**Common Pile v0.1 code subset** — https://huggingface.co/datasets/common-pile/stackv2_edu_filtered
- 255 GB inline text, ~69.6M docs (~60B tokens est.): the *openly licensed only* slice of Stack v2 (every detected license in the repo must be Blue-Oak-certified), further edu-filtered. Inline content — no SWH S3 step. Per-doc license + full SWH metadata. The cleanest licensing story of any code corpus surveyed. Natural. Paper: https://arxiv.org/abs/2506.05209 (Comma v0.1 7B models trained on it are the quality evidence — decent but not code-specialized).

**OpenCoder RefineCode** — https://huggingface.co/datasets/OpenCoder-LLM/RefineCode-code-corpus-meta
- **Not released as content.** 960B tokens described in the paper; only *metadata* + the ~130 language-specific filtering rules are public — reproduction requires your own GitHub crawl. Ablations showed better training efficiency than a Stack v2 subset, so the released *heuristics* are valuable even though the corpus isn't. The companion opc-fineweb-code-corpus is released (below); opc-annealing-corpus contains synthetic portions (rule-conflicted).

**Arctic-SnowCoder** — https://arxiv.org/abs/2409.02326 — **data not released** (checked; paper-only). Its lesson (quality-annotator selection of 50B from 500B, then 5B synthetic) is method evidence, not a data source. Phase-3 is LLM-synthetic anyway.

**SwallowCode / SwallowCode-v2** — https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2
- v2: 49.8B tokens Python, Apache-2.0, +17 HumanEval pass@1 over Stack-Edu in 50B-token continual-pretrain ablations — but it is **Qwen3-235B-rewritten code: LLM-synthetic, rule-conflicted for this lab**. Flagged, not recommended.
- Usable nuance: the released ablation intermediates in swallow-code (exp1 raw / exp2 syntax-filtered / exp3 pylint>=7) are *pre-rewrite natural code*, and the paper's ablations quantify exactly how much lift comes from cheap non-LLM filtering (syntax + linter) before any rewriting — that pipeline is directly adoptable.

**Nemotron-Pretraining-Code v1/v2/v3 (NVIDIA)** — https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v1
- The natural-code portion ships as **metadata to reproduce ~747B tokens** (re-crawl needed), while the ready-to-use portions are LLM-synthetic (175B synthetic code QA via Mixtral-8x22B, 58.5B code SFT) — **rule-conflicted**. NVIDIA Open Data License. v2/v3 exist; I did not verify their composition in detail (honest unknown), but the pattern (synthetic + metadata) appears to continue.

## 5. Documentation / technical prose

- **starcoder2data-extras** — https://huggingface.co/datasets/bigcode/starcoder2data-extras — 151 GB / 51.6M rows total, ungated. Relevant subsets: `documentation` (59.7K rows, popular library docs), `stackoverflow` (10.4M conversation rows), `issues`, `kaggle`, plus `ir_*` compiler intermediate representations (an unusual execution-adjacent extra). Proven in StarCoder2. Licensing is per-subset and under-documented on the card — SO is CC BY-SA-derived, docs vary by project.
- **StackOverflow/StackExchange status**: in July 2024 SE stopped public archive.org dumps; official dumps are now login-gated with an added "no LLM training" term (widely argued to be incompatible with CC BY-SA 4.0, but that is their position). The defensible path is a **pre-change archive.org dump (through early/mid-2024) under CC BY-SA 4.0** — attribution per post (author + link) fits the lab's manifest; note ShareAlike obligations on redistribution. Existing HF derivatives (RedPajama stackexchange ~20B tokens, extras `stackoverflow`) derive from pre-change dumps. HuggingFaceTB/stackexchange_2025_md (127 GB, used for SmolLM3) exists but has an empty card — provenance/license undocumented (honest unknown; I would not adopt it without asking HF how it was sourced). Sources: https://devclass.com/2024/07/30/stack-exchange-restricts-access-to-dump-of-user-contributed-data-as-critics-complain-license-permits-reuse-for-any-purpose/, https://wiki.archiveteam.org/index.php/Stack_Exchange
- **opc-fineweb-code-corpus** — https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-code-corpus — ~255 GB / 101M pages (~60–75B tokens) of code-related web prose recalled from FineWeb; MIT-labeled but CommonCrawl-derived, so attribution is web-crawl-grade (URL-level only). Ablation evidence: OpenCoder's 10% web mix.
- **Man pages / API-doc corpora**: no substantial curated open corpus found beyond the extras `documentation` subset and Common Pile's PEPs — genuine gap; a self-built readthedocs/devdocs crawl (most docs are permissively licensed per-project) would be a small bespoke job.

---

# Ranked shortlist for this lab

1. **Stack-Edu (125B tokens)** — the highest quality-per-token natural code available, with the most independent ablation evidence (SmolLM2/3, OLMo 3, used as the baseline others beat); SWHID-based, so the lab's existing Stack attribution-manifest pipeline works unchanged; alone it covers the 60B+ target with room to select. https://huggingface.co/datasets/HuggingFaceTB/stack-edu
2. **The Stack v3 train (4.9T tokens)** — the successor to the lab's own base corpus: two years fresher, contents inline, ODC-By with per-file repo/commit/license fields that map one-to-one onto the manifest, and effectively unlimited headroom for language-targeted selection; discount only for its youth (no third-party ablations yet). https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train
3. **CommitPack full (4 TB, ~1T tokens est.)** — directly relieves the stated commit-data exhaustion at ~2000x CommitPackFT scale, with per-sample license fields and identical attribution mechanics to what the lab already ships; budget for an intermediate quality filter between FT-grade and raw. https://huggingface.co/datasets/bigcode/commitpack
4. **Common Pile stackv2_edu_filtered (~60B tokens est., inline)** — the bulletproof-licensing tranche: only fully openly-licensed repos, edu-filtered, inline text with per-doc license metadata; ideal as the portion of the corpus where the attribution manifest must be unimpeachable. https://huggingface.co/datasets/common-pile/stackv2_edu_filtered
5. **starcoder2data-extras: issues + kaggle + documentation + stackoverflow (within 151 GB)** — the proven StarCoder2 recipe for code-adjacent conversational/notebook/docs data in one ungated download; flag that issues attribution rests on GitHub ToS rather than per-file licenses, and SO carries CC BY-SA ShareAlike. https://huggingface.co/datasets/bigcode/starcoder2data-extras

Honorable mentions: Project CodeNet (CDLA-P-2.0, execution-labeled, ~5B tokens) + CodeContests (CC BY 4.0) as a small execution-adjacent tranche; opc-fineweb-code-corpus for code-related web prose if URL-level attribution is acceptable.

**Rule-conflicted (flagged, not recommended)**: SwallowCode v1/v2 (Qwen/Llama-rewritten code — despite the best published ablation numbers), Nemotron synthetic code QA/SFT, OpenCoder opc-annealing-corpus (synthetic portions), Arctic-SnowCoder phase-3 (unreleased anyway), and all 2025 reasoning-code sets (KodCode, OpenCodeReasoning, codeforces-cots). Borderline-but-acceptable: Stack-Edu/Common Pile edu filters use LLM-*judged* classifiers over natural content, which matches the lab's judging/eval carve-out.

Key unknowns to resolve before committing: Stack v3's untested quality (run a 1x-Chinchilla ablation vs a Stack-v1 slice before migrating); whether stackexchange_2025_md is legitimately sourced; CommitPack's true token count; whether any StarCoder2 PR corpus will ever be released.


# SECTION: General, math, reasoning sources

All research complete. Here is the survey.

---

# Pretraining data survey: general-web, math, reasoning-adjacent (for a quality-first 60B-token corpus, 3B code-heavy model)

## 1. Curated general web

**FineWeb-Edu** — `HuggingFaceFW/fineweb-edu` (1.3T tokens, score >=3) and `HuggingFaceFW/fineweb-edu-score-2` (5.4T, score >=2). License ODC-By + CC ToU. Methodology: Llama3-70B-Instruct scored 450K FineWeb samples 0-5 for educational value; a Snowflake-arctic-embed-m regression head was trained on those annotations (F1 82% at threshold 3) and run over all of FineWeb. Quality evidence is the best-documented of any open set: published 350B-token ablations (`HuggingFaceFW/ablation-model-fineweb-edu`) show it beating FineWeb and all other open web sets on MMLU, ARC, OpenBookQA. Provenance: 100% natural web text; the LLM was used only for *judging* (consistent with your no-distillation rule — the classifier is a scorer, not a text source). Refreshed with new CC snapshots periodically.

**DCLM-baseline** — `mlfoundations/dclm-baseline-1.0`, 3.8T tokens, CC-BY-4.0. Filtered from CC via a fastText classifier trained on OpenHermes-2.5 + r/ExplainLikeImFive as positives. DCLM-7B trained on it hit 64% MMLU (5-shot), state of the art for open-data 7Bs at release. Natural text. Note its classifier optimizes "instruction-like/helpful" rather than "educational," so it's *complementary* to FineWeb-Edu — SmolLM2 and Zyda-2 both found the best results mixing the two (FineWeb-Edu stronger on MMLU/ARC, DCLM stronger on commonsense/world knowledge). Caveat: positives derived from GPT-4-generated OpenHermes — the classifier's taste is LLM-shaped, but the retained text is natural.

**Nemotron-CC (v1)** — 6.3T tokens = 4.4T deduped organic + 1.9T synthetic (rephrased low-quality + diverse-QA generated by LLMs). Distributed via Common Crawl (data.commoncrawl.org/contrib/Nemotron), NVIDIA Data Agreement for Model Training (permits training any model, incl. commercial). Evidence: HQ subset gives +5.6 MMLU vs DCLM at 1T scale; used for 7.2T of a 15T-token 8B run. The **HQ-organic** slice is natural and excellent; the synthetic 1.9T is **rule-conflicted** (LLM-generated text as training source).

**Nemotron-CC-v2 / v2.1** — `nvidia/Nemotron-CC-v2` (6.59T) and `nvidia/Nemotron-CC-v2.1` (2.5T). Same NVIDIA license. Composition is ~51% organic / **49% synthetic-rephrased-translated** (Qwen3-30B-A3B, Mistral-Nemo-12B generators; explicit SFT-style synthetic subsets). **Rule-conflicted for roughly half its tokens** — usable only if you filter to the organic partitions.

**TxT360** — `LLM360/TxT360`, ODC-By. ~5T deduped (15T with upsampling recipe): 99 CC snapshots globally deduped + 14 curated sources with per-source token counts: Papers 155B, Wikipedia 36B, **StackExchange 27.8B**, FreeLaw 16.7B, USPTO 4.95B, DM Math 5.2B, PG-19 books 2.6B, Europarl, HackerNews, Ubuntu IRC. Evidence: beat FineWeb at 1.5T-token scale on an 8x8B MoE (their own eval; weaker third-party validation than FineWeb-Edu/DCLM). Natural. Its real value for you is the **cleanly packaged curated sources**, not the web portion.

**Zyda-2** — `Zyphra/Zyda-2`, 5T tokens, ODC-By. Not new data: cross-deduplicated + model-filtered merge of DCLM + FineWeb-Edu + Dolma-CC + Zyda-1. Zamba2-7B trained on it was strong for its class. Natural. Useful shortcut if you want "DCLM+FineWeb-Edu, already deduped against each other."

**RedPajama-V2** — `togethercomputer/RedPajama-Data-V2`, 30T tokens raw with **40+ precomputed quality signals** (incl. fastText scores, dedup clusters) rather than a filtered corpus. License: CC ToU pass-through. No curation applied — it's a pool for rolling your own slices. Superseded for that purpose, in my view, by Essential-Web (below), whose labels are richer.

**Dolma / "Dolma 2" / Dolma 3** — note: there is **no dataset literally named "Dolma 2"**; the OLMo 2 era used `allenai/olmo-mix-1124` (~3.9T) + `allenai/dolmino-mix-1124` (mid-training). The current release is **Dolma 3** (Nov 2025, with OLMo 3): ~9.3T pool, `allenai/dolma3_mix-6T-1025-7B` = 5.93T mix, ODC-By, fully downloadable. Composition: 76% CC web, **13.6% olmOCR-processed science PDFs**, 6.9% stack-edu code, 2.6% FineMath-3+, 0.9% proof-pile-arXiv. Natural, heavily decontaminated, backed by the fully open OLMo 3 7B/32B results. The **Dolmino mid-training pools** (high-quality math/science/QA) are exactly the kind of thing a 60B quality-first budget can raid.

**Essential-Web v1.0** — `EssentialAI/essential-web-v1.0`, 24T tokens / 23.6B docs / 75.3TB, ODC-By + CC ToU. Every document labeled by EAI-Distill-0.5b with a 12-field taxonomy (subject FDC code, Bloom's cognitive level, document type, reasoning depth, technical correctness, education level, extraction quality). SQL-style filters yield competitive domain sets: math within 8% of SOTA-curated, web code +14.3%, STEM +24.5%, medical +8.6% vs prior web-curated baselines. Natural text; labels are model-generated metadata (judging, not generation — rule-compatible). This is the single best substrate for a lab that wants to **curate its own slices**.

**HPLT 3.0** (Jul 2025) — `HPLT/HPLT3.0`, ~30T tokens, 198 languages, ~half non-English; ~3x larger than HPLT v2 (8T mono). Aimed at multilingual coverage; English quality is not competitive with FineWeb-Edu/DCLM for a quality-first English corpus. Skip unless you want multilingual.

**Ultra-FineWeb** (OpenBMB, 2025) — `openbmb/Ultra-FineWeb`: ~1T EN + 120B ZH re-filtered from FineWeb via an efficient verification-trained fastText classifier; +3.6 MMLU vs FineWeb baseline in MiniCPM-1.2B 100B-token ablations. ODC-By(-ish, check card). Natural. A credible alternative top-slice of FineWeb.

## 2. Math

**FineMath** — `HuggingFaceTB/finemath`, ODC-By. Subsets: FineMath-3+ 34B, FineMath-4+ 9.6B, plus re-extracted InfiMM-WebMath-3+ 20.5B / 4+ 8.5B. Classifier: multilingual-e5-small regression trained on Llama-3.1-70B annotations (judging only — rule-compatible; the text itself is natural CC math). Evidence: 60B-token continued-pretraining ablations on Llama-3.2-3B show it clearly beating OpenWebMath and InfiMM-WebMath on GSM8K/MATH/MMLU-STEM. **Direct, strict upgrade over your current open-web-math.**

**OpenWebMath** — `open-web-math/open-web-math`, 14.7B tokens, ODC-By. The 2023 baseline; both FineMath and InfiMM ablations show it's now dominated (e.g., InfiMM-40B text: GSM8K 26.1 vs OWM 11.0 in matched training). Keep only for dedup lineage awareness (FineMath/MegaMath partially overlap it).

**InfiMM-WebMath-40B** — `Infi-MM/InfiMM-WebMath-40B`, 40B text tokens (+85M image URLs), natural, ODC-By. Matched DeepSeekMath's private corpus in their ablation; now effectively superseded by FineMath's cleaner re-extraction of the same pages.

**Proof-Pile-2 / AlgebraicStack** — `EleutherAI/proof-pile-2`, 55B tokens (arXiv 29B + OpenWebMath + **AlgebraicStack 11B math code** — Lean, Isabelle, Coq, math-heavy Python). Trained Llemma 7B/34B (beat Minerva-parity at scale). Mixed underlying licenses (arXiv/ per-repo code licenses). Natural. AlgebraicStack is still the best formal/math-code slice and very on-theme for a code-heavy model.

**MegaMath** (COLM 2025) — `LLM360/MegaMath`, 371B total, Apache-2.0 (repo): MegaMath-Web ~263B (natural, re-extracted CC with math-preserving HTML handling), MegaMath-Web-Pro ~15B (**LLM-refined — rule-conflicted**), MegaMath-Code ~28B (natural, Stack-V2 filtered), MegaMath-Synth ~80B (**LLM-generated — rule-conflicted**). Up-to-20% math-reasoning boosts in their ablations. Use the Web/Code partitions only.

**Nemotron-CC-Math** (Aug 2025) — `nvidia/Nemotron-CC-Math-v1`: 3+ = 133B, 4+ = 52B. NVIDIA Data Agreement. Pipeline: Lynx text-mode rendering of CC pages + an LLM pass to standardize equations to LaTeX and strip boilerplate. **Best measured open math corpus**: +4.8 MATH over FineMath-4+ at 100B-token scale (40.6 EM), +9.6 MATH over FineMath-3+ / +12.6 over MegaMath-Web at 300B scale, decontaminated vs MATH/GSM8K/MMLU. Provenance gray area for your rule: content is natural web math, but an LLM rewrote formatting — flag as **LLM-postprocessed natural** and decide; it is not synthetic content generation, but the token stream did pass through a bigger model.

**AutoMathText** — `math-ai/AutoMathText`, ~200GB pool scored by Qwen-72B zero-shot (lm_q1q2_score per doc). Natural text, model-scored (rule-compatible); 2x token-efficiency claims. More interesting as a scoring methodology than as a corpus today.

**UltraData-Math** (OpenBMB, 2026 — newest thing found) — `openbmb/UltraData-Math`, 290B+ tokens in tiers: L1 170.5B layout-preserved web, L2 33.7B embedding-classifier-selected high-density math, L3 88B "multi-format refined" (**likely LLM-refined — treat as rule-conflicted pending card check**). Ships a MathML/KaTeX/AsciiMath→LaTeX parser; used for MiniCPM5 math pretraining. L2 looks like a genuine FineMath competitor; third-party ablations not yet available (honest unknown).

**Textbooks/OER math**: no large clean-licensed math-textbook token dump exists. OpenStax is CC BY-NC-SA per their licensing page (NC clause — problematic if you ever commercialize; fine for a research lab, but flag). LibreTexts is per-page mixed CC licensing. The practical clean path is the **Common Pile v0.1** educational subsets (below) and public-domain classics (e.g., older analysis/geometry texts in Institutional Books).

## 3. Reasoning-adjacent natural data

**peS2o** — `allenai/peS2o`, ODC-By. v2 = 42B tokens / 39M docs of cleaned open-access academic full text + abstracts (S2ORC). A v3 directory exists in the repo (~120GB, olmOCR-era) but the card still documents v2 — v3 token count undocumented (unknown). This can *replace* your homegrown arXiv pipeline for non-arXiv academic text; Dolma 3's olmOCR science-PDF partition (13.6% of 5.9T ≈ 800B tokens) is the newer, bigger sibling and is downloadable inside the Dolma 3 mix.

**Wikipedia/Wikibooks** — current XML dumps at dumps.wikimedia.org (twice monthly), CC BY-SA 4.0. `wikimedia/wikipedia` on HF is frozen at 2023-11; `wikimedia/structured-wikipedia` (Enterprise HTML, 2024) is cleaner to parse. Common Pile's `wikimedia`/`wikiteam` subsets are already-parsed alternatives incl. Wikibooks/Wikiversity. ~5-6B tokens English Wikipedia; small but dense and standard.

**StackExchange** — key licensing fact: since July 2024 official dumps require login and carry a no-LLM-training rider (widely argued to violate CC BY-SA, but contested). Practical clean paths: (a) **pre-July-2024 dumps on archive.org** (CC BY-SA 3.0/4.0, unencumbered), (b) **TxT360's StackExchange slice (27.8B tokens, ODC-By packaging)**, (c) **Common Pile's stackexchange subset** (built from openly licensed dumps). High-value natural QA/argumentation for a code-heavy model.

**Law/argumentation** — Pile of Law (`pile-of-law/pile-of-law`, ~256GB, 31 sources) is CC BY-NC-SA (NC flag); the clean-license alternatives are TxT360's **FreeLaw 16.7B** and Common Pile's **Caselaw Access Project** (5.5M public-domain court opinions). Court opinions are genuinely good long-form argumentation data.

**Common Pile v0.1** (EleutherAI, Jun 2025) — `common-pile` org, 8TB, 30 sources, all public-domain/openly-licensed (Open Definition 2.1): StackExchange, CAP caselaw, arXiv, wikis, government docs, books, educational resources. Comma 7B models trained on 1-2T tokens of it matched Llama-1/2-7B-class performance — proof the clean-license slice is viable, though it trails FineWeb-Edu-class web on per-token quality.

**Institutional Books 1.0** (Harvard, 2025) — `institutional/institutional-books-1.0`, 242B tokens, 983K public-domain books, OCR post-processed, topic-classified. Public domain. Older prose; good for depth/long-context, weak on modern STEM.

**FinePDFs** (HF, Sept 2025) — `HuggingFaceFW/finepdfs`, ~3T tokens / 475M docs / 1733 langs from PDFs (Docling + RolmOCR dual pipeline), ODC-By. Near-SOTA mixture results despite mild filtering, and *additive* when mixed with HTML corpora; also the best open source of naturally long documents. English science/legal slices are very reasoning-adjacent.

## 4. Quality-classifier tooling (all locally runnable — yes, you can rescore your own slices)

- `HuggingFaceFW/fineweb-edu-classifier` — Snowflake-arctic-embed-m + regression head (~109M params). Plain transformers; ~2K docs/s/GPU. The exact model that built FineWeb-Edu; outputs 0-5 educational score on arbitrary text.
- `HuggingFaceTB/finemath-classifier` — e5-small-based math-value scorer; same usage pattern.
- `mlfoundations/fasttext-oh-eli5` — the DCLM fastText bin; CPU-only, ~GB/hour/core class throughput. Reproduces DCLM-baseline's filter.
- `openbmb/Ultra-FineWeb-classifier` — fastText, CPU; their pipeline note: 1,000 CPU-hours to filter FineWeb-scale.
- `nvidia/quality-classifier-deberta` — DeBERTa quality scorer used in the Nemotron-CC ensemble (NeMo Curator integrates it).
- `EssentialAI/eai-taxonomy` models (EAI-Distill-0.5b) — 0.5B labeler emitting the full 12-field taxonomy (reasoning depth, education level, etc.) — the most expressive open scorer.
- Bonus: `kenhktsui/fineweb-edu-fasttext-classifier` — fastText distillation of the FineWeb-Edu classifier for CPU-scale prefiltering.

Recommended local stack: fastText (DCLM or Ultra-FineWeb) as cheap first pass, FineWeb-Edu + FineMath classifiers as second pass, EAI-taxonomy labels where you want facet control. All weights are downloadable; nothing requires an API.

## 5. House-rule (no-distillation) flags

**Conflicted (LLM-generated content as training text):** Cosmopedia/Cosmopedia-2, Nemotron-CC synthetic partitions (1.9T in v1; ~49% of v2; most of v2.1), MegaMath-Synth + MegaMath-Web-Pro, UltraData-Math L3 (probable), all Nemotron SFT-style pretraining subsets. These are the *strongest-benchmarking* tokens in several ablations — that's exactly the tension — but they are distilled model output.
**Gray area:** Nemotron-CC-Math (natural content, LLM-normalized formatting).
**Compatible:** classifier-curated natural text (FineWeb-Edu, FineMath, DCLM, Essential-Web slices) — LLMs used only as judges.

## Ranked shortlist — 5 best additions for a quality-first 60B corpus

1. **FineWeb-Edu (sampled slice)** — replace the generic FineWeb sample; the best-evidenced quality upgrade per web token (MMLU/ARC/OpenBookQA ablations), ODC-By, trivially streamable.
2. **FineMath-3+ (34B, upweight the 4+ 9.6B core)** — strict, ablation-proven upgrade over your open-web-math on GSM8K/MATH; natural provenance; drop-in.
3. **DCLM-baseline (top-scored slice)** — mix with FineWeb-Edu rather than choose (SmolLM2/Zyda-2 both show the blend beats either alone); covers commonsense/world-knowledge where Edu-filtering under-selects.
4. **StackExchange 27.8B via TxT360 (or Common Pile)** — highest-value natural reasoning-adjacent data for a *code-heavy* model: real Q&A, argumentation, accepted-answer signal; clean pre-2024 licensing.
5. **peS2o v2 (42B) + Dolma 3's olmOCR science-PDF partition** — retire your homegrown arXiv-only pipeline for a cleaned, deduped academic corpus; phi-style curation logic says dense expository prose is where the quality lever lives.

Runner-up: **Nemotron-CC-Math-4+ (52B)** — the best math numbers available (+4.8 MATH over FineMath-4+), if you accept the LLM-postprocessing gray area and NVIDIA's data agreement. And keep **Essential-Web v1.0** in your toolkit as the substrate for custom slices (its STEM/code filters beat prior curated baselines) — it plus the local classifiers is exactly the "curate our own" capability you want.

**Honest unknowns:** peS2o v3 token count (undocumented); UltraData-Math has no third-party ablations yet and L3's synthesis method needs a card read; OpenStax per-title license split (help page says CC BY-NC-SA globally; older titles were CC BY — verify per book before ingesting); whether TxT360's StackExchange packaging is truly free of the 2024 rider (it predates the change, but confirm dump date).

Sources: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | [FineWeb-Edu classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) | [DCLM-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | [DCLM paper](https://arxiv.org/abs/2406.11794) | [Nemotron-CC](https://research.nvidia.com/labs/adlr/Nemotron-CC/) | [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) | [Nemotron-CC-Math](https://arxiv.org/abs/2508.15096) | [TxT360](https://huggingface.co/datasets/LLM360/TxT360) | [Zyda-2](https://arxiv.org/abs/2411.06068) | [RedPajama-V2](https://www.together.ai/blog/redpajama-data-v2) | [Dolma 3 / OLMo 3](https://allenai.org/blog/olmo3) | [dolma3_mix](https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B) | [Essential-Web v1.0](https://arxiv.org/abs/2506.14111) | [HPLT 3.0](https://hplt-project.org/datasets/v3.0) | [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) | [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | [InfiMM-WebMath-40B](https://openreview.net/forum?id=Twzrpa6V2o) | [Proof-Pile-2](https://www.eleuther.ai/artifacts/proof-pile-2) | [MegaMath](https://github.com/LLM360/MegaMath) | [AutoMathText](https://huggingface.co/datasets/math-ai/AutoMathText) | [UltraData-Math](https://huggingface.co/datasets/openbmb/UltraData-Math) | [peS2o](https://huggingface.co/datasets/allenai/peS2o) | [Common Pile](https://blog.eleuther.ai/common-pile/) | [Institutional Books](https://huggingface.co/papers/2506.08300) | [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) | [Pile of Law](https://huggingface.co/datasets/pile-of-law/pile-of-law) | [StackExchange dump policy](https://devclass.com/2024/07/30/stack-exchange-restricts-access-to-dump-of-user-contributed-data-as-critics-complain-license-permits-reuse-for-any-purpose/) | [OpenStax licensing](https://help.openstax.org/s/article/Licensing-information-of-OpenStax-textbooks)
