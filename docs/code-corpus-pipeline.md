# Code-corpus pipeline (`scripts/build_code_corpus.py`)

Builds the coding-specialist pretraining corpus from **bigcode/the-stack-dedup** (gated;
the configured HF token opens it -- see `docs/code-corpus-comparison.md` for why this
source and not starcoderdata or the non-gated substitutes).

## Pipeline

```
per-language parquet files (hf_hub_download one file at a time, deleted after use)
  -> language check (partition dir + `lang` column; mismatch raises)
  -> cleaning gates REUSED from build_code_tokenizer_corpora
     (non-UTF-8, minified/generated blobs, 64 B..1 MB size bounds)
  -> license filter: permissive allowlist (MIT/Apache-2.0/BSD-2,3/ISC/Unlicense/0BSD/
     Zlib/CC0/MIT-0/BSD-3-Clear; ALL listed licenses must pass, so MIT+GPL dual tags
     are rejected) + attribution completeness (repo + hexsha + path required)
  -> exact dedup on sha256(content) (near-dedup: documented hook `near_dup_reason`,
     not implemented; `--near-dedup` raises)
  -> deterministic train/val routing by content hash (--val-fraction)
  -> tokenize (--tokenizer, EOT after each doc)
  -> uint16 .bin shards, byte-exact ShardDataset layout (see below)
```

Every kept file appends `{lang, repo, hexsha, path, licenses, split, tokens}` to
`attribution.jsonl` -- the shipped attribution manifest (hard requirement). Invariant
(unit-tested and verified on the smoke corpus): `sum(record.tokens)` equals the train+val
manifest totals, so every emitted token is attributable.

## Output format

Exactly the `data/shards/fineweb-100bt` layout, so `ShardDataset` consumes it unchanged:
`train-NNNNN.bin` / `val-NNNNN.bin` (uint16), `train-manifest.json` / `val-manifest.json`
(`{split, shards: [{file, tokens}], total_tokens, dtype}`), and `tokenizer.json` copied
alongside. Plus corpus extras: `attribution.jsonl`, `seen-hashes.u64` (dedup state),
`build-state.json` (cursor/config/stats).

## Resumability (house rule: long jobs write as they go)

Shards and manifests are written progressively (a valid manifest always exists). A
checkpoint every `--checkpoint-rows` rows (and at language boundaries) persists partial
token buffers, flushes attribution records + dedup hashes, then atomically replaces
`build-state.json` (written LAST). On restart with the same `--out`, everything written
after the last checkpoint is rolled back (orphan shards deleted, attribution/hash files
truncated to the checkpointed offsets) and the parquet stream resumes at the recorded
(file index, row) cursor via row-group skipping. Verified two ways: SIGKILL mid-build and
resume produced **bit-identical** shards and attribution vs an uninterrupted run; unit
tests cover interrupt/resume, post-checkpoint rollback, and config-mismatch refusal.

## Smoke corpus (built 2026-07-29)

```
python scripts/build_code_corpus.py --out data/shards/code-stack-smoke \
    --tokenizer data/tokenizers/code-49k.json \
    --languages python javascript typescript \
    --target-tokens 200000000 --val-fraction 0.005
```

| | |
|---|---|
| tokens | 201,392,516 (train 199,977,955 in 2 shards; val 1,414,561 in 1) |
| per language | py 67.5M / js 66.9M / ts 67.0M |
| rows streamed | py 50,001 / js 70,051 / ts 95,847 (survival 94-95% after all gates) |
| disk | 431 MB (400 MB shards + 43.4 MB attribution.jsonl, 201,306 records) |
| verification | ShardDataset loads train+val unchanged; get_batch x/y offset checked; decoded samples at several offsets are clean Python/JS/TS; attribution token-sum invariant holds |

The smoke `attribution.jsonl` (43 MB) is too large to commit; it lives at
`data/shards/code-stack-smoke/attribution.jsonl` (gitignored with the shards).

## Throughput and full-corpus projections

Measured pipeline throughput on the smoke build: **~9.0B tokens/hour** (201.4M tokens in
80.9 s, single process; per-file `hf_hub_download` at ~40 MB/s dominates, tokenization
overlaps poorly since the design is serial). Long sustained runs may see lower CDN
throughput; the projections below quote measured-rate and a conservative half-rate bound.

| build | wall clock (9B/h .. 4.5B/h) | shards | attribution | dedup RAM |
|---|---|---|---|---|
| 40B tokens | ~4.5 h .. ~9 h | 80 GB | ~8.6 GB | ~4 GB |
| 90B tokens | ~10 h .. ~20 h | 180 GB | ~19.4 GB | ~9 GB |

Transient disk: one parquet file (~200-500 MB) + pending buffers (<= 2x shard bytes).
Well within the ~1.1 TB free.

**Availability ceiling (important).** Estimated single-epoch yield of the-stack-dedup
after cleaning + license filter + dedup, from smoke-measured tokens/row x parquet row
counts: python ~17.6B, javascript ~20.1B, typescript ~7.4B -- **~45B tokens total**.
Consequences:

- A **40B** dual-focus build is feasible in a single epoch only with
  availability-weighted `--lang-weights` (e.g. `17.6 20.1 7.4`); equal thirds would
  exhaust TypeScript at ~7.4B (the build then marks the language `exhausted` and reports
  the shortfall rather than silently rebalancing).
- A **90B** build is NOT reachable single-epoch from py/js/ts the-stack-dedup alone: it
  needs ~2 epochs of repetition, more languages, or an additional clean-provenance
  source. Decide before launching the real build.

## Knobs

`--languages`, `--target-tokens`, `--val-fraction`, `--lang-weights`, `--shard-size`
(default 100M tokens, = fineweb-100bt), `--batch-docs`, `--checkpoint-rows`,
`--min-chars`/`--max-chars` (cleaning size gates), `--near-dedup` (raises; hook
documented at `near_dup_reason`). Token budgets overshoot by at most one tokenize batch
(~1M tokens at defaults).
