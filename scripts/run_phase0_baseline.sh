#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_ID="phase0-ollama-qwen3-6-27b-$(date -u +%Y%m%dT%H%M%SZ)"

/home/rje/anaconda3/bin/conda run -n microlab python -m microlab.evals.cli \
  --suite evals/suites/phase0-core.jsonl \
  --config configs/eval/ollama-qwen3_6_27b.json \
  --output-dir "runs/evals/${RUN_ID}"

echo "Wrote runs/evals/${RUN_ID}"
