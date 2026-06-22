#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_ID="phase0-smoke-$(date -u +%Y%m%dT%H%M%SZ)"

/home/rje/anaconda3/bin/conda run -n microlab python -m microlab.evals.cli \
  --suite evals/suites/smoke.jsonl \
  --config configs/eval/smoke-fixture.json \
  --output-dir "runs/evals/${RUN_ID}"

echo "Wrote runs/evals/${RUN_ID}"
