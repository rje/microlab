#!/usr/bin/env bash
# Microlab regression guardrail.
#
# Runs the FULL test + lint + build suite. Every phase's tests live in this
# suite; it must stay green. Run before pushing, and any time you want to be
# sure earlier work still works after a change.
#
#   scripts/check.sh
#
# Exits non-zero if anything fails.
set -uo pipefail
cd "$(dirname "$0")/.."
CONDA=/home/rje/anaconda3/bin/conda
fail=0

run() {
  echo ""
  echo "== $1 =="
  shift
  if ! "$@"; then
    echo "FAILED: $*"
    fail=1
  fi
}

run "ruff (lint)"      $CONDA run -n microlab ruff check .
run "pytest (python)"  $CONDA run -n microlab pytest -q -m "not exercise"
run "vitest (spa)"     bash -c 'cd site && npx vitest run'
run "vite build (spa)" bash -c 'cd site && npm run build'

echo ""
if [ "$fail" -ne 0 ]; then
  echo "GUARDRAIL: FAIL — do not push. Fix the failures above."
  exit 1
fi
echo "GUARDRAIL: PASS"
