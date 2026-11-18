#!/usr/bin/env bash
# Re-run frontier MetaMedQA consistency samples (e.g. fix o3-mini empty parses).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/scripts/frontier"

PYTHON="${PYTHON:-python3}"

PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-o3-mini}"

SLUG="${PROVIDER}_${MODEL//\//_}"
SRC="$ROOT/results/metamedqa_frontier_${SLUG}.jsonl"
if [[ ! -f "$SRC" ]]; then
  echo "ERROR: prior results not found: $SRC"
  exit 1
fi

echo "==> Frontier metacognition CONSISTENCY RERUN: $PROVIDER / $MODEL"

"$PYTHON" eval_metacognition_api.py \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --rerun_consistency_from "$SRC" \
  --n_consistency_samples "${N_CONSISTENCY:-3}"

echo "Done."
