#!/usr/bin/env bash
# Frontier MetaMedQA metacognition via API (Claude, ChatGPT).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/scripts/frontier"

PYTHON="${PYTHON:-python3}"

PROVIDER="${PROVIDER:-anthropic}"
MODEL="${MODEL:-claude-sonnet-4-6}"
PILOT_N="${PILOT_N:-0}"
SUBSET="${SUBSET:-eval}"
N_CONSISTENCY="${N_CONSISTENCY:-3}"

EXTRA=()
if [[ "$PILOT_N" -gt 0 ]]; then
  EXTRA+=(--pilot_n "$PILOT_N")
  echo "==> Frontier metacognition PILOT (N=$PILOT_N): $PROVIDER / $MODEL"
else
  echo "==> Frontier metacognition FULL (N=200): $PROVIDER / $MODEL"
fi

"$PYTHON" eval_metacognition_api.py \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --subset "$SUBSET" \
  --n_consistency_samples "$N_CONSISTENCY" \
  --resume \
  "${EXTRA[@]}"

echo "Done. Results in $ROOT/results/metamedqa_frontier_${PROVIDER}_${MODEL//\//_}*.jsonl"
