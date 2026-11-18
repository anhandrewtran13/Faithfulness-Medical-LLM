#!/usr/bin/env bash
# Frontier MedXpertQA faithfulness via API (Claude, ChatGPT, o3-mini).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/scripts/frontier"

PYTHON="${PYTHON:-python3}"

PROVIDER="${PROVIDER:-anthropic}"
MODEL="${MODEL:-claude-sonnet-4-6}"
N_ITEMS="${N_ITEMS:-200}"
SEED="${SEED:-42}"
MANIFEST="$ROOT/data/manifests/medxpertqa_reasoning_n200_seed42.json"
MODE="${MODE:-full}"  # full | truncation_rerun

EXTRA=()
EXTRA+=(--manifest "$MANIFEST")

if [[ "$MODE" == "truncation_rerun" ]]; then
  SLUG="${PROVIDER}_${MODEL//\//_}"
  SRC="$ROOT/results/n${N_ITEMS}_frontier_${SLUG}.jsonl"
  if [[ ! -f "$SRC" ]]; then
    echo "ERROR: prior results not found: $SRC"
    exit 1
  fi
  echo "==> Frontier faithfulness TRUNCATION RERUN: $PROVIDER / $MODEL"
  EXTRA+=(--rerun_truncation_from "$SRC")
else
  echo "==> Frontier faithfulness FULL (N=$N_ITEMS): $PROVIDER / $MODEL"
  EXTRA+=(--resume)
fi

"$PYTHON" eval_faithfulness_api.py \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --n_items "$N_ITEMS" \
  --seed "$SEED" \
  "${EXTRA[@]}"

echo "Done."
