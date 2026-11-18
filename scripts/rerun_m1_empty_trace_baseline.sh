#!/usr/bin/env bash
# Re-extract m1 CoT traces for items with empty trace_full (thinking-prefill fix).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PRIOR="${PRIOR:-results/n200_m1-7b-23k_fixed.jsonl}"
OUT="${OUT:-results/n200_m1-7b-23k_fixed.jsonl}"

EXTRA=()
if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  EXTRA+=(--load_in_4bit)
fi

python scripts/eval_faithfulness.py \
  --model UCSC-VLAA/m1-7B-23K \
  --n_items 200 \
  --seed 42 \
  --truncation_ks 25 50 75 90 \
  --truncation_max_new_tokens 16 \
  --max_new_tokens 4096 \
  --rebaseline_empty_from "$PRIOR" \
  --out_path "$OUT" \
  --summary_path "${OUT%.jsonl}_summary.txt" \
  "${EXTRA[@]}"

echo "Done. Re-run PS on N=50 with:"
echo "  bash scripts/rerun_m1_n50_paraphrase.sh"
