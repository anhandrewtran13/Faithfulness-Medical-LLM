#!/usr/bin/env bash
# Re-run m1 PS on N=50 items still missing paraphrase_stable (after rebaseline).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

nvidia-smi -L || { echo "ERROR: no GPU visible" >&2; exit 1; }

PRIOR="${PARA_PRIOR:-results/n50_m1-7b-23k_paraphrase.jsonl}"
OUT="${OUT:-$PRIOR}"

EXTRA=()
if [[ "${LOAD_IN_4BIT:-1}" == "1" ]]; then
  EXTRA+=(--load_in_4bit)
fi

python scripts/eval_faithfulness.py \
  --model UCSC-VLAA/m1-7B-23K \
  --n_items 50 \
  --seed 42 \
  --truncation_ks 25 50 75 90 \
  --truncation_max_new_tokens 16 \
  --enable_paraphrase \
  --paraphrase_only_from "$PRIOR" \
  --paraphrase_trace_fallback "${TRACE_FALLBACK:-results/n200_m1-7b-23k_fixed.jsonl}" \
  --paraphrase_only_missing \
  --out_path "$OUT" \
  --summary_path "${OUT%.jsonl}_summary.txt" \
  "${EXTRA[@]}"

echo "Done. Summary: ${OUT%.jsonl}_summary.txt"
