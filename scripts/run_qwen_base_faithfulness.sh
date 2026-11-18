#!/usr/bin/env bash
# Phase-1 faithfulness on unfine-tuned Qwen2.5-7B-Instruct (MedXpertQA N=200).
# Uses the same harness as HuatuoGPT/m1 (letter-constrained decoding), NOT eval_accuracy.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/scripts"

PYTHON="${PYTHON:-python3}"
MANIFEST="$ROOT/data/manifests/medxpertqa_reasoning_n200_seed42.json"
MODEL="Qwen/Qwen2.5-7B-Instruct"
OUT="${OUT:-$ROOT/results/n200_qwen2.5-7b-instruct_faithfulness.jsonl}"
N_ITEMS="${N_ITEMS:-200}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: manifest not found: $MANIFEST" >&2
  exit 1
fi

echo "==> Qwen base faithfulness (MedXpertQA N=$N_ITEMS)"
echo "    manifest: $MANIFEST"
echo "    output:   $OUT"

"$PYTHON" eval_faithfulness.py \
  --model "$MODEL" \
  --manifest "$MANIFEST" \
  --n_items "$N_ITEMS" \
  --load_in_4bit \
  --max_new_tokens 2048 \
  --truncation_max_new_tokens 16 \
  --out_path "$OUT"

echo "Done. Summary: ${OUT%.jsonl}.txt"
