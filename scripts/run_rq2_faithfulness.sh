#!/usr/bin/env bash
# Phase-1 faithfulness on RQ2 adapters (MedXpertQA N=200, frozen manifest).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/scripts"

PYTHON="${PYTHON:-python3}"
MANIFEST="$ROOT/data/manifests/medxpertqa_reasoning_n200_seed42.json"
BASE="Qwen/Qwen2.5-7B-Instruct"
N_ITEMS="${N_ITEMS:-200}"
VARIANT="${VARIANT:-both}"  # dense | moe | both

run_one() {
  local adapter_type="$1"
  local adapter_dir="$2"
  local out_path="$3"
  if [[ ! -d "$adapter_dir" ]]; then
    echo "ERROR: adapter not found: $adapter_dir" >&2
    exit 1
  fi
  echo "==> RQ2 faithfulness ($adapter_type): $out_path"
  "$PYTHON" eval_faithfulness.py \
    --model "$BASE" \
    --base_model "$BASE" \
    --adapter_type "$adapter_type" \
    --adapter_dir "$adapter_dir" \
    --manifest "$MANIFEST" \
    --n_items "$N_ITEMS" \
    --load_in_4bit \
    --max_new_tokens 2048 \
    --truncation_max_new_tokens 16 \
    --out_path "$out_path"
}

if [[ "$VARIANT" == "dense" || "$VARIANT" == "both" ]]; then
  run_one dense "$ROOT/checkpoints/rq1/dense_lora/adapter" "$ROOT/results/n200_rq2_dense_faithfulness.jsonl"
fi

if [[ "$VARIANT" == "moe" || "$VARIANT" == "both" ]]; then
  run_one moe "$ROOT/checkpoints/rq1/moe_lora/adapter" "$ROOT/results/n200_rq2_moe_faithfulness.jsonl"
fi

echo "Done."
