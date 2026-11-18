#!/usr/bin/env bash
# Full RQ1 dense LoRA training run on the pinned manifest.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/scripts/rq1"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MANIFEST="$ROOT/data/manifests/rq1_training_manifest.jsonl"
OUT="$ROOT/checkpoints/rq1/dense_lora"
MAX_STEPS="${MAX_STEPS:-2000}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Building full training manifest..."
  python3 build_training_manifest.py \
    --out_jsonl "$MANIFEST" \
    --out_meta "$ROOT/data/manifests/rq1_training_manifest_meta.json"
fi

python3 train.py \
  --model "$MODEL" \
  --adapter dense \
  --manifest "$MANIFEST" \
  --output_dir "$OUT" \
  --max_steps "$MAX_STEPS" \
  --lora_rank 8 \
  --lora_alpha 16 \
  --gradient_accumulation_steps 8 \
  --run_name rq1-dense-lora

python3 eval_accuracy.py \
  --base_model "$MODEL" \
  --adapter_type dense \
  --adapter_dir "$OUT/adapter" \
  --benchmarks medqa medmcqa medxpertqa \
  --n_per_benchmark 200 \
  --out_json "$OUT/eval.json"
