#!/usr/bin/env bash
# Evaluate the unfine-tuned Qwen2.5-7B base model on RQ1 benchmarks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/scripts/rq1"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUT="$ROOT/checkpoints/rq1/base"

python3 eval_accuracy.py \
  --base_model "$MODEL" \
  --adapter_type none \
  --benchmarks medqa medmcqa medxpertqa \
  --medxpertqa_manifest "$ROOT/data/manifests/medxpertqa_reasoning_n200_seed42.json" \
  --n_per_benchmark 200 \
  --out_json "$OUT/eval.json"
