#!/usr/bin/env bash
# Re-run consistency samples only (fast) on an existing metacognition JSONL.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/scripts/rq2"

MODEL="${MODEL:-FreedomIntelligence/HuatuoGPT-o1-8B}"
SRC="${1:-$ROOT/results/metamedqa_pilot_huatuogpt_v2.jsonl}"
OUT="${2:-$SRC}"
SUMMARY="${OUT%.jsonl}_summary.json"

python3 eval_metacognition.py \
  --model "$MODEL" \
  --manifest "$ROOT/data/manifests/metamedqa_eval_seed42.json" \
  --subset pilot \
  --prompt_style metamedqa_full \
  --consistency_only_from "$SRC" \
  --out_jsonl "$OUT" \
  --summary_json "$SUMMARY" \
  --n_consistency_samples "${N_CONSISTENCY:-3}" \
  --consistency_temperature "${CONSISTENCY_TEMPERATURE:-0.7}"

echo "Consistency re-run complete -> $OUT"
