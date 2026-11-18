#!/usr/bin/env bash
# Re-run MetaMedQA metacognition on RQ3 abstention-recovered adapters.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/scripts/rq2"

BASE="${BASE:-Qwen/Qwen2.5-7B-Instruct}"
MANIFEST="$ROOT/data/manifests/metamedqa_eval_seed42.json"
SUBSET="${SUBSET:-eval}"
ENABLE_DMC="${ENABLE_DMC:-0}"

run_variant() {
  local name="$1"
  local adapter_type="$2"
  local adapter_dir="$3"
  local out_jsonl="$ROOT/results/metamedqa_rq3_${name}.jsonl"
  local summary_json="${out_jsonl%.jsonl}_summary.json"

  local -a extra=(--base_model "$BASE" --adapter_type "$adapter_type" --adapter_dir "$adapter_dir")
  if [[ "$ENABLE_DMC" == "1" ]]; then
    extra+=(--enable_dmc_binary)
  fi

  echo "==> RQ3 metacognition: $name"
  python3 eval_metacognition.py \
    --model "$BASE" \
    --manifest "$MANIFEST" \
    --subset "$SUBSET" \
    --n_consistency_samples 3 \
    --resume \
    --out_jsonl "$out_jsonl" \
    --summary_json "$summary_json" \
    "${extra[@]}"
}

if [[ "${SKIP_BASE:-0}" != "1" ]]; then
  run_variant "qwen_base" "none" ""
fi

run_variant "dense_lora" "dense" "$ROOT/checkpoints/rq3/dense_lora/adapter"
run_variant "moe_lora" "moe" "$ROOT/checkpoints/rq3/moe_lora/adapter"

python3 aggregate_rq1.py \
  --summaries \
  "$ROOT/results/metamedqa_rq3_dense_lora_summary.json" \
  "$ROOT/results/metamedqa_rq3_moe_lora_summary.json" \
  --out_json "$ROOT/results/rq3_metacognition_aggregate.json" \
  --out_table "$ROOT/results/rq3_metacognition_table.txt"

echo "RQ3 metacognition re-run complete."
