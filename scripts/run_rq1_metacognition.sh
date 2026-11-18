#!/usr/bin/env bash
# RQ1 inference: MetaMedQA metacognition suite (DMC + consistency + ECE).
# Phase 1 faithfulness (EAR/BIFR) is linked post-hoc from cached N=200 JSONL.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/scripts/rq2"

MANIFEST="$ROOT/data/manifests/metamedqa_eval_seed42.json"
SUBSET="${SUBSET:-eval}"          # eval | pilot | all
ENABLE_DMC="${ENABLE_DMC:-1}"     # 1 = --enable_dmc_binary (slow)
N_CONSISTENCY="${N_CONSISTENCY:-3}"
RESUME="${RESUME:-1}"
PROMPT_STYLE="${PROMPT_STYLE:-metamedqa_full}"

# Format: "hf_model_id" or "hf_model_id|base_model|adapter_type|adapter_dir"
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "FreedomIntelligence/HuatuoGPT-o1-8B"
  "UCSC-VLAA/m1-7B-23K"
)

run_model() {
  local spec="$1"
  local model base adapter_type adapter_dir
  IFS='|' read -r model base adapter_type adapter_dir <<< "$spec"
  base="${base:-$model}"
  adapter_type="${adapter_type:-none}"
  adapter_dir="${adapter_dir:-}"

  local slug="${model//\//_}"
  local out_jsonl="$ROOT/results/metamedqa_${slug}.jsonl"
  local summary_json="${out_jsonl%.jsonl}_summary.json"

  local -a extra=()
  if [[ "$ENABLE_DMC" == "1" ]]; then
    extra+=(--enable_dmc_binary)
  fi
  if [[ "$RESUME" == "1" ]]; then
    extra+=(--resume)
  fi
  if [[ -n "$adapter_dir" ]]; then
    extra+=(--base_model "$base" --adapter_type "$adapter_type" --adapter_dir "$adapter_dir")
  fi

  echo "==> Metacognition eval: $model (subset=$SUBSET, prompt=$PROMPT_STYLE)"
  python3 eval_metacognition.py \
    --model "$model" \
    --manifest "$MANIFEST" \
    --subset "$SUBSET" \
    --prompt_style "$PROMPT_STYLE" \
    --n_consistency_samples "$N_CONSISTENCY" \
    --out_jsonl "$out_jsonl" \
    --summary_json "$summary_json" \
    "${extra[@]}"
}

if [[ ! -f "$MANIFEST" ]]; then
  python3 freeze_metamedqa_manifest.py --out_json "$MANIFEST"
fi

for spec in "${MODELS[@]}"; do
  run_model "$spec"
done

echo "==> Link Phase 1 faithfulness metrics"
python3 link_phase1_faithfulness.py --link_all

echo "==> Aggregate comparison table"
python3 aggregate_rq1.py

echo "RQ1 metacognition suite complete."
