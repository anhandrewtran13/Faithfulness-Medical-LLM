#!/usr/bin/env bash
# Quick smoke test for abstention-recovery training (continues from existing dense LoRA).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/scripts/rq1"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SEED=42
MANIFEST="$ROOT/data/manifests/rq3_abstention_smoke_manifest.jsonl"
META="$ROOT/data/manifests/rq3_abstention_smoke_manifest_meta.json"
OUT="$ROOT/checkpoints/rq3_smoke/dense_lora"
INIT_ADAPTER="${INIT_ADAPTER:-$ROOT/checkpoints/rq1/dense_lora/adapter}"
MAX_STEPS="${MAX_STEPS:-100}"

echo "==> [1/3] Build smoke abstention manifest (500 MCQ + abstention x30)"
python3 build_abstention_manifest.py \
  --out_jsonl "$MANIFEST" \
  --out_meta "$META" \
  --mcq_max 500 \
  --abstention_repeat 30 \
  --no_metamedqa_original \
  --seed "$SEED"

echo "==> [2/3] Continue dense LoRA on abstention mix (${MAX_STEPS} steps)"
python3 train.py \
  --model "$MODEL" \
  --adapter dense \
  --manifest "$MANIFEST" \
  --output_dir "$OUT" \
  --init_adapter_dir "$INIT_ADAPTER" \
  --max_steps "$MAX_STEPS" \
  --max_seq_len 1536 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --learning_rate 1e-4 \
  --run_name rq3-smoke-abstention-dense

echo "==> [3/3] MetaMedQA metacognition eval (N=200 held-out)"
cd "$ROOT/scripts/rq2"
python3 eval_metacognition.py \
  --model "$MODEL" \
  --base_model "$MODEL" \
  --adapter_type dense \
  --adapter_dir "$OUT/adapter" \
  --manifest "$ROOT/data/manifests/metamedqa_eval_seed42.json" \
  --subset eval \
  --n_consistency_samples 3 \
  --resume \
  --out_jsonl "$ROOT/results/metamedqa_rq3_smoke_dense_lora.jsonl" \
  --summary_json "$ROOT/results/metamedqa_rq3_smoke_dense_lora_summary.json"

python3 aggregate_rq1.py \
  --summaries "$ROOT/results/metamedqa_rq3_smoke_dense_lora_summary.json" \
  --out_json "$ROOT/results/rq3_smoke_metacognition_aggregate.json" \
  --out_table "$ROOT/results/rq3_smoke_metacognition_table.txt"

echo "Smoke abstention recovery complete. See $ROOT/results/rq3_smoke_metacognition_table.txt"
