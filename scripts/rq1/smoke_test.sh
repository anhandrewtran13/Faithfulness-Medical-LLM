#!/usr/bin/env bash
# RQ1 week-1 smoke test: pinned manifest + dense LoRA + MoE-LoRA end-to-end.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/scripts/rq1"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SEED=42
SMOKE_N=500
MAX_STEPS=100
MANIFEST="$ROOT/data/manifests/rq1_training_smoke_500.jsonl"
META="$ROOT/data/manifests/rq1_training_smoke_500_meta.json"

echo "==> [1/5] Build smoke training manifest (${SMOKE_N} rows)"
python3 build_training_manifest.py \
  --out_jsonl "$MANIFEST" \
  --out_meta "$META" \
  --medqa_max 100 \
  --medmcqa_max 400 \
  --seed "$SEED"

echo "==> [2/5] Freeze MedXpertQA eval manifest (N=200, seed=42)"
python3 freeze_eval_manifest.py \
  --results_jsonl "$ROOT/results/n200_huatuogpt-o1-8b_fixed.jsonl" \
  --out_json "$ROOT/data/manifests/medxpertqa_reasoning_n200_seed42.json"

echo "==> [3/5] Dense LoRA smoke train"
python3 train.py \
  --model "$MODEL" \
  --adapter dense \
  --manifest "$MANIFEST" \
  --output_dir "$ROOT/checkpoints/rq1_smoke/dense_lora" \
  --max_samples "$SMOKE_N" \
  --max_steps "$MAX_STEPS" \
  --lora_rank 8 \
  --lora_alpha 16 \
  --run_name rq1-smoke-dense

echo "==> [4/5] MoE-LoRA smoke train (4 experts, rank 8)"
python3 train.py \
  --model "$MODEL" \
  --adapter moe \
  --manifest "$MANIFEST" \
  --output_dir "$ROOT/checkpoints/rq1_smoke/moe_lora" \
  --max_samples "$SMOKE_N" \
  --max_steps "$MAX_STEPS" \
  --lora_rank 8 \
  --lora_alpha 16 \
  --num_experts 4 \
  --run_name rq1-smoke-moe

echo "==> [5/5] Load-check + tiny accuracy eval (20 items each)"
python3 eval_accuracy.py \
  --base_model "$MODEL" \
  --adapter_type dense \
  --adapter_dir "$ROOT/checkpoints/rq1_smoke/dense_lora/adapter" \
  --benchmarks medqa medmcqa \
  --n_per_benchmark 20 \
  --out_json "$ROOT/checkpoints/rq1_smoke/dense_lora/eval_smoke.json"

python3 eval_accuracy.py \
  --base_model "$MODEL" \
  --adapter_type moe \
  --adapter_dir "$ROOT/checkpoints/rq1_smoke/moe_lora/adapter" \
  --benchmarks medqa medmcqa \
  --n_per_benchmark 20 \
  --out_json "$ROOT/checkpoints/rq1_smoke/moe_lora/eval_smoke.json"

echo "Smoke test complete."
