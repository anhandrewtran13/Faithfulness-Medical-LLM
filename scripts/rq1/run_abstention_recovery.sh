#!/usr/bin/env bash
# Full abstention-recovery run: continue existing RQ2 adapters on mixed manifest.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/scripts/rq1"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SEED=42
MANIFEST="$ROOT/data/manifests/rq3_abstention_training_manifest.jsonl"
META="$ROOT/data/manifests/rq3_abstention_training_manifest_meta.json"
MAX_STEPS="${MAX_STEPS:-2000}"
MCQ_MAX="${MCQ_MAX:-20000}"
ABSTENTION_REPEAT="${ABSTENTION_REPEAT:-30}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

DENSE_INIT="${DENSE_INIT:-$ROOT/checkpoints/rq1/dense_lora/adapter}"
MOE_INIT="${MOE_INIT:-$ROOT/checkpoints/rq1/moe_lora/adapter}"
DENSE_OUT="$ROOT/checkpoints/rq3/dense_lora"
MOE_OUT="$ROOT/checkpoints/rq3/moe_lora"

echo "==> [1/4] Build abstention recovery manifest"
python3 build_abstention_manifest.py \
  --out_jsonl "$MANIFEST" \
  --out_meta "$META" \
  --mcq_max "$MCQ_MAX" \
  --abstention_repeat "$ABSTENTION_REPEAT" \
  --include_metamedqa_original \
  --seed "$SEED"

echo "==> [2/4] Continue dense LoRA (${MAX_STEPS} steps, lr=${LEARNING_RATE})"
python3 train.py \
  --model "$MODEL" \
  --adapter dense \
  --manifest "$MANIFEST" \
  --output_dir "$DENSE_OUT" \
  --init_adapter_dir "$DENSE_INIT" \
  --max_steps "$MAX_STEPS" \
  --max_seq_len 1536 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --learning_rate "$LEARNING_RATE" \
  --run_name rq3-abstention-dense

python3 eval_accuracy.py \
  --base_model "$MODEL" \
  --adapter_type dense \
  --adapter_dir "$DENSE_OUT/adapter" \
  --benchmarks medqa medmcqa medxpertqa \
  --n_per_benchmark 200 \
  --out_json "$DENSE_OUT/eval.json"

echo "==> [3/4] Continue MoE-LoRA (${MAX_STEPS} steps, lr=${LEARNING_RATE})"
python3 train.py \
  --model "$MODEL" \
  --adapter moe \
  --manifest "$MANIFEST" \
  --output_dir "$MOE_OUT" \
  --init_adapter_dir "$MOE_INIT" \
  --max_steps "$MAX_STEPS" \
  --max_seq_len 1536 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --num_experts 4 \
  --learning_rate "$LEARNING_RATE" \
  --run_name rq3-abstention-moe

python3 eval_accuracy.py \
  --base_model "$MODEL" \
  --adapter_type moe \
  --adapter_dir "$MOE_OUT/adapter" \
  --benchmarks medqa medmcqa medxpertqa \
  --n_per_benchmark 200 \
  --out_json "$MOE_OUT/eval.json"

echo "==> [4/4] MetaMedQA metacognition re-eval"
cd "$ROOT/scripts"
SKIP_BASE=1 bash run_rq3_metacognition.sh

echo "Abstention recovery complete."
