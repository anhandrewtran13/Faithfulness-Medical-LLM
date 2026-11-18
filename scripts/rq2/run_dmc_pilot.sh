#!/usr/bin/env bash
# Week-1 DMC n=50 pilot on MetaMedQA (after freeze_metamedqa_manifest.py).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/scripts/rq2"

MODEL="${MODEL:-FreedomIntelligence/HuatuoGPT-o1-8B}"
MANIFEST="$ROOT/data/manifests/metamedqa_eval_seed42.json"
OUT_JSONL="$ROOT/results/metamedqa_pilot_${MODEL//\//_}.jsonl"

echo "==> [1/2] Freeze MetaMedQA eval manifest (if missing)"
if [[ ! -f "$MANIFEST" ]]; then
  python3 freeze_metamedqa_manifest.py --out_json "$MANIFEST"
else
  echo "Using existing manifest: $MANIFEST"
fi

echo "==> [2/2] DMC pilot (N~50, subset=pilot, prompt=metamedqa_full)"
python3 eval_metacognition.py \
  --model "$MODEL" \
  --manifest "$MANIFEST" \
  --subset pilot \
  --prompt_style metamedqa_full \
  --n_consistency_samples 3 \
  --enable_dmc_binary \
  --out_jsonl "$OUT_JSONL" \
  --summary_json "${OUT_JSONL%.jsonl}_summary.json"

echo "Pilot complete."
