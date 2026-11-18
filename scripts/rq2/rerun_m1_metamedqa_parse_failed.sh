#!/usr/bin/env bash
# Re-run MetaMedQA items where m1 parse_ok=false (parser + generation fixes).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/scripts/rq2"

MODEL="${MODEL:-UCSC-VLAA/m1-7B-23K}"
OUT="$ROOT/results/metamedqa_UCSC-VLAA_m1-7B-23K.jsonl"
MANIFEST="$ROOT/data/manifests/metamedqa_eval_seed42.json"

EXTRA=()
if [[ "${LOAD_IN_4BIT:-1}" == "1" ]]; then
  EXTRA+=(--load_in_4bit)
fi
if [[ "${LOAD_IN_4BIT:-1}" == "0" ]]; then
  EXTRA+=(--no_4bit)
fi

python3 eval_metacognition.py \
  --model "$MODEL" \
  --manifest "$MANIFEST" \
  --subset eval \
  --prompt_style metamedqa_full \
  --out_jsonl "$OUT" \
  --summary_json "${OUT%.jsonl}_summary.json" \
  --enable_dmc_binary \
  --n_consistency_samples "${N_CONSISTENCY:-3}" \
  --resume \
  --rerun_failed \
  "${EXTRA[@]}"

echo "==> Link Phase 1 faithfulness + aggregate"
python3 link_phase1_faithfulness.py --link_all
python3 aggregate_rq1.py

echo "Done. Summary: ${OUT%.jsonl}_summary_linked.json"
