#!/usr/bin/env bash
# Fix m1 parsing gaps:
#   1) Re-extract CoT traces for N=200 items with empty trace_full (PS prerequisite)
#   2) Re-run PS on N=50 items still missing paraphrase_stable (Claude API + short GPU decode)
#   3) Re-run MetaMedQA N=200 items with parse_ok=false (force-answer retry)
#   4) Link Phase 1 faithfulness + aggregate RQ1 table
#
# Requires: 1× GPU (≥24GB; 4-bit optional), ANTHROPIC_API_KEY in .env for PS step.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

nvidia-smi -L || { echo "ERROR: no GPU visible" >&2; exit 1; }

EXTRA=()
if [[ "${LOAD_IN_4BIT:-1}" == "1" ]]; then
  EXTRA+=(--load_in_4bit)
fi

N200="results/n200_m1-7b-23k_fixed.jsonl"
METAMEDQA="results/metamedqa_UCSC-VLAA_m1-7B-23K.jsonl"

echo "==> [1/4] Re-baseline empty trace_full rows (N=200, m1 thinking-prefill fix)"
python scripts/eval_faithfulness.py \
  --model UCSC-VLAA/m1-7B-23K \
  --n_items 200 \
  --seed 42 \
  --truncation_ks 25 50 75 90 \
  --truncation_max_new_tokens 16 \
  --max_new_tokens 4096 \
  --rebaseline_empty_from "$N200" \
  --out_path "$N200" \
  --summary_path "${N200%.jsonl}_summary.txt" \
  "${EXTRA[@]}"

echo ""
echo "==> [2/4] PS on N=50 — only items still missing paraphrase (28 expected)"
PARA_PRIOR="results/n50_m1-7b-23k_paraphrase.jsonl"
PARA_EXTRA=(--paraphrase_only_missing --paraphrase_trace_fallback "$N200")
if [[ ! -f "$PARA_PRIOR" ]]; then
  PARA_EXTRA=(--paraphrase_trace_fallback "$N200")
  PARA_PRIOR="$N200"
fi
python scripts/eval_faithfulness.py \
  --model UCSC-VLAA/m1-7B-23K \
  --n_items 50 \
  --seed 42 \
  --truncation_ks 25 50 75 90 \
  --truncation_max_new_tokens 16 \
  --enable_paraphrase \
  --paraphrase_only_from "$PARA_PRIOR" \
  "${PARA_EXTRA[@]}" \
  --out_path "$PARA_PRIOR" \
  --summary_path "${PARA_PRIOR%.jsonl}_summary.txt" \
  "${EXTRA[@]}"

echo ""
echo "==> [3/4] MetaMedQA — re-run parse_ok=false items (force-answer retry)"
bash scripts/rq2/rerun_m1_metamedqa_parse_failed.sh

echo ""
echo "==> [4/4] Done. Check:"
echo "  ${N200%.jsonl}_summary.txt"
echo "  ${PARA_PRIOR%.jsonl}_summary.txt"
echo "  ${METAMEDQA%.jsonl}_summary_linked.json"
echo "  results/rq1_metacognition_table.txt"
