#!/usr/bin/env bash
# Week 1 Phase 2 gate: smoke train, base eval, DMC pilot.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "======== Phase 2 Week 1 ========"

echo "==> [1/3] RQ2 training smoke test (dense + MoE-LoRA)"
bash scripts/rq1/smoke_test.sh

echo "==> [2/3] Qwen2.5-7B base accuracy (MedQA / MedMCQA / MedXpertQA N=200)"
bash scripts/rq1/run_base_eval.sh

echo "==> [3/3] DMC n=50 pilot on MetaMedQA"
bash scripts/rq2/run_dmc_pilot.sh

echo "==> Link Phase 1 faithfulness to pilot summary (if present)"
if compgen -G "$ROOT/results/metamedqa_pilot_*_summary.json" > /dev/null; then
  python3 scripts/rq2/link_phase1_faithfulness.py --link_all
fi

echo "Week 1 complete. Review:"
echo "  - checkpoints/rq1_smoke/*/eval_smoke.json"
echo "  - checkpoints/rq1/base/eval.json"
echo "  - results/metamedqa_pilot_*_summary.json"
