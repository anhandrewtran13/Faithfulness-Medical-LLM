#!/usr/bin/env bash
# Compute Wilson CIs, answerable-item abstention, and McNemar paired tests.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/scripts/analysis"
python3 compute_uncertainty.py "$@"
