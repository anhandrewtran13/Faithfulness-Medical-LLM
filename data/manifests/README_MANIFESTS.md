# Manifests

Frozen eval manifests (`seed=42`) and smoke training subsets are included.
Full RQ2/RQ3 training JSONL (~58MB) omitted for the AAAI 50MB upload limit.

Rebuild:
```bash
python3 scripts/rq1/build_training_manifest.py \
  --out_jsonl data/manifests/rq1_training_manifest.jsonl \
  --out_meta data/manifests/rq1_training_manifest_meta.json

python3 scripts/rq1/build_abstention_manifest.py \
  --out_jsonl data/manifests/rq3_abstention_training_manifest.jsonl \
  --out_meta data/manifests/rq3_abstention_training_manifest_meta.json \
  --mcq_max 20000 --abstention_repeat 30 --include_metamedqa_original --seed 42
```
