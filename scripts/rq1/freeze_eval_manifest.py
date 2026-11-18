#!/usr/bin/env python3
"""Freeze the Phase-1 MedXpertQA Reasoning eval slice (N=200, seed=42)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_jsonl",
        default="results/n200_huatuogpt-o1-8b_fixed.jsonl",
        help="Source JSONL whose qid order defines the frozen eval set.",
    )
    p.add_argument(
        "--out_json",
        default="data/manifests/medxpertqa_reasoning_n200_seed42.json",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    qids: list[str] = []
    with open(args.results_jsonl) as f:
        for line in f:
            qids.append(json.loads(line)["qid"])

    ds = load_dataset("TsinghuaC3I/MedXpertQA", "Text", split="test")
    by_id = {row["id"]: row for row in ds if row.get("question_type") == "Reasoning"}
    items = []
    missing = []
    for qid in qids:
        row = by_id.get(qid)
        if row is None:
            missing.append(qid)
            continue
        options = row["options"]
        if isinstance(options, list):
            options = {o["letter"]: o["content"] for o in options}
        gold = row["label"]
        if isinstance(gold, list):
            gold = gold[0]
        items.append(
            {
                "qid": row["id"],
                "gold": gold,
                "question": row["question"],
                "options": options,
                "medical_task": row.get("medical_task"),
                "body_system": row.get("body_system"),
            }
        )
    if missing:
        raise RuntimeError(f"Missing {len(missing)} qids from MedXpertQA: {missing[:5]}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "MedXpertQA Text Reasoning",
        "hf_id": "TsinghuaC3I/MedXpertQA",
        "split": "test",
        "seed": args.seed,
        "n_items": len(items),
        "source_results": args.results_jsonl,
        "qids": qids,
        "items": items,
        "note": (
            "Immutable eval set for RQ1 out-of-domain accuracy and RQ2 faithfulness. "
            "Phase-1 BIFR/EAR results used these exact qids."
        ),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(items)} items -> {out}")


if __name__ == "__main__":
    main()
