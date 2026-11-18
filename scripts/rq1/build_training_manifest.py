#!/usr/bin/env python3
"""Pin the RQ1 training set: MedQA-USMLE train + MedMCQA train."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from data import medmcqa_row_to_example, medqa_row_to_example


def build_manifest(
    out_jsonl: Path,
    out_meta: Path,
    medqa_max: int | None,
    medmcqa_max: int | None,
    seed: int,
) -> dict:
    medqa = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    medmcqa = load_dataset("openlifescienceai/medmcqa", split="train")

    examples = []
    medqa_kept = 0
    for idx, row in enumerate(medqa):
        examples.append(medqa_row_to_example(row, idx).to_dict())
        medqa_kept += 1
        if medqa_max is not None and medqa_kept >= medqa_max:
            break

    medmcqa_kept = 0
    for row in medmcqa:
        ex = medmcqa_row_to_example(row)
        if ex is None:
            continue
        examples.append(ex.to_dict())
        medmcqa_kept += 1
        if medmcqa_max is not None and medmcqa_kept >= medmcqa_max:
            break

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_rows": len(examples),
        "sources": {
            "medqa_usmle": {
                "hf_id": "GBaker/MedQA-USMLE-4-options",
                "split": "train",
                "rows_included": medqa_kept,
                "max_rows": medqa_max,
            },
            "medmcqa": {
                "hf_id": "openlifescienceai/medmcqa",
                "split": "train",
                "rows_included": medmcqa_kept,
                "max_rows": medmcqa_max,
                "filter": "choice_type == single and cop in 0..3",
            },
        },
        "manifest_path": str(out_jsonl),
        "note": (
            "All RQ1 adapter variants (dense LoRA, MoE-LoRA) must train on this "
            "exact manifest to avoid matched-data confounds."
        ),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out_jsonl",
        default="data/manifests/rq1_training_manifest.jsonl",
    )
    p.add_argument(
        "--out_meta",
        default="data/manifests/rq1_training_manifest_meta.json",
    )
    p.add_argument(
        "--medqa_max",
        type=int,
        default=None,
        help="Optional cap on MedQA-USMLE train rows (default: all).",
    )
    p.add_argument(
        "--medmcqa_max",
        type=int,
        default=None,
        help="Optional cap on MedMCQA rows (default: include all single-choice train rows).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta = build_manifest(
        out_jsonl=Path(args.out_jsonl),
        out_meta=Path(args.out_meta),
        medqa_max=args.medqa_max,
        medmcqa_max=args.medmcqa_max,
        seed=args.seed,
    )
    print(f"Wrote {meta['total_rows']} rows -> {args.out_jsonl}")
    print(f"Wrote metadata -> {args.out_meta}")


if __name__ == "__main__":
    main()
