#!/usr/bin/env python3
"""Build a mixed training manifest for abstention recovery.

Combines MedQA-USMLE + MedMCQA MCQ rows with held-out MetaMedQA abstention
examples (excluding the frozen N=200 eval split). Abstention rows are repeated
to counter class imbalance against ~130k in-domain MCQ rows.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from data import (
    TrainingExample,
    load_training_manifest,
    medmcqa_row_to_example,
    medqa_row_to_example,
    metamedqa_row_to_example,
    repeat_examples,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_MANIFEST = REPO_ROOT / "data/manifests/metamedqa_eval_seed42.json"
DEFAULT_MCQ_MANIFEST = REPO_ROOT / "data/manifests/rq1_training_manifest.jsonl"
UNKNOWN_KINDS = frozenset({"Bad", "Question modified", "Answer removed", "Answer modified", "Fictional"})


def _load_eval_qids(eval_manifest: Path) -> set[str]:
    with open(eval_manifest) as f:
        manifest = json.load(f)
    return {str(qid) for qid in manifest["qids"]}


def _metamedqa_train_rows(eval_qids: set[str]) -> list[TrainingExample]:
    ds = load_dataset("maximegmd/MetaMedQA", split="test")
    rows: list[TrainingExample] = []
    for idx, row in enumerate(ds):
        qid = str(idx)
        if qid in eval_qids:
            continue
        rows.append(metamedqa_row_to_example(row, qid=qid))
    return rows


def build_manifest(
    out_jsonl: Path,
    out_meta: Path,
    eval_manifest: Path,
    mcq_manifest: Path | None,
    medqa_max: int | None,
    medmcqa_max: int | None,
    mcq_max: int | None,
    abstention_repeat: int,
    include_metamedqa_original: bool,
    original_repeat: int,
    seed: int,
) -> dict:
    eval_qids = _load_eval_qids(eval_manifest)
    metamedqa_rows = _metamedqa_train_rows(eval_qids)

    abstention_rows = [row for row in metamedqa_rows if row.kind in UNKNOWN_KINDS]
    original_rows = [row for row in metamedqa_rows if row.kind == "Original"]

    mcq_rows: list[TrainingExample] = []
    if mcq_manifest is not None and mcq_manifest.exists():
        mcq_rows = load_training_manifest(mcq_manifest)
    else:
        medqa = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
        medmcqa = load_dataset("openlifescienceai/medmcqa", split="train")
        medqa_kept = 0
        for idx, row in enumerate(medqa):
            mcq_rows.append(medqa_row_to_example(row, idx))
            medqa_kept += 1
            if medqa_max is not None and medqa_kept >= medqa_max:
                break
        medmcqa_kept = 0
        for row in medmcqa:
            ex = medmcqa_row_to_example(row)
            if ex is None:
                continue
            mcq_rows.append(ex)
            medmcqa_kept += 1
            if medmcqa_max is not None and medmcqa_kept >= medmcqa_max:
                break

    if mcq_max is not None and len(mcq_rows) > mcq_max:
        rng = random.Random(seed)
        mcq_rows = rng.sample(mcq_rows, mcq_max)

    abstention_rows = repeat_examples(abstention_rows, abstention_repeat)
    if include_metamedqa_original:
        original_rows = repeat_examples(original_rows, original_repeat)
    else:
        original_rows = []

    examples = mcq_rows + abstention_rows + original_rows
    rng = random.Random(seed)
    rng.shuffle(examples)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for row in examples:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    kind_counts = Counter(ex.kind or "mcq" for ex in examples)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_rows": len(examples),
        "eval_manifest": str(eval_manifest.resolve()),
        "held_out_eval_qids": len(eval_qids),
        "sources": {
            "medqa_medmcqa": {
                "manifest": str(mcq_manifest) if mcq_manifest else "built_inline",
                "rows_included": len(mcq_rows),
                "mcq_max": mcq_max,
            },
            "metamedqa_abstention": {
                "hf_id": "maximegmd/MetaMedQA",
                "kinds": sorted(UNKNOWN_KINDS),
                "unique_rows": len(abstention_rows) // max(abstention_repeat, 1),
                "repeat": abstention_repeat,
                "rows_included": len(abstention_rows),
            },
            "metamedqa_original": {
                "included": include_metamedqa_original,
                "unique_rows": len(original_rows) // max(original_repeat, 1) if include_metamedqa_original else 0,
                "repeat": original_repeat if include_metamedqa_original else 0,
                "rows_included": len(original_rows),
            },
        },
        "kind_counts": dict(sorted(kind_counts.items())),
        "manifest_path": str(out_jsonl),
        "note": (
            "Mixed manifest for abstention recovery. MetaMedQA eval qids are excluded. "
            "Use prompt_style=metamedqa_full rows to teach E/F abstention under the "
            "same preamble as Phase-2 eval."
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
        default=str(REPO_ROOT / "data/manifests/rq3_abstention_training_manifest.jsonl"),
    )
    p.add_argument(
        "--out_meta",
        default=str(REPO_ROOT / "data/manifests/rq3_abstention_training_manifest_meta.json"),
    )
    p.add_argument("--eval_manifest", default=str(DEFAULT_EVAL_MANIFEST))
    p.add_argument("--mcq_manifest", default=str(DEFAULT_MCQ_MANIFEST))
    p.add_argument("--medqa_max", type=int, default=None)
    p.add_argument("--medmcqa_max", type=int, default=None)
    p.add_argument(
        "--mcq_max",
        type=int,
        default=20000,
        help="Cap MedQA+MedMCQA rows after loading (default: 20000). 0 = no cap.",
    )
    p.add_argument(
        "--abstention_repeat",
        type=int,
        default=30,
        help="Repeat each MetaMedQA abstention row this many times (default: 30).",
    )
    p.add_argument(
        "--include_metamedqa_original",
        action="store_true",
        help="Include MetaMedQA Original items with metamedqa_full prompt.",
    )
    p.add_argument(
        "--no_metamedqa_original",
        action="store_true",
        help="Exclude MetaMedQA Original rows (default for recovery runs).",
    )
    p.add_argument(
        "--original_repeat",
        type=int,
        default=1,
        help="Repeat factor for MetaMedQA Original rows (default: 1).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    include_original = args.include_metamedqa_original and not args.no_metamedqa_original
    mcq_max = args.mcq_max if args.mcq_max > 0 else None
    meta = build_manifest(
        out_jsonl=Path(args.out_jsonl),
        out_meta=Path(args.out_meta),
        eval_manifest=Path(args.eval_manifest),
        mcq_manifest=Path(args.mcq_manifest) if args.mcq_manifest else None,
        medqa_max=args.medqa_max,
        medmcqa_max=args.medmcqa_max,
        mcq_max=mcq_max,
        abstention_repeat=args.abstention_repeat,
        include_metamedqa_original=include_original,
        original_repeat=args.original_repeat,
        seed=args.seed,
    )
    print(f"Wrote {meta['total_rows']} rows -> {args.out_jsonl}")
    print(f"  MCQ rows: {meta['sources']['medqa_medmcqa']['rows_included']}")
    print(
        "  Abstention rows: "
        f"{meta['sources']['metamedqa_abstention']['rows_included']} "
        f"({meta['sources']['metamedqa_abstention']['unique_rows']} unique × "
        f"{meta['sources']['metamedqa_abstention']['repeat']})"
    )
    if include_original:
        print(
            "  Original MetaMedQA rows: "
            f"{meta['sources']['metamedqa_original']['rows_included']}"
        )
    print(f"Wrote metadata -> {args.out_meta}")


if __name__ == "__main__":
    main()
