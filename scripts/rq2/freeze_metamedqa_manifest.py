#!/usr/bin/env python3
"""Freeze a stratified MetaMedQA eval split (N~200) and DMC pilot subset (N~50)."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]

PILOT_QUOTAS = {
    "Original": 35,
    "Bad": 4,
    "Question modified": 2,
    "Answer removed": 2,
    "Answer modified": 2,
    "Fictional": 5,
}

EVAL_TARGET = 200


def _row_to_item(row: dict, qid: str) -> dict:
    return {
        "qid": qid,
        "question": row["question"],
        "options": dict(row["options"]),
        "gold": row["answer_idx"],
        "kind": row.get("kind", "Original"),
        "meta_info": row.get("meta_info", ""),
        "answer_text": row.get("answer", ""),
    }


def stratified_sample(
    rows: list[dict],
    target_n: int,
    seed: int,
) -> list[dict]:
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)
    rng = random.Random(seed)
    kinds = sorted(by_kind.keys())
    selected: list[dict] = []
    per_kind = max(1, target_n // len(kinds))
    for kind in kinds:
        pool = by_kind[kind][:]
        rng.shuffle(pool)
        selected.extend(pool[:per_kind])
    if len(selected) < target_n:
        remaining = [row for row in rows if row not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_n - len(selected)])
    return selected[:target_n]


def pilot_sample(rows: list[dict], seed: int) -> list[dict]:
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)
    rng = random.Random(seed)
    selected: list[dict] = []
    for kind, quota in PILOT_QUOTAS.items():
        pool = by_kind.get(kind, [])
        if not pool:
            continue
        rng.shuffle(pool)
        selected.extend(pool[:quota])
    return selected


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out_json",
        default=str(REPO_ROOT / "data/manifests/metamedqa_eval_seed42.json"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_n", type=int, default=EVAL_TARGET)
    args = p.parse_args()

    ds = load_dataset("maximegmd/MetaMedQA", split="test")
    rows = []
    for idx, row in enumerate(ds):
        rows.append(_row_to_item(row, qid=str(idx)))

    eval_rows = stratified_sample(rows, target_n=args.eval_n, seed=args.seed)
    pilot_rows = pilot_sample(eval_rows, seed=args.seed)
    eval_qids = [row["qid"] for row in eval_rows]
    pilot_qids = [row["qid"] for row in pilot_rows]

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "MetaMedQA",
        "hf_id": "maximegmd/MetaMedQA",
        "split": "test",
        "seed": args.seed,
        "n_items": len(eval_rows),
        "n_pilot": len(pilot_rows),
        "pilot_qids": pilot_qids,
        "qids": eval_qids,
        "items": eval_rows,
        "kind_counts": {
            kind: sum(1 for row in eval_rows if row["kind"] == kind)
            for kind in sorted({row["kind"] for row in eval_rows})
        },
        "pilot_kind_counts": {
            kind: sum(1 for row in pilot_rows if row["kind"] == kind)
            for kind in sorted({row["kind"] for row in pilot_rows})
        },
        "note": (
            "Immutable MetaMedQA eval split for RQ1 metacognition. "
            "pilot_qids is the Week-1 DMC n=50 superset."
        ),
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(eval_rows)} eval items ({len(pilot_rows)} pilot) -> {out}")


if __name__ == "__main__":
    main()
