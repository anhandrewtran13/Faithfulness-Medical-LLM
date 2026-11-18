#!/usr/bin/env python3
"""Re-apply answer/trace parsers to an existing MetaMedQA JSONL (no GPU)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RQ2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RQ2_DIR))

from data import items_from_manifest, load_manifest, resolve_repo_path  # noqa: E402
from metrics import BinaryTrial, compute_dmc, confidence_to_unit, expected_calibration_error, sample_consistency  # noqa: E402
from parsing import parse_answer, parse_confidence, parse_trace  # noqa: E402


def aggregate_rows(rows: list[dict]) -> dict:
    n = len(rows)
    parsed = [r for r in rows if r.get("parse_ok")]
    confidences = [confidence_to_unit(r["confidence"]) for r in parsed]
    correctness = [r["correct"] for r in parsed]
    binary_trials = [
        BinaryTrial(
            qid=t["qid"],
            gold=t["gold"],
            pred=t["pred"],
            confidence=t["confidence"],
            correct=t["correct"],
        )
        for r in rows
        for t in r.get("binary_trials", [])
    ]
    unknown_items = [r for r in rows if r.get("kind") in {"Bad", "Fictional", "Question modified"}]
    idk_gold_items = [r for r in rows if r.get("gold") == "F"]
    missing_gold_items = [r for r in rows if r.get("gold") == "E"]
    return {
        "n_items": n,
        "accuracy": sum(1 for r in rows if r["correct"]) / n if n else 0.0,
        "parse_rate": len(parsed) / n if n else 0.0,
        "ece": expected_calibration_error(confidences, correctness, n_bins=5),
        "mean_confidence": sum(r["confidence"] for r in parsed) / len(parsed) if parsed else 0.0,
        "mean_consistency": sum(r.get("consistency_score", 0.0) for r in rows) / n if n else 0.0,
        "abstention_recall": sum(1 for r in rows if r.get("gold") in {"E", "F"} and r.get("abstained"))
        / max(1, sum(1 for r in rows if r.get("gold") in {"E", "F"})),
        "unknown_item_abstention": sum(1 for r in unknown_items if r.get("answer") == "F")
        / len(unknown_items)
        if unknown_items
        else 0.0,
        "idk_recall": sum(1 for r in idk_gold_items if r.get("answer") == "F") / len(idk_gold_items)
        if idk_gold_items
        else 0.0,
        "missing_answer_recall": sum(1 for r in missing_gold_items if r.get("answer") == "E")
        / len(missing_gold_items)
        if missing_gold_items
        else 0.0,
        "dmc": compute_dmc(binary_trials).__dict__ if binary_trials else {},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--manifest", default="data/manifests/metamedqa_eval_seed42.json")
    p.add_argument("--summary_json", default="")
    p.add_argument("--only_failed", action="store_true")
    args = p.parse_args()

    items_by_qid = {
        item.qid: item for item in items_from_manifest(load_manifest(resolve_repo_path(args.manifest)))
    }

    jsonl_path = Path(args.jsonl)
    rows: list[dict] = []
    updated = 0
    for line in jsonl_path.read_text().splitlines():
        row = json.loads(line)
        if args.only_failed and row.get("parse_ok", True):
            rows.append(row)
            continue
        item = items_by_qid.get(row["qid"])
        if item is None:
            rows.append(row)
            continue
        raw_trace = row.get("trace") or ""
        answer = parse_answer(raw_trace, valid_letters=item.option_letters)
        if answer and (not row.get("parse_ok") or answer != row.get("answer")):
            updated += 1
        row = {
            **row,
            "answer": answer,
            "confidence": parse_confidence(raw_trace) if answer else row.get("confidence", 3),
            "correct": answer == row["gold"],
            "abstained": answer in {item.idk_letter, item.none_of_above_letter},
            "parse_ok": bool(answer),
            "trace": parse_trace(raw_trace) or raw_trace,
        }
        rows.append(row)

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = Path(args.summary_json) if args.summary_json else jsonl_path.with_name(
        jsonl_path.stem + "_summary.json"
    )
    prior_meta: dict = {}
    if summary_path.exists():
        with open(summary_path) as f:
            prior_meta = json.load(f)
    summary = aggregate_rows(rows)
    for key in ("model", "base_model", "adapter_dir", "adapter_type", "manifest", "subset", "prompt_style", "dmc", "by_kind"):
        if key in prior_meta:
            summary[key] = prior_meta[key]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Updated {updated} rows -> {jsonl_path}")
    print(f"parse_rate={summary['parse_rate']:.3f} accuracy={summary['accuracy']:.3f}")


if __name__ == "__main__":
    main()
