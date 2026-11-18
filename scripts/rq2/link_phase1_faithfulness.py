#!/usr/bin/env python3
"""Attach Phase 1 MedXpertQA faithfulness metrics (EAR, BIFR, PS) to RQ2 summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical Phase 1 fixed runs (N=200, seed=42).
DEFAULT_PHASE1_JSONL: dict[str, str] = {
    "FreedomIntelligence/HuatuoGPT-o1-8B": "results/n200_huatuogpt-o1-8b_fixed.jsonl",
    "UCSC-VLAA/m1-7B-23K": "results/n200_m1-7b-23k_fixed.jsonl",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "results/n200_deepseek-r1-distill-llama-8b_fixed.jsonl",
}

PARAPHRASE_JSONL: dict[str, str] = {
    "FreedomIntelligence/HuatuoGPT-o1-8B": "results/n50_huatuogpt-o1-8b_paraphrase.jsonl",
    "UCSC-VLAA/m1-7B-23K": "results/n50_m1-7b-23k_paraphrase.jsonl",
}


def resolve(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p.resolve()
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return p


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_faithfulness(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n_items": 0, "note": "empty jsonl"}

    correct = sum(1 for r in rows if r.get("correct_full"))
    ks = sorted({int(k) for r in rows for k in (r.get("truncated") or {}).keys()})

    ear: dict[str, float] = {}
    empty_parse: dict[str, float] = {}
    for k in ks:
        key = str(k)
        unchanged = 0
        empty = 0
        for r in rows:
            full = r.get("answer_full", "")
            trunc = (r.get("truncated") or {}).get(key, "")
            if (r.get("truncated_empty") or {}).get(key):
                empty += 1
            elif full and trunc and full == trunc:
                unchanged += 1
        ear[f"EAR_{k}"] = unchanged / n
        empty_parse[f"empty@{k}%"] = empty / n

    biased = [r for r in rows if r.get("biased_flipped") is not None]
    flipped = sum(1 for r in biased if r.get("biased_flipped"))
    originally_correct = sum(1 for r in biased if r.get("correct_full"))

    paraphrase = [r for r in rows if r.get("paraphrase_stable") is not None]
    ps_hits = sum(1 for r in paraphrase if r.get("paraphrase_stable"))
    trace_coverage = sum(1 for r in paraphrase if r.get("trace_full"))

    return {
        "n_items": n,
        "medxpertqa_accuracy": correct / n,
        "medxpertqa_correct": correct,
        "ear": ear,
        "empty_parse": empty_parse,
        "bifr": flipped / originally_correct if originally_correct else None,
        "bifr_n_correct": originally_correct,
        "bifr_n_flipped": flipped,
        "ps": ps_hits / len(paraphrase) if paraphrase else None,
        "ps_n": len(paraphrase),
        "ps_trace_coverage": trace_coverage / len(paraphrase) if paraphrase else None,
    }


def merge_summary(rq2_summary_path: Path, model: str, phase1_path: Path | None) -> dict:
    with open(rq2_summary_path) as f:
        summary = json.load(f)

    if phase1_path is None or not phase1_path.exists():
        summary["phase1_faithfulness"] = {"note": "No Phase 1 JSONL mapped for this model."}
        return summary

    rows = load_jsonl(phase1_path)
    faith = compute_faithfulness(rows)
    faith["source_jsonl"] = str(phase1_path)

    paraphrase_path = PARAPHRASE_JSONL.get(model)
    if paraphrase_path:
        ppath = resolve(paraphrase_path)
        if ppath.exists():
            faith["paraphrase_subset"] = compute_faithfulness(load_jsonl(ppath))

    summary["phase1_faithfulness"] = faith
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rq2_summary", action="append", default=[], help="RQ2 summary JSON path.")
    p.add_argument("--model", default="", help="HF model id (auto-map Phase 1 JSONL).")
    p.add_argument("--phase1_jsonl", default="", help="Override Phase 1 JSONL path.")
    p.add_argument(
        "--out_suffix",
        default="_linked",
        help="Write merged copy as <stem><suffix>.json (empty = update summary in place).",
    )
    p.add_argument(
        "--link_all",
        action="store_true",
        help="Link all results/metamedqa_*_summary.json with known Phase 1 runs.",
    )
    args = p.parse_args()

    targets: list[tuple[Path, str]] = []
    if args.link_all:
        for path in sorted((REPO_ROOT / "results").glob("metamedqa_*_summary.json")):
            with open(path) as f:
                model = json.load(f).get("model", "")
            if model:
                targets.append((path, model))
    elif args.rq2_summary:
        for spath in args.rq2_summary:
            path = resolve(spath)
            with open(path) as f:
                model = args.model or json.load(f).get("model", "")
            targets.append((path, model))
    else:
        p.error("Pass --rq2_summary, or use --link_all.")

    for summary_path, model in targets:
        if args.phase1_jsonl:
            phase1 = resolve(args.phase1_jsonl)
        else:
            rel = DEFAULT_PHASE1_JSONL.get(model)
            phase1 = resolve(rel) if rel else None

        merged = merge_summary(summary_path, model, phase1)
        if args.out_suffix:
            out_path = summary_path.with_name(summary_path.stem + args.out_suffix + ".json")
        else:
            out_path = summary_path
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
            f.write("\n")
        if out_path != summary_path:
            with open(summary_path, "w") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
                f.write("\n")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
