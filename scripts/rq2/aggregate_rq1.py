#!/usr/bin/env python3
"""Aggregate RQ1 metacognition summaries into a comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p.resolve()
    candidate = REPO_ROOT / p
    if candidate.exists():
        return candidate
    return p


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def short_name(summary: dict) -> str:
    model = summary.get("model", "unknown")
    adapter = summary.get("adapter_type", "none")
    if adapter and adapter != "none":
        return f"{Path(summary.get('adapter_dir', 'adapter')).name} ({adapter})"
    return model.split("/")[-1]


def fmt_float(x: float | None, ndigits: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{ndigits}f}"


def fmt_dmc(dmc: dict | None) -> str:
    if not dmc:
        return "—"
    if dmc.get("note"):
        return "N/A"
    score = dmc.get("score")
    if score is None:
        return "—"
    return f"{score:.3f}"


def build_table(summaries: list[dict]) -> str:
    headers = [
        "Model",
        "N",
        "Acc",
        "ECE",
        "DMC",
        "Consistency",
        "Unk.Abstain",
        "EAR_50",
        "BIFR",
        "PS",
    ]
    rows: list[list[str]] = []
    for s in summaries:
        phase1 = s.get("phase1_faithfulness") or s.get("phase1") or {}
        ear = phase1.get("ear") or {}
        ps = phase1.get("ps")
        if ps is None:
            ps = (phase1.get("paraphrase_subset") or {}).get("ps")
        rows.append(
            [
                short_name(s),
                str(s.get("n_items", "")),
                fmt_float(s.get("accuracy")),
                fmt_float(s.get("ece")),
                fmt_dmc(s.get("dmc")),
                fmt_float(s.get("mean_consistency")),
                fmt_float(s.get("unknown_item_abstention")),
                fmt_float(ear.get("EAR_50")),
                fmt_float(phase1.get("bifr")),
                fmt_float(ps),
            ]
        )

    widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    lines: list[str] = []
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--summaries",
        nargs="*",
        default=[],
        help="Summary JSON paths (default: results/metamedqa_*_summary.json).",
    )
    p.add_argument("--out_json", default=str(REPO_ROOT / "results/rq1_metacognition_aggregate.json"))
    p.add_argument("--out_table", default=str(REPO_ROOT / "results/rq1_metacognition_table.txt"))
    args = p.parse_args()

    paths = [resolve(p) for p in args.summaries] if args.summaries else sorted(
        (REPO_ROOT / "results").glob("metamedqa_*_summary.json")
    )
    if not paths:
        raise SystemExit("No summary JSON files found.")

    summaries = [load_summary(path) for path in paths]
    table = build_table(summaries)

    aggregate = {
        "n_models": len(summaries),
        "models": summaries,
        "table": table,
    }
    out_json = resolve(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)
        f.write("\n")

    out_table = resolve(args.out_table)
    with open(out_table, "w") as f:
        f.write(table + "\n")

    print(table)
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_table}")


if __name__ == "__main__":
    main()
