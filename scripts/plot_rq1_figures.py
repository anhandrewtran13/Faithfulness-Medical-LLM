#!/usr/bin/env python3
"""Generate RQ1 figures from rq1_metacognition_aggregate.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "Paper" / "figures"
AGGREGATE = REPO_ROOT / "results" / "rq1_metacognition_aggregate.json"

COLORS = {
    "HuatuoGPT-o1-8B": "#4477AA",
    "Qwen2.5-7B-Instruct": "#66AA55",
    "m1-7B-23K": "#CC6677",
}


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def load_rows() -> list[dict]:
    agg = json.loads(AGGREGATE.read_text())
    rows: list[dict] = []
    for s in agg["models"]:
        label = short_name(s["model"])
        p1 = s.get("phase1_faithfulness") or {}
        ps = p1.get("ps")
        if ps is None:
            ps = (p1.get("paraphrase_subset") or {}).get("ps")
        rows.append(
            {
                "label": label,
                "color": COLORS.get(label, "#888888"),
                "medx_acc": p1.get("medxpertqa_accuracy"),
                "ear50": (p1.get("ear") or {}).get("EAR_50"),
                "bifr": p1.get("bifr"),
                "ps": ps,
                "meta_acc": s.get("accuracy"),
                "ece": s.get("ece"),
                "consistency": s.get("mean_consistency"),
                "abstain": s.get("unknown_item_abstention"),
            }
        )
    return rows


def plot_faithfulness_accuracy(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    metrics = [
        ("EAR$_{50}$", "ear50", "higher → verbose guessing"),
        ("BIFR", "bifr", "higher → sycophancy"),
        ("PS", "ps", "higher → semantic stability"),
    ]
    xs = [r["medx_acc"] * 100 for r in rows if r["medx_acc"] is not None]
    xmin = max(0, min(xs) - 2) if xs else 8
    xmax = max(xs) + 2 if xs else 16

    for ax, (title, key, subtitle) in zip(axes, metrics):
        for row in rows:
            if row["medx_acc"] is None or row[key] is None:
                continue
            ax.scatter(
                row["medx_acc"] * 100,
                row[key],
                s=100,
                color=row["color"],
                edgecolors="white",
                linewidths=0.8,
                label=row["label"],
                zorder=3,
            )
            ax.annotate(
                row["label"].replace("-8B", "").replace("-Instruct", "").replace("-23K", ""),
                (row["medx_acc"] * 100, row[key]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=row["color"],
            )
        ax.set_xlabel("MedXpertQA accuracy (%)")
        ax.set_title(title)
        ax.set_xlim(xmin, xmax)
        ax.grid(True, alpha=0.25)
        ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, fontsize=7, color="0.35")
    axes[0].set_ylabel("Metric value")
    fig.suptitle("Phase 1: faithfulness vs. board-level accuracy", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_abstention_comparison(rows: list[dict], out: Path) -> None:
    labels = [r["label"].replace("-8B", "").replace("-Instruct", "").replace("-23K", "") for r in rows]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    abstain = [r["abstain"] for r in rows]
    consistency = [r["consistency"] for r in rows]
    bars1 = ax.bar(x - width / 2, abstain, width, label="Unknown-item abstention", color="#4477AA", alpha=0.85)
    bars2 = ax.bar(x + width / 2, consistency, width, label="Sample-consistency", color="#66AA55", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Phase 2: abstention vs. consistency (MetaMedQA $N{=}200$)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_metacognition_bars(rows: list[dict], out: Path) -> None:
    labels = [r["label"].replace("-8B", "").replace("-Instruct", "").replace("-23K", "") for r in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    abstain = [r["abstain"] for r in rows]
    consistency = [r["consistency"] for r in rows]
    ece = [r["ece"] for r in rows]
    ax.bar(x - width, abstain, width, label="Unknown-item abstention", color="#88CCEE")
    ax.bar(x, consistency, width, label="Sample-consistency", color="#44AA99")
    ax.bar(x + width, ece, width, label="ECE", color="#DDCC77")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Phase 2 metacognition (MetaMedQA $N{=}200$)")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not AGGREGATE.exists():
        raise FileNotFoundError(f"Missing aggregate: {AGGREGATE}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    plot_faithfulness_accuracy(rows, FIG_DIR / "fig_faithfulness_accuracy.pdf")
    plot_abstention_comparison(rows, FIG_DIR / "fig_abstention_comparison.pdf")
    plot_metacognition_bars(rows, FIG_DIR / "fig_metacognition_bars.pdf")
    print(f"Wrote {FIG_DIR / 'fig_faithfulness_accuracy.pdf'}")
    print(f"Wrote {FIG_DIR / 'fig_abstention_comparison.pdf'}")
    print(f"Wrote {FIG_DIR / 'fig_metacognition_bars.pdf'}")


if __name__ == "__main__":
    main()
