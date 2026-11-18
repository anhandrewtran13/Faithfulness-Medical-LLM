#!/usr/bin/env python3
"""Generate RQ3 abstention-recovery figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "Paper" / "figures"
RQ2_AGG = REPO_ROOT / "results" / "rq2_metacognition_aggregate.json"
RQ3_AGG = REPO_ROOT / "results" / "rq3_metacognition_aggregate.json"
RQ3_DENSE_EVAL = REPO_ROOT / "checkpoints" / "rq3" / "dense_lora" / "eval.json"
RQ3_MOE_EVAL = REPO_ROOT / "checkpoints" / "rq3" / "moe_lora" / "eval.json"
RQ1_DENSE_EVAL = REPO_ROOT / "checkpoints" / "rq1" / "dense_lora" / "eval.json"


def _model_row(agg: dict, adapter_type: str) -> dict:
    for s in agg["models"]:
        if s.get("adapter_type", "none") == adapter_type:
            return s
    raise KeyError(adapter_type)


def plot_abstention_recovery(out: Path) -> None:
    rq2 = json.loads(RQ2_AGG.read_text())
    rq3 = json.loads(RQ3_AGG.read_text())
    rows = [
        ("Qwen base", _model_row(rq2, "none"), "#66AA55"),
        ("RQ2 dense", _model_row(rq2, "dense"), "#CC6677"),
        ("RQ3 dense", _model_row(rq3, "dense"), "#4477AA"),
        ("RQ3 MoE", _model_row(rq3, "moe"), "#AA4499"),
    ]
    labels = [r[0] for r in rows]
    abstain = [r[1]["unknown_item_abstention"] for r in rows]
    meta_acc = [r[1]["accuracy"] for r in rows]
    colors = [r[2] for r in rows]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    b1 = ax.bar(x - width / 2, abstain, width, label="Unknown-item abstention", color=colors, alpha=0.75)
    b2 = ax.bar(x + width / 2, meta_acc, width, label="MetaMedQA accuracy", color=colors, alpha=0.45, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("RQ3: abstention recovery after mixed fine-tuning ($N{=}200$)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_rq3_accuracy(out: Path) -> None:
    rq1_dense = json.loads(RQ1_DENSE_EVAL.read_text())
    rq3_dense = json.loads(RQ3_DENSE_EVAL.read_text())
    rq3_moe = json.loads(RQ3_MOE_EVAL.read_text())
    rows = [
        ("RQ2 dense", rq1_dense, "#CC6677"),
        ("RQ3 dense", rq3_dense, "#4477AA"),
        ("RQ3 MoE", rq3_moe, "#AA4499"),
    ]
    labels = [r[0] for r in rows]
    keys = ["medqa_usmle_test", "medmcqa_validation", "medxpertqa_reasoning_n200"]
    titles = ["MedQA-USMLE", "MedMCQA val", "MedXpertQA"]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    bar_colors = ["#88CCEE", "#44AA99", "#DDCC77"]
    for i, (key, title, color) in enumerate(zip(keys, titles, bar_colors)):
        vals = [r[1][key]["accuracy"] * 100 for r in rows]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=title, color=color, alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 80)
    ax.set_title("RQ3: in-domain accuracy after abstention recovery ($N{=}200$)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for path in (RQ2_AGG, RQ3_AGG, RQ3_DENSE_EVAL, RQ3_MOE_EVAL, RQ1_DENSE_EVAL):
        if not path.exists():
            raise FileNotFoundError(path)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_abstention_recovery(FIG_DIR / "fig_rq3_abstention_recovery.pdf")
    plot_rq3_accuracy(FIG_DIR / "fig_rq3_accuracy.pdf")
    print(f"Wrote {FIG_DIR / 'fig_rq3_abstention_recovery.pdf'}")
    print(f"Wrote {FIG_DIR / 'fig_rq3_accuracy.pdf'}")


if __name__ == "__main__":
    main()
