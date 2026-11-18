#!/usr/bin/env python3
"""Generate RQ2 figures from training eval.json and rq2_metacognition_aggregate.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "Paper" / "figures"
AGGREGATE = REPO_ROOT / "results" / "rq2_metacognition_aggregate.json"
BASE_EVAL = REPO_ROOT / "checkpoints" / "rq1" / "base" / "eval.json"
DENSE_EVAL = REPO_ROOT / "checkpoints" / "rq1" / "dense_lora" / "eval.json"
MOE_EVAL = REPO_ROOT / "checkpoints" / "rq1" / "moe_lora" / "eval.json"

VARIANTS = [
    ("Qwen base", BASE_EVAL, "#66AA55"),
    ("Dense-LoRA", DENSE_EVAL, "#4477AA"),
    ("MoE-LoRA", MOE_EVAL, "#CC6677"),
]

BENCHMARK_KEYS = [
    ("medqa_usmle_test", "MedQA-USMLE"),
    ("medmcqa_validation", "MedMCQA val"),
    ("medxpertqa_reasoning_n200", "MedXpertQA"),
]


def load_accuracy_rows() -> list[dict]:
    rows: list[dict] = []
    for label, path, color in VARIANTS:
        data = json.loads(path.read_text())
        rows.append(
            {
                "label": label,
                "color": color,
                "medqa": data["medqa_usmle_test"]["accuracy"],
                "medmcqa": data["medmcqa_validation"]["accuracy"],
                "medxpert": data["medxpertqa_reasoning_n200"]["accuracy"],
            }
        )
    return rows


def load_metacognition_rows() -> list[dict]:
    agg = json.loads(AGGREGATE.read_text())
    labels = {
        "Qwen/Qwen2.5-7B-Instruct": ("Qwen base", "#66AA55"),
        "dense": ("Dense-LoRA", "#4477AA"),
        "moe": ("MoE-LoRA", "#CC6677"),
    }
    rows: list[dict] = []
    for s in agg["models"]:
        adapter = s.get("adapter_type", "none")
        if adapter == "none":
            label, color = labels["Qwen/Qwen2.5-7B-Instruct"]
        elif adapter == "dense":
            label, color = labels["dense"]
        else:
            label, color = labels["moe"]
        rows.append(
            {
                "label": label,
                "color": color,
                "meta_acc": s["accuracy"],
                "ece": s["ece"],
                "consistency": s["mean_consistency"],
                "abstain": s["unknown_item_abstention"],
            }
        )
    order = ["Qwen base", "Dense-LoRA", "MoE-LoRA"]
    return sorted(rows, key=lambda r: order.index(r["label"]))


def plot_accuracy_transfer(rows: list[dict], out: Path) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    keys = ["medqa", "medmcqa", "medxpert"]
    titles = ["MedQA-USMLE", "MedMCQA val", "MedXpertQA"]
    colors = ["#88CCEE", "#44AA99", "#DDCC77"]
    for i, (key, title, color) in enumerate(zip(keys, titles, colors)):
        vals = [r[key] * 100 for r in rows]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=title, color=color, alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.8,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 70)
    ax.set_title("RQ2: in-domain gains vs. OOD transfer ($N{=}200$)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_metacognition_tradeoff(rows: list[dict], out: Path) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    abstain = [r["abstain"] for r in rows]
    consistency = [r["consistency"] for r in rows]
    bars1 = ax.bar(x - width / 2, abstain, width, label="Unknown-item abstention", color="#4477AA", alpha=0.85)
    bars2 = ax.bar(x + width / 2, consistency, width, label="Sample-consistency", color="#66AA55", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("RQ2: medical fine-tuning erodes abstention (MetaMedQA $N{=}200$)")
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


def main() -> None:
    if not AGGREGATE.exists():
        raise FileNotFoundError(f"Missing aggregate: {AGGREGATE}")
    for _, path, _ in VARIANTS:
        if not path.exists():
            raise FileNotFoundError(f"Missing eval: {path}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    acc_rows = load_accuracy_rows()
    meta_rows = load_metacognition_rows()
    plot_accuracy_transfer(acc_rows, FIG_DIR / "fig_rq2_accuracy_transfer.pdf")
    plot_metacognition_tradeoff(meta_rows, FIG_DIR / "fig_rq2_metacognition.pdf")
    print(f"Wrote {FIG_DIR / 'fig_rq2_accuracy_transfer.pdf'}")
    print(f"Wrote {FIG_DIR / 'fig_rq2_metacognition.pdf'}")


if __name__ == "__main__":
    main()
