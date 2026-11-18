"""MetaMedQA manifest loading for RQ1/RQ2 metacognition eval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

UNKNOWN_KINDS = frozenset({"Bad", "Question modified", "Answer removed", "Answer modified"})
ABSTAIN_LETTERS = frozenset({"E", "F"})
IDK_PHRASES = ("don't know", "cannot answer", "can not answer")


@dataclass
class MetaMedQAItem:
    qid: str
    question: str
    options: dict[str, str]
    gold: str
    kind: str
    meta_info: str = ""

    @property
    def option_letters(self) -> list[str]:
        return sorted(self.options.keys())

    @property
    def idk_letter(self) -> str:
        """Letter for 'I don't know or cannot answer' (usually F)."""
        for letter, text in self.options.items():
            lower = text.lower()
            if any(phrase in lower for phrase in IDK_PHRASES):
                return letter
        return "F"

    @property
    def none_of_above_letter(self) -> str:
        """Letter for 'None of the above' (usually E)."""
        for letter, text in self.options.items():
            if "none of the above" in text.lower():
                return letter
        return "E"

    @property
    def expects_abstention(self) -> bool:
        return self.gold in ABSTAIN_LETTERS or self.kind in UNKNOWN_KINDS or self.kind == "Fictional"

    @property
    def expects_idk(self) -> bool:
        """Gold is the I-don't-know option (MetaMedQA unknown recall)."""
        return self.gold == self.idk_letter

    @property
    def expects_none_of_above(self) -> bool:
        return self.gold == self.none_of_above_letter

    @property
    def clinical_distractors(self) -> list[str]:
        """Letters A–D excluding gold (for DMC binary pairing)."""
        return [letter for letter in ("A", "B", "C", "D") if letter in self.options and letter != self.gold]


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p.resolve()
    candidate = REPO_ROOT / p
    if candidate.exists():
        return candidate
    return p


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = resolve_repo_path(path)
    with open(manifest_path) as f:
        return json.load(f)


def items_from_manifest(manifest: dict[str, Any]) -> list[MetaMedQAItem]:
    items: list[MetaMedQAItem] = []
    for row in manifest["items"]:
        items.append(
            MetaMedQAItem(
                qid=str(row["qid"]),
                question=row["question"],
                options=dict(row["options"]),
                gold=row["gold"],
                kind=row.get("kind", "Original"),
                meta_info=row.get("meta_info", ""),
            )
        )
    return items


def select_subset(
    items: list[MetaMedQAItem],
    subset: str,
    manifest: dict[str, Any],
) -> list[MetaMedQAItem]:
    if subset == "all":
        return items
    if subset == "pilot":
        pilot_qids = set(manifest.get("pilot_qids", []))
        if not pilot_qids:
            raise ValueError("Manifest has no pilot_qids; run freeze_metamedqa_manifest.py first.")
        return [item for item in items if item.qid in pilot_qids]
    if subset == "eval":
        return items
    raise ValueError(f"Unknown subset: {subset}")
