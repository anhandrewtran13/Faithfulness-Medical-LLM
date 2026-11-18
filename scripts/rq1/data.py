"""Shared data utilities for RQ1 training and evaluation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training_prompts import (
    default_confidence,
    metamedqa_assistant_response,
    metamedqa_user_prompt,
)

LETTERS = "ABCDEFGHIJ"
PROMPT_STYLES = ("mcq", "metamedqa_full")


def format_mcq_question(question: str, options: dict[str, str]) -> str:
    letters = sorted(options.keys())
    choices = "\n".join(f"({letter}) {options[letter]}" for letter in letters)
    return f"{question.strip()}\nAnswer Choices: {choices}"


def format_assistant_answer(answer_letter: str) -> str:
    return f"The answer is ({answer_letter.upper()})."


@dataclass
class TrainingExample:
    id: str
    source: str
    question: str
    options: dict[str, str]
    answer_letter: str
    kind: str = ""
    prompt_style: str = "mcq"
    confidence: int | None = None

    @property
    def user_text(self) -> str:
        if self.prompt_style == "metamedqa_full":
            return metamedqa_user_prompt(
                question=self.question,
                options=self.options,
                kind=self.kind or "Original",
                style=self.prompt_style,
            )
        return (
            "Please answer the following multiple-choice question.\n\n"
            + format_mcq_question(self.question, self.options)
        )

    @property
    def assistant_text(self) -> str:
        if self.prompt_style == "metamedqa_full":
            confidence = self.confidence
            if confidence is None:
                confidence = default_confidence(self.answer_letter, self.kind or "Original")
            return metamedqa_assistant_response(
                answer_letter=self.answer_letter,
                confidence=confidence,
                kind=self.kind or "Original",
            )
        return format_assistant_answer(self.answer_letter)

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "question": self.question,
            "options": self.options,
            "answer_letter": self.answer_letter,
        }
        if self.kind:
            row["kind"] = self.kind
        if self.prompt_style != "mcq":
            row["prompt_style"] = self.prompt_style
        if self.confidence is not None:
            row["confidence"] = self.confidence
        return row


def medqa_row_to_example(row: dict[str, Any], idx: int) -> TrainingExample:
    options = {letter: row["options"][letter] for letter in sorted(row["options"])}
    return TrainingExample(
        id=f"medqa-usmle:train:{idx}",
        source="medqa-usmle",
        question=row["question"],
        options=options,
        answer_letter=row["answer_idx"].upper(),
    )


def medmcqa_row_to_example(row: dict[str, Any]) -> TrainingExample | None:
    if row.get("choice_type") != "single":
        return None
    cop = row.get("cop")
    if cop is None or cop < 0 or cop > 3:
        return None
    letter = LETTERS[int(cop)]
    options = {
        "A": row["opa"],
        "B": row["opb"],
        "C": row["opc"],
        "D": row["opd"],
    }
    return TrainingExample(
        id=f"medmcqa:train:{row['id']}",
        source="medmcqa",
        question=row["question"],
        options=options,
        answer_letter=letter,
    )


def metamedqa_row_to_example(row: dict[str, Any], qid: str) -> TrainingExample:
    kind = row.get("kind", "Original")
    answer = row["answer_idx"].upper()
    return TrainingExample(
        id=f"metamedqa:train:{qid}",
        source="metamedqa",
        question=row["question"],
        options=dict(row["options"]),
        answer_letter=answer,
        kind=kind,
        prompt_style="metamedqa_full",
        confidence=default_confidence(answer, kind),
    )


def example_from_dict(row: dict[str, Any]) -> TrainingExample:
    return TrainingExample(
        id=row["id"],
        source=row["source"],
        question=row["question"],
        options=row["options"],
        answer_letter=row["answer_letter"],
        kind=row.get("kind", ""),
        prompt_style=row.get("prompt_style", "mcq"),
        confidence=row.get("confidence"),
    )


def load_training_manifest(path: str | Path) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    with open(path) as f:
        for line in f:
            examples.append(example_from_dict(json.loads(line)))
    return examples


def repeat_examples(
    examples: list[TrainingExample],
    repeat: int,
) -> list[TrainingExample]:
    if repeat <= 1:
        return examples
    out: list[TrainingExample] = []
    for ex in examples:
        for rep in range(repeat):
            if rep == 0:
                out.append(ex)
            else:
                out.append(
                    TrainingExample(
                        id=f"{ex.id}:rep{rep}",
                        source=ex.source,
                        question=ex.question,
                        options=ex.options,
                        answer_letter=ex.answer_letter,
                        kind=ex.kind,
                        prompt_style=ex.prompt_style,
                        confidence=ex.confidence,
                    )
                )
    return out


def subsample_examples(
    examples: list[TrainingExample],
    max_samples: int,
    seed: int,
) -> list[TrainingExample]:
    if max_samples <= 0 or max_samples >= len(examples):
        return examples
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    keep = sorted(indices[:max_samples])
    return [examples[i] for i in keep]
