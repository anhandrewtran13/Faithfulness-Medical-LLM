"""Prompt templates for abstention-aware RQ1 training."""

from __future__ import annotations

from dataclasses import dataclass

ABSTAIN_LETTERS = frozenset({"E", "F"})
UNKNOWN_KINDS = frozenset({"Bad", "Question modified", "Answer removed", "Answer modified", "Fictional"})

ROLE_FULL = (
    "You are a medical assistant answering multiple-choice questions to help a doctor.\n"
    "Some questions are intentionally designed to test your metacognition. They may:\n"
    "- Contain fictional medical entities (diseases, drugs, organs, or lab tests that do not exist)\n"
    "- Be incomplete (e.g., reference an image, ECG, slide, or result that is not provided)\n"
    "- Have no correct option among (A)–(D), or have been modified so the true answer is absent\n"
    "Before selecting an answer, decide whether the question is answerable with the information given.\n"
)

OPTION_SEMANTICS = (
    "About the answer choices:\n"
    "- (A)–(D): standard clinical options when the question is well-posed\n"
    "- (E) None of the above: the question is answerable, but none of A–D is correct\n"
    "- (F) I don't know or cannot answer: the question is malformed, incomplete, fictional, "
    "or you lack the information needed to answer safely — do NOT guess\n"
)

ANSWERABILITY_CHECKLIST = (
    "Before answering, check:\n"
    "1. Are all medical terms real (not fictional)?\n"
    "2. Is required information present (no missing figures, labs, or images)?\n"
    "3. Is the question clinically coherent?\n"
    "4. Can one of A–D be justified? If not, consider (E) or (F) as appropriate.\n"
)

RESPONSE_FORMAT = (
    "Think step by step inside a '## Thinking' section.\n"
    "Then give a '## Final Response' section with EXACTLY these two lines:\n"
    "Answer: X\n"
    "Confidence: N\n"
    "where X is one letter A–F and N is an integer from 1 (not confident) to 5 (absolutely sure).\n"
)

KIND_SCREENING: dict[str, str] = {
    "Fictional": (
        "Item note: this vignette may describe a fictional organ, disease, drug, or lab test. "
        "If key entities are not real, answer (F).\n"
    ),
    "Bad": (
        "Item note: this question may rely on missing media (image, ECG, slide) or lack information "
        "needed to answer. If critical data are absent, answer (F).\n"
    ),
    "Question modified": (
        "Item note: the stem may be incomplete, nonsensical, or missing context. "
        "If you cannot answer safely, answer (F).\n"
    ),
    "Answer modified": (
        "Item note: the listed options may not include the true clinical answer even if the "
        "vignette looks standard. If none of A–D is defensible, answer (E).\n"
    ),
    "Answer removed": (
        "Item note: the correct clinical option may have been removed from the list. "
        "If none of A–D applies, answer (E) or (F) as appropriate.\n"
    ),
}

THINKING_BY_KIND: dict[str, str] = {
    "Fictional": (
        "The vignette references fictional medical entities (disease, drug, organ, or test). "
        "These are not real, so the question is not answerable with clinical knowledge."
    ),
    "Bad": (
        "Critical information needed to answer safely (image, ECG, slide, or lab result) "
        "is missing from the vignette."
    ),
    "Question modified": (
        "The question stem is incomplete or nonsensical and cannot be answered safely."
    ),
    "Answer modified": (
        "The vignette looks standard, but none of options A–D is clinically defensible. "
        "The true answer is absent from the list."
    ),
    "Answer removed": (
        "The correct clinical option appears to have been removed from the answer list."
    ),
    "Original": (
        "The question is well-posed with real medical entities and sufficient information."
    ),
}


@dataclass
class _PromptItem:
    question: str
    options: dict[str, str]
    kind: str


def _format_options(options: dict[str, str]) -> str:
    return "\n".join(f"({letter}) {options[letter]}" for letter in sorted(options.keys()))


def metamedqa_user_prompt(
    question: str,
    options: dict[str, str],
    kind: str = "Original",
    style: str = "metamedqa_full",
) -> str:
    del style  # only metamedqa_full is supported for training
    item = _PromptItem(question=question, options=options, kind=kind)
    parts = [ROLE_FULL, OPTION_SEMANTICS, ANSWERABILITY_CHECKLIST]
    kind_note = KIND_SCREENING.get(item.kind, "")
    if kind_note:
        parts.append(kind_note)
    parts.extend([RESPONSE_FORMAT, item.question, _format_options(item.options)])
    return "\n\n".join(p.strip() for p in parts if p.strip())


def metamedqa_assistant_response(
    answer_letter: str,
    confidence: int,
    kind: str = "Original",
) -> str:
    thinking = THINKING_BY_KIND.get(kind, THINKING_BY_KIND["Original"])
    return (
        f"## Thinking\n{thinking}\n"
        f"## Final Response\n"
        f"Answer: {answer_letter.upper()}\n"
        f"Confidence: {confidence}"
    )


def default_confidence(answer_letter: str, kind: str) -> int:
    if answer_letter == "F":
        return 5
    if answer_letter == "E":
        return 4
    if kind in UNKNOWN_KINDS:
        return 4
    return 4


def is_abstention_example(kind: str, answer_letter: str) -> bool:
    return kind in UNKNOWN_KINDS or answer_letter in ABSTAIN_LETTERS
