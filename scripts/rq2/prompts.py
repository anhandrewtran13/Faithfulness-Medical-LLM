"""Prompt templates for MetaMedQA metacognition eval.

Prompt styles follow Griot et al. (MetaMedQA, Nature Communications 2025).
The paper reports that explicitly warning models about malformed, fictional,
and incomplete questions significantly improves unknown recall (answer F).
See: maximegmd/MetaMedQA-benchmark README (role-full, role-explicit-unknown-missing).
"""

from __future__ import annotations

from data import MetaMedQAItem

PROMPT_STYLES = ("baseline", "metamedqa_warn", "metamedqa_full", "metamedqa_explicit_unknown")


def format_options(options: dict[str, str]) -> str:
    lines = []
    for letter in sorted(options.keys()):
        lines.append(f"({letter}) {options[letter]}")
    return "\n".join(lines)


# --- MetaMedQA paper prompt fragments (system preamble) ---

ROLE_BASE = (
    "You are a medical assistant answering multiple-choice questions to help a doctor.\n"
)

ROLE_WARN = (
    "You are a medical assistant answering multiple-choice questions to help a doctor.\n"
    "The questions may be tricky, so be careful.\n"
)

ROLE_EXPLICIT_UNKNOWN = (
    "You are a medical assistant answering multiple-choice questions to help a doctor.\n"
    "The questions may ask about knowledge you do not possess, be incomplete, "
    "or be impossible for you to answer safely.\n"
)

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

# Kind-specific screening (does not reveal the keyed answer).
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

_STYLE_PREAMBLE = {
    "baseline": "",
    "metamedqa_warn": ROLE_WARN,
    "metamedqa_full": ROLE_FULL,
    "metamedqa_explicit_unknown": ROLE_EXPLICIT_UNKNOWN,
}


def system_preamble(style: str) -> str:
    return _STYLE_PREAMBLE.get(style, ROLE_FULL)


def metamedqa_prompt(item: MetaMedQAItem, style: str = "metamedqa_full") -> str:
    """Build the user prompt for a MetaMedQA item."""
    parts: list[str] = []

    preamble = system_preamble(style)
    if preamble:
        parts.append(preamble)

    if style != "baseline":
        parts.append(OPTION_SEMANTICS)
        parts.append(ANSWERABILITY_CHECKLIST)

    kind_note = KIND_SCREENING.get(item.kind, "")
    if kind_note:
        parts.append(kind_note)

    parts.append(RESPONSE_FORMAT)
    parts.append(item.question)
    parts.append(format_options(item.options))

    return "\n\n".join(p.strip() for p in parts if p.strip())


CONSISTENCY_INSTRUCTION = (
    "{preamble}"
    "Answer the following multiple-choice medical question.\n"
    "Give ONLY the final answer letter — no reasoning trace.\n"
    "Use exactly this format (the assistant response will continue after 'Answer:'):\n"
    "{prefill_hint}"
    "where X is one of: {option_list}.\n\n"
    "{question}\n\n{options}"
)


def consistency_prompt(
    item: MetaMedQAItem,
    style: str = "metamedqa_full",
    prefill: str = "## Final Response\nAnswer: ",
) -> str:
    preamble = system_preamble(style)
    if style != "baseline":
        preamble = preamble + "\n" + OPTION_SEMANTICS if preamble else OPTION_SEMANTICS
    if "## Final Response" in prefill:
        prefill_hint = "## Final Response\nAnswer: X\n"
    else:
        prefill_hint = "Answer: X\n"
    return CONSISTENCY_INSTRUCTION.format(
        preamble=preamble + "\n\n" if preamble else "",
        prefill_hint=prefill_hint,
        option_list=", ".join(item.option_letters),
        question=item.question,
        options=format_options(item.options),
    )


BINARY_DMC_INSTRUCTION = (
    "You are answering a binary multiple-choice medical question.\n"
    "Think briefly inside a '## Thinking' section.\n"
    "Then give a '## Final Response' section with:\n"
    "1) Your answer: A or B only\n"
    "2) Confidence from 1 (not confident) to 5 (absolutely sure)\n\n"
    "Use exactly this format in ## Final Response:\n"
    "Answer: A\n"
    "Confidence: N\n\n"
    "{question}\n\n"
    "(A) {option_a}\n"
    "(B) {option_b}\n\n"
    "Respond with Answer: A or Answer: B, then Confidence: 1-5."
)


def binary_dmc_prompt(
    stem: str,
    option_a_letter: str,
    option_a_text: str,
    option_b_letter: str,
    option_b_text: str,
) -> str:
    return BINARY_DMC_INSTRUCTION.format(
        question=stem,
        option_a=option_a_text,
        option_b=option_b_text,
    )
