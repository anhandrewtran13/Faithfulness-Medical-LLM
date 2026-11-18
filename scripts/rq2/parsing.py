"""Parse model outputs for MetaMedQA metacognition and MedXpertQA faithfulness eval."""

from __future__ import annotations

import re

THINK_HEADER_PATTERNS = [
    re.compile(
        r"##\s*Thinking\s*\n(.*?)(?=##\s*Final Response|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"###\s*Thinking\s*\n(.*?)(?=##\s*Final Response|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"<\s*think(?:ing)?\s*>(.*?)(?:<\s*/\s*think(?:ing)?\s*>|##\s*Final Response|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
]
FINAL_BLOCK_RE = re.compile(r"##\s*Final Response\s*\n(.*)", re.DOTALL | re.IGNORECASE)

# Most specific first; letter capture is filtered against valid_letters at runtime.
ANSWER_PATTERNS = [
    re.compile(r"(?im)^Answer:\s*\(?([A-J])\)?\s*$"),
    re.compile(r"(?im)^Answer:\s*\(?([A-J])\)?"),
    re.compile(r"[Tt]he answer is[:\s]*\(?([A-J])\)?"),
    re.compile(r"[Ff]inal answer[:\s]*\(?([A-J])\)?"),
    re.compile(r"[Aa]nswer[:\s]*\(?([A-J])\)?"),
    re.compile(r"[Oo]ption\s+\(?([A-J])\)?"),
    re.compile(r"[Cc]hoose\s+\(?([A-J])\)?"),
    re.compile(r"\b(?:option|letter)\s+([A-J])\b"),
    re.compile(r"\(([A-J])\)(?:\.|,|\s|$)"),
    re.compile(r"\b([A-J])\."),
    re.compile(r"(?m)^([A-J])\b"),
]

IDK_TEXT_PATTERNS = [
    re.compile(r"(?im)^Answer:\s*\(?F\)?\s*$"),
    re.compile(r"(?im)^Answer:\s*F(?:\s|$)"),
    re.compile(r"(?i)\bi\s+don'?t\s+know\b"),
    re.compile(r"(?i)\bcannot\s+answer\b"),
    re.compile(r"(?i)\bunable\s+to\s+answer\b"),
]

CONFIDENCE_PATTERNS = [
    re.compile(r"(?im)^Confidence:\s*([1-5])\s*$"),
    re.compile(r"[Cc]onfidence[:\s]*([1-5])"),
    re.compile(r"[Cc]onfidence[:\s]*([1-5])\s*/\s*5"),
    re.compile(r"\b([1-5])\s*/\s*5\b"),
]

TRAILING_ANSWER_RE = re.compile(
    r"(?is)\s*(?:Answer|The answer is|Final answer)\s*[:\s]*\(?([A-J])\)?"
    r"\s*(?:\n\s*Confidence\s*:\s*[1-5].*)?\s*$"
)


def truncate_words(text: str, k_pct: int) -> str:
    """Keep the first k%% of whitespace-delimited tokens."""
    toks = text.split()
    if not toks:
        return ""
    keep = max(1, int(len(toks) * k_pct / 100))
    return " ".join(toks[:keep])


def _strip_trailing_answer_lines(text: str) -> str:
    return TRAILING_ANSWER_RE.sub("", text.strip()).strip()


def parse_trace(text: str, min_reasoning_chars: int = 40) -> str:
    """Extract CoT trace from structured (## Thinking) or headerless model output (e.g. m1)."""
    if not text or not text.strip():
        return ""

    for pattern in THINK_HEADER_PATTERNS:
        match = pattern.search(text)
        if match and match.group(1).strip():
            return match.group(1).strip()

    final = FINAL_BLOCK_RE.search(text)
    prefix = text[: final.start()].strip() if final else text.strip()
    if prefix:
        candidate = _strip_trailing_answer_lines(prefix)
        if len(candidate) >= min_reasoning_chars:
            return candidate
        return ""

    candidate = _strip_trailing_answer_lines(text.strip())
    if len(candidate) >= min_reasoning_chars:
        return candidate
    return ""


def parse_trace_or_fallback(text: str, min_reasoning_chars: int = 40) -> str:
    """Like parse_trace, but keep stripped output when structured extraction returns empty."""
    trace = parse_trace(text, min_reasoning_chars=min_reasoning_chars)
    if trace:
        return trace
    stripped = text.strip()
    if not stripped:
        return ""
    candidate = _strip_trailing_answer_lines(stripped)
    if len(candidate) >= min_reasoning_chars:
        return candidate
    return stripped


def _search_block(text: str) -> str:
    final = FINAL_BLOCK_RE.search(text)
    return final.group(1) if final else text


def _last_valid_letter(text: str, valid: set[str]) -> str:
    """Prefer the last explicit answer marker (avoids early option-letter mentions in CoT)."""
    found = ""
    for pattern in ANSWER_PATTERNS:
        for match in pattern.finditer(text):
            letter = match.group(1)
            if letter in valid:
                found = letter
    return found


def _idk_answer(text: str, idk_letter: str | None) -> str:
    if not idk_letter:
        return ""
    for block in (_search_block(text), text):
        for pattern in IDK_TEXT_PATTERNS:
            if pattern.search(block):
                return idk_letter
    return ""


def parse_answer(text: str, valid_letters: list[str] | None = None) -> str:
    valid = set(valid_letters or list("ABCDEF"))
    idk_letter = "F" if "F" in valid else None

    idk = _idk_answer(text, idk_letter)
    if idk:
        return idk

    search_in = _search_block(text)
    ans = _last_valid_letter(search_in, valid)
    if ans:
        return ans

    ans = _last_valid_letter(text, valid)
    if ans:
        return ans

    # Tail window: m1 often states the final choice at the end of a long headerless trace.
    tail = text[-1500:]
    ans = _last_valid_letter(tail, valid)
    if ans:
        return ans

    stripped = text.strip()
    if stripped and stripped[0] in valid:
        return stripped[0]
    return ""


def parse_truncated_answer(text: str, valid_letters: list[str]) -> str:
    """Parse force-answer continuation after a ## Final Response prefill."""
    valid = set(valid_letters)
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped[0] in valid:
        return stripped[0]
    for prefill in (
        f"## Final Response\nAnswer: {stripped}",
        f"## Final Response\nThe answer is {stripped}",
    ):
        ans = parse_answer(prefill, valid_letters=valid_letters)
        if ans:
            return ans
    return parse_answer(stripped, valid_letters=valid_letters)


def parse_consistency_sample(text: str, valid_letters: list[str]) -> str:
    """Parse a short consistency re-sample (often a single constrained letter token)."""
    valid = set(valid_letters)
    stripped = text.strip()
    if len(stripped) == 1 and stripped.upper() in valid:
        return stripped.upper()
    if len(stripped) <= 2:
        for ch in stripped:
            if ch.upper() in valid:
                return ch.upper()
    return parse_answer(text, valid_letters=valid_letters)


def parse_confidence(text: str, default: int = 3) -> int:
    search_in = _search_block(text)
    for pattern in CONFIDENCE_PATTERNS:
        match = pattern.search(search_in)
        if match:
            return int(match.group(1))
    for pattern in CONFIDENCE_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return default
