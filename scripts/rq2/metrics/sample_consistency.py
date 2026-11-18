"""Sample-consistency proxy — Savage et al. style answer agreement across samples."""

from __future__ import annotations

import math
from collections import Counter


def sample_consistency(answers: list[str]) -> float:
    """
    Fraction of samples agreeing with the majority answer.
    Returns 0.0 if no valid parsed answers.
    """
    valid = [answer for answer in answers if answer]
    if not valid:
        return 0.0
    counts = Counter(valid)
    majority = counts.most_common(1)[0][1]
    return majority / len(valid)


def sample_consistency_entropy(answers: list[str]) -> float:
    """Normalized entropy of the answer distribution (0 = deterministic, 1 = uniform)."""
    valid = [answer for answer in answers if answer]
    if len(valid) <= 1:
        return 0.0
    counts = Counter(valid)
    n = len(valid)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log(p)
    max_entropy = math.log(len(counts))
    if max_entropy <= 0:
        return 0.0
    return entropy / max_entropy
