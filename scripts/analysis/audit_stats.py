"""Wilson confidence intervals and paired significance tests for audit metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProportionCI:
    successes: int
    trials: int
    rate: float
    lower: float
    upper: float

    def fmt_rate(self, ndigits: int = 3) -> str:
        return f"{self.rate:.{ndigits}f}"

    def fmt_ci(self, ndigits: int = 3) -> str:
        return f"{self.fmt_rate(ndigits)} [{self.lower:.{ndigits}f}, {self.upper:.{ndigits}f}]"

    def fmt_latex(self, ndigits: int = 3) -> str:
        """Compact LaTeX-friendly interval, e.g. 0.535 [0.47--0.60]."""
        lo = f"{self.lower:.{ndigits}f}".rstrip("0").rstrip(".")
        hi = f"{self.upper:.{ndigits}f}".rstrip("0").rstrip(".")
        return f"{self.fmt_rate(ndigits)} [{lo}--{hi}]"


def wilson_ci(successes: int, trials: int, alpha: float = 0.05) -> ProportionCI:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return ProportionCI(0, 0, 0.0, 0.0, 0.0)
    z = 1.959963984540054 if alpha == 0.05 else abs(
        math.sqrt(2) * math.erfinv(1 - alpha)
    )
    p_hat = successes / trials
    denom = 1 + z * z / trials
    center = (p_hat + z * z / (2 * trials)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / trials + z * z / (4 * trials * trials))
    return ProportionCI(
        successes=successes,
        trials=trials,
        rate=p_hat,
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
    )


@dataclass(frozen=True)
class McNemarResult:
    n_discordant: int
    b: int  # A correct, B incorrect
    c: int  # A incorrect, B correct
    chi2: float
    p_value: float

    @property
    def significant_005(self) -> bool:
        return self.p_value < 0.05


def mcnemar_test(b: int, c: int) -> McNemarResult:
    """McNemar test with continuity correction (chi-square approximation)."""
    n = b + c
    if n == 0:
        return McNemarResult(0, b, c, 0.0, 1.0)
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2))
    return McNemarResult(n_discordant=n, b=b, c=c, chi2=chi2, p_value=p)


def paired_mcnemar(
  outcomes_a: dict[str, bool],
  outcomes_b: dict[str, bool],
) -> McNemarResult:
    """Paired McNemar on shared question ids."""
    shared = sorted(set(outcomes_a) & set(outcomes_b))
    b = sum(1 for qid in shared if outcomes_a[qid] and not outcomes_b[qid])
    c = sum(1 for qid in shared if not outcomes_a[qid] and outcomes_b[qid])
    return mcnemar_test(b, c)
