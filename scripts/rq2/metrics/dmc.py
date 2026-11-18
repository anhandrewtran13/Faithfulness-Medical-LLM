"""DMC (Decoupling Metacognition from Cognition) scoring — Wang et al. AAAI 2025."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class BinaryTrial:
    """One binary-choice trial with discrete confidence rating."""

    qid: str
    gold: str  # "A" or "B"
    pred: str  # "A" or "B"
    confidence: int  # 1..k
    correct: bool


@dataclass
class DMCResult:
    d_hat: float
    d_star: float
    dmc_score: float
    n_trials: int
    n_bins: int
    counts: dict[str, int] = field(default_factory=dict)
    fit_loss: float = 0.0
    note: str = ""


def _phi_inv(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(norm.ppf(p))


def _counts_from_trials(trials: list[BinaryTrial], k: int = 5) -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in range(1, k + 1):
        for prefix in ("T_A", "F_A", "T_B", "F_B"):
            counts[f"{prefix}{i}"] = 0
    for trial in trials:
        if trial.pred not in {"A", "B"} or trial.gold not in {"A", "B"}:
            continue
        rating = int(np.clip(trial.confidence, 1, k))
        if trial.pred == "A" and trial.gold == "A":
            key = f"T_A{rating}"
        elif trial.pred == "A" and trial.gold == "B":
            key = f"F_A{rating}"
        elif trial.pred == "B" and trial.gold == "B":
            key = f"T_B{rating}"
        else:
            key = f"F_B{rating}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _observed_matrix(counts: dict[str, int], k: int = 5) -> np.ndarray:
    """Return shape (4, k): rows T_A, F_A, T_B, F_B."""
    mat = np.zeros((4, k), dtype=float)
    row_keys = ("T_A", "F_A", "T_B", "F_B")
    for row_idx, prefix in enumerate(row_keys):
        for bin_idx in range(k):
            mat[row_idx, bin_idx] = counts.get(f"{prefix}{bin_idx + 1}", 0)
    return mat


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    out = mat.copy()
    for i in range(out.shape[0]):
        row_sum = out[i].sum()
        if row_sum > 0:
            out[i] /= row_sum
    return out


def _estimate_cognitive_d(counts: dict[str, int], k: int = 5) -> tuple[float, float]:
    mat = _observed_matrix(counts, k)
    t_a = mat[0].sum()
    f_a = mat[1].sum()
    t_b = mat[2].sum()
    f_b = mat[3].sum()
    t_rate = t_a / max(t_a + f_b, 1.0)
    f_rate = f_b / max(f_a + t_b, 1.0)
    d_hat = _phi_inv(t_rate) - _phi_inv(f_rate)
    c0_hat = -0.5 * (_phi_inv(t_rate) + _phi_inv(f_rate))
    return d_hat, c0_hat


def _lms_predict(d: float, c_boundaries: np.ndarray, k: int) -> np.ndarray:
    """Predict normalized (4, k) matrix from LMS parameters."""
    c0 = c_boundaries[0]
    c_bins_a = np.concatenate(([c0], c_boundaries[1 : k + 1]))
    c_bins_b = np.concatenate(([c0], c_boundaries[k + 1 :]))
    mat = np.zeros((4, k), dtype=float)
    scale = max(abs(d), 1e-3) / 2.0
    mu_a = -scale
    mu_b = scale
    sigma = 1.0
    for i in range(k):
        low_a, high_a = c_bins_a[i], c_bins_a[i + 1]
        low_b, high_b = c_bins_b[i], c_bins_b[i + 1]
        mat[0, i] = norm.cdf(high_a, loc=mu_a, scale=sigma) - norm.cdf(low_a, loc=mu_a, scale=sigma)
        mat[1, i] = norm.cdf(high_a, loc=mu_b, scale=sigma) - norm.cdf(low_a, loc=mu_b, scale=sigma)
        mat[2, i] = norm.cdf(high_b, loc=mu_b, scale=sigma) - norm.cdf(low_b, loc=mu_b, scale=sigma)
        mat[3, i] = norm.cdf(high_b, loc=mu_a, scale=sigma) - norm.cdf(low_b, loc=mu_a, scale=sigma)
    return _normalize_rows(mat)


def _monotonic_penalty(pred: np.ndarray) -> float:
    """Penalize confidence bins whose accuracy does not increase with rating."""
    penalty = 0.0
    for row_a, row_b in ((0, 1), (2, 3)):
        acc = []
        for i in range(pred.shape[1]):
            denom = pred[row_a, i] + pred[row_b, i]
            acc.append(pred[row_a, i] / denom if denom > 0 else 0.0)
        for i in range(1, len(acc)):
            if acc[i] + 1e-6 < acc[i - 1]:
                penalty += (acc[i - 1] - acc[i]) ** 2
    return penalty


def compute_dmc(trials: list[BinaryTrial], k: int = 5) -> DMCResult:
    valid = [t for t in trials if t.pred in {"A", "B"} and t.gold in {"A", "B"}]
    if len(valid) < max(20, k * 4):
        return DMCResult(
            d_hat=0.0,
            d_star=0.0,
            dmc_score=0.0,
            n_trials=len(valid),
            n_bins=k,
            note=f"Insufficient binary trials for DMC fit (need >= {max(20, k * 4)}).",
        )

    counts = _counts_from_trials(valid, k=k)
    observed = _normalize_rows(_observed_matrix(counts, k))
    d_hat, c0_hat = _estimate_cognitive_d(counts, k=k)
    if not np.isfinite(d_hat) or abs(d_hat) < 1e-3:
        d_hat = 0.5

    init_bounds = np.linspace(c0_hat - abs(d_hat), c0_hat + abs(d_hat), k)
    x0 = np.concatenate(([c0_hat], init_bounds, init_bounds))

    def objective(x: np.ndarray) -> float:
        d_star = max(float(x[0]), 1e-3)
        boundaries = x[1:]
        c0 = boundaries[0]
        c_a = boundaries[1 : k + 1]
        c_b = boundaries[k + 1 :]
        if not np.all(np.diff(c_a) > -1e-3) or not np.all(np.diff(c_b) > -1e-3):
            return 1e3
        if abs(c0 - c0_hat) > abs(d_hat) * 2:
            return 1e2 + abs(c0 - c0_hat)
        pred = _lms_predict(d_star, boundaries, k)
        loss = float(np.sum((pred - observed) ** 2))
        loss += 10.0 * _monotonic_penalty(pred)
        return loss

    result = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 5000})
    d_star = max(float(result.x[0]), 1e-3)
    fit_loss = float(result.fun)

    note_parts: list[str] = []
    if d_hat <= 0 or not np.isfinite(d_hat):
        note_parts.append(f"negative or non-finite cognitive d' (d_hat={d_hat:.3f})")
    if fit_loss >= 999.0:
        note_parts.append("LMS metacognitive fit did not converge")

    if note_parts:
        return DMCResult(
            d_hat=float(d_hat),
            d_star=float(d_star),
            dmc_score=0.0,
            n_trials=len(valid),
            n_bins=k,
            counts=counts,
            fit_loss=fit_loss,
            note="Unreliable DMC: " + "; ".join(note_parts) + ".",
        )

    dmc_score = float(d_star / max(abs(d_hat), 1e-3))
    return DMCResult(
        d_hat=float(d_hat),
        d_star=float(d_star),
        dmc_score=dmc_score,
        n_trials=len(valid),
        n_bins=k,
        counts=counts,
        fit_loss=fit_loss,
    )
