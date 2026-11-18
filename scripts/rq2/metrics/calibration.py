"""Calibration metrics for metacognition eval."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 5,
) -> float:
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (conf >= bins[i]) & (conf < bins[i + 1])
        else:
            mask = (conf >= bins[i]) & (conf <= bins[i + 1])
        if not mask.any():
            continue
        bin_acc = acc[mask].mean()
        bin_conf = conf[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return float(ece)


def confidence_to_unit(confidence_1_to_5: int) -> float:
    return float(np.clip(confidence_1_to_5, 1, 5) - 1) / 4.0
