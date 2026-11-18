"""Metric helpers for RQ2 metacognition eval."""

from .calibration import confidence_to_unit, expected_calibration_error
from .dmc import BinaryTrial, DMCResult, compute_dmc
from .sample_consistency import sample_consistency, sample_consistency_entropy

__all__ = [
    "BinaryTrial",
    "DMCResult",
    "compute_dmc",
    "sample_consistency",
    "sample_consistency_entropy",
    "expected_calibration_error",
    "confidence_to_unit",
]
