"""ELBO metrology — Phase 3 (tasks 3.1–3.7, design §5).

The instrument: per-language NELBO scored with stratified timesteps and
common random numbers across language conditions (``scoring``), per-document
mean-and-spread aggregation, and the versioned bound-calibration table with
its single application point (``calibration``).
"""

from .calibration import (
    CALIBRATION_VERSION,
    FAMILY,
    CalibrationTable,
    calibrate_bits,
    derive_report_only,
    family_of,
    rank_languages,
)
from .scoring import (
    CONDITION_UNCOND,
    DocumentScores,
    ScoreSettings,
    per_document,
    score_conditions,
)

__all__ = [
    "CALIBRATION_VERSION",
    "CONDITION_UNCOND",
    "FAMILY",
    "CalibrationTable",
    "DocumentScores",
    "ScoreSettings",
    "calibrate_bits",
    "derive_report_only",
    "family_of",
    "per_document",
    "rank_languages",
    "score_conditions",
]
