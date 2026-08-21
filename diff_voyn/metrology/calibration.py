"""Per-language bound calibration — task 3.4 (design §5b) — the *one* place
where calibration offsets are applied.

A calibration table (``DATA_ROOT/calibration/calibration_<version>.json``,
written by ``scripts/calibrate.py``) stores, per language,

    offset_bits = NELBO(own-language condition) − NLL_AR

measured on the full tiled held-out split. The calibrated score of a
candidate plaintext scored under language ``L`` is

    NELBO_calibrated = NELBO − offset_L

Two sign conventions meet here: the stored table is in the §5b "bound gap"
sign (positive = bound looser than the AR reference), while the evaluator
hook (:meth:`diff_voyn.heads.evaluator.EvaluatorBase.calibrated_bits_per_char`)
has always carried *additive* offsets (the n-gram evaluator adds
−heldout-bits). :func:`calibrate_bits` is the single arithmetic both paths
call; :meth:`CalibrationTable.additive_offsets` converts the table into the
hook's convention. Nothing else in the codebase may add or subtract an
offset.

A table carries two offset sets: ``measured`` (the §5b.3 estimate, always
stored with its s.e.m.) and ``applied`` (what the ranking subtracts). The
default policy is ``apply`` (applied = measured). The policy
``report-only`` zeroes the applied offsets and keeps the measured ones as
the *systematic uncertainty* of a ranking margin — adopted in Phase 3 after
the synthetic suite (3.6) showed the measured offsets are coarser than the
same-text conditioning margin between close languages (see
``docs/phase3_status.md``). Either way the arithmetic below is the only
place offsets enter a score.

``CALIBRATION_VERSION`` names the table the instrument applies. Bump it
deliberately (a new phase's table), never edit a table in place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..data.loader import LANG_TO_INDEX

CALIBRATION_VERSION = "v1"

# Task 3.7: language-family granularity. Latin and Italian are Italic/Romance
# (the close pair, the expected dominant confusion); German is Germanic.
FAMILY: dict[str, str] = {
    "latin": "romance",
    "italian": "romance",
    "german": "germanic",
}


def family_of(language: str) -> str:
    return FAMILY[language]


def calibrate_bits(
    raw_bits: float, language: str, additive_offsets_bits: dict[str, float]
) -> float:
    """THE calibration arithmetic: ``raw + additive_offset[language]``
    (missing language ⇒ uncalibrated). Called by the evaluator hook and by
    every ranking in the metrology scripts — nowhere else."""
    return float(raw_bits) + float(additive_offsets_bits.get(language, 0.0))


def rank_languages(
    raw_bits_by_language: dict[str, float], additive_offsets_bits: dict[str, float]
) -> list[tuple[str, float]]:
    """Ascending calibrated bits/char: ``[(language, calibrated_bits), ...]``;
    the first entry is the ranking's winner."""
    scored = [
        (lang, calibrate_bits(b, lang, additive_offsets_bits))
        for lang, b in raw_bits_by_language.items()
    ]
    return sorted(scored, key=lambda kv: kv[1])


@dataclass(frozen=True)
class CalibrationTable:
    version: str
    phase: str
    backbone_path: str
    backbone_step: int
    reference: str
    offsets_bits: dict[str, float]  # measured NELBO − NLL_AR (stored sign)
    offsets_sem: dict[str, float]
    nelbo_bits: dict[str, float]
    nll_ar_bits: dict[str, float]
    created_utc: str
    path: str
    policy: str = "apply"  # "apply" | "report-only"
    applied_offsets_bits: dict[str, float] | None = None

    @staticmethod
    def file_for(version: str, root: Path | None = None) -> Path:
        if root is None:
            from ..ciphers.external import data_root

            root = data_root()
        return root / "calibration" / f"calibration_{version}.json"

    @classmethod
    def load(cls, version: str | None = None, root: Path | None = None):
        version = version or CALIBRATION_VERSION
        path = cls.file_for(version, root)
        d = json.loads(path.read_text())
        langs = d["languages"]
        missing = set(LANG_TO_INDEX) - set(langs)
        if missing:
            raise ValueError(f"calibration {version} lacks languages {missing}")
        policy = d.get("policy", "apply")
        if policy not in ("apply", "report-only"):
            raise ValueError(f"unknown calibration policy {policy!r}")
        measured = {l: float(v["offset_bits"]) for l, v in langs.items()}
        if "applied_offsets_bits" in d:
            applied = {l: float(v) for l, v in d["applied_offsets_bits"].items()}
        else:
            applied = dict(measured) if policy == "apply" else {l: 0.0 for l in langs}
        return cls(
            policy=policy,
            applied_offsets_bits=applied,
            version=d["calibration_version"],
            phase=d.get("phase", "?"),
            backbone_path=d["backbone"]["path"],
            backbone_step=int(d["backbone"]["step"]),
            reference=d.get("reference", "?"),
            offsets_bits={l: float(v["offset_bits"]) for l, v in langs.items()},
            offsets_sem={l: float(v.get("offset_sem", 0.0)) for l, v in langs.items()},
            nelbo_bits={l: float(v["nelbo_bits"]) for l, v in langs.items()},
            nll_ar_bits={l: float(v["nll_ar_bits"]) for l, v in langs.items()},
            created_utc=d.get("created_utc", "?"),
            path=str(path),
        )

    def additive_offsets(self) -> dict[str, float]:
        """The *applied* offsets in the evaluator hook's additive convention
        (``NELBO − offset`` ⇔ add ``−offset``); all zeros under the
        ``report-only`` policy."""
        applied = self.applied_offsets_bits or self.offsets_bits
        return {l: -v for l, v in applied.items()}

    def margin_uncertainty_bits(self, lang_a: str, lang_b: str) -> float:
        """Systematic uncertainty of a between-language margin implied by
        the *measured* offsets: |measured_a − measured_b| if those offsets
        are not applied (report-only), else the quadrature sum of their
        s.e.m. — the number a ranking margin must exceed to be resolvable at
        the calibration's precision."""
        if self.policy == "report-only":
            return abs(self.offsets_bits[lang_a] - self.offsets_bits[lang_b])
        return float(
            (self.offsets_sem[lang_a] ** 2 + self.offsets_sem[lang_b] ** 2) ** 0.5
        )

    def apply(self, raw_bits: float, language: str) -> float:
        return calibrate_bits(raw_bits, language, self.additive_offsets())

    def rank(self, raw_bits_by_language: dict[str, float]) -> list[tuple[str, float]]:
        return rank_languages(raw_bits_by_language, self.additive_offsets())

    @property
    def spread_bits(self) -> float:
        """Spread of the *measured* offsets across languages."""
        v = list(self.offsets_bits.values())
        return max(v) - min(v)

    def summary(self) -> dict:
        return {
            "version": self.version,
            "phase": self.phase,
            "policy": self.policy,
            "backbone": {"path": self.backbone_path, "step": self.backbone_step},
            "reference": self.reference,
            "offsets_bits": self.offsets_bits,
            "offsets_sem": self.offsets_sem,
            "applied_offsets_bits": self.applied_offsets_bits,
            "spread_bits": self.spread_bits,
            "path": self.path,
        }


def derive_report_only(
    source_version: str, new_version: str, root: Path | None = None, note: str = ""
) -> Path:
    """Write ``calibration_<new_version>.json`` = the source table with the
    ``report-only`` policy (applied offsets zero, measured offsets kept).
    The source file is never modified (tables are immutable once written)."""
    src = CalibrationTable.file_for(source_version, root)
    d = json.loads(src.read_text())
    d["calibration_version"] = new_version
    d["derived_from"] = source_version
    d["policy"] = "report-only"
    d["applied_offsets_bits"] = {l: 0.0 for l in d["languages"]}
    d["policy_note"] = note or (
        "measured offsets are reported as the systematic uncertainty of a "
        "ranking margin and are NOT subtracted from scores (Phase 3, task 3.6)"
    )
    out = CalibrationTable.file_for(new_version, root)
    if out.exists():
        raise FileExistsError(f"{out} exists — tables are immutable; pick a new version")
    out.write_text(json.dumps(d, indent=2))
    return out
