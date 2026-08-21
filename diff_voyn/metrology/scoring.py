"""Scoring harness — task 3.1 (design §5a).

Per-window bits/char of a frozen backbone under several *conditions*
(conditioning languages, or the unconditional NULL language), estimated with
stratified timestep sampling (``n_strata × samples_per_stratum``) and
**common random numbers across conditions**: every condition of a chunk of
windows is scored with the identical masking realizations (same t values,
same masked positions). The language ranking consumes score *differences*,
and CRN removes the shared Monte-Carlo noise from exactly those differences
— the acceptance experiment of task 3.1 (``scripts/crn_check.py``) measures
how much.

Two input shapes are supported:

- one text per window shared by all conditions (clean-text LID, calibration):
  ``ids`` is a single ``[N, L]`` array;
- one text per (window, condition) — the trial-decipherment case, where each
  language hypothesis yields its own candidate plaintext of the same length:
  ``ids`` is ``{condition: [N, L]}``. CRN then means the *same masks* are
  applied to every hypothesis's text.

Per-document aggregation (task 3.3): :func:`per_document` turns per-window
scores plus a window→document index into mean / std / s.e.m. per document,
so every ranking can carry an uncertainty statement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch

from ..data.loader import LANG_TO_INDEX, NULL_LANG_INDEX
from ..infra.nelbo import per_window_nelbo_bits

CONDITION_UNCOND = "uncond"
DEFAULT_CONDITIONS: tuple[str, ...] = tuple(LANG_TO_INDEX) + (CONDITION_UNCOND,)
_INDEPENDENT_STRIDE = 7919  # seed offset between conditions when crn=False


def condition_index(condition: str) -> int:
    if condition == CONDITION_UNCOND:
        return NULL_LANG_INDEX
    return LANG_TO_INDEX[condition]


@dataclass(frozen=True)
class ScoreSettings:
    """The sample budget and CRN seed of one scoring pass. ``budget`` =
    ``n_strata × samples_per_stratum`` forward passes per window and
    condition (design §5a: "64 strata × k")."""

    n_strata: int = 64
    samples_per_stratum: int = 1
    seed: int = 0
    batch: int = 16
    t_floor: float = 1e-3
    autocast: bool = True

    @property
    def budget(self) -> int:
        return self.n_strata * self.samples_per_stratum

    def as_dict(self) -> dict:
        return {
            "n_strata": self.n_strata,
            "samples_per_stratum": self.samples_per_stratum,
            "budget": self.budget,
            "seed": self.seed,
            "batch": self.batch,
            "t_floor": self.t_floor,
        }


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.long()
    return torch.from_numpy(np.asarray(x).astype(np.int64))


@torch.no_grad()
def score_conditions(
    model,
    ids,
    conditions: tuple[str, ...] | list[str] = DEFAULT_CONDITIONS,
    *,
    settings: ScoreSettings | None = None,
    device: str | torch.device = "cpu",
    crn: bool = True,
) -> np.ndarray:
    """``[N, len(conditions)]`` bits/char.

    ``ids``: ``[N, L]`` (shared text) or ``{condition: [N, L]}`` (one text
    per condition, equal ``N`` and ``L``). Chunk ``i`` of ``settings.batch``
    windows uses seed ``settings.seed + i`` for every condition (CRN); with
    ``crn=False`` each condition draws its own independent masks — only for
    the task-3.1 variance experiment, never for ranking.
    """
    settings = settings or ScoreSettings()
    conditions = list(conditions)
    shared = not isinstance(ids, Mapping)
    if shared:
        ids_t = _as_tensor(ids)
        n = ids_t.shape[0]
        per_cond = {c: ids_t for c in conditions}
    else:
        per_cond = {c: _as_tensor(ids[c]) for c in conditions}
        shapes = {tuple(v.shape) for v in per_cond.values()}
        if len(shapes) != 1:
            raise ValueError(f"per-condition texts must share a shape: {shapes}")
        n = next(iter(per_cond.values())).shape[0]
    device = torch.device(device)
    use_autocast = settings.autocast and device.type == "cuda"
    out = np.zeros((n, len(conditions)), dtype=np.float64)
    for ci, start in enumerate(range(0, n, settings.batch)):
        for j, cond in enumerate(conditions):
            chunk = per_cond[cond][start : start + settings.batch]
            seed = settings.seed + ci
            if not crn:
                seed += _INDEPENDENT_STRIDE * (j + 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                out[start : start + len(chunk), j] = per_window_nelbo_bits(
                    model,
                    chunk,
                    condition_index(cond),
                    n_strata=settings.n_strata,
                    samples_per_stratum=settings.samples_per_stratum,
                    seed=seed,
                    device=device,
                    t_floor=settings.t_floor,
                ).numpy()
    return out


@dataclass
class DocumentScores:
    """Mean and spread of per-window scores within one document."""

    doc_id: str
    n_windows: int
    mean: dict[str, float]  # condition -> mean bits/char
    std: dict[str, float]  # window-to-window std (0 for a single window)
    sem: dict[str, float]  # standard error of the mean
    window_rows: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "n_windows": self.n_windows,
            "mean": self.mean,
            "std": self.std,
            "sem": self.sem,
        }


def per_document(
    scores: np.ndarray,
    doc_index: np.ndarray,
    doc_ids: list[str],
    conditions: tuple[str, ...] | list[str],
) -> list[DocumentScores]:
    """Aggregate ``[N, C]`` window scores by document (task 3.3)."""
    out = []
    for di in np.unique(doc_index):
        rows = np.flatnonzero(doc_index == di)
        block = scores[rows]
        n = len(rows)
        std = block.std(axis=0, ddof=1) if n > 1 else np.zeros(block.shape[1])
        out.append(
            DocumentScores(
                doc_id=doc_ids[int(di)],
                n_windows=n,
                mean={c: float(block[:, j].mean()) for j, c in enumerate(conditions)},
                std={c: float(std[j]) for j, c in enumerate(conditions)},
                sem={c: float(std[j] / np.sqrt(n)) for j, c in enumerate(conditions)},
                window_rows=[int(r) for r in rows],
            )
        )
    return out


def spread_summary(values: np.ndarray) -> dict:
    """mean / std / s.e.m. / quantiles of a 1-D array — the "spread" every
    reported number carries (task 3.3)."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    return {
        "n": int(n),
        "mean": float(v.mean()) if n else float("nan"),
        "std": float(v.std(ddof=1)) if n > 1 else 0.0,
        "sem": float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "q05": float(np.quantile(v, 0.05)) if n else float("nan"),
        "median": float(np.median(v)) if n else float("nan"),
        "q95": float(np.quantile(v, 0.95)) if n else float("nan"),
    }
