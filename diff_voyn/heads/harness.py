"""Shared, evaluator-agnostic test harness — task CH.2 (prototyping doc §5).

One harness, parameterized by ``Evaluator``; built against ``NgramEvaluator``
now and re-run unchanged against ``DiffusionEvaluator`` post-G4. Emits, per
(cipher kind x language x length x trial):

- letter-map accuracy and SER (ground-truth metrics),
- evals-per-solve and wall-clock (R6 cost realism, task X.3),
- optionally the trial-decipherment language ranking (CH.9): solve under
  every candidate language, compare CALIBRATED bits/char via the single
  calibration hook, check the true language wins.

Calibration for the n-gram evaluator (§6): the per-language offset is minus
the LM's own held-out bits/char, so the ranked quantity is *excess* bits over
the language's intrinsic compressibility — without this, the language with
the lowest-entropy LM (German here) wins every ranking by construction (the
R1 fairness point, in n-gram form).
"""

from __future__ import annotations

import json
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .rung1_sinkhorn import SinkhornSubstitutionHead
from .rung2_homophonic import HomophonicHead
from .synth import HeldoutSampler, gen_homophonic, gen_substitution, map_accuracy, ser

DEFAULT_LENGTHS = (50, 100, 200, 400, 700)


@dataclass
class CellResult:
    kind: str
    language: str
    length: int
    trial: int
    ser: float
    map_accuracy: float
    hard_score: float
    n_evals: int
    wall_seconds: float
    # language probe (empty unless probe_language=True)
    ranking: dict[str, float] = field(default_factory=dict)  # lang -> calib bits
    ranked_first: str | None = None
    true_language_won: bool | None = None


def ngram_calibration_offsets(lms: dict) -> dict[str, float]:
    """The n-gram evaluator's §3.4-style offsets: -heldout bits/char."""
    return {
        lang: -lm.meta["heldout_bits_per_char"]
        for lang, lm in lms.items()
        if "heldout_bits_per_char" in lm.meta
    }


def _solve(head_kind: str, evaluator, cipher, language: str, seed: int):
    if head_kind == "sub1to1":
        head = SinkhornSubstitutionHead(evaluator, seed=seed)
        res = head.solve(cipher.cipher_ids, language=language)
    elif head_kind == "homophonic":
        head = HomophonicHead(evaluator, seed=seed)
        res = head.solve(cipher.cipher_ids, cipher.n_symbols, language=language)
    else:
        raise ValueError(head_kind)
    return res


def run_cell(
    evaluator,
    cipher,
    *,
    seed: int = 0,
    probe_language: bool = False,
) -> CellResult:
    """Solve one synthetic cipher under its true language; optionally solve
    under every candidate language and rank them (common seed across language
    conditions — the CRN discipline of non-negotiable #4)."""
    t0 = time.time()
    res = _solve(cipher.kind, evaluator, cipher, cipher.language, seed)
    out = CellResult(
        kind=cipher.kind,
        language=cipher.language,
        length=len(cipher.plain_ids),
        trial=seed,
        ser=ser(cipher, res.sym_to_letter),
        map_accuracy=map_accuracy(cipher, res.sym_to_letter),
        hard_score=res.hard_score,
        n_evals=res.n_evals,
        wall_seconds=time.time() - t0,
    )
    if probe_language:
        n = len(cipher.plain_ids)
        for lang in evaluator.languages:
            r = (
                res
                if lang == cipher.language
                else _solve(cipher.kind, evaluator, cipher, lang, seed)
            )
            out.ranking[lang] = evaluator.calibrated_bits_per_char(
                r.hard_score, n, lang
            )
        out.ranked_first = min(out.ranking, key=out.ranking.get)
        out.true_language_won = out.ranked_first == cipher.language
    return out


def run_grid(
    evaluator,
    corpus_dir: Path,
    splits: dict,
    *,
    kinds=("sub1to1", "homophonic"),
    languages=("latin", "italian", "german"),
    lengths=DEFAULT_LENGTHS,
    trials: int = 5,
    n_symbols: int = 54,
    probe_language: bool = False,
    seed: int = 0,
    progress=print,
) -> list[CellResult]:
    results: list[CellResult] = []
    for lang in languages:
        sampler = HeldoutSampler(corpus_dir, splits, lang)
        for kind in kinds:
            for L in lengths:
                for trial in range(trials):
                    # stable across processes (builtin hash() is salted)
                    key = f"{seed}/{lang}/{kind}/{L}/{trial}".encode()
                    rng = np.random.default_rng(zlib.crc32(key))
                    plain = sampler.sample(L, rng)
                    cipher = (
                        gen_substitution(plain, lang, rng)
                        if kind == "sub1to1"
                        else gen_homophonic(plain, lang, rng, n_symbols=n_symbols)
                    )
                    cell = run_cell(
                        evaluator, cipher, seed=trial, probe_language=probe_language
                    )
                    results.append(cell)
                    progress(
                        f"{kind} {lang} L={L} t={trial}: SER={cell.ser:.4f} "
                        f"map={cell.map_accuracy:.3f} "
                        f"{'lang=' + str(cell.ranked_first) if probe_language else ''} "
                        f"{cell.wall_seconds:.0f}s"
                    )
    return results


def summarize(results: list[CellResult]) -> dict:
    """Per-cell aggregates (mean/max SER, map accuracy, language-win rate)."""
    cells: dict[tuple, list[CellResult]] = {}
    for r in results:
        cells.setdefault((r.kind, r.language, r.length), []).append(r)
    out = {}
    for (kind, lang, L), rs in sorted(cells.items()):
        entry = {
            "trials": len(rs),
            "ser_mean": float(np.mean([r.ser for r in rs])),
            "ser_max": float(np.max([r.ser for r in rs])),
            "map_accuracy_mean": float(np.mean([r.map_accuracy for r in rs])),
            "evals_per_solve_mean": float(np.mean([r.n_evals for r in rs])),
            "wall_seconds_mean": float(np.mean([r.wall_seconds for r in rs])),
        }
        wins = [r.true_language_won for r in rs if r.true_language_won is not None]
        if wins:
            entry["language_win_rate"] = float(np.mean(wins))
        out[f"{kind}/{lang}/L{L}"] = entry
    return out


def save_results(results: list[CellResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"results": [asdict(r) for r in results], "summary": summarize(results)},
            indent=2,
        )
    )
