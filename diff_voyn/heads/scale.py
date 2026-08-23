"""Uniform cross-head scoring scale — tasks 5.1 / 5.6 (design §8, R5).

Every (cipher × language) cell is reported on ONE scale:

    cell_bits_per_char = calibrated_bits_per_plaintext_char
                         + (key_bits + choice_bits) / n_plaintext_chars

- ``calibrated_bits_per_plaintext_char``: the frozen evaluator's bits/char
  of the head's hard decode under the language condition, through the single
  calibration hook (``metrology.calibrate_bits``; report-only under the
  adopted table, so raw own-condition NELBO).
- ``key_bits``: description length of the cipher key — log2 of the size of
  the key space the head searches (design §8: "parameter count of the head /
  description length of the map, so verbose heads cannot win by capacity
  alone"). Rung 1: log2(A!); rung 2: n_symbols·log2(A); rung 3: 18 block
  bijections, 18·log2(23!); rung 4: log2(16!) for the 16 cipher-character
  values plus 25 letter values over the generator's integer range.
- ``choice_bits``: the cipher's own encoding freedom — the bits needed to
  name, given plaintext and key, WHICH ciphertext the encipherer emitted
  (zero for a bijection; the homophone choice for homophonic ciphers; the
  deck draw + parse state for Naibbe; the Zipf-weighted homophone draw for
  the arithmetic cipher). This is the second leg of the MDL argument: a
  verbose cipher explains any ciphertext's statistics more easily, and the
  price of that freedom is paid here, per plaintext character. It is a
  constant per cipher hypothesis, so it never touches the within-cipher
  language ranking — it orders cipher hypotheses against each other.

The three terms are always reported separately next to the total; the
language ranking inside a cipher hypothesis uses the first term only (same
text, same key class ⇒ identical penalties).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .ngram import A

HEAD_KINDS = ("sub1to1", "homophonic", "naibbe", "arithmetic")


def log2_factorial(n: int) -> float:
    return float(math.lgamma(n + 1) / math.log(2.0))


def key_bits(kind: str, **p) -> float:
    """Description length (bits) of one key of the head's key class."""
    if kind == "sub1to1":
        return log2_factorial(A)
    if kind == "homophonic":
        return float(p["n_symbols"]) * math.log2(A)
    if kind == "naibbe":
        n_blocks = int(p.get("n_blocks", 18))
        support = int(p.get("support", 23))
        return n_blocks * log2_factorial(support)
    if kind == "arithmetic":
        # 16 cipher characters carry a permutation of the 16 integer values
        # (-2..13) and each of the A letters an integer value in the
        # generator's range (upstream default a=3..z=28 → 26 values).
        n_chars = int(p.get("n_chars", 16))
        value_range = int(p.get("letter_value_range", 26))
        return log2_factorial(n_chars) + A * math.log2(value_range)
    raise ValueError(kind)


def choice_bits(kind: str, decoded: np.ndarray, **p) -> float:
    """Bits to name the emitted ciphertext given plaintext + key (total over
    the decode). Exact where the generator is deterministic given its draws;
    the Naibbe / arithmetic terms use the pinned generators' published draw
    distributions (documented approximations, see module docstring)."""
    decoded = np.asarray(decoded)
    n = len(decoded)
    if kind == "sub1to1":
        return 0.0
    if kind == "homophonic":
        # sym_to_letter (n_symbols,): uniform choice among a letter's symbols
        sym_to_letter = np.asarray(p["sym_to_letter"])
        n_hom = np.bincount(sym_to_letter, minlength=A)
        return float(np.log2(np.maximum(n_hom[decoded], 1)).sum())
    if kind == "naibbe":
        # per token: deck draw (published card weights) + unigram/bigram
        # state; tokens ≈ n / (2 - p_unigram) for the generator's p_unigram
        w = np.asarray(list(p["card_weights"].values()), float)
        w = w / w.sum()
        h_deck = float(-(w * np.log2(w)).sum())
        p_uni = float(p.get("p_unigram", 0.5))
        h_state = -(p_uni * math.log2(p_uni) + (1 - p_uni) * math.log2(1 - p_uni))
        n_tokens = p.get("n_tokens", n / (2.0 - p_uni))
        return float(n_tokens * (h_deck + h_state))
    if kind == "arithmetic":
        # one token per letter drawn Zipf-weighted from ~n_hom homophones
        n_hom = int(p.get("n_homophones", 500))
        s = float(p.get("zipf_exponent", 1.0))
        ranks = np.arange(1, n_hom + 1, dtype=float)
        z = ranks ** (-s)
        z /= z.sum()
        h = float(-(z * np.log2(z)).sum())
        return float(n * h)
    raise ValueError(kind)


@dataclass
class CellScore:
    kind: str
    language: str
    n_plain: int
    calibrated_bits: float  # bits per plaintext char, calibrated
    key_bits: float
    choice_bits: float
    extra: dict = field(default_factory=dict)

    @property
    def penalty_bits_per_char(self) -> float:
        return (self.key_bits + self.choice_bits) / max(self.n_plain, 1)

    @property
    def total_bits_per_char(self) -> float:
        return self.calibrated_bits + self.penalty_bits_per_char

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "language": self.language,
            "n_plain": self.n_plain,
            "calibrated_bits": self.calibrated_bits,
            "key_bits": self.key_bits,
            "choice_bits": self.choice_bits,
            "penalty_bits_per_char": self.penalty_bits_per_char,
            "total_bits_per_char": self.total_bits_per_char,
            **self.extra,
        }


def cell_score(
    evaluator,
    decoded: np.ndarray,
    *,
    kind: str,
    language: str,
    n_strata: int = 64,
    seed: int = 0,
    key_params: dict | None = None,
    choice_params: dict | None = None,
    raw_bits: float | None = None,
) -> CellScore:
    """Score one hard decode on the uniform scale. ``raw_bits`` may be
    supplied (already measured under CRN) to skip the evaluator call."""
    decoded = np.asarray(decoded, dtype=np.int64)
    if raw_bits is None:
        raw_bits = evaluator.score_stream(
            decoded, language=language, n_strata=n_strata, seed=seed
        )
    from ..metrology.calibration import calibrate_bits

    cal = calibrate_bits(raw_bits, language, evaluator.calibration_offsets_bits)
    return CellScore(
        kind=kind,
        language=language,
        n_plain=len(decoded),
        calibrated_bits=float(cal),
        key_bits=key_bits(kind, **(key_params or {})),
        choice_bits=choice_bits(kind, decoded, **(choice_params or {})),
        extra={"raw_bits": float(raw_bits)},
    )
