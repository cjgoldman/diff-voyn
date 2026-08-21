"""Decipherment-noise generators — tasks 2.1, 2.2, 2.3 and the NULL-frame
exposure of 2.5 (design §7.3, §8, requirement R2).

Phase B turns the clean backbone into a noise-robust evaluator by training on
*simulated partial decipherments*: the corruption the model meets at
deployment is the output of a cipher head holding a partly wrong key and a
partly wrong parse, read from a partly wrong transcription. Each generator
below is one of those three error sources, parameterized by a single
``severity`` so sweeps and curricula share one scale:

- :class:`SubstitutionNoise` (2.1) — a **self-consistent many-to-one wrong
  key**: a random subset of letters is remapped, every occurrence of a remapped
  letter goes to the *same* wrong letter (targets may collide, so the map is
  many-to-one — exactly what a wrong homophonic key produces). ``severity`` is
  the expected fraction of positions altered. This is *not* i.i.d. flip noise.
- :class:`SegmentationNoise` (2.2) — **wrong unigram-vs-bigram parses** of a
  Naibbe-style token stream: the letters are grouped into 1- or 2-letter
  tokens at the measured Naibbe unigram rate; a misparsed bigram loses its
  second letter (deletion), a misparsed unigram gains a spurious one
  (duplication of the letter, or a letter drawn from the window's own letter
  distribution). ``severity`` is the expected number of edits per source
  letter. No space noise exists: whitespace is stripped upstream (task 0.3).
- :class:`TranscriptionNoise` (2.3) — i.i.d. reading errors at ~5%: mostly
  substitutions, some deletions/insertions. ``severity`` is the per-character
  event rate.
- :func:`frame_with_nulls` (2.5) — the 2N-slot frame of design §8 on hard
  tokens: each token owns two slots, slot 2 is ``NULL`` for unigram tokens.
  NULL therefore lands on ~P_UNIGRAM/2 of all slots, always in slot-2
  position, never adjacent to another NULL — the pattern Phase-5 heads emit.

All generators work on 1-D ``uint8`` arrays of *letter ids* (the encoded
corpus), are pure functions of the ``numpy.random.Generator`` they are given
(so CRN/replay work), and return ``(ids, info)`` where ``info`` records the
realized rates for logging and tests. Length is not preserved by 2.2/2.3/2.5;
:class:`NoiseMixture` over-samples the source window and crops.

Noise is applied to the *data*, not to the masking: the noised stream is the
training target, so the loss stays a proper NELBO of whatever text the model
is shown — at deployment the partially decrypted text is scored the same way.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from ..vocab import LETTER_IDS, NULL_ID

# Letter ids are contiguous (specials first, then the 25 letters) — the
# generators rely on ``letter index = id − LETTER_BASE``.
LETTER_BASE: int = LETTER_IDS[0]
N_LETTERS: int = len(LETTER_IDS)
assert LETTER_IDS == list(range(LETTER_BASE, LETTER_BASE + N_LETTERS))

# Fraction of Naibbe ciphertext tokens that encode ONE plaintext letter,
# measured with the pinned naibbe_v2 (greshko @ df3d074) on 20k chars of each
# frozen language: latin 0.479, italian 0.477, german 0.473 (2026-08-21).
P_UNIGRAM_NAIBBE: float = 0.476


def tokens_per_letter(p_unigram: float) -> float:
    """Naibbe tokens per plaintext letter: 1 / (p·1 + (1−p)·2)."""
    return 1.0 / (2.0 - p_unigram)


def _check_letters(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids)
    if ids.ndim != 1:
        raise ValueError("noise generators take a 1-D id array")
    if ids.size and ((ids < LETTER_BASE) | (ids >= LETTER_BASE + N_LETTERS)).any():
        raise ValueError("noise generators expect letter ids only (no specials)")
    return ids.astype(np.uint8, copy=False)


def _random_other_letter(letters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """For each letter index, a uniformly random *different* letter index."""
    r = rng.integers(0, N_LETTERS - 1, size=letters.shape)
    return np.where(r >= letters, r + 1, r)


def _parse_tokens(
    n_letters: int, p_unigram: float, rng: np.random.Generator
) -> np.ndarray:
    """Token lengths (1 or 2) covering exactly ``n_letters`` letters, each
    token independently a unigram with probability ``p_unigram`` (the last
    token is clipped to the remaining letter if needed)."""
    if n_letters == 0:
        return np.zeros(0, dtype=np.int64)
    lengths = np.where(rng.random(n_letters) < p_unigram, 1, 2)
    cum = np.cumsum(lengths)
    n_tok = int(np.searchsorted(cum, n_letters, side="left")) + 1
    lengths = lengths[:n_tok].copy()
    lengths[-1] -= int(cum[n_tok - 1] - n_letters)  # clip the overflow
    assert lengths.sum() == n_letters and lengths[-1] in (1, 2)
    return lengths


# ---------------------------------------------------------------------------
# 2.1 structured substitution noise
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WrongKey:
    """A sampled many-to-one letter map: ``target[i]`` is the letter index
    letter ``i`` decodes to (``target[i] == i`` for correctly keyed letters)."""

    target: np.ndarray  # [N_LETTERS] int64
    p_apply: float  # per-position application probability (1.0 = fully consistent)

    @property
    def wrong_letters(self) -> np.ndarray:
        return np.flatnonzero(self.target != np.arange(N_LETTERS))

    @property
    def is_many_to_one(self) -> bool:
        return len(np.unique(self.target)) < N_LETTERS


class SubstitutionNoise:
    """Self-consistent wrong-key noise (task 2.1).

    ``severity`` ∈ [0, 1]: expected fraction of positions whose letter changes.
    Letters are drawn (in random order) into the wrong-key set until their
    frequency *in this window* covers ``severity``; with ``exact_rate`` the
    map is then applied to each occurrence with probability
    ``severity / covered`` so the realized rate is unbiased for ``severity``
    (a homophonic key with some wrong homophones behaves exactly like this:
    the wrong target is fixed per letter, only some occurrences hit it).
    With ``exact_rate=False`` every occurrence of a wrong-key letter is
    remapped (realized rate ≥ severity).
    """

    name = "substitution"

    def __init__(self, severity: float, exact_rate: bool = True):
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")
        self.severity = float(severity)
        self.exact_rate = exact_rate

    def sample_key(self, ids: np.ndarray, rng: np.random.Generator) -> WrongKey:
        letters = _check_letters(ids) - LETTER_BASE
        target = np.arange(N_LETTERS)
        if self.severity <= 0.0 or letters.size == 0:
            return WrongKey(target, 1.0)
        freq = np.bincount(letters, minlength=N_LETTERS) / letters.size
        present = np.flatnonzero(freq > 0)
        order = present[rng.permutation(len(present))]
        cum = np.cumsum(freq[order])
        n_wrong = int(np.searchsorted(cum, self.severity, side="left")) + 1
        n_wrong = min(n_wrong, len(order))
        wrong = order[:n_wrong]
        covered = float(cum[n_wrong - 1])
        target = target.copy()
        target[wrong] = _random_other_letter(wrong, rng)
        p_apply = min(1.0, self.severity / covered) if self.exact_rate else 1.0
        return WrongKey(target, p_apply)

    def apply_key(
        self, ids: np.ndarray, key: WrongKey, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:
        letters = _check_letters(ids) - LETTER_BASE
        mapped = key.target[letters]
        hit = mapped != letters
        if key.p_apply < 1.0:
            hit &= rng.random(letters.shape) < key.p_apply
        out = np.where(hit, mapped, letters).astype(np.uint8) + LETTER_BASE
        info = {
            "severity": self.severity,
            "n_wrong_letters": len(key.wrong_letters),
            "many_to_one": bool(key.is_many_to_one),
            "p_apply": key.p_apply,
            "changed_fraction": float(hit.mean()) if letters.size else 0.0,
        }
        return out.astype(np.uint8), info

    def __call__(
        self, ids: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:
        return self.apply_key(ids, self.sample_key(ids, rng), rng)


# ---------------------------------------------------------------------------
# 2.2 segmentation noise (wrong unigram/bigram parses)
# ---------------------------------------------------------------------------


class SegmentationNoise:
    """Letter-stream parse errors of a Naibbe-style token stream (task 2.2).

    ``severity``: expected edits per *source letter*. Internally the stream is
    parsed into 1/2-letter tokens (unigram probability ``p_unigram``), and each
    token is misparsed with probability ``severity / tokens_per_letter`` —
    a bigram read as a unigram drops its second letter, a unigram read as a
    bigram gains a second letter: a duplicate of itself with probability
    ``p_duplicate``, otherwise a letter drawn from the window's empirical
    letter distribution. Output is letters only — whitespace-free by
    construction (there is no space symbol to emit).
    """

    name = "segmentation"

    def __init__(
        self,
        severity: float,
        p_unigram: float = P_UNIGRAM_NAIBBE,
        p_duplicate: float = 0.5,
    ):
        if severity < 0.0:
            raise ValueError("severity must be ≥ 0")
        self.severity = float(severity)
        self.p_unigram = p_unigram
        self.p_duplicate = p_duplicate
        self.p_misparse = min(1.0, self.severity / tokens_per_letter(p_unigram))

    def __call__(
        self, ids: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:
        ids = _check_letters(ids)
        n = ids.size
        zero = {
            "severity": self.severity,
            "n_deletions": 0,
            "n_insertions": 0,
            "n_duplications": 0,
            "edit_rate": 0.0,
        }
        if n == 0 or self.severity <= 0.0:
            return ids.copy(), zero
        lengths = _parse_tokens(n, self.p_unigram, rng)
        starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        misparse = rng.random(len(lengths)) < self.p_misparse
        uni = lengths == 1
        # output length per token: 1 + (kept second letter | spurious letter)
        keep_second = ~uni & ~misparse
        spurious = uni & misparse
        deleted = ~uni & misparse
        out_len = 1 + (keep_second | spurious).astype(np.int64)
        offsets = np.concatenate(([0], np.cumsum(out_len)[:-1]))
        out = np.empty(int(out_len.sum()), dtype=np.uint8)
        first = ids[starts]
        out[offsets] = first
        out[offsets[keep_second] + 1] = ids[starts[keep_second] + 1]
        n_sp = int(spurious.sum())
        n_dup = 0
        if n_sp:
            dup = rng.random(n_sp) < self.p_duplicate
            n_dup = int(dup.sum())
            letters = ids - LETTER_BASE
            freq = np.bincount(letters, minlength=N_LETTERS).astype(np.float64)
            freq /= freq.sum()
            drawn = rng.choice(N_LETTERS, size=n_sp, p=freq).astype(np.uint8)
            drawn += LETTER_BASE
            extra = np.where(dup, first[spurious], drawn)
            out[offsets[spurious] + 1] = extra
        n_del = int(deleted.sum())
        info = {
            "severity": self.severity,
            "n_deletions": n_del,
            "n_insertions": n_sp - n_dup,
            "n_duplications": n_dup,
            "edit_rate": (n_del + n_sp) / n,
        }
        return out, info


# ---------------------------------------------------------------------------
# 2.3 transcription noise
# ---------------------------------------------------------------------------


class TranscriptionNoise:
    """i.i.d. transcription (reading) errors at ``severity`` per character
    (task 2.3; ~0.05 is the Bruton-2026 level). Event mix: substitution with a
    uniformly random other letter (``p_sub``), deletion (``p_del``),
    insertion of a uniformly random letter after the character (``p_ins``).
    """

    name = "transcription"

    def __init__(
        self,
        severity: float = 0.05,
        p_sub: float = 0.8,
        p_del: float = 0.1,
        p_ins: float = 0.1,
    ):
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")
        if abs(p_sub + p_del + p_ins - 1.0) > 1e-9:
            raise ValueError("event probabilities must sum to 1")
        self.severity = float(severity)
        self.p_sub, self.p_del, self.p_ins = p_sub, p_del, p_ins

    def __call__(
        self, ids: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:
        ids = _check_letters(ids)
        n = ids.size
        if n == 0 or self.severity <= 0.0:
            return ids.copy(), {
                "severity": self.severity,
                "n_substitutions": 0,
                "n_deletions": 0,
                "n_insertions": 0,
                "edit_rate": 0.0,
            }
        event = rng.random(n) < self.severity
        kind = rng.choice(3, size=n, p=[self.p_sub, self.p_del, self.p_ins])
        sub = event & (kind == 0)
        dele = event & (kind == 1)
        ins = event & (kind == 2)
        letters = (ids - LETTER_BASE).astype(np.int64)
        letters = np.where(sub, _random_other_letter(letters, rng), letters)
        out_len = np.where(dele, 0, np.where(ins, 2, 1))
        offsets = np.concatenate(([0], np.cumsum(out_len)[:-1]))
        out = np.empty(int(out_len.sum()), dtype=np.uint8)
        keep = ~dele
        out[offsets[keep]] = (letters[keep] + LETTER_BASE).astype(np.uint8)
        n_ins = int(ins.sum())
        if n_ins:
            out[offsets[ins] + 1] = (
                rng.integers(0, N_LETTERS, size=n_ins) + LETTER_BASE
            ).astype(np.uint8)
        info = {
            "severity": self.severity,
            "n_substitutions": int(sub.sum()),
            "n_deletions": int(dele.sum()),
            "n_insertions": n_ins,
            "edit_rate": float(event.mean()),
        }
        return out, info


# ---------------------------------------------------------------------------
# 2.5 NULL-frame exposure (2N-slot scheme, design §8)
# ---------------------------------------------------------------------------


def frame_with_nulls(
    ids: np.ndarray, rng: np.random.Generator, p_unigram: float = P_UNIGRAM_NAIBBE
) -> tuple[np.ndarray, dict]:
    """Lay a letter stream onto the 2N-slot frame with hard tokens: the
    stream is parsed into Naibbe-style 1/2-letter tokens; every token emits
    two slots — ``[letter, NULL]`` for a unigram, ``[letter, letter]`` for a
    bigram. Output length is ``2 × n_tokens`` (≈ 1.31 × n_letters).

    Invariants (tested): NULL only ever in odd (slot-2) positions, no two
    adjacent NULLs, and dropping the NULLs recovers the input exactly.
    """
    ids = _check_letters(ids)
    lengths = _parse_tokens(ids.size, p_unigram, rng)
    n_tok = len(lengths)
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    out = np.full(2 * n_tok, NULL_ID, dtype=np.uint8)
    out[0::2] = ids[starts]
    bi = lengths == 2
    out[1::2][bi] = ids[starts[bi] + 1]
    info = {
        "n_tokens": n_tok,
        "n_null": int(n_tok - bi.sum()),
        "null_fraction": float((n_tok - bi.sum()) / (2 * n_tok)) if n_tok else 0.0,
    }
    return out, info


# ---------------------------------------------------------------------------
# Phase-B mixture (task 2.4 / 2.5)
# ---------------------------------------------------------------------------

KIND_CLEAN, KIND_NOISED, KIND_FRAMED, KIND_FRAMED_NOISED = 0, 1, 2, 3
KIND_NAMES = {
    KIND_CLEAN: "clean",
    KIND_NOISED: "noised",
    KIND_FRAMED: "framed",
    KIND_FRAMED_NOISED: "framed_noised",
}


@dataclass(frozen=True)
class NoiseConfig:
    """Example-type mix and severity ranges for Phase B (design §7.3).

    The clean fraction (no noise, no frame) is the calibration anchor and is
    never reduced below one half. Noised examples carry 30–50% of the stream
    (``p_noised + p_framed_noised``); framed examples expose NULL at the
    2N-slot rate (``p_framed + p_framed_noised``).
    """

    p_clean: float = 0.50
    p_noised: float = 0.30
    p_framed: float = 0.10
    p_framed_noised: float = 0.10
    # Within a noised example each family is applied independently with these
    # probabilities (resampled if none is drawn), in deployment order:
    # wrong key → wrong parse → transcription errors.
    p_substitution: float = 0.75
    p_segmentation: float = 0.50
    p_transcription: float = 0.50
    # Severities are drawn uniformly from these ranges per example.
    substitution_severity: tuple[float, float] = (0.02, 0.50)
    segmentation_severity: tuple[float, float] = (0.01, 0.20)
    transcription_severity: tuple[float, float] = (0.01, 0.10)
    p_unigram: float = P_UNIGRAM_NAIBBE
    # Source windows for length-changing kinds are over-sampled by this factor
    # and the output cropped back to seq_len.
    source_margin: float = 1.5

    def __post_init__(self) -> None:
        ps = (self.p_clean, self.p_noised, self.p_framed, self.p_framed_noised)
        if abs(sum(ps) - 1.0) > 1e-9 or min(ps) < 0:
            raise ValueError("kind probabilities must be non-negative and sum to 1")
        if self.p_clean < 0.5:
            raise ValueError("clean fraction must stay ≥ 0.5 (calibration anchor)")
        if not 0.3 <= self.p_noised + self.p_framed_noised <= 0.5:
            raise ValueError("noised fraction must be within 30–50% (design §7.3)")
        if self.p_substitution + self.p_segmentation + self.p_transcription <= 0:
            raise ValueError("at least one noise family must be enabled")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def kind_probs(self) -> np.ndarray:
        return np.array(
            [self.p_clean, self.p_noised, self.p_framed, self.p_framed_noised]
        )


class NoiseMixture:
    """Samples an example kind and applies the matching corruption."""

    def __init__(self, cfg: NoiseConfig | None = None):
        self.cfg = cfg or NoiseConfig()

    def sample_kind(self, rng: np.random.Generator) -> int:
        return int(rng.choice(4, p=self.cfg.kind_probs))

    def source_length(self, kind: int, seq_len: int) -> int:
        if kind == KIND_CLEAN:
            return seq_len
        return math.ceil(seq_len * self.cfg.source_margin)

    def _noise(
        self, ids: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:
        cfg = self.cfg
        while True:
            use = rng.random(3) < (
                cfg.p_substitution,
                cfg.p_segmentation,
                cfg.p_transcription,
            )
            if use.any():
                break
        info: dict = {}
        if use[0]:
            sev = rng.uniform(*cfg.substitution_severity)
            ids, i = SubstitutionNoise(sev)(ids, rng)
            info["substitution"] = i
        if use[1]:
            sev = rng.uniform(*cfg.segmentation_severity)
            ids, i = SegmentationNoise(sev, cfg.p_unigram)(ids, rng)
            info["segmentation"] = i
        if use[2]:
            sev = rng.uniform(*cfg.transcription_severity)
            ids, i = TranscriptionNoise(sev)(ids, rng)
            info["transcription"] = i
        return ids, info

    def apply(
        self, ids: np.ndarray, kind: int, seq_len: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:
        """Corrupt ``ids`` (a source window of :meth:`source_length`) for
        ``kind`` and crop to ``seq_len``. Returns (ids[seq_len], info)."""
        info: dict = {"kind": kind}
        if kind == KIND_CLEAN:
            assert len(ids) >= seq_len
            return np.asarray(ids[:seq_len], dtype=np.uint8), info
        if kind in (KIND_NOISED, KIND_FRAMED_NOISED):
            ids, ninfo = self._noise(ids, rng)
            info.update(ninfo)
        if kind in (KIND_FRAMED, KIND_FRAMED_NOISED):
            ids, finfo = frame_with_nulls(ids, rng, self.cfg.p_unigram)
            info["frame"] = finfo
        if len(ids) < seq_len:  # cannot happen within the margin; be safe
            reps = math.ceil(seq_len / max(1, len(ids)))
            ids = np.tile(ids, reps)
            info["tiled"] = True
        return np.asarray(ids[:seq_len], dtype=np.uint8), info


# ---------------------------------------------------------------------------
# Fixed evaluation variants (canary / G2)
# ---------------------------------------------------------------------------

EVAL_NOISE_SEED = 2024
# A "typical moderate partial decipherment" for the in-training canary: 20%
# of positions under a wrong key, 5% parse edits, 5% transcription errors.
EVAL_NOISED_SEVERITIES = {
    "substitution": 0.20,
    "segmentation": 0.05,
    "transcription": 0.05,
}


def noised_variant(
    source: np.ndarray,
    seq_len: int,
    rng: np.random.Generator,
    severities: dict[str, float] = EVAL_NOISED_SEVERITIES,
    p_unigram: float = P_UNIGRAM_NAIBBE,
) -> np.ndarray:
    """Apply the three families at fixed severities (deployment order) to a
    source window longer than ``seq_len`` and crop."""
    ids = _check_letters(source)
    ids, _ = SubstitutionNoise(severities.get("substitution", 0.0))(ids, rng)
    ids, _ = SegmentationNoise(severities.get("segmentation", 0.0), p_unigram)(ids, rng)
    ids, _ = TranscriptionNoise(severities.get("transcription", 0.0))(ids, rng)
    if len(ids) < seq_len:
        ids = np.tile(ids, math.ceil(seq_len / len(ids)))
    return ids[:seq_len].copy()


def framed_variant(
    source: np.ndarray,
    seq_len: int,
    rng: np.random.Generator,
    p_unigram: float = P_UNIGRAM_NAIBBE,
) -> np.ndarray:
    ids, _ = frame_with_nulls(_check_letters(source), rng, p_unigram)
    if len(ids) < seq_len:
        ids = np.tile(ids, math.ceil(seq_len / len(ids)))
    return ids[:seq_len].copy()
