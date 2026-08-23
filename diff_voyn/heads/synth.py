"""Synthetic cipher ground truth + metrics — the CH.2 harness core.

Plaintexts are sampled from the HELD-OUT side of splits v1 — the n-gram LMs
never trained on these documents, so map recovery is never "the LM memorized
this text". Generators emit (cipher_ids, plain_ids, true_map) with full
ground truth (task 0.7's paired-corpus convention).

Metrics (prototyping doc §5): letter-map accuracy (occurrence-weighted), SER
(decode-vs-truth symbol error rate, Kambhatla/ALICE convention), and the
language-recovery probe.
"""

from __future__ import annotations

import random as pyrandom
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ngram import A, encode_letters


@dataclass
class SyntheticCipher:
    kind: str  # "sub1to1" | "homophonic"
    language: str
    plain_ids: np.ndarray  # (L,) 0..A-1
    cipher_ids: np.ndarray  # (L,) 0..V_sym-1
    n_symbols: int
    true_map: np.ndarray  # (n_symbols,) cipher symbol -> plaintext letter


class HeldoutSampler:
    """Random plaintext windows from held-out docs of one language."""

    def __init__(self, corpus_dir: Path, splits: dict, language: str):
        self.language = language
        self.docs = [
            encode_letters(
                (corpus_dir / language / "docs" / f"{d['doc_id']}.txt").read_text()
            )
            for d in splits["languages"][language]["heldout"]
        ]
        self.weights = np.array([len(d) for d in self.docs], float)
        self.weights /= self.weights.sum()

    def sample(self, length: int, rng: np.random.Generator) -> np.ndarray:
        d = self.docs[rng.choice(len(self.docs), p=self.weights)]
        start = rng.integers(0, len(d) - length + 1)
        return d[start : start + length].astype(np.int64)


def gen_substitution(
    plain_ids: np.ndarray, language: str, rng: np.random.Generator
) -> SyntheticCipher:
    """1:1 bijective substitution over the 25-letter alphabet (rung 1)."""
    perm = rng.permutation(A)  # perm[letter] = cipher symbol
    inverse = np.argsort(perm)  # cipher symbol -> letter
    return SyntheticCipher("sub1to1", language, plain_ids, perm[plain_ids], A, inverse)


def gen_homophonic(
    plain_ids: np.ndarray,
    language: str,
    rng: np.random.Generator,
    n_symbols: int = 54,
) -> SyntheticCipher:
    """Unigram homophonic cipher (rung 2): homophone counts allocated
    proportional to the plaintext's letter frequencies (Zodiac-408-style
    flat-output construction), each present letter >= 1 symbol, uniform
    random choice among a letter's symbols at encipherment."""
    counts = np.bincount(plain_ids, minlength=A).astype(float)
    present = counts > 0
    alloc = np.zeros(A, dtype=int)
    alloc[present] = 1
    extra = n_symbols - alloc.sum()
    if extra < 0:
        raise ValueError("n_symbols smaller than distinct plaintext letters")
    frac = counts / counts.sum()
    for _ in range(extra):  # largest-remainder-ish greedy allocation
        deficit = frac - alloc / max(alloc.sum(), 1)
        deficit[~present] = -np.inf
        alloc[int(np.argmax(deficit))] += 1
    true_map = np.repeat(np.arange(A), alloc)  # symbol -> letter
    rng.shuffle(true_map)
    symbols_of = [np.flatnonzero(true_map == a) for a in range(A)]
    cipher = np.array(
        [symbols_of[p][rng.integers(len(symbols_of[p]))] for p in plain_ids],
        dtype=np.int64,
    )
    return SyntheticCipher(
        "homophonic", language, plain_ids, cipher, n_symbols, true_map
    )


@dataclass
class ArithmeticInstance:
    """Rung-4 ground truth: the head consumes ``char_ids`` (whitespace
    stripped, symbol ids under a random permutation so no code path can leak
    the natural hex order); everything else is validation-only."""

    language: str
    plain_ids: np.ndarray  # (n_letters,) 0..A-1
    char_ids: np.ndarray  # (n_chars,) 0..15 — unsegmented stream
    token_starts: np.ndarray  # (n_letters,) true segment start positions
    true_v: np.ndarray  # (16,) char id -> integer value
    true_u: np.ndarray  # (A,) letter -> integer value
    true_rank: np.ndarray  # (16,) char id -> canonical-order position


def gen_arithmetic(
    plain_ids: np.ndarray, language: str, cipher, rng: np.random.Generator
) -> ArithmeticInstance:
    """Encipher held-out plaintext with a pinned ``ArithmeticCipher`` and
    emit the whitespace-stripped stream + full ground truth."""
    from ..vocab import LETTERS
    from .ngram import LETTER_TO_IDX

    text = "".join(LETTERS[i] for i in plain_ids)
    pyrng = pyrandom.Random(int(rng.integers(2**31)))
    tokens = cipher.enc.encode(text, rng=pyrng).split()
    assert len(tokens) == len(plain_ids)

    from voynpy.pseudo_vms.encoder import HEX_CHARS, HEX_VALUE

    hex_order = "EFDCBA9876543210"  # canonical within-token order
    perm = rng.permutation(len(HEX_CHARS))  # hex index -> shuffled symbol id
    sym_of = {c: int(perm[i]) for i, c in enumerate(HEX_CHARS)}
    true_v = np.empty(len(HEX_CHARS), dtype=np.int64)
    true_rank = np.empty(len(HEX_CHARS), dtype=np.int64)
    for c in HEX_CHARS:
        true_v[sym_of[c]] = HEX_VALUE[c]
        true_rank[sym_of[c]] = hex_order.index(c)
    char_ids = np.array([sym_of[c] for t in tokens for c in t], dtype=np.int64)
    lens = np.array([len(t) for t in tokens])
    token_starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    true_u = np.full(A, -(10**6), dtype=np.int64)
    for letter, value in cipher.enc.alphabet.items():
        true_u[LETTER_TO_IDX[letter]] = value
    return ArithmeticInstance(
        language, plain_ids, char_ids, token_starts, true_v, true_u, true_rank
    )


@dataclass
class NaibbeInstance:
    """Rung-3 ground truth: ``tokens`` is what the head sees (the
    whitespace-free stream is their concatenation); ``plain_ids`` is the
    23-letter pre-mapped plaintext (k→c, w→uu) in frozen-alphabet ids — the
    text the decode is compared against; the glyph→letter key is the
    published apparatus (``NaibbeParser.block_truth``)."""

    language: str
    plain_ids: np.ndarray
    tokens: list[str]
    segments: list[str]
    cipher_seed: int


def gen_naibbe(
    plain_ids: np.ndarray, language: str, rng: np.random.Generator
) -> NaibbeInstance:
    from ..ciphers.naibbe import NaibbeCipher
    from ..vocab import LETTERS
    from .ngram import LETTER_TO_IDX

    text = "".join(LETTERS[i] for i in plain_ids)
    seed = int(rng.integers(2**31))
    tokens, segments = NaibbeCipher(seed=seed).encipher(text)
    plain23 = np.array([LETTER_TO_IDX[c] for c in "".join(segments)], dtype=np.int64)
    return NaibbeInstance(language, plain23, tokens, segments, seed)


# -- metrics ----------------------------------------------------------------


def decode(cipher_ids: np.ndarray, sym_to_letter: np.ndarray) -> np.ndarray:
    return sym_to_letter[cipher_ids]


def ser(cipher: SyntheticCipher, sym_to_letter: np.ndarray) -> float:
    """Symbol error rate of the induced decipherment (field convention)."""
    return float(np.mean(decode(cipher.cipher_ids, sym_to_letter) != cipher.plain_ids))


def map_accuracy(cipher: SyntheticCipher, sym_to_letter: np.ndarray) -> float:
    """Occurrence-weighted letter-map accuracy over symbols that occur."""
    occ = np.bincount(cipher.cipher_ids, minlength=cipher.n_symbols)
    correct = (sym_to_letter == cipher.true_map).astype(float)
    return float((correct * occ).sum() / occ.sum())
