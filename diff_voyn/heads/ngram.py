"""Per-language character n-gram LMs — task CH.0.

Interpolated Witten-Bell models over the frozen 25-letter alphabet (vocab v1),
trained on the *train* side of splits v1 only (the held-out split is the
calibration set for every evaluator, n-gram included — same hygiene as the
backbone). Two roles (prototyping doc §4):

- **DP order (trigram)** — the semi-Markov forward DP state space is
  A^(order-1); trigram (625 states) is the tractable inner-loop scorer.
- **Anchor order (pentagram)** — the literature anchors (Zodiac-408, Borg,
  BnF fr2988) were solved with pentagram-scale scorers; fixed-alignment heads
  score against this directly (no DP needed).

Tables are dense ``float32`` log-probs, shape ``(A**(k-1), A)`` per order k,
with context codes big-endian in time (oldest letter most significant), so
dropping the oldest context letter is ``code % A**(k-2)``. All orders 1..k_max
are stored — scoring uses the highest order the position's history allows.

Persisted under ``DATA_ROOT/ngram_lms/<version>/<lang>.npz`` with the corpus /
splits / vocab versions recorded; ``load_lm`` verifies them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..vocab import LETTERS, VOCAB_VERSION

NGRAM_LM_VERSION = "v1"

A = len(LETTERS)  # 25
LETTER_TO_IDX = {c: i for i, c in enumerate(LETTERS)}


def encode_letters(text: str) -> np.ndarray:
    """Normalized letter stream -> letter indices 0..A-1 (uint8)."""
    try:
        arr = np.frombuffer(
            text.encode("ascii").translate(
                bytes(LETTER_TO_IDX.get(chr(b), 255) for b in range(256))
            ),
            dtype=np.uint8,
        )
    except UnicodeEncodeError as e:
        raise ValueError("text must be normalized (25-letter alphabet)") from e
    if len(arr) and arr.max() >= A:
        bad = chr(text.encode("ascii")[int(np.argmax(arr >= A))])
        raise ValueError(f"non-alphabet character {bad!r}; normalize first")
    return arr


def _ngram_counts(streams: list[np.ndarray], order: int) -> np.ndarray:
    """Dense counts (A**(order-1), A). Windows never cross document bounds."""
    counts = np.zeros(A**order, dtype=np.int64)
    for arr in streams:
        if len(arr) < order:
            continue
        x = arr.astype(np.int64)
        code = np.zeros(len(arr) - order + 1, dtype=np.int64)
        for j in range(order):
            code += x[j : len(arr) - order + 1 + j] * A ** (order - 1 - j)
        counts += np.bincount(code, minlength=A**order)
    return counts.reshape(A ** (order - 1), A)


def _witten_bell(counts: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Interpolated Witten-Bell: p(w|h) = lam*ML + (1-lam)*p_lower(w|h').

    counts: (C, A) for this order; lower: (C // A, A) log-probs of the
    next-lower order, indexed by ``h % (C // A)`` (drop oldest letter).
    Returns log-probs (C, A).
    """
    C = counts.shape[0]
    total = counts.sum(axis=1, keepdims=True).astype(np.float64)
    distinct = (counts > 0).sum(axis=1, keepdims=True).astype(np.float64)
    lam = np.divide(total, total + distinct, out=np.zeros_like(total), where=total > 0)
    ml = np.divide(counts, total, out=np.zeros_like(total * counts), where=total > 0)
    p_low = np.exp(lower[np.arange(C) % lower.shape[0]])
    p = lam * ml + (1.0 - lam) * p_low
    return np.log(p).astype(np.float32)


@dataclass
class NgramLM:
    """All-orders interpolated LM for one language."""

    language: str
    k_max: int
    logp: dict[int, np.ndarray]  # order -> (A**(k-1), A) float32 log-probs
    meta: dict

    def table(self, order: int) -> np.ndarray:
        return self.logp[order]

    def score_ids(self, ids: np.ndarray, order: int | None = None) -> float:
        """Total log-prob (nats) of a letter-index stream; initial positions
        use the highest order their history allows."""
        order = order or self.k_max
        ids = ids.astype(np.int64)
        total = 0.0
        for k in range(1, order):  # warm-up positions 0..order-2
            if len(ids) < k:
                return total
            ctx = 0
            for j in range(k - 1):
                ctx = ctx * A + ids[k - 1 - (k - 1) + j]  # ids[0..k-2]
            total += float(self.logp[k][ctx, ids[k - 1]])
        if len(ids) < order:
            return total
        x = ids
        code = np.zeros(len(x) - order + 1, dtype=np.int64)
        for j in range(order - 1):
            code += x[j : len(x) - order + 1 + j] * A ** (order - 2 - j)
        total += float(self.logp[order][code, x[order - 1 :]].sum())
        return total

    def bits_per_char(self, ids: np.ndarray, order: int | None = None) -> float:
        return -self.score_ids(ids, order) / (len(ids) * np.log(2.0))


def train_lm(
    language: str, streams: list[np.ndarray], k_max: int = 5, meta: dict | None = None
) -> NgramLM:
    logp: dict[int, np.ndarray] = {}
    uni = _ngram_counts(streams, 1)  # (1, A)
    # Unigram gets add-1 so no letter has -inf mass at the interpolation floor.
    p1 = (uni + 1.0) / (uni + 1.0).sum()
    logp[1] = np.log(p1).astype(np.float32)
    for k in range(2, k_max + 1):
        logp[k] = _witten_bell(_ngram_counts(streams, k), logp[k - 1])
    return NgramLM(language, k_max, logp, meta or {})


# ---------------------------------------------------------------------------
# Corpus plumbing + persistence


def _read_docs(corpus_dir: Path, lang: str, doc_ids: list[str]) -> list[np.ndarray]:
    return [
        encode_letters((corpus_dir / lang / "docs" / f"{d}.txt").read_text())
        for d in doc_ids
    ]


def train_from_corpus(
    corpus_dir: Path, splits: dict, lang: str, k_max: int = 5
) -> tuple[NgramLM, float]:
    """Train on the train split; return (lm, heldout_bits_per_char)."""
    train_ids = [d["doc_id"] for d in splits["languages"][lang]["train"]]
    held_ids = [d["doc_id"] for d in splits["languages"][lang]["heldout"]]
    lm = train_lm(
        lang,
        _read_docs(corpus_dir, lang, train_ids),
        k_max,
        meta={
            "ngram_lm_version": NGRAM_LM_VERSION,
            "corpus_version": splits["corpus_version"],
            "splits_version": splits["splits_version"],
            "vocab_version": VOCAB_VERSION,
            "k_max": k_max,
            "train_docs": len(train_ids),
            "smoothing": "interpolated witten-bell",
        },
    )
    held = _read_docs(corpus_dir, lang, held_ids)
    n = sum(len(h) for h in held)
    bits = sum(-lm.score_ids(h) for h in held) / (n * np.log(2.0))
    lm.meta["heldout_bits_per_char"] = float(bits)
    return lm, float(bits)


def save_lm(lm: NgramLM, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{lm.language}.npz"
    np.savez_compressed(
        path,
        meta=json.dumps({**lm.meta, "language": lm.language}),
        **{f"logp_{k}": v for k, v in lm.logp.items()},
    )
    return path


def load_lm(path: Path) -> NgramLM:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        if meta["vocab_version"] != VOCAB_VERSION:
            raise RuntimeError(f"{path}: vocab {meta['vocab_version']} != current")
        logp = {int(k.split("_")[1]): z[k] for k in z.files if k.startswith("logp_")}
    return NgramLM(meta["language"], meta["k_max"], logp, meta)


def lm_dir(version: str = NGRAM_LM_VERSION) -> Path:
    from ..ciphers.external import data_root

    return data_root() / "ngram_lms" / version
