"""Per-position *confidence masks* for a trial decipherment.

A cipher head hands the judge a full plaintext in which some letters are
right and some are wrong. This module builds the boolean "observed" masks the
:mod:`heads.masked_bits` scorer consumes — one entry per plaintext position,
``True`` = show the letter, ``False`` = hand the judge a blank.

All rules here are for the 1:1 position heads (``sub1to1``/``homophonic``),
where plaintext position *i* comes from cipher symbol ``symbols[i]``.

Deployable rules (no truth needed):

- :func:`freq_mask` — keep the most frequent cipher symbols. Language-neutral
  by construction (a property of the ciphertext alone, so all three language
  conditions get the *identical* mask — the R1 fairness requirement), and
  motivated by the project's own failures: rare symbols are where keys go
  wrong (Borg's < 20-occurrence types, the ``elbo_polish`` winner's curse).
- :func:`ngram_sensitivity` + :func:`sensitivity_mask` — keep the symbols
  whose assignment the inner n-gram search is most sure of. ``per_language``
  gives the *biased* form (one mask per hypothesis, the thing the n-gram
  judge's German drift can leak into); the default ``min`` reduction over
  languages is the fair form (one shared mask).

Controls:

- :func:`random_symbol_mask` / :func:`random_position_mask` — matched-coverage
  thinning with no confidence signal. If a confidence rule does not beat
  these, the confidence measure contributed nothing and only the thinning
  did.
- :func:`oracle_mask` — keep the positions that are actually correct. Not
  deployable (needs the plaintext); it is the ceiling on what any confidence
  measure could buy.
"""

from __future__ import annotations

import numpy as np

from .ngram import A


def _coverage(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else 0.0


def symbols_by_count(symbols: np.ndarray) -> np.ndarray:
    """Distinct symbols, most frequent first (ties by symbol id)."""
    vals, cnt = np.unique(symbols, return_counts=True)
    return vals[np.lexsort((vals, -cnt))]


def mask_from_symbol_order(
    symbols: np.ndarray, order: np.ndarray, target: float
) -> tuple[np.ndarray, list[int]]:
    """Keep symbols from ``order`` until the covered fraction of positions
    first reaches ``target``. Returns (position mask, kept symbols)."""
    symbols = np.asarray(symbols)
    n = len(symbols)
    counts = {int(s): int((symbols == s).sum()) for s in np.unique(symbols)}
    kept: list[int] = []
    covered = 0
    for s in order:
        if covered / n >= target:
            break
        kept.append(int(s))
        covered += counts[int(s)]
    mask = np.isin(symbols, kept)
    return mask, kept


def freq_mask(symbols: np.ndarray, target: float) -> tuple[np.ndarray, dict]:
    """Frequency rule: keep the most frequent cipher symbols."""
    order = symbols_by_count(symbols)
    mask, kept = mask_from_symbol_order(symbols, order, target)
    return mask, {
        "rule": "freq",
        "target": target,
        "n_kept_symbols": len(kept),
        "kept_symbols": kept,
        "coverage": _coverage(mask),
    }


def random_symbol_mask(
    symbols: np.ndarray, target: float, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Matched-coverage control that drops whole symbols at random."""
    order = rng.permutation(np.unique(symbols))
    mask, kept = mask_from_symbol_order(symbols, order, target)
    return mask, {
        "rule": "random_symbol",
        "target": target,
        "n_kept_symbols": len(kept),
        "kept_symbols": kept,
        "coverage": _coverage(mask),
    }


def random_position_mask(
    n: int, target: float, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Matched-coverage control that drops individual positions at random."""
    k = int(round(target * n))
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=k, replace=False)] = True
    return mask, {
        "rule": "random_position",
        "target": target,
        "coverage": _coverage(mask),
    }


def oracle_mask(
    decode: np.ndarray,
    truth: np.ndarray,
    target: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Keep correct positions. ``target=None`` keeps exactly the correct set;
    otherwise the mask is resized to ``target`` coverage — subsampling correct
    positions when the key is better than the target, topping up with wrong
    ones when it is worse (so the coverage matches the other rules)."""
    correct = np.asarray(decode) == np.asarray(truth)
    acc = float(correct.mean())
    if target is None:
        return correct.copy(), {
            "rule": "oracle",
            "target": None,
            "accuracy": acc,
            "coverage": _coverage(correct),
        }
    n = len(correct)
    k = int(round(target * n))
    idx_c = np.flatnonzero(correct)
    idx_w = np.flatnonzero(~correct)
    mask = np.zeros(n, dtype=bool)
    if k <= len(idx_c):
        mask[rng.choice(idx_c, size=k, replace=False)] = True
        purity = 1.0
    else:
        mask[idx_c] = True
        extra = k - len(idx_c)
        mask[rng.choice(idx_w, size=extra, replace=False)] = True
        purity = len(idx_c) / k if k else 1.0
    return mask, {
        "rule": "oracle",
        "target": target,
        "accuracy": acc,
        "purity": purity,
        "coverage": _coverage(mask),
    }


def ngram_sensitivity(
    symbols: np.ndarray,
    sym_to_letter: np.ndarray,
    lm,
    order: int | None = None,
) -> np.ndarray:
    """Per-symbol bits the n-gram model loses when a symbol is reassigned to
    its best *alternative* letter, every other assignment held fixed:
    ``min_{l' != l(s)} [ score(decode) - score(decode with s -> l') ]``.
    Large = the search is confident about this symbol. Indexed by symbol id.
    """
    symbols = np.asarray(symbols, dtype=np.int64)
    decode = np.asarray(sym_to_letter, dtype=np.int64)[symbols]
    base = lm.score_ids(decode, order)
    uniq = np.unique(symbols)
    out = np.full(int(symbols.max()) + 1, -np.inf)
    for s in uniq:
        where = symbols == s
        cur = int(sym_to_letter[s])
        best = np.inf
        for l in range(A):
            if l == cur:
                continue
            alt = decode.copy()
            alt[where] = l
            best = min(best, base - lm.score_ids(alt, order))
        out[int(s)] = best
    return out


def sensitivity_mask(
    symbols: np.ndarray, sens: np.ndarray, target: float
) -> tuple[np.ndarray, dict]:
    """Keep the symbols the n-gram search is most confident about."""
    uniq = np.unique(symbols)
    order = uniq[np.argsort(-sens[uniq], kind="stable")]
    mask, kept = mask_from_symbol_order(symbols, order, target)
    return mask, {
        "rule": "ngram_sensitivity",
        "target": target,
        "n_kept_symbols": len(kept),
        "kept_symbols": kept,
        "coverage": _coverage(mask),
    }


def shuffle_within_mask(
    decode: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """The structure-margin control for a masked decode: permute the observed
    letters among the observed positions and the blanked letters among the
    blanked positions.

    Permuting *within* the groups is what keeps the comparison honest — the
    observed subsequence of the shuffled copy carries exactly the same letter
    multiset as the observed subsequence of the decode, so the margin measures
    sequential structure only, never the unigram distribution the mask
    selected. With an all-observed mask this is a plain permutation of the
    window, i.e. the frozen Phase-6 control.
    """
    out = np.asarray(decode).copy()
    for grp in (mask, ~mask):
        idx = np.flatnonzero(grp)
        if idx.size > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


# -- shared (fair) reduction, controlled-wrongness keys, and the E1 metric ---


def shared_sensitivity(sens_by_language: dict[str, np.ndarray]) -> np.ndarray:
    """The R1-fair reduction: one confidence value per symbol, the ``min``
    over the language hypotheses' sensitivities — a symbol is trusted only
    if every hypothesis' key is sure of it, so the resulting mask is
    identical for all three conditions."""
    arrs = list(sens_by_language.values())
    n = max(len(a) for a in arrs)
    out = np.full(n, np.inf)
    for a in arrs:
        b = np.full(n, np.inf)
        b[: len(a)] = a
        out = np.minimum(out, b)
    return out


def derange_key(
    true_map: np.ndarray, symbols: np.ndarray, f: float, rng: np.random.Generator
) -> np.ndarray:
    """A key with a fraction ``f`` of the *occurring* symbols' assignments
    permuted among themselves (self-consistent; the ``control6a`` derangement
    generalized to non-bijective maps). ``f = 0`` returns the true key."""
    key = np.asarray(true_map).copy()
    occ = np.unique(symbols)
    n = int(round(f * len(occ)))
    if n < 2:
        return key
    idx = rng.choice(occ, size=n, replace=False)
    while True:
        p = rng.permutation(n)
        if (p != np.arange(n)).all():
            break
    key[idx] = np.asarray(true_map)[idx[p]]
    return key


def symbol_correct(
    symbols: np.ndarray, key: np.ndarray, true_map: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(occurring symbols, bool correct) — the E1 label."""
    occ = np.unique(symbols)
    return occ, np.asarray(key)[occ] == np.asarray(true_map)[occ]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC of ``scores`` for ``labels`` (True = positive);
    ties count half. ``nan`` if one class is absent."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
