"""Confidence-mask probe, mask side (docs/confidence_mask_probe.md §8):
oracle AUROC = 1, masks identical across language conditions for the
deployable shared rules, coverage targets hit, derangement wrongness."""

from __future__ import annotations

import numpy as np
import pytest

from diff_voyn.heads.confidence import (
    auroc,
    derange_key,
    freq_mask,
    ngram_sensitivity,
    oracle_mask,
    random_position_mask,
    random_symbol_mask,
    sensitivity_mask,
    shared_sensitivity,
    symbol_correct,
)
from diff_voyn.heads.ngram import A, train_lm


@pytest.fixture(scope="module")
def lms():
    rng = np.random.default_rng(0)
    out = {}
    for lang in ("latin", "italian", "german"):
        trans = rng.dirichlet(np.full(A, 0.15), size=A)
        s = [int(rng.integers(A))]
        for _ in range(60_000):
            s.append(int(rng.choice(A, p=trans[s[-1]])))
        out[lang] = train_lm(lang, [np.array(s, np.uint8)], k_max=3)
    return out


@pytest.fixture(scope="module")
def instance(lms):
    """A sub1to1 cipher of a 'latin' sample and a 30 %-wrong key."""
    rng = np.random.default_rng(1)
    lm = lms["latin"]
    p = np.exp(lm.table(2))
    s = [int(rng.integers(A))]
    for _ in range(600):
        s.append(int(rng.choice(A, p=p[s[-1]])))
    plain = np.array(s)
    perm = rng.permutation(A)
    symbols = perm[plain]
    true_map = np.argsort(perm)
    key = derange_key(true_map, symbols, 0.3, rng)
    return plain, symbols, true_map, key


def test_auroc_basic():
    assert auroc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]) == 0.75
    assert auroc([1, 1, 1, 1], [True, False, True, False]) == 0.5
    assert np.isnan(auroc([1, 2], [True, True]))


def test_derange_key_hits_the_requested_fraction():
    symbols = np.repeat(np.arange(A), 5)
    true_map = np.random.default_rng(0).permutation(A)
    for f in (0.0, 0.2, 0.5, 0.65):
        key = derange_key(true_map, symbols, f, np.random.default_rng(3))
        occ, ok = symbol_correct(symbols, key, true_map)
        assert abs((~ok).mean() - f) < 1.0 / A + 1e-9
        assert set(key) == set(true_map)  # still bijective


def test_oracle_mask_auroc_is_one(instance):
    plain, symbols, true_map, key = instance
    decode = key[symbols]
    mask, info = oracle_mask(decode, plain, None, np.random.default_rng(0))
    assert auroc(mask.astype(float), decode == plain) == 1.0
    assert info["coverage"] == pytest.approx((decode == plain).mean())
    # resized to a target: coverage hit, purity reported
    m2, i2 = oracle_mask(decode, plain, 0.5, np.random.default_rng(0))
    assert abs(m2.mean() - 0.5) < 1e-3 and i2["purity"] == 1.0
    m3, i3 = oracle_mask(decode, plain, 0.95, np.random.default_rng(0))
    assert abs(m3.mean() - 0.95) < 1e-3 and i3["purity"] < 1.0


def test_sensitivity_finds_wrong_symbols_and_shared_mask_is_language_free(
    instance, lms
):
    plain, symbols, true_map, key = instance
    occ, ok = symbol_correct(symbols, key, true_map)
    sens = {l: ngram_sensitivity(symbols, key, lm) for l, lm in lms.items()}
    assert auroc(sens["latin"][occ], ok) > 0.8  # the true language's LM
    shared = shared_sensitivity(sens)
    assert (shared[occ] <= sens["latin"][occ]).all()
    # one mask for all three conditions, by construction
    masks = [sensitivity_mask(symbols, shared, 0.7)[0] for _ in range(3)]
    assert all(np.array_equal(masks[0], m) for m in masks)
    fmask = [freq_mask(symbols, 0.7)[0] for _ in range(3)]
    assert all(np.array_equal(fmask[0], m) for m in fmask)


def test_coverage_targets_hit(instance):
    plain, symbols, true_map, key = instance
    n = len(symbols)
    counts = np.bincount(symbols)
    rng = np.random.default_rng(0)
    for target in (0.9, 0.8, 0.7, 0.5):
        for mask, info in (
            freq_mask(symbols, target),
            random_symbol_mask(symbols, target, rng),
            sensitivity_mask(symbols, rng.random(A), target),
        ):
            cov = mask.mean()
            assert cov >= target - 1e-9
            # overshoot bounded by the last symbol added
            assert cov - target <= counts[info["kept_symbols"][-1]] / n + 1e-9
            assert info["coverage"] == pytest.approx(cov)
        m, _ = random_position_mask(n, target, rng)
        assert abs(m.mean() - target) < 1.0 / n + 1e-9
