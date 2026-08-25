"""Confidence-mask probe, scorer side (docs/confidence_mask_probe.md §8):
the all-true mask reproduces ``paired_bits`` bit-for-bit, blanked positions
never enter the loss, ``shuffle_within_mask`` preserves both multisets, and
the NaN smoke test."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diff_voyn.heads.confidence import shuffle_within_mask
from diff_voyn.heads.masked_bits import paired_bits_masked
from diff_voyn.heads.ngram import A
from diff_voyn.heads.two_tier import paired_bits

LANGS = ["latin", "italian", "german"]


@pytest.fixture(scope="module")
def ev():
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator
    from diff_voyn.infra.config import ModelConfig
    from diff_voyn.model.backbone import Backbone

    torch.manual_seed(0)
    cfg = ModelConfig(n_layers=2, d_model=64, n_heads=4, d_ffn=128, seq_len=64)
    return DiffusionEvaluator(Backbone(cfg), n_strata=4, seed=0, window=64)


@pytest.fixture(scope="module")
def rows():
    return np.random.default_rng(1).integers(0, A, size=(3, 40))


def test_all_true_mask_reproduces_paired_bits(ev, rows):
    ref = paired_bits(ev, rows, LANGS, n_strata=8, seed=3)
    got = paired_bits_masked(
        ev, rows, np.ones_like(rows, bool), LANGS, n_strata=8, seed=3
    )
    assert got.shape == ref.shape == (3, 3)
    np.testing.assert_array_equal(got, ref)  # identical draws, identical sums


def test_all_true_mask_reproduces_paired_bits_across_windows(ev):
    rows = np.random.default_rng(2).integers(0, A, size=(2, 150))  # 3 windows
    ref = paired_bits(ev, rows, LANGS, n_strata=8, seed=5)
    got = paired_bits_masked(
        ev, rows, np.ones_like(rows, bool), LANGS, n_strata=8, seed=5
    )
    np.testing.assert_allclose(got, ref, rtol=1e-12)


def test_blanked_positions_never_enter_the_loss(ev, rows):
    obs = np.random.default_rng(3).random(rows.shape) < 0.6
    a = paired_bits_masked(ev, rows, obs, LANGS, n_strata=8, seed=1)
    alt = rows.copy()
    alt[~obs] = (alt[~obs] + 7) % A  # rewrite every blanked letter
    b = paired_bits_masked(ev, alt, obs, LANGS, n_strata=8, seed=1)
    np.testing.assert_array_equal(a, b)
    # ... but they do shape the context: unblanking them changes the score
    c = paired_bits_masked(ev, rows, np.ones_like(obs), LANGS, n_strata=8, seed=1)
    assert not np.allclose(a, c)


def test_masked_bits_are_per_observed_char_and_finite(ev, rows):
    obs = np.zeros_like(rows, bool)
    obs[:, :5] = True  # heavy blanking — NaN smoke
    out = paired_bits_masked(ev, rows, obs, LANGS, n_strata=8, seed=0)
    assert np.isfinite(out).all() and (out > 0).all()
    with pytest.raises(ValueError):
        paired_bits_masked(ev, rows, np.zeros_like(rows, bool), LANGS, n_strata=8)


def test_row_vector_mask_broadcasts(ev, rows):
    m = np.random.default_rng(4).random(rows.shape[1]) < 0.7
    a = paired_bits_masked(ev, rows, m, LANGS, n_strata=8, seed=2)
    b = paired_bits_masked(
        ev, rows, np.broadcast_to(m, rows.shape), LANGS, n_strata=8, seed=2
    )
    np.testing.assert_array_equal(a, b)


def test_shuffle_within_mask_preserves_both_multisets():
    rng = np.random.default_rng(0)
    dec = rng.integers(0, A, size=500)
    mask = rng.random(500) < 0.6
    out = shuffle_within_mask(dec, mask, rng)
    assert np.array_equal(np.sort(out[mask]), np.sort(dec[mask]))
    assert np.array_equal(np.sort(out[~mask]), np.sort(dec[~mask]))
    assert not np.array_equal(out, dec)
    # all-true mask: a plain permutation of the window
    full = shuffle_within_mask(dec, np.ones(500, bool), rng)
    assert np.array_equal(np.sort(full), np.sort(dec))
