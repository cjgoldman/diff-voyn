"""Task 0.5 acceptance: sampler frequencies within 2% of target over 1M draws;
masking sampler and conditioning-injection mechanics."""

import numpy as np
import torch

from diff_voyn.data.loader import (
    NULL_LANG_INDEX,
    LanguageConditioning,
    LanguageSampler,
    MaskingSampler,
)
from diff_voyn.vocab import MASK_ID


def test_language_sampler_1m_draws_within_2pct():
    # Realistic char counts: german >> latin >> italian
    sampler = LanguageSampler(
        {"german": 200_000_000, "latin": 60_000_000, "italian": 8_000_000},
        temperature=0.7,
    )
    rng = np.random.default_rng(0)
    draws = rng.choice(len(sampler.languages), size=1_000_000, p=sampler.weights)
    counts = np.bincount(draws, minlength=len(sampler.languages)) / 1_000_000
    for target, got in zip(sampler.weights, counts):
        assert abs(got - target) < 0.02, (target, got)


def test_temperature_flattens_distribution():
    counts = {"a": 100, "b": 10_000}
    flat = LanguageSampler(counts, temperature=0.0).weights
    raw = LanguageSampler(counts, temperature=1.0).weights
    assert abs(flat[0] - 0.5) < 1e-9
    assert raw[0] < 0.01


def test_masking_matches_t_and_weight():
    ms = MaskingSampler()
    ids = torch.randint(6, 31, (8, 2048))
    g = torch.Generator().manual_seed(0)
    t = torch.full((8,), 0.3)
    z_t, masked, weight = ms.mask(ids, t, g)
    frac = masked.float().mean().item()
    assert abs(frac - 0.3) < 0.02
    assert (z_t[masked] == MASK_ID).all()
    assert (z_t[~masked] == ids[~masked]).all()
    assert torch.allclose(weight, torch.tensor(1.0 / 0.3))


def test_masking_common_random_numbers_reproducible():
    """Same generator seed → identical masking realization (CRN, design §5a)."""
    ms = MaskingSampler()
    ids = torch.randint(6, 31, (4, 512))
    t = torch.tensor([0.1, 0.4, 0.7, 0.9])
    z1, m1, _ = ms.mask(ids, t, torch.Generator().manual_seed(7))
    z2, m2, _ = ms.mask(ids, t, torch.Generator().manual_seed(7))
    assert torch.equal(m1, m2) and torch.equal(z1, z2)


def test_language_conditioning_dropout_rate():
    cond = LanguageConditioning(d_model=16, p_dropout=0.1)
    cond.train()
    lang_idx = torch.zeros(100_000, dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        out = cond(lang_idx, g)
        null_emb = cond.embedding(torch.tensor([NULL_LANG_INDEX]))
    is_null = (out[:, 0, :] == null_emb).all(dim=-1).float().mean().item()
    assert abs(is_null - 0.1) < 0.005  # 10% conditioning dropout (design §4)

    cond.eval()  # eval mode: no dropout
    with torch.no_grad():
        out = cond(lang_idx[:1000], g)
    assert not (out[:, 0, :] == null_emb).all(dim=-1).any()


def test_conditioning_is_additive_shape():
    cond = LanguageConditioning(d_model=32)
    out = cond(torch.tensor([0, 1, 2]))
    assert out.shape == (3, 1, 32)  # broadcasts additively over positions
