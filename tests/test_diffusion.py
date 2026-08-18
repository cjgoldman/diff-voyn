"""Task 1.1 acceptance: the continuous-time Rao-Blackwellized NELBO matches a
reference MDLM computation to numerical tolerance.

Reference #1 — *exact* NELBO: for a tiny model and a short sequence, the inner
expectation E_q[Σ_masked −log p_θ(x_i|z_t)] is enumerated over all 2^L mask
patterns, and the time integral ∫ (1/t)·E(t) dt is taken by fine midpoint
quadrature. No sampling anywhere.

Reference #2 — the *discrete-time* T-step MDLM bound
L_T = Σ_i (1/T)(1/t_i)·E(t_i) (α_t = 1−t), whose T→∞ limit is the
continuous-time objective; we check convergence toward the exact value.

Both references are computed independently of the package code paths.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from diff_voyn.infra.config import ModelConfig
from diff_voyn.infra.nelbo import estimate_nelbo_bits_per_char
from diff_voyn.model.backbone import Backbone
from diff_voyn.model.diffusion import LN2, mdlm_loss, mdlm_nelbo_terms
from diff_voyn.vocab import LETTER_IDS, MASK_ID, VOCAB_SIZE

TINY = ModelConfig(
    n_layers=2, d_model=64, n_heads=4, d_ffn=128, dropout=0.0, seq_len=64
)


def tiny_model(seed: int = 0) -> Backbone:
    torch.manual_seed(seed)
    return Backbone(TINY).eval()


def exact_expected_masked_nll(model, x: torch.Tensor, lang: torch.Tensor, t: float):
    """E_q[Σ_masked −log p(x_i|z_t)] at time t, enumerated over mask patterns."""
    length = x.shape[0]
    total = 0.0
    for bits in range(1, 1 << length):
        pattern = torch.tensor(
            [(bits >> i) & 1 for i in range(length)], dtype=torch.bool
        )
        k = int(pattern.sum())
        prob = t**k * (1 - t) ** (length - k)
        z = x.masked_fill(pattern, MASK_ID)
        logp = F.log_softmax(model(z[None], lang).float(), dim=-1)[0]
        nll = -logp.gather(-1, x[:, None]).squeeze(-1)[pattern].sum()
        total += prob * float(nll)
    return total


@torch.no_grad()
def exact_continuous_nelbo_bits(model, x, lang, n_quad: int = 256) -> float:
    """∫₀¹ (1/t)·E(t) dt per char, midpoint quadrature, in bits/char."""
    length = x.shape[0]
    total = 0.0
    for j in range(n_quad):
        t = (j + 0.5) / n_quad
        total += exact_expected_masked_nll(model, x, lang, t) / t / n_quad
    return total / length / LN2


@torch.no_grad()
def discrete_time_nelbo_bits(model, x, lang, n_steps: int) -> float:
    """The T-step MDLM bound Σ_i (1/T)(1/t_i)·E(t_i), t_i = i/T, bits/char."""
    length = x.shape[0]
    total = 0.0
    for i in range(1, n_steps + 1):
        t = i / n_steps
        total += exact_expected_masked_nll(model, x, lang, t) / t / n_steps
    return total / length / LN2


@pytest.fixture(scope="module")
def setup():
    model = tiny_model()
    torch.manual_seed(1)
    x = torch.tensor(LETTER_IDS, dtype=torch.long)[torch.randint(0, 25, (6,))]
    lang = torch.zeros(1, dtype=torch.long)
    exact = exact_continuous_nelbo_bits(model, x, lang)
    return model, x, lang, exact


def test_stratified_estimator_matches_exact_reference(setup):
    model, x, _lang, exact = setup
    est = estimate_nelbo_bits_per_char(
        model, x[None], 0, n_strata=128, samples_per_stratum=8, seed=0
    )
    assert abs(est - exact) / exact < 0.05, (est, exact)


def test_training_loss_matches_exact_reference(setup):
    """Monte-Carlo average of the training-loss integrand over the *training*
    forward-process draws (uniform t, i.i.d. masking) converges to the exact
    NELBO — the loss and the metric estimate the same integral."""
    model, x, _lang, exact = setup
    g = torch.Generator().manual_seed(0)
    n = 6000
    ids = x[None].expand(n, -1)
    # antithetic-stratified t over (0,1] to keep the 1/t-weighted MC stable
    t = (torch.arange(n) + torch.rand(n, generator=g)) / n
    t = t.clamp_min(1e-4)
    masked = torch.rand(ids.shape, generator=g) < t[:, None]
    z = ids.masked_fill(masked, MASK_ID)
    with torch.no_grad():
        logits = model(z, torch.zeros(n, dtype=torch.long))
        terms = mdlm_nelbo_terms(logits, ids, masked, t)
    mc = float(terms.mean()) / LN2
    assert abs(mc - exact) / exact < 0.05, (mc, exact)


def test_discrete_time_bound_converges_to_continuous(setup):
    model, x, lang, exact = setup
    err_64 = abs(discrete_time_nelbo_bits(model, x, lang, 64) - exact)
    err_512 = abs(discrete_time_nelbo_bits(model, x, lang, 512) - exact)
    assert err_512 < err_64  # tightening with T
    assert err_512 / exact < 0.01, (err_512, exact)


def test_loss_analytic_uniform_full_mask():
    """Zero logits, everything masked at t=1: exactly log(V) nats/char."""
    ids = torch.randint(6, 31, (3, 16))
    logits = torch.zeros(3, 16, VOCAB_SIZE)
    masked = torch.ones(3, 16, dtype=torch.bool)
    t = torch.ones(3)
    loss = mdlm_loss(logits, ids, masked, t)
    assert abs(float(loss) - math.log(VOCAB_SIZE)) < 1e-6


def test_loss_weighting_partial_mask():
    """k masked of L at time t contributes (1/t)·k·log(V)/L for zero logits."""
    ids = torch.randint(6, 31, (1, 8))
    masked = torch.zeros(1, 8, dtype=torch.bool)
    masked[0, :3] = True
    t = torch.tensor([0.25])
    loss = mdlm_loss(torch.zeros(1, 8, VOCAB_SIZE), ids, masked, t)
    assert abs(float(loss) - (1 / 0.25) * 3 * math.log(VOCAB_SIZE) / 8) < 1e-6


def test_unmasked_example_contributes_zero():
    ids = torch.randint(6, 31, (2, 8))
    masked = torch.zeros(2, 8, dtype=torch.bool)
    terms = mdlm_nelbo_terms(
        torch.zeros(2, 8, VOCAB_SIZE), ids, masked, torch.full((2,), 0.5)
    )
    assert terms.abs().max() == 0.0


def test_subs_zero_masking_probability():
    """Backbone logits carry −inf at MASK; probabilities renormalize over the
    rest and stay finite (no NaN — the design-§8 smoke-test family)."""
    model = tiny_model()
    z = torch.randint(6, 31, (2, 32))
    z[:, ::4] = MASK_ID
    logits = model(z, torch.zeros(2, dtype=torch.long))
    assert torch.isneginf(logits[..., MASK_ID]).all()
    probs = logits.softmax(dim=-1)
    assert torch.isfinite(probs).all()
    assert (probs[..., MASK_ID] == 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-5)


def test_tiny_backbone_overfits_repeated_text():
    """Wiring check: 150 steps on one repeated window cuts the loss sharply."""
    torch.manual_seed(0)
    model = Backbone(TINY).train()
    ids = torch.tensor(LETTER_IDS, dtype=torch.long)[
        torch.randint(0, 25, (64,), generator=torch.Generator().manual_seed(7))
    ][None].expand(16, -1)
    lang = torch.zeros(16, dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    first = last = None
    for step in range(150):
        t = torch.rand(16, generator=g).clamp_min(1e-3)
        masked = torch.rand(ids.shape, generator=g) < t[:, None]
        z = ids.masked_fill(masked, MASK_ID)
        loss = mdlm_loss(model(z, lang), ids, masked, t)
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    assert last < 0.5 * first, (first, last)
