"""NELBO canary estimator sanity (task 0.6): uniform model ≈ log2(V) bits/char;
common random numbers across language conditions."""

import torch

from diff_voyn.infra.nelbo import estimate_nelbo_bits_per_char
from diff_voyn.vocab import VOCAB_SIZE


class UniformModel(torch.nn.Module):
    def forward(self, z_t, lang):
        return torch.zeros(*z_t.shape, VOCAB_SIZE)


class LangSensitiveModel(torch.nn.Module):
    """Logit pattern depends on the language index — used to verify CRN."""

    def forward(self, z_t, lang):
        logits = torch.zeros(*z_t.shape, VOCAB_SIZE)
        logits[..., int(lang[0]) % VOCAB_SIZE] = 1.0
        return logits


def test_uniform_model_scores_log2_vocab():
    ids = torch.randint(6, 31, (4, 256))
    bits = estimate_nelbo_bits_per_char(UniformModel(), ids, 0, n_strata=32, seed=0)
    assert abs(bits - 5.0) < 0.15  # log2(32) = 5


def test_same_seed_same_estimate():
    ids = torch.randint(6, 31, (2, 128))
    a = estimate_nelbo_bits_per_char(UniformModel(), ids, 0, n_strata=16, seed=3)
    b = estimate_nelbo_bits_per_char(UniformModel(), ids, 0, n_strata=16, seed=3)
    assert a == b


def test_crn_isolates_language_effect():
    """With a shared seed, the between-language difference must be identical
    across repeats (the masking noise cancels out of the difference)."""
    ids = torch.randint(6, 31, (2, 128))
    model = LangSensitiveModel()
    d1 = estimate_nelbo_bits_per_char(
        model, ids, 0, n_strata=16, seed=11
    ) - estimate_nelbo_bits_per_char(model, ids, 1, n_strata=16, seed=11)
    d2 = estimate_nelbo_bits_per_char(
        model, ids, 0, n_strata=16, seed=99
    ) - estimate_nelbo_bits_per_char(model, ids, 1, n_strata=16, seed=99)
    # different seeds, but the language-difference is a model property:
    # CRN makes each difference exact per seed; across seeds they stay close.
    assert abs(d1 - d2) < 0.05
