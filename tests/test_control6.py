"""Control experiments 6a / 6b — library-level behaviour the scripts rely on."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from diff_voyn.data.loader import LANG_TO_INDEX, NULL_LANG_INDEX
from diff_voyn.data.noise import LETTER_BASE, NoiseConfig, NoiseMixture
from diff_voyn.heads.ngram import A, NgramLM, _ngram_counts, _witten_bell
from diff_voyn.heads.two_tier import CONDITION_UNCOND, condition_index


def test_condition_index_maps_uncond_to_null_language():
    assert condition_index(CONDITION_UNCOND) == NULL_LANG_INDEX
    for lang, idx in LANG_TO_INDEX.items():
        assert condition_index(lang) == idx
    with pytest.raises(KeyError):
        condition_index("english")


def test_noise_config_without_wrong_key_family_never_substitutes():
    cfg = NoiseConfig(**{**NoiseConfig().to_dict(), "p_substitution": 0.0})
    mix = NoiseMixture(cfg)
    rng = np.random.default_rng(0)
    ids = (rng.integers(0, A, size=600) + LETTER_BASE).astype(np.uint8)
    for _ in range(50):
        _, info = mix._noise(ids.copy(), rng)
        assert "substitution" not in info
        assert info  # at least one of the remaining families applied


def _toy_stream(rng, n=5000):
    # a peaked Markov chain so the LM has structure to interpolate over
    x = [int(rng.integers(0, A))]
    for _ in range(n - 1):
        x.append((x[-1] + int(rng.integers(0, 3))) % A)
    return np.asarray(x, dtype=np.int64)


def test_weighted_pooled_counts_match_equal_weight_lm():
    """Scaling counts by a per-stream weight is what the pooled LM does;
    scaling every stream by the same factor leaves Witten-Bell's ML term
    unchanged and moves only its interpolation weight."""
    rng = np.random.default_rng(1)
    s = [_toy_stream(rng) for _ in range(3)]
    c2 = sum(_ngram_counts([x], 2) for x in s)
    uni = sum(_ngram_counts([x], 1) for x in s)
    p1 = np.log((uni + 1.0) / (uni + 1.0).sum()).astype(np.float32)
    lp = _witten_bell(c2, p1)
    lp_scaled = _witten_bell(0.5 * c2, p1)
    assert lp.shape == (A, A)
    assert np.allclose(np.exp(lp).sum(1), 1.0, atol=1e-4)
    assert np.allclose(np.exp(lp_scaled).sum(1), 1.0, atol=1e-4)
    # fractional counts are accepted and the ML ranking of continuations holds
    assert (np.argsort(lp, 1) == np.argsort(lp_scaled, 1))[:, -1].mean() > 0.9
    lm = NgramLM("pooled", 2, {1: p1, 2: lp}, {})
    assert np.isfinite(lm.score_ids(s[0][:100], 2))


class _ToyEvaluator:
    """Stands in for the diffusion evaluator inside ``paired_bits``: a
    backbone whose logits reward decodes equal to a fixed target."""

    def __init__(self, target: np.ndarray):
        from diff_voyn.vocab import VOCAB_SIZE

        self.device = torch.device("cpu")
        self.t_floor = 1e-3
        self.autocast = False
        self.window = 1024
        tgt = torch.from_numpy(target).long()
        V = VOCAB_SIZE

        class _B(torch.nn.Module):
            def forward(self, z, lang):
                B, L = z.shape
                logits = torch.full((B, L, V), -3.0)
                logits[
                    torch.arange(B)[:, None],
                    torch.arange(L)[None],
                    tgt[None].expand(B, L),
                ] = 3.0
                return logits

        self.backbone = _B()

    def _windows(self, n):
        return [(0, n)]


def test_elbo_polish_set_moves_false_keeps_key_bijective():
    from diff_voyn.heads.ladder import elbo_polish
    from diff_voyn.vocab import LETTER_IDS

    rng = np.random.default_rng(0)
    plain = rng.integers(0, A, size=120)
    perm = rng.permutation(A)  # letter -> symbol
    cipher = perm[plain]
    true_key = np.argsort(perm)
    start = true_key.copy()
    start[[0, 1]] = start[[1, 0]]  # one swap away
    letters = np.asarray(LETTER_IDS)
    ev = _ToyEvaluator(letters[plain])
    out, info = elbo_polish(
        ev,
        cipher,
        start,
        language="uncond",
        sweeps=3,
        budget=2,
        confirm_budget=2,
        pair_swaps=True,
        set_moves=False,
    )
    assert sorted(out.tolist()) == list(range(A))  # still a bijection
    assert info["accepted"] and (out == true_key).all()


def test_swap_search_recovers_key_under_exact_objective():
    from control6a_judge_vs_target import swap_search

    rng = np.random.default_rng(3)
    plain = rng.integers(0, A, size=80)
    perm = rng.permutation(A)
    cipher = perm[plain]
    true_key = np.argsort(perm)

    def objective(rows):  # hamming distance to the plaintext, per row
        return (rows != plain[None]).mean(1)

    start = true_key.copy()
    idx = rng.choice(A, size=6, replace=False)
    start[idx] = true_key[np.roll(idx, 1)]
    found, trace, n_evals = swap_search(
        objective, cipher, plain, start, sweeps=20, kicks=2
    )
    assert (found[cipher] == plain).all()
    assert trace[0]["ser"] > 0 and n_evals > 0
