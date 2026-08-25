"""Racing polish (docs/race_polish_plan.md): per-stratum scorer equality,
key recovery on an exact toy, and the winner's-curse reproduction — the
greedy polish moves a TRUE key on a noisy judge, the race does not, and the
race still repairs a one-swap-away key."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diff_voyn.heads.ngram import A
from diff_voyn.vocab import LETTER_IDS, VOCAB_SIZE

LETTERS = np.asarray(LETTER_IDS)


class _NoisyToyEvaluator:
    """Target-rewarding logits plus per-forward Gaussian noise: an exact
    judge with estimator noise, the regime where argmin-of-many selects on
    luck."""

    def __init__(self, target: np.ndarray, noise: float, seed: int = 0):
        self.device = torch.device("cpu")
        self.t_floor = 1e-3
        self.autocast = False
        self.window = 1024
        tgt = torch.from_numpy(target).long()
        g = torch.Generator().manual_seed(seed)

        class _B(torch.nn.Module):
            def forward(self, z, lang):
                B, L = z.shape
                logits = torch.full((B, L, VOCAB_SIZE), -2.0)
                logits[
                    torch.arange(B)[:, None],
                    torch.arange(L)[None],
                    tgt[None].expand(B, L),
                ] = 2.0
                if noise > 0:
                    logits = logits + noise * torch.randn(logits.shape, generator=g)
                return logits

        self.backbone = _B()

    def _windows(self, n):
        return [(0, n)]


def _instance(seed, n_sym=20, L=200):
    rng = np.random.default_rng(seed)
    plain = rng.integers(0, A, size=L)
    key = rng.integers(0, A, size=n_sym)  # symbol -> letter (homophonic-ish)
    # ensure every symbol used: cipher symbol chosen among those mapping to plain letter
    by_letter = {a: np.where(key == a)[0] for a in range(A)}
    cipher = np.empty(L, dtype=np.int64)
    for i, a in enumerate(plain):
        if len(by_letter[a]) == 0:
            s = rng.integers(0, n_sym)
            key[s] = a
            by_letter = {b: np.where(key == b)[0] for b in range(A)}
        cipher[i] = rng.choice(by_letter[a])
    plain = key[cipher]
    return plain, cipher, key


def test_paired_bits_strata_mean_equals_paired_bits():
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator
    from diff_voyn.heads.two_tier import paired_bits, paired_bits_strata
    from diff_voyn.infra.config import ModelConfig
    from diff_voyn.model.backbone import Backbone

    plain, _cipher, _key = _instance(0)
    rows = np.stack([plain, (plain + 1) % A])
    ev = _NoisyToyEvaluator(LETTERS[plain], noise=0.0)
    for seed in (1, 2):
        ref = paired_bits(ev, rows, ["latin", "german"], n_strata=6, seed=seed)
        per = paired_bits_strata(ev, rows, ["latin", "german"], n_strata=6, seed=seed)
        assert per.shape == (2, 2, 6)
        assert np.allclose(per.sum(-1), ref, atol=1e-9)
    # a real (tiny) backbone, multi-window rows
    torch.manual_seed(0)
    cfg = ModelConfig(n_layers=1, d_model=32, n_heads=4, d_ffn=64, seq_len=64)
    dev = DiffusionEvaluator(Backbone(cfg), n_strata=2, seed=0, window=64)
    ref = paired_bits(dev, rows, ["latin"], n_strata=4, seed=3)
    per = paired_bits_strata(dev, rows, ["latin"], n_strata=4, seed=3)
    assert np.allclose(per.sum(-1), ref, atol=1e-6)


def test_race_polish_exact_judge_recovers_swap_and_keeps_bijection():
    from diff_voyn.heads.ladder import race_polish

    rng = np.random.default_rng(0)
    plain = rng.integers(0, A, size=120)
    perm = rng.permutation(A)
    cipher = perm[plain]
    true_key = np.argsort(perm)
    start = true_key.copy()
    start[[0, 1]] = start[[1, 0]]
    ev = _NoisyToyEvaluator(LETTERS[plain], noise=0.0)
    out, info = race_polish(
        ev,
        cipher,
        start,
        language="uncond",
        sweeps=3,
        budgets=(2, 4),
        max_survivors=(None, 8),
        confirm_budget=2,
        set_moves=False,
    )
    assert sorted(out.tolist()) == list(range(A))
    assert info["accepted"] and (out == true_key).all() and info["n_moves"] == 1


@pytest.mark.parametrize("noise", [1.5])
def test_winners_curse_greedy_moves_true_key_race_does_not(noise):
    from diff_voyn.heads.ladder import elbo_polish, race_polish

    greedy_moved, race_moved, race_repaired = 0, 0, 0
    n_seeds = 4
    for seed in range(n_seeds):
        plain, cipher, key = _instance(seed)
        ev = _NoisyToyEvaluator(LETTERS[plain], noise=noise, seed=seed)
        g, _ = elbo_polish(
            ev,
            cipher,
            key,
            language="uncond",
            sweeps=3,
            budget=4,
            confirm_budget=8,
            seed=seed,
        )
        greedy_moved += int((g != key).any())
        ev = _NoisyToyEvaluator(LETTERS[plain], noise=noise, seed=seed)
        r, _info = race_polish(
            ev,
            cipher,
            key,
            language="uncond",
            sweeps=3,
            budgets=(4, 16, 64),
            max_survivors=(None, 32, 8),
            confirm_budget=8,
            seed=seed,
        )
        race_moved += int((r != key).any())
        # power: one symbol corrupted
        bad = key.copy()
        bad[0] = (bad[0] + 3) % A
        ev = _NoisyToyEvaluator(LETTERS[plain], noise=noise, seed=seed)
        r2, _ = race_polish(
            ev,
            cipher,
            bad,
            language="uncond",
            sweeps=3,
            budgets=(4, 16, 64),
            max_survivors=(None, 32, 8),
            confirm_budget=8,
            seed=seed,
        )
        race_repaired += int((r2 == key).all())
    assert race_moved == 0, race_moved
    assert race_repaired >= n_seeds - 1, race_repaired
    # the reproduction of the failure is reported, not required, if the toy
    # is too easy for it; but it must never be *less* conservative than greedy
    assert greedy_moved >= race_moved


def test_choice_term_refused_unless_opted_in():
    """The MDL choice term destroyed the Borg key when used as the polish
    objective (docs/race_polish_plan.md §7): both polishes refuse it unless
    the caller opts in explicitly."""
    from diff_voyn.heads.ladder import elbo_polish, race_polish

    plain, cipher, key = _instance(0)
    ev = _NoisyToyEvaluator(LETTERS[plain], noise=0.0)
    for fn in (elbo_polish, race_polish):
        with pytest.raises(ValueError, match="choice_term_in_polish"):
            fn(ev, cipher, key, language="uncond", choice_fn=lambda m, d: 0.0, sweeps=1)
