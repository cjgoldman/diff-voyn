"""Toy tests for the denoiser-posterior proposer and the alternating loop
(docs/alt_loop_plan.md §5)."""

import numpy as np
import torch

from diff_voyn.heads.altloop import alternate
from diff_voyn.heads.posterior import (
    A,
    disagreements,
    position_posterior,
    symbol_scores,
    unit_scores,
)
from diff_voyn.heads.wordhom import UnitTargets, expand_units
from diff_voyn.vocab import LETTER_IDS, VOCAB_SIZE


class _Toy:
    """Denoiser whose logits favour a fixed target stream (letter ids)."""

    def __init__(self, target):
        self.device = torch.device("cpu")
        self.autocast = False
        self.window = 1024
        self.stratum_batch = 8
        tgt = torch.tensor([LETTER_IDS[t] for t in target])

        class _B(torch.nn.Module):
            def forward_soft(self, z, lang):
                B, L, _ = z.shape
                logits = torch.full((B, L, VOCAB_SIZE), -3.0)
                logits[torch.arange(B)[:, None], torch.arange(L)[None], tgt[None]] = 3.0
                return logits

        self.backbone = _B()

    def _windows(self, n):
        return [(0, n)]


def _instance(seed=0, L=300, n_sym=20):
    rng = np.random.default_rng(seed)
    true_map = rng.integers(0, A, size=n_sym)
    cipher = rng.integers(0, n_sym, size=L)
    return cipher, true_map, true_map[cipher]


def test_posterior_agrees_with_true_key():
    cipher, true_map, plain = _instance()
    ev = _Toy(plain)
    P = position_posterior(ev, plain, "latin", n_draws=48, mask_rate=0.5, seed=1)
    assert P.shape == (len(plain), A)
    assert (P.argmax(1) == plain).all()
    assert disagreements(symbol_scores(P, cipher, len(true_map)), true_map) == []


def test_posterior_flags_the_swapped_pair():
    cipher, true_map, plain = _instance()
    ev = _Toy(plain)
    key = true_map.copy()
    s1, s2 = 0, 1
    while true_map[s1] == true_map[s2]:
        s2 += 1
    key[s1], key[s2] = true_map[s2], true_map[s1]
    P = position_posterior(ev, key[cipher], "latin", n_draws=8, mask_rate=0.5, seed=1)
    D = disagreements(symbol_scores(P, cipher, len(key)), key)
    assert {d[0] for d in D} == {s1, s2}
    assert all(d[1] == true_map[d[0]] for d in D)


def test_alternate_recovers_swap_with_toy_objective():
    cipher, true_map, plain = _instance()
    ev = _Toy(plain)
    key = true_map.copy()
    key[0], key[1] = key[1], key[0]
    if key[0] == key[1]:
        key[0] = (key[0] + 1) % A

    def objective(m):  # a "n-gram" that likes the truth
        return -float((m[cipher] != plain).sum())

    def scores_fn(m):
        return symbol_scores(
            position_posterior(ev, m[cipher], "latin", n_draws=8, mask_rate=0.5),
            cipher,
            len(m),
        )

    def short_sa(m, rng):  # identity "search": accept what the proposer says
        return m, objective(m)

    out, info = alternate(
        key,
        mechanism="posterior",
        objective=objective,
        short_sa=short_sa,
        scores_fn=scores_fn,
        k=8,
    )
    assert (out == true_map).all()
    assert info["n_accepted"] == 1
    # null: from the truth nothing moves
    out2, info2 = alternate(
        true_map,
        mechanism="posterior",
        objective=objective,
        short_sa=short_sa,
        scores_fn=scores_fn,
    )
    assert (out2 == true_map).all() and info2["n_accepted"] == 0


def test_unit_scores_wordhom_shapes():
    targets = UnitTargets.from_list([[0, 0], [1, 1]])
    sym_map = np.array([0, 3, A, A + 1, 5])
    symbols = np.array([0, 1, 2, 3, 4, 2, 0])
    plain = expand_units(sym_map[symbols], targets)
    ev = _Toy(plain)
    P = position_posterior(ev, plain, "latin", n_draws=8, mask_rate=0.5)
    S = unit_scores(P, symbols, sym_map, targets)
    assert S.shape == (5, targets.n)
    assert np.isfinite(S[0, :A]).all() and np.isinf(S[0, A:]).all()
    assert np.isinf(S[2, :A]).all() and np.isfinite(S[2, A:]).all()
    assert disagreements(S, sym_map) == []
