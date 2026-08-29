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


# -- docs/altloop_vms_plan.md: pair-swap proposer and the §5 tiers -----------


def test_pair_swaps_ranks_disjoint_transpositions():
    from diff_voyn.heads.altloop import pair_swaps

    key = np.array([0, 1, 2, 3])
    S = np.zeros((4, A))
    # the judge wants 0<->1 swapped (gain 4) and 2->letter 3 (index 3 unused, gain 2)
    S[0, 1] = 2.0
    S[1, 0] = 2.0
    S[2, 3] = 2.0
    sw = pair_swaps(S, key, k=4)
    assert [(i, j) for i, j, _ in sw] == [(0, 1), (2, 3)]
    assert sw[0][2] == 4.0 and sw[1][2] == 2.0
    assert pair_swaps(S, key, k=1) == [sw[0]]
    assert pair_swaps(np.zeros((4, A)), key, k=4) == []


def test_alternate_pair_swap_keeps_bijection():
    from diff_voyn.heads.altloop import alternate

    key = np.arange(6)
    S = np.zeros((6, A))
    S[0, 1] = S[1, 0] = 3.0
    seen = []
    out, info = alternate(
        key,
        mechanism="pair_swap",
        objective=lambda m: float(m[0] == 1),
        short_sa=lambda m, rng: (m, float(m[0] == 1)),
        scores_fn=lambda m: S,
        k=2,
        rounds=2,
        on_round=lambda i: seen.append(i["round"]),
    )
    assert info["n_accepted"] == 1 and seen == [0, 1]
    assert sorted(out.tolist()) == list(range(6)) and out[0] == 1 and out[1] == 0
    out, _ = alternate(
        key,
        mechanism="random_swap",
        objective=lambda m: 0.0,
        short_sa=lambda m, rng: (m, 1.0),
        k=2,
        rounds=1,
        seed=3,
    )
    assert sorted(out.tolist()) == list(range(6)) and (out != key).sum() == 4


def test_classify_tier():
    from diff_voyn.heads.altloop import classify_tier

    assert classify_tier(2.0, 2.5, None) == "PENDING"  # control not in: never flag
    assert classify_tier(1.20, 2.5, 0.9) == "NOISE"  # below the manuscript ceiling
    assert classify_tier(1.30, 2.6, 1.10) == "NOTABLE"
    assert classify_tier(1.30, 2.6, 1.28) == "NOISE"  # a control reached it too
    assert classify_tier(1.60, 2.6, 1.50) == "NOISE"
    assert classify_tier(1.55, 2.6, 1.20) == "PROMISING"
    assert classify_tier(1.55, 2.6, 1.30) == "NOTABLE"  # control above the ceiling
    assert classify_tier(1.55, 2.6, 1.20, flip_rate=0.0) == "LANGUAGE-LIKE"
    assert classify_tier(1.55, 2.6, 1.20, flip_rate=0.25) == "PROMISING"
    assert classify_tier(1.55, 3.2, 1.20, flip_rate=0.0) == "PROMISING"
    assert (
        classify_tier(1.55, 2.6, 1.20, flip_rate=0.0, controls_language_like=True)
        == "PROMISING"
    )


def test_alternate_schedule_rebaselines_objective():
    """A schedule that changes the objective mid-run must re-score the
    incumbent so ``obj_in`` of the next round is on the new scale, and the
    patience counter restarts on the change."""
    key = np.arange(4)
    scale = {"v": 1.0}

    def objective(m):
        return -scale["v"] * float(m.sum())

    def short_sa(m, rng):  # never improves: every round is rejected
        return m, objective(m)

    def schedule(r):
        if r == 2:
            scale["v"] = 10.0
            return {"scale": 10.0}
        return None

    _, info = alternate(
        key,
        mechanism="none",
        objective=objective,
        short_sa=short_sa,
        rounds=4,
        patience=3,
        schedule=schedule,
    )
    tr = info["trace"]
    assert tr[0]["obj_in"] == -6.0 and tr[1]["obj_in"] == -6.0
    assert tr[2]["obj_in"] == -60.0 and tr[2]["schedule"] == {"scale": 10.0}
    # patience 3 would have stopped after round 2; the reset lets round 3 run
    assert len(tr) == 4 and info["n_accepted"] == 0
