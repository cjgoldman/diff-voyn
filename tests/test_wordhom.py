"""Word-level homophonic head (Boxer's hypothesis without the arithmetic):
unit targets, expansion / segmentation round trip, the repeat rule
bookkeeping, the synthetic cipher, and the apply-pipeline plumbing."""

from __future__ import annotations

import numpy as np
import pytest

from diff_voyn.heads.ngram import A
from diff_voyn.heads.wordhom import (
    N_BIGRAMS,
    UnitTargets,
    WordHomCipher,
    adjacency,
    choice_bits_total,
    doubling_rate,
    expand_units,
    repeat_positions,
    rule_violations,
    segment_units,
    targets_from_ids,
)
from diff_voyn.vms.apply import WORDHOM, head_key_bits, make_jobs


def _targets():
    return UnitTargets(((0, 0), (3, 3)))  # "aa", "dd" as extra units


def test_expand_segment_round_trip():
    t = _targets()
    rng = np.random.default_rng(0)
    plain = rng.integers(0, 5, size=500)
    units = segment_units(plain, t)
    assert (expand_units(units, t) == plain).all()
    assert (units >= A).sum() > 0  # some doubled units were formed
    # no doubled-letter unit ever straddles: greedy segmentation is left-to-right
    assert len(units) < len(plain)


def test_targets_from_ids_picks_doubled_letters():
    ids = np.array([1, 1, 1, 1, 2, 2, 3, 4, 5, 5, 5, 5, 5, 5])
    t = targets_from_ids(ids, n_bigrams=2)
    assert all(a == b for a, b in t.bigrams)
    assert {a for a, _ in t.bigrams} == {1, 5}
    assert t.n == A + 2


def test_repeat_rule_bookkeeping():
    sym = np.array([0, 0, 1, 2, 2, 3])
    pos = np.array([0, 1, 2, 5, 6, 7])  # tokens 3,4 dropped between 2 and 5
    adj = adjacency(sym, pos)
    assert adj.tolist() == [True, True, False, True, True]
    rep = repeat_positions(sym, adj)
    assert rep.tolist() == [False, True, False, False, True, False]
    # map sending symbols 1 and 2 to the same unit: (1,2) not adjacent, so
    # only the (2->3) pair can violate — it does when 3 shares unit with 2
    m = np.array([5, 7, 7, 7])
    assert rule_violations(m[sym], sym, adj) == 1
    m = np.array([5, 7, 7, 8])
    assert rule_violations(m[sym], sym, adj) == 0
    # choice bits: zero at repeat positions, log2(#homophones) elsewhere
    cb = choice_bits_total(m, sym, rep, A + 2)
    # unit 7 has two homophones (symbols 1, 2): positions 2, 3 pay 1 bit, 4 is a repeat
    assert cb == pytest.approx(2.0)


def test_synthetic_cipher_respects_rule_and_inverts():
    t = _targets()
    rng = np.random.default_rng(1)
    plain = rng.integers(0, 6, size=3000)
    units, toks, s2u = WordHomCipher(t, n_types=300).encipher(plain, rng)
    assert (expand_units(s2u[toks], t) == plain).all()
    assert rule_violations(s2u[toks], toks, adjacency(toks, None)) == 0
    # a doubled unit is always a repeated token
    same_unit = units[1:] == units[:-1]
    assert (toks[1:][same_unit] == toks[:-1][same_unit]).all()
    assert len(set(toks.tolist())) <= 300
    assert doubling_rate(units) == pytest.approx(same_unit.mean())


def test_key_bits_and_jobs():
    assert head_key_bits(WORDHOM, 100) == pytest.approx(100 * np.log2(A + N_BIGRAMS))
    inst = {
        "name": "x/A",
        "kind": "wordtypesall",
        "n_symbols": 50,
        "n_stream": 30000,
        "path": "p",
        "coverage": {},
    }
    jobs = make_jobs(
        inst, heads=(WORDHOM, "sub1to1", "homophonic"), n_windows=1, w5=12000
    )
    assert {j["head"] for j in jobs} == {WORDHOM}
    assert len(jobs) == 3 and jobs[0]["window_span"] == [0, 12000]
    # eva instances never get the word head
    inst["kind"] = "eva"
    assert not [j for j in make_jobs(inst, heads=(WORDHOM,)) if j["head"] == WORDHOM]
