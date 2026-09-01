"""Word-level homophonic head (Boxer's hypothesis without the arithmetic):
unit targets, expansion / segmentation round trip, the repeat rule
bookkeeping, the synthetic cipher, and the apply-pipeline plumbing."""

from __future__ import annotations

import itertools

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


# -- incremental objective (wordhom_state) -----------------------------------


def _toy_evaluator():
    from diff_voyn.heads.evaluator import NgramEvaluator
    from diff_voyn.heads.ngram import train_lm

    rng = np.random.default_rng(0)
    # a stream with structure so that the pentagram table is non-trivial
    stream = (rng.integers(0, 8, size=20_000) + (np.arange(20_000) % 3)) % A
    lm = train_lm("toy", [stream.astype(np.uint8)], k_max=5)
    return NgramEvaluator({"toy": lm})


def test_incremental_state_matches_full_objective():
    from diff_voyn.heads.wordhom import WordHomophonicHead
    from diff_voyn.heads.wordhom_state import WordHomObjectiveState

    ev = _toy_evaluator()
    t = UnitTargets(((0, 0), (3, 3), (5, 5)))
    hd = WordHomophonicHead(ev, targets=t, seed=0)
    rng = np.random.default_rng(3)
    n_sym, n_tok = 120, 1500
    # Zipf-ish symbol stream: heavy and singleton types, some in tokens 0..3
    w = 1.0 / np.arange(1, n_sym + 1)
    symbols = rng.choice(n_sym, size=n_tok, p=w / w.sum())
    token_pos = np.cumsum(rng.random(n_tok) < 0.9)  # some non-adjacent pairs
    adj = adjacency(symbols, token_pos)
    key = rng.integers(0, t.n, size=n_sym)
    st = WordHomObjectiveState(hd, symbols, adj, key, "toy", t)
    full = lambda m: hd.objective(m, symbols, adj, "toy", t)
    assert st.score == pytest.approx(full(key), abs=1e-3)
    occ = np.bincount(symbols, minlength=n_sym)
    heavy = int(np.argmax(occ))
    early = [int(s) for s in symbols[:4]]
    for it in range(300):
        if it < 4:
            syms, units = np.array([early[it]]), rng.integers(t.n, size=1)
        elif it < 12:
            syms, units = np.array([heavy]), rng.integers(t.n, size=1)
        elif rng.random() < 0.8:
            syms, units = rng.choice(n_sym, size=1), rng.integers(t.n, size=1)
        else:  # the SA's swap move
            syms = rng.choice(n_sym, size=2, replace=False)
            units = np.array([st.sym_map[syms[1]], st.sym_map[syms[0]]])
        d = st.delta(syms, units)
        m2 = st.sym_map.copy()
        m2[syms] = units
        assert d == pytest.approx(full(m2) - full(st.sym_map), abs=2e-3)
        if rng.random() < 0.5:
            assert st.commit(syms, units) == pytest.approx(full(st.sym_map), abs=2e-3)
    for s in [heavy, *early, *rng.choice(n_sym, size=4).tolist()]:
        d = st.deltas_all(s)
        base = full(st.sym_map)
        for u in range(t.n):
            m2 = st.sym_map.copy()
            m2[s] = u
            assert d[u] == pytest.approx(full(m2) - base, abs=2e-3), (s, u)
        assert d[st.sym_map[s]] == 0.0


def test_polish_and_sa_use_incremental_state_and_return_full_score():
    from diff_voyn.heads.wordhom import WordHomophonicHead

    ev = _toy_evaluator()
    t = UnitTargets(((0, 0), (3, 3)))
    hd = WordHomophonicHead(ev, targets=t, seed=0)
    rng = np.random.default_rng(5)
    symbols = rng.integers(0, 40, size=600)
    adj = adjacency(symbols, None)
    key = rng.integers(0, t.n, size=40)
    m, sc, _ = hd.polish(symbols, adj, key, "toy", t)
    assert sc == pytest.approx(hd.objective(m, symbols, adj, "toy", t), abs=1e-6)
    assert sc >= hd.objective(key, symbols, adj, "toy", t)
    # a local optimum: no single reassignment improves
    for s in range(40):
        for u in range(t.n):
            m2 = m.copy()
            m2[s] = u
            assert hd.objective(m2, symbols, adj, "toy", t) <= sc + 1e-6
    m2, sc2, _ = hd.sa_phase(symbols, adj, key.copy(), "toy", t, rng, steps=2000)
    assert sc2 == pytest.approx(hd.objective(m2, symbols, adj, "toy", t), abs=1e-6)
    assert sc2 >= hd.objective(key, symbols, adj, "toy", t)


# -- unit-set specs (doubles + general bigrams, 2026-08-30) -------------------


def test_parse_units_and_suffix():
    from diff_voyn.heads.wordhom import parse_units, units_suffix

    assert parse_units(None) == (5, 0)
    assert parse_units("d5") == (5, 0)
    assert parse_units("d5b20") == (5, 20)
    assert parse_units("d3b7") == (3, 7)
    assert units_suffix(None) == "" and units_suffix("d5") == ""
    assert units_suffix("d5b20") == "_d5b20"
    with pytest.raises(ValueError):
        parse_units("b20")


def test_language_targets_general_bigrams_extend_the_doubles():
    from diff_voyn.heads.wordhom import language_targets

    ev = _toy_evaluator()
    d5 = language_targets(ev, "toy")
    big = language_targets(ev, "toy", units="d5b20")
    assert len(d5.bigrams) == 5 and len(big.bigrams) == 25
    # doubles first, unchanged, so a d5 key is a d5b20 key
    assert big.bigrams[:5] == d5.bigrams
    gen = big.bigrams[5:]
    assert all(a != b for a, b in gen)
    assert len(set(gen)) == 20
    # ranked by the LM's bigram probability
    l1 = ev.logT("toy", 1).cpu().numpy().reshape(A)
    l2 = ev.logT("toy", 2).cpu().numpy().reshape(A, A)
    j = [l1[a] + l2[a, b] for a, b in gen]
    assert all(x >= y - 1e-9 for x, y in itertools.pairwise(j))
    # the numeric form agrees with the spec form
    assert language_targets(ev, "toy", 5, 20).bigrams == big.bigrams


def test_project_key_between_unit_spaces():
    from diff_voyn.heads.wordhom import project_key

    cipher = UnitTargets(((0, 0), (3, 3), (1, 2)))
    hyp = UnitTargets(((0, 0), (3, 3), (1, 2), (4, 5)))  # superset, same order
    tm = np.array([0, 7, A, A + 1, A + 2])
    assert (project_key(tm, cipher, hyp) == tm).all()
    # subset: the missing unit falls back to its first letter
    sub = UnitTargets(((0, 0),))
    assert project_key(tm, cipher, sub).tolist() == [0, 7, A, 3, 1]
    # reordered: units follow their pair, not their index
    re_ = UnitTargets(((1, 2), (0, 0)))
    assert project_key(tm, cipher, re_).tolist() == [0, 7, A + 1, 3, A]
    # identical spaces: identity object semantics not required, values equal
    assert (project_key(tm, cipher, cipher) == tm).all()


def test_round_trip_and_incremental_state_with_general_bigrams():
    from diff_voyn.heads.wordhom import WordHomophonicHead
    from diff_voyn.heads.wordhom_state import WordHomObjectiveState

    t = UnitTargets(((0, 0), (3, 3), (0, 1), (2, 5), (5, 2)))
    rng = np.random.default_rng(1)
    plain = rng.integers(0, 7, size=800)
    units = segment_units(plain, t)
    assert (expand_units(units, t) == plain).all()
    assert (units >= A + 2).sum() > 0  # general bigram units were formed
    ev = _toy_evaluator()
    hd = WordHomophonicHead(ev, targets=t, seed=0)
    n_sym, n_tok = 80, 900
    w = 1.0 / np.arange(1, n_sym + 1)
    symbols = rng.choice(n_sym, size=n_tok, p=w / w.sum())
    adj = adjacency(symbols, None)
    key = rng.integers(0, t.n, size=n_sym)
    st = WordHomObjectiveState(hd, symbols, adj, key, "toy", t)
    full = lambda m: hd.objective(m, symbols, adj, "toy", t)
    assert st.score == pytest.approx(full(key), abs=1e-3)
    for _ in range(150):
        syms, us = rng.choice(n_sym, size=1), rng.integers(t.n, size=1)
        d = st.delta(syms, us)
        m2 = st.sym_map.copy()
        m2[syms] = us
        assert d == pytest.approx(full(m2) - full(st.sym_map), abs=2e-3)
        if rng.random() < 0.5:
            st.commit(syms, us)
    # the synthetic cipher over the wide unit set inverts under its own key
    _, toks, s2u = WordHomCipher(t, n_types=200).encipher(plain, rng)
    assert (expand_units(s2u[toks], t) == plain).all()
