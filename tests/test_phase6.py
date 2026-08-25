"""Phase 6 — VMS presentations, the segmented rung-4 lattice, the apply
pipeline's job / scale / cell arithmetic and the no-cipher baselines."""

from __future__ import annotations

import numpy as np
import pytest

from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.ngram import A
from diff_voyn.heads.rung4_arithmetic import (
    SEG_LENGTHS,
    positional_rank,
    segmented_admissible_mask,
)
from diff_voyn.vms.apply import (
    ABSTAIN_RULE,
    KEY,
    ciphertext_baselines,
    head_key_bits,
    make_jobs,
    order0_entropy_bits,
    rank_table,
    window_slices,
)

IT_PATH = data_root() / "raw" / "vms" / "IT2a-n.txt"
BOXER = data_root() / "external" / "voynich-attack" / "transcription" / "vms.csv"


# ---------------------------------------------------------------- rung 4 seg


def test_segmented_admissible_mask_marks_exactly_the_observed_tokens():
    ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    starts = np.array([0, 2, 5, 6, 9])  # lengths 2, 3, 1, 3, 1
    adm = segmented_admissible_mask(ids, starts)
    assert adm.shape == (10, len(SEG_LENGTHS))
    assert adm[0, SEG_LENGTHS.index(2)] and adm[2, SEG_LENGTHS.index(3)]
    assert adm[6, SEG_LENGTHS.index(3)]
    assert adm.sum() == 3  # length-1 tokens leave no admissible segment
    assert not adm[5].any() and not adm[9].any()


def test_positional_rank_orders_initial_before_final():
    # symbol 0 always token-initial, symbol 2 always final, symbol 1 middle
    ids = np.array([0, 1, 2, 0, 1, 2, 0, 1, 1, 2])
    starts = np.array([0, 3, 6])
    rank = positional_rank(ids, starts, n_sym=3)
    assert rank[0] < rank[1] < rank[2]


# ---------------------------------------------------------------- apply


def test_window_slices_cover_and_space_evenly():
    assert window_slices(100, 40, 2) == [(0, 40), (60, 100)]
    assert window_slices(30, 40, 3) == [(0, 30)]
    w = window_slices(1000, 100, 4)
    assert len(w) == 4 and w[0] == (0, 100) and w[-1] == (900, 1000)


def test_key_bits_injective_map_is_below_full_permutation():
    assert head_key_bits("sub1to1", 20) < head_key_bits("sub1to1", A)
    assert head_key_bits("homophonic", 54) == pytest.approx(54 * np.log2(A))
    assert head_key_bits("naibbe", 0) > head_key_bits("arithmetic", 16)


def test_make_jobs_respects_head_applicability(tmp_path):
    inst = {
        "name": "x/A",
        "kind": "eva",
        "n_symbols": 30,
        "n_stream": 5000,
        "path": str(tmp_path / "x.json"),
    }
    jobs = make_jobs(
        inst, heads=("sub1to1", "homophonic", "naibbe", "arithmetic"), n_windows=2
    )
    heads = {j["head"] for j in jobs}
    assert heads == {
        "homophonic"
    }  # 30 > 25 symbols: no bijection; no words; no 16-value stream
    assert (
        len(jobs) == 2 * 3 and {tuple(j[k] for k in KEY) for j in jobs}.__len__() == 6
    )
    words = {
        "name": "x/A",
        "kind": "words",
        "n_symbols": 0,
        "n_stream": 300,
        "path": "p",
    }
    assert {
        j["head"] for j in make_jobs(words, heads=("sub1to1", "naibbe"), n_windows=2)
    } == {"naibbe"}


def test_order0_and_ngram_baselines_are_sane():
    rng = np.random.default_rng(0)
    # a deterministic cycle: order-1 model explains it perfectly
    sym = np.tile(np.arange(5), 400)
    b = ciphertext_baselines(sym, 5, orders=(1, 2))
    assert b["order0_entropy_bits"] == pytest.approx(np.log2(5))
    assert b["ngram1_heldout_bits"] < 0.1
    iid = rng.integers(0, 5, size=4000)
    b2 = ciphertext_baselines(iid, 5, orders=(1, 2))
    assert abs(b2["ngram2_heldout_bits"] - np.log2(5)) < 0.15
    assert order0_entropy_bits(np.zeros(10, dtype=int)) == 0.0


def _cell(head, hyp, total, plain, margin, like, baseline=2.5):
    return {
        "instance": "x/A",
        "presentation": "eva",
        "head": head,
        "window": 0,
        "hypothesis": hyp,
        "total_per_all_symbols": total,
        "total_per_covered_symbol": total,
        "no_cipher_baseline_bits_per_symbol": baseline,
        "plain_bits": plain,
        "structure_margin": margin,
        "language_like": like,
    }


def test_rank_table_ranks_by_total_and_abstains_when_nothing_is_language_like():
    cells = [
        _cell("sub1to1", "latin", 2.9, 2.9, 0.8, False),
        _cell("sub1to1", "german", 2.8, 2.8, 0.9, False),
        _cell("homophonic", "latin", 3.1, 2.6, 0.9, False),
        _cell("homophonic", "german", 3.2, 2.7, 0.9, False),
        _cell("homophonic", "italian", 3.3, 2.8, 0.9, False),
    ]
    tab = rank_table(cells)
    assert (
        tab["ranked"][0]["head"] == "sub1to1"
        and tab["ranked"][0]["hypothesis"] == "german"
    )
    assert tab["abstain"] is True
    assert tab["per_head"]["homophonic"]["language_order"] == [
        "latin",
        "german",
        "italian",
    ]
    cells[2]["language_like"] = True
    assert rank_table(cells)["abstain"] is False
    # coverage accounting: the ranking key is the all-symbols total, and each
    # row records whether it beats the stream's own no-cipher baseline
    assert rank_table(cells)["ranked"][0]["beats_no_cipher_baseline"] is False


def test_abstain_rule_is_frozen():
    assert ABSTAIN_RULE["max_plain_bits"] == 3.0
    assert ABSTAIN_RULE["min_structure_margin"] == 1.5


# ---------------------------------------------------------------- presentations


@pytest.mark.skipif(not IT_PATH.exists(), reason="voynich.nu files not fetched")
def test_eva_presentation_matches_ingest_and_fits_rung1():
    from diff_voyn.heads.naibbe_parse import NaibbeParser
    from diff_voyn.vms.presentations import eva_presentation, words_presentation

    for d in ("A", "B"):
        p = eva_presentation("IT2a", d)
        assert p.n_symbols <= A and len(p.symbols) > 50_000
        assert all(len(c) == 1 and c.isalpha() for c in p.alphabet)
        w = words_presentation("IT2a", d, NaibbeParser())
        assert 0.6 < w.coverage["word_fraction"] < 0.9
        assert all(len(t) >= 1 for t in w.tokens)
    r = eva_presentation("RF1b", "A")
    assert all(c.isalpha() and c.islower() for c in r.alphabet)  # residue stripped


@pytest.mark.skipif(
    not (IT_PATH.exists() and BOXER.exists()), reason="data not fetched"
)
def test_boxer_presentation_dialects_and_coverage():
    from diff_voyn.vms.presentations import boxer_presentation, boxer_tokens

    toks = boxer_tokens()
    assert len(toks["A"]) > 5000 and len(toks["B"]) > 10000
    pA = boxer_presentation("A", n_symbols=16, tokens=toks)
    pB = boxer_presentation("B", n_symbols=16, tokens=toks)
    assert pA.alphabet == pB.alphabet  # one manuscript-wide alphabet
    assert 0.85 < pA.coverage["covered_fraction"] < 1.0
    assert "?" not in pA.alphabet
    # token starts are consistent with the kept stream
    assert pA.token_starts[0] == 0 and pA.token_starts[-1] < len(pA.symbols)
    assert np.all(np.diff(pA.token_starts) >= 1)
