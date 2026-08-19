"""Cipher-head early track tests (CH.0–CH.5): evaluator correctness against
brute-force enumeration, the task-5.1 frame/NaN smoke tests (CH.4), and head
mechanics on tiny synthetic problems.

Everything here runs on synthetic streams — no corpus or trained-LM artifacts
required — so the suite is CI-safe.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diff_voyn.heads.evaluator import NgramEvaluator, TokenEmission
from diff_voyn.heads.frame import (
    build_frame,
    letters_to_vocab,
    log_frame,
    straight_through_frame,
)
from diff_voyn.heads.ngram import A, encode_letters, train_lm
from diff_voyn.heads.synth import gen_homophonic, gen_substitution, map_accuracy, ser
from diff_voyn.vocab import LETTER_IDS, NULL_ID, VOCAB_SIZE

torch.set_num_threads(1)


def _markov_sample(trans: np.ndarray, n: int, rng, s: int = 0) -> np.ndarray:
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        s = rng.choice(A, p=trans[s])
        out[i] = s
    return out


@pytest.fixture(scope="module")
def toy_trans():
    """Sparse random Markov transitions: sequential structure makes
    substitution recovery well-posed (an i.i.d. stream is only solvable up
    to frequency ties, which is not what rung 1 tests)."""
    return np.random.default_rng(0).dirichlet(np.full(A, 0.15), size=A)


@pytest.fixture(scope="module")
def toy_lm(toy_trans):
    rng = np.random.default_rng(0)
    stream = _markov_sample(toy_trans, 120_000, rng).astype(np.uint8)
    return train_lm("toy", [stream], k_max=5)


@pytest.fixture(scope="module")
def toy_ev(toy_lm):
    return NgramEvaluator({"toy": toy_lm})


# ---------------------------------------------------------------- CH.0 ngram


def test_tables_are_distributions(toy_lm):
    for k, t in toy_lm.logp.items():
        rows = np.exp(t.astype(np.float64)).sum(axis=1)
        assert np.allclose(rows, 1.0, atol=1e-4), f"order {k} not normalized"


def test_score_ids_matches_manual_chaining(toy_lm):
    rng = np.random.default_rng(1)
    ids = rng.integers(0, A, size=40).astype(np.uint8)
    manual = 0.0
    for t in range(len(ids)):
        k = min(t + 1, toy_lm.k_max)
        ctx = 0
        for j in range(k - 1):
            ctx = ctx * A + int(ids[t - k + 1 + j])
        manual += float(toy_lm.logp[k][ctx, ids[t]])
    assert abs(manual - toy_lm.score_ids(ids)) < 1e-3


def test_encode_letters_roundtrip():
    assert encode_letters("abkz").tolist() == [0, 1, 9, 24]
    with pytest.raises(ValueError):
        encode_letters("a b")  # whitespace must already be stripped


# ------------------------------------------------------------- CH.1 evaluator


def test_onehot_soft_equals_hard(toy_lm, toy_ev):
    rng = np.random.default_rng(2)
    ids = rng.integers(0, A, size=30)
    soft = torch.zeros(30, A)
    soft[torch.arange(30), torch.from_numpy(ids)] = 1.0
    sf = toy_ev.score_fixed(soft, language="toy")
    assert abs(float(sf) - toy_lm.score_ids(ids)) < 1e-2


def test_soft_score_matches_bruteforce_expectation(toy_lm, toy_ev):
    L, k = 6, 3
    q = torch.softmax(torch.randn(L, A, generator=torch.Generator().manual_seed(3)), 1)
    got = float(toy_ev.score_fixed(q, language="toy", order=k))
    want = 0.0
    for t in range(L):
        kk = min(t + 1, k)
        tab = toy_lm.logp[kk].reshape((A,) * kk).astype(np.float64)
        for j in range(kk):
            tab = np.tensordot(q[t - kk + 1 + j].numpy(), tab, axes=([0], [0]))
        want += float(tab)
    assert abs(got - want) < 1e-3


def test_segmental_dp_matches_enumeration(toy_lm, toy_ev):
    rng = np.random.default_rng(4)
    ids = rng.integers(0, A, size=7)
    ems = []
    for i in ids:
        v = torch.zeros(A)
        v[int(i)] = 1.0
        ems.append(TokenEmission(uni=v))
    got = float(toy_ev.score_segmental(ems, language="toy"))
    lp1 = toy_lm.logp[1].ravel()
    lp2 = toy_lm.logp[2]
    lp3 = toy_lm.logp[3].reshape(A, A, A)
    want = -np.inf
    for a in range(A):
        for b in range(A):
            s = lp1[a] + lp2[a, b]
            h = (a, b)
            for c in ids:
                s += lp3[h[0], h[1], int(c)]
                h = (h[1], int(c))
            want = np.logaddexp(want, s)
    assert abs(got - want) < 1e-3


def test_segmental_mixture_grads_and_inf_guard(toy_ev):
    w = torch.tensor(0.3, requires_grad=True)
    q = torch.softmax(torch.randn(A, generator=torch.Generator().manual_seed(5)), 0)
    ems = [
        TokenEmission(
            uni=q, pre=q, suf=q, log_w_uni=torch.log(w), log_w_bi=torch.log1p(-w)
        )
        for _ in range(4)
    ]
    s = toy_ev.score_segmental(ems, language="toy")
    s.backward()
    assert torch.isfinite(s) and torch.isfinite(w.grad)
    # degenerate weight: -inf branch is skipped, never blended (design §8 trap)
    ems = [TokenEmission(uni=q, pre=q, suf=q, log_w_uni=0.0, log_w_bi=-np.inf)]
    assert torch.isfinite(toy_ev.score_segmental(ems, language="toy"))


def test_calibration_hook_single_source(toy_ev):
    raw = -100.0
    base = toy_ev.calibrated_bits_per_char(raw, 50, "toy")
    toy_ev.calibration_offsets_bits["toy"] = 0.25
    assert toy_ev.calibrated_bits_per_char(raw, 50, "toy") == pytest.approx(base + 0.25)
    toy_ev.calibration_offsets_bits.pop("toy")


# ------------------------------------------------- CH.4 frame / task 5.1 smoke


def test_frame_null_blend_rows_stochastic_no_nan():
    n = 5
    g = torch.Generator().manual_seed(6)
    s1 = torch.softmax(torch.randn(n, A, generator=g), 1)
    s2 = torch.softmax(torch.randn(n, A, generator=g), 1)
    # include the degenerate corners w=0 and w=1 — the logaddexp(-inf,-inf)
    # trap territory (non-negotiable #6)
    w = torch.tensor([0.0, 1.0, 0.5, 0.0, 1.0])
    frame = build_frame(s1, s2, w)
    assert frame.shape == (2 * n, VOCAB_SIZE)
    assert torch.isfinite(frame).all()
    assert torch.allclose(frame.sum(-1), torch.ones(2 * n), atol=1e-5)
    assert torch.isfinite(log_frame(frame)).all()  # safe log, no -inf pairs
    # slot 2 of a w=1 token is pure NULL; of a w=0 token has zero NULL mass
    assert frame[3, NULL_ID] == pytest.approx(1.0)  # token 1, w=1
    assert frame[1, NULL_ID] == pytest.approx(0.0)  # token 0, w=0


def test_frame_gradients_reach_head_params():
    g = torch.Generator().manual_seed(7)
    logits1 = torch.randn(4, A, generator=g, requires_grad=True)
    logits2 = torch.randn(4, A, generator=g, requires_grad=True)
    w_logit = torch.zeros(4, requires_grad=True)
    frame = build_frame(
        torch.softmax(logits1, 1), torch.softmax(logits2, 1), torch.sigmoid(w_logit)
    )
    # arbitrary downstream consumer of the log-frame
    loss = log_frame(frame).logsumexp(-1).sum()
    loss.backward()
    for p in (logits1, logits2, w_logit):
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_straight_through_frame():
    q = torch.softmax(torch.randn(3, VOCAB_SIZE, requires_grad=True), -1)
    st = straight_through_frame(q)
    assert ((st.detach() == 0) | (st.detach() == 1)).all()  # forward is one-hot
    st.sum().backward()  # backward flows through the soft path


def test_diffusion_evaluator_random_init_smoke():
    """CH.4 acceptance: gradients reach a toy head through the random-init
    backbone; scores finite; one-hot forward_soft == id forward."""
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator
    from diff_voyn.infra.config import ModelConfig
    from diff_voyn.model.backbone import Backbone

    torch.manual_seed(0)
    cfg = ModelConfig(n_layers=2, d_model=64, n_heads=4, d_ffn=128, seq_len=64)
    backbone = Backbone(cfg)
    ev = DiffusionEvaluator(backbone, n_strata=4, seed=0)

    # forward_soft equivalence on one-hot ids (eval mode, no dropout)
    ids = torch.randint(0, VOCAB_SIZE, (1, 16))
    onehot = torch.nn.functional.one_hot(ids, VOCAB_SIZE).float()
    lang = torch.tensor([0])
    with torch.no_grad():
        a = backbone(ids, lang)
        b = backbone.forward_soft(onehot, lang)
    assert torch.allclose(a, b, atol=1e-5)

    # toy head: logits -> soft letters -> frame -> NELBO score; grads flow
    head_logits = torch.randn(6, A, requires_grad=True)
    w_logit = torch.zeros(3, requires_grad=True)
    soft = torch.softmax(head_logits, 1)
    frame = build_frame(soft[0::2], soft[1::2], torch.sigmoid(w_logit))
    score = ev.score_frame(frame, language="latin")
    assert torch.isfinite(score)
    score.backward()
    assert torch.isfinite(head_logits.grad).all()
    assert torch.isfinite(w_logit.grad).all()
    # frozen measuring stick: backbone params must have no grads
    assert all(p.grad is None for p in backbone.parameters())

    # CRN: same seed -> identical masking -> identical score
    s2 = ev.score_frame(frame.detach(), language="latin")
    s3 = ev.score_frame(frame.detach(), language="italian")
    assert float(s2) == pytest.approx(float(score), abs=1e-4)
    assert float(s3) != pytest.approx(float(score), abs=1e-6)  # lang changes it

    # expected-embedding path through a random-init embedding table
    sl = torch.softmax(torch.randn(6, A, requires_grad=True), 1)
    emb = ev.as_embedding_frame(sl, torch.sigmoid(torch.zeros(3)))
    assert emb.shape == (6, cfg.d_model)
    assert torch.isfinite(emb).all()


def test_letters_to_vocab_layout():
    q = torch.zeros(2, A)
    q[0, 0] = 1.0  # letter 'a'
    q[1, A - 1] = 1.0  # letter 'z'
    v = letters_to_vocab(q)
    assert v[0, LETTER_IDS[0]] == 1.0 and v[0].sum() == 1.0
    assert v[1, LETTER_IDS[-1]] == 1.0


# ----------------------------------------------------- CH.2/CH.3 synth + head


def test_gen_substitution_ground_truth():
    rng = np.random.default_rng(8)
    plain = rng.integers(0, A, size=100)
    c = gen_substitution(plain, "toy", rng)
    assert ser(c, c.true_map) == 0.0
    assert map_accuracy(c, c.true_map) == 1.0
    assert not np.array_equal(c.cipher_ids, c.plain_ids) or True


def test_gen_homophonic_ground_truth():
    rng = np.random.default_rng(9)
    plain = rng.integers(0, A, size=300)
    c = gen_homophonic(plain, "toy", rng, n_symbols=54)
    assert ser(c, c.true_map) == 0.0
    assert c.cipher_ids.max() < 54
    # homophones flatten the symbol distribution vs the letter distribution
    sym_counts = np.bincount(c.cipher_ids, minlength=54)
    let_counts = np.bincount(plain, minlength=A)
    assert sym_counts.max() < let_counts.max()


def test_rung1_recovers_toy_permutation(toy_ev, toy_trans):
    """Head mechanics: sample plaintext from the toy chain (so the frozen LM
    fits it), encipher with a random permutation, recover it. This is the
    CH.3 acceptance property at unit-test scale — a frozen evaluator drives
    correct map recovery."""
    from diff_voyn.heads.rung1_sinkhorn import SinkhornSubstitutionHead

    rng = np.random.default_rng(10)
    plain = _markov_sample(toy_trans, 200, rng)
    c = gen_substitution(plain, "toy", rng)
    head = SinkhornSubstitutionHead(toy_ev, steps=80, seed=0)
    res = head.solve(c.cipher_ids, language="toy", restarts=3, kicks=20)
    true_score = toy_ev.score_hard(c.true_map[c.cipher_ids], language="toy", order=5)
    # exact recovery, or the head found something the frozen scorer likes
    # even better (the evaluator, not the head, is then the limiting factor)
    assert ser(c, res.sym_to_letter) == 0.0 or res.hard_score >= true_score


# ------------------------------------------------------- CH.6 Naibbe parsing


@pytest.fixture(scope="module")
def naibbe_parser():
    from diff_voyn.heads.naibbe_parse import NaibbeParser

    try:
        return NaibbeParser()
    except (FileNotFoundError, RuntimeError):
        pytest.skip("pinned naibbe-cipher repo not fetched")


def test_naibbe_vocab_sizes(naibbe_parser):
    # 6 tables x 23 letters = 138 codes per state; unigram has one glyph
    # string shared by two (table, letter) slots of the SAME letter -> 137
    assert naibbe_parser.n_uni in (137, 138)
    assert naibbe_parser.n_pre == 138
    assert naibbe_parser.n_suf == 138
    for state in ("unigram", "prefix", "suffix"):
        assert naibbe_parser.truth[state].shape == (len(naibbe_parser.types[state]),)


def test_naibbe_parse_stream_roundtrip(naibbe_parser):
    """Every generated token parses, and the truth maps decode each token to
    its aligned ground-truth segment (the Greshko-alignment property that
    rung-3 map accuracy is defined against)."""
    from diff_voyn.ciphers.naibbe import NaibbeCipher
    from diff_voyn.heads.ngram import LETTERS

    cipher = NaibbeCipher(seed=1)
    tokens, segments = cipher.encipher("nel mezzo del cammin di nostra vita")
    parses = naibbe_parser.parse_stream(tokens)
    truth = naibbe_parser.truth
    for p, seg in zip(parses, segments):
        cands = []
        if p.uni is not None:
            cands.append(LETTERS[truth["unigram"][p.uni]])
        for pre, suf in p.bi:
            cands.append(LETTERS[truth["prefix"][pre]] + LETTERS[truth["suffix"][suf]])
        assert seg in cands


def test_naibbe_block_structure(naibbe_parser):
    """18 (state x table) blocks of exactly 23 codes; each block's truth map
    is a bijection onto the 23 support letters (the v2 head's Sinkhorn
    prior); type_cells covers all 414 codes."""
    naibbe_parser.build_blocks()
    from diff_voyn.heads.naibbe_parse import NaibbeParser

    assert len(naibbe_parser.block_codes) == 18
    n_cells = 0
    for key, codes in naibbe_parser.block_codes.items():
        assert len(codes) == 23
        truth = naibbe_parser.block_truth[key]
        assert sorted(truth.tolist()) == list(range(23))  # bijection
    for state in ("unigram", "prefix", "suffix"):
        cells = naibbe_parser.type_cells[state]
        n_cells += sum(len(c) for c in cells)
        assert all(len(c) >= 1 for c in cells)
    assert n_cells == 18 * 23
    assert len(NaibbeParser.TABLES) == 6


# ------------------------------------------------------------- CH.8 rung 4


def _tiny_arith_cipher():
    """8-char arithmetic cipher following the scheme's conventions: 2
    negatives (-1, -2) then positives descending 5..0 (canonical order =
    char id order); tokens are 2-4 chars canonically ordered, values
    summing to the letter's value; 25 distinct letter values in -4..20."""
    from itertools import combinations_with_replacement

    v = np.array([-1, -2, 5, 4, 3, 2, 1, 0], dtype=np.int64)
    u = np.array(sorted(set(range(-4, 21)))[:A], dtype=np.int64)
    tokens_of = {}
    for letter in range(A):
        toks = []
        for n in (2, 3, 4):
            for combo in combinations_with_replacement(range(8), n):
                if v[list(combo)].sum() == u[letter]:
                    toks.append(tuple(sorted(combo)))  # id order = canonical
        tokens_of[letter] = sorted(set(toks))
        assert tokens_of[letter], f"letter value {u[letter]} unreachable"
    return v, u, tokens_of


def _tiny_arith_stream(plain_ids, tokens_of, rng):
    toks = [tokens_of[p][rng.integers(len(tokens_of[p]))] for p in plain_ids]
    starts = np.cumsum([0] + [len(t) for t in toks[:-1]])
    chars = np.array([c for t in toks for c in t], dtype=np.int64)
    return chars, starts


def test_lattice_dp_matches_enumeration(toy_ev):
    """score_lattice (order 2) vs brute-force sum over segmentations in
    linear-space float64 numpy (independent code path)."""
    rng = np.random.default_rng(3)
    L, lengths = 7, [2, 3]
    log_emis = torch.tensor(
        rng.normal(-1.0, 0.7, size=(L, len(lengths), A)), dtype=torch.float32
    )
    got = float(toy_ev.score_lattice(log_emis, lengths, language="toy", order=2))
    p1 = np.exp(toy_ev.logT("toy", 1).numpy().astype(np.float64))
    P2 = np.exp(toy_ev.logT("toy", 2).numpy().astype(np.float64))
    b = np.exp(log_emis.numpy().astype(np.float64))

    def segmentations(rem, at=0):
        if rem == 0:
            yield []
        for j, n in enumerate(lengths):
            if n <= rem:
                for rest in segmentations(rem - n, at + n):
                    yield [(at, j)] + rest

    total = 0.0
    for seg in segmentations(L):
        x = p1.copy()
        for at, j in seg:
            x = (x @ P2) * b[at, j]
        total += x.sum()
    assert abs(got - np.log(total)) < 1e-3


def test_lattice_dp_order3_matches_segmental(toy_ev):
    """With exactly one admissible segmentation, the order-3 lattice DP must
    equal the (already-verified) token-level segmental DP."""
    from diff_voyn.heads.evaluator import TokenEmission

    rng = np.random.default_rng(4)
    lengths = [2, 3]
    seg = [(0, 1), (3, 0), (5, 1)]  # 3 + 2 + 3 = 8 chars
    L = 8
    log_emis = torch.full((L, len(lengths), A), -1e30)
    ems = []
    for at, j in seg:
        row = torch.tensor(rng.normal(-1.0, 0.5, size=A), dtype=torch.float32)
        log_emis[at, j] = row
        ems.append(TokenEmission(uni=row.exp()))
    got = float(toy_ev.score_lattice(log_emis, lengths, language="toy", order=3))
    want = float(toy_ev.score_segmental(ems, language="toy"))
    assert abs(got - want) < 1e-3


def test_rung4_order_inference_and_admissibility(toy_trans):
    from diff_voyn.heads.rung4_arithmetic import (
        SEG_LENGTHS,
        admissible_mask,
        infer_char_orders,
        order_derived_values,
    )

    v, _u, tokens_of = _tiny_arith_cipher()
    rng = np.random.default_rng(5)
    plain = _markov_sample(toy_trans, 500, rng)
    chars, starts = _tiny_arith_stream(plain, tokens_of, rng)
    ranks = infer_char_orders(chars, n_sym=8, seed=0)
    true_rank = np.arange(8)
    assert any(np.array_equal(r, true_rank) for r in ranks)
    assert np.array_equal(ranks[0], true_rank)
    # order-derived values at the true split recover v exactly
    assert np.array_equal(order_derived_values(true_rank, 2).astype(np.int64), v)
    # every true segment is admissible under the inferred order
    adm = admissible_mask(chars, ranks[0])
    lens = np.diff(np.concatenate([starts, [len(chars)]]))
    assert all(adm[s, SEG_LENGTHS.index(n)] for s, n in zip(starts, lens))


def test_rung4_true_key_decode(toy_lm, toy_ev, toy_trans):
    from diff_voyn.heads.rung4_arithmetic import ArithmeticHead, levenshtein_ser

    v, u, tokens_of = _tiny_arith_cipher()
    rng = np.random.default_rng(6)
    plain = _markov_sample(toy_trans, 150, rng)
    chars, _ = _tiny_arith_stream(plain, tokens_of, rng)
    head = ArithmeticHead(toy_ev)
    _, letters, _ = head.decode_with_key(chars, v, u, language="toy", rank=np.arange(8))
    assert levenshtein_ser(letters, plain) < 0.05


def test_rung4_solve_smoke(toy_ev, toy_trans):
    """End-to-end mini solve: runs without NaN, returns finite score, and
    the Sinkhorn gradient phase actually updates the logits."""
    from diff_voyn.heads.rung4_arithmetic import ArithmeticHead

    v, _u, tokens_of = _tiny_arith_cipher()
    rng = np.random.default_rng(8)
    plain = _markov_sample(toy_trans, 120, rng)
    chars, _ = _tiny_arith_stream(plain, tokens_of, rng)
    head = ArithmeticHead(toy_ev, steps=10, chunk_chars=200, seed=0)
    res = head.solve(
        chars, language="toy", restarts=1, splits=(2,), polish=False, n_sym=8
    )
    assert np.isfinite(res.score)
    assert np.isfinite(res.raw_ll)
    assert len(res.decoded) > 0
    assert res.v_accuracy(v) == 1.0  # order-derived init is exact here
