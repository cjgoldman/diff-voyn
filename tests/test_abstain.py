"""Task 4.3 data: voynichesque pool → in-vocabulary letter streams; shuffled
windows keep the unigram histogram; the LID example stream emits labelled
batches with both classes and a single length per batch."""

import numpy as np
import pytest
import torch

from diff_voyn.data.abstain import (
    ABSTAIN_LABEL,
    KIND_SHUFFLED,
    KIND_UNIFORM,
    KIND_VOYNICHESQUE,
    LIDDataConfig,
    LIDExampleStream,
    build_lid_eval_set,
    sample_from_pool,
    shuffled_window,
    uniform_random_letters,
)
from diff_voyn.data.loader import LANG_TO_INDEX, CorpusWindows
from diff_voyn.data.noise import LETTER_BASE, N_LETTERS
from diff_voyn.vocab import LETTERS


@pytest.fixture
def tiny_corpus(tmp_path):
    rng = np.random.default_rng(0)
    doc_ids = {}
    for lang in LANG_TO_INDEX:
        d = tmp_path / lang / "docs"
        d.mkdir(parents=True)
        for i in range(2):
            text = "".join(rng.choice(list(LETTERS), size=3000))
            (d / f"{lang}{i}.txt").write_text(text)
        doc_ids[lang] = [f"{lang}0", f"{lang}1"]
    return CorpusWindows(tmp_path, doc_ids)


def _fake_pool(rng, n=6, length=1500):
    letters = np.arange(LETTER_BASE, LETTER_BASE + N_LETTERS, dtype=np.uint8)
    return [rng.choice(letters, size=length) for _ in range(n)]


def test_shuffled_window_keeps_histogram():
    rng = np.random.default_rng(0)
    ids = rng.integers(LETTER_BASE, LETTER_BASE + N_LETTERS, size=500).astype(np.uint8)
    s = shuffled_window(ids, rng)
    assert np.array_equal(np.bincount(s, minlength=32), np.bincount(ids, minlength=32))
    assert not np.array_equal(s, ids)


def test_pool_sampling_stays_inside_one_encryption():
    rng = np.random.default_rng(0)
    pool = [np.full(1200, LETTER_BASE + i, dtype=np.uint8) for i in range(3)]
    for _ in range(20):
        w = sample_from_pool(pool, 1024, rng)
        assert len(w) == 1024 and len(np.unique(w)) == 1
    with pytest.raises(ValueError):
        sample_from_pool(pool, 2000, rng)


def test_uniform_random_letters_in_vocab():
    w = uniform_random_letters(300, np.random.default_rng(1))
    assert ((w >= LETTER_BASE) & (w < LETTER_BASE + N_LETTERS)).all()


def test_stream_batches_are_labelled_and_mixed(tiny_corpus):
    pool = _fake_pool(np.random.default_rng(1))
    cfg = LIDDataConfig(
        p_abstain=0.4, lengths=(64, 128), length_probs=(0.5, 0.5), batch=16
    )
    stream = LIDExampleStream(tiny_corpus, pool, cfg, seed=0)
    it = iter(stream)
    seen_kinds, seen_labels, seen_lengths = set(), set(), set()
    for _ in range(12):
        b = next(it)
        assert b["ids"].shape == (16, b["length"]) and b["ids"].dtype == torch.int64
        assert b["label"].shape == (16,)
        abst = b["label"] == ABSTAIN_LABEL
        assert (b["kind"][abst] >= KIND_VOYNICHESQUE).all()
        assert (b["kind"][~abst] < KIND_VOYNICHESQUE).all()
        seen_kinds |= set(b["kind"].tolist())
        seen_labels |= set(b["label"].tolist())
        seen_lengths.add(b["length"])
    assert {KIND_VOYNICHESQUE, KIND_SHUFFLED, KIND_UNIFORM, 0} <= seen_kinds
    assert seen_labels == {0, 1, 2, ABSTAIN_LABEL}
    assert seen_lengths == {64, 128}


def test_stream_is_deterministic_in_seed(tiny_corpus):
    pool = _fake_pool(np.random.default_rng(1))
    cfg = LIDDataConfig(lengths=(64,), length_probs=(1.0,), batch=4)
    a = next(iter(LIDExampleStream(tiny_corpus, pool, cfg, seed=3)))
    b = next(iter(LIDExampleStream(tiny_corpus, pool, cfg, seed=3)))
    assert torch.equal(a["ids"], b["ids"]) and torch.equal(a["label"], b["label"])


def test_eval_set_shapes(tiny_corpus):
    pool = _fake_pool(np.random.default_rng(2))
    sets = build_lid_eval_set(tiny_corpus, pool, n_per_language=2, lengths=(64,))
    assert set(sets) == {
        "clean_L64",
        "noised_L64",
        "framed_L64",
        "voynichesque_L64",
        "shuffled_L64",
        "uniform_L64",
    }
    ids, y = sets["clean_L64"]
    assert ids.shape == (6, 64) and y.tolist() == [0, 0, 1, 1, 2, 2]
    assert (sets["voynichesque_L64"][1] == ABSTAIN_LABEL).all()
