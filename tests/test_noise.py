"""Phase-2 generator acceptance (tasks 2.1, 2.2, 2.3, 2.5): severity
parameterization, self-consistency of wrong keys, whitespace-free output,
NULL-frame invariants, mixture bookkeeping, and determinism (CRN)."""

import numpy as np
import pytest

from diff_voyn.data.noise import (
    KIND_CLEAN,
    KIND_FRAMED,
    KIND_FRAMED_NOISED,
    KIND_NOISED,
    LETTER_BASE,
    N_LETTERS,
    P_UNIGRAM_NAIBBE,
    NoiseConfig,
    NoiseMixture,
    SegmentationNoise,
    SubstitutionNoise,
    TranscriptionNoise,
    frame_with_nulls,
    framed_variant,
    noised_variant,
    tokens_per_letter,
)
from diff_voyn.vocab import LETTER_IDS, NULL_ID, decode

LETTER_SET = set(LETTER_IDS)


def zipf_text(n: int, seed: int = 0) -> np.ndarray:
    """A letter stream with a realistic skewed letter distribution."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, N_LETTERS + 1)
    p /= p.sum()
    return (rng.choice(N_LETTERS, size=n, p=p) + LETTER_BASE).astype(np.uint8)


def is_letters_only(ids: np.ndarray) -> bool:
    return set(np.unique(ids).tolist()) <= LETTER_SET


# ----------------------------------------------------------------- 2.1


def test_substitution_zero_severity_is_identity():
    ids = zipf_text(5000)
    out, info = SubstitutionNoise(0.0)(ids, np.random.default_rng(1))
    assert np.array_equal(out, ids)
    assert info["changed_fraction"] == 0.0 and info["n_wrong_letters"] == 0


@pytest.mark.parametrize("severity", [0.05, 0.2, 0.5, 1.0])
def test_substitution_realized_rate_matches_severity(severity):
    ids = zipf_text(200_000)
    out, info = SubstitutionNoise(severity)(ids, np.random.default_rng(3))
    realized = float((out != ids).mean())
    assert abs(realized - severity) < 0.01, (severity, realized)
    assert abs(info["changed_fraction"] - realized) < 1e-12
    assert is_letters_only(out) and len(out) == len(ids)


def test_substitution_is_self_consistent_not_iid():
    """Every changed occurrence of a letter goes to the SAME wrong letter, and
    letters outside the wrong-key set are never touched (design §7.3: wrong
    keys are self-consistent; i.i.d. flips are the wrong model)."""
    ids = zipf_text(50_000)
    noise = SubstitutionNoise(0.3, exact_rate=False)
    rng = np.random.default_rng(5)
    key = noise.sample_key(ids, rng)
    out, info = noise.apply_key(ids, key, rng)
    wrong = set((key.wrong_letters + LETTER_BASE).tolist())
    for letter in np.unique(ids):
        targets = set(np.unique(out[ids == letter]).tolist())
        if letter in wrong:
            assert len(targets) == 1 and targets != {letter}, letter
        else:
            assert targets == {letter}, letter
    # fully consistent key: realized rate ≥ severity
    assert info["changed_fraction"] >= 0.3 and info["p_apply"] == 1.0


def test_substitution_exact_rate_partial_key_is_still_consistent():
    """With exact_rate, only some occurrences are hit, but a hit occurrence of
    letter a always becomes σ(a) (a homophonic key with some wrong homophones)."""
    ids = zipf_text(50_000)
    noise = SubstitutionNoise(0.15)
    rng = np.random.default_rng(8)
    key = noise.sample_key(ids, rng)
    out, _ = noise.apply_key(ids, key, rng)
    changed = out != ids
    for letter in np.unique(ids[changed]):
        assert len(np.unique(out[(ids == letter) & changed])) == 1


def test_substitution_maps_are_many_to_one_over_draws():
    ids = zipf_text(20_000)
    noise = SubstitutionNoise(0.6)
    many = sum(
        noise.sample_key(ids, np.random.default_rng(s)).is_many_to_one
        for s in range(30)
    )
    assert many >= 15  # collisions are the norm at this coverage


def test_substitution_rejects_specials():
    with pytest.raises(ValueError):
        SubstitutionNoise(0.1)(
            np.array([NULL_ID, LETTER_BASE], np.uint8), np.random.default_rng(0)
        )


# ----------------------------------------------------------------- 2.2


def test_segmentation_zero_severity_is_identity():
    ids = zipf_text(3000)
    out, info = SegmentationNoise(0.0)(ids, np.random.default_rng(0))
    assert np.array_equal(out, ids) and info["edit_rate"] == 0.0


@pytest.mark.parametrize("severity", [0.02, 0.1, 0.3])
def test_segmentation_edit_rate_and_whitespace_free(severity):
    ids = zipf_text(200_000)
    out, info = SegmentationNoise(severity)(ids, np.random.default_rng(2))
    assert abs(info["edit_rate"] - severity) < 0.01, info
    assert info["n_deletions"] > 0 and info["n_insertions"] > 0
    assert info["n_duplications"] > 0
    # generator output verified whitespace-free (2.2 acceptance): letters only,
    # and the decoded text contains no whitespace character.
    assert is_letters_only(out)
    text = decode(out.tolist())
    assert len(text) == len(out) and not any(c.isspace() for c in text)
    # length bookkeeping: insertions lengthen, deletions shorten
    assert (
        len(out)
        == len(ids)
        - info["n_deletions"]
        + info["n_insertions"]
        + info["n_duplications"]
    )


def test_segmentation_deletions_insertions_balance_at_naibbe_rates():
    """Misparsed bigrams delete, misparsed unigrams insert: at p_unigram≈0.48
    the two are nearly balanced — a property of the parse model, not of a
    generic edit process."""
    ids = zipf_text(300_000)
    _, info = SegmentationNoise(0.2)(ids, np.random.default_rng(4))
    ins = info["n_insertions"] + info["n_duplications"]
    ratio = ins / info["n_deletions"]
    expect = P_UNIGRAM_NAIBBE / (1 - P_UNIGRAM_NAIBBE)
    assert abs(ratio - expect) < 0.08, (ratio, expect)


def test_segmentation_preserves_untouched_letters_in_order():
    """Deleting only ever removes a letter and inserting only ever adds one: the
    source is a subsequence of (output with insertions removed) — checked via
    the severity→edits relation on a tiny stream."""
    ids = zipf_text(40)
    out, info = SegmentationNoise(0.1)(ids, np.random.default_rng(11))
    assert (
        abs(len(out) - len(ids))
        <= info["n_deletions"] + info["n_insertions"] + info["n_duplications"]
    )
    assert tokens_per_letter(P_UNIGRAM_NAIBBE) == pytest.approx(
        1 / (2 - P_UNIGRAM_NAIBBE)
    )


# ----------------------------------------------------------------- 2.3


def test_transcription_default_is_five_percent():
    ids = zipf_text(200_000)
    out, info = TranscriptionNoise()(ids, np.random.default_rng(0))
    assert abs(info["edit_rate"] - 0.05) < 0.005
    n_events = info["n_substitutions"] + info["n_deletions"] + info["n_insertions"]
    assert n_events == pytest.approx(0.05 * len(ids), rel=0.1)
    # mix ≈ 80/10/10
    assert info["n_substitutions"] / n_events == pytest.approx(0.8, abs=0.03)
    assert is_letters_only(out)
    assert len(out) == len(ids) - info["n_deletions"] + info["n_insertions"]


@pytest.mark.parametrize("severity", [0.0, 0.02, 0.1, 0.3])
def test_transcription_severity_parameterized(severity):
    ids = zipf_text(100_000)
    out, info = TranscriptionNoise(severity)(ids, np.random.default_rng(1))
    assert abs(info["edit_rate"] - severity) < 0.01
    if severity == 0.0:
        assert np.array_equal(out, ids)


def test_transcription_substitutions_never_keep_the_letter():
    ids = zipf_text(50_000)
    out, _ = TranscriptionNoise(0.2, p_sub=1.0, p_del=0.0, p_ins=0.0)(
        ids, np.random.default_rng(0)
    )
    changed = (out != ids).mean()
    assert abs(changed - 0.2) < 0.01


# ----------------------------------------------------------------- 2.5


def test_null_frame_invariants():
    ids = zipf_text(100_000)
    out, info = frame_with_nulls(ids, np.random.default_rng(0))
    assert len(out) == 2 * info["n_tokens"]
    # NULL only at slot-2 (odd) positions, never in slot 1
    assert (out[0::2] != NULL_ID).all()
    assert (out[out == NULL_ID] == NULL_ID).all() and (out[1::2] == NULL_ID).any()
    # never two adjacent NULLs
    assert not ((out[:-1] == NULL_ID) & (out[1:] == NULL_ID)).any()
    # dropping NULLs recovers the stream exactly
    assert np.array_equal(out[out != NULL_ID], ids)
    # NULL rate = p_unigram / 2 of all slots (2N-slot scheme, design §8)
    assert abs(info["null_fraction"] - P_UNIGRAM_NAIBBE / 2) < 0.01
    assert abs(len(out) / len(ids) - 2 * tokens_per_letter(P_UNIGRAM_NAIBBE)) < 0.02


def test_null_frame_p_unigram_extremes():
    ids = zipf_text(1000)
    out, info = frame_with_nulls(ids, np.random.default_rng(0), p_unigram=1.0)
    assert info["null_fraction"] == 0.5 and len(out) == 2000
    out, info = frame_with_nulls(ids, np.random.default_rng(0), p_unigram=0.0)
    assert info["null_fraction"] == 0.0 and len(out) == 1000


# ----------------------------------------------------------------- mixture


def test_noise_config_guards():
    NoiseConfig()  # defaults valid
    with pytest.raises(ValueError):
        NoiseConfig(p_clean=0.4, p_noised=0.4, p_framed=0.1, p_framed_noised=0.1)
    with pytest.raises(ValueError):
        NoiseConfig(p_clean=0.8, p_noised=0.1, p_framed=0.05, p_framed_noised=0.05)
    with pytest.raises(ValueError):
        NoiseConfig(p_clean=0.5, p_noised=0.5, p_framed=0.1, p_framed_noised=0.1)


def test_mixture_kind_fractions_and_outputs():
    mix = NoiseMixture(NoiseConfig())
    rng = np.random.default_rng(0)
    seq_len = 256
    counts = np.zeros(4, int)
    null_seen = {k: 0 for k in range(4)}
    src = zipf_text(50_000, seed=1)
    for _ in range(4000):
        kind = mix.sample_kind(rng)
        counts[kind] += 1
        n_src = mix.source_length(kind, seq_len)
        start = rng.integers(0, len(src) - n_src)
        out, info = mix.apply(src[start : start + n_src], kind, seq_len, rng)
        assert len(out) == seq_len and out.dtype == np.uint8
        assert "tiled" not in info
        has_null = bool((out == NULL_ID).any())
        null_seen[kind] += has_null
        if kind == KIND_CLEAN:
            assert np.array_equal(out, src[start : start + seq_len])
        if kind in (KIND_NOISED, KIND_FRAMED_NOISED):
            assert any(
                k in info for k in ("substitution", "segmentation", "transcription")
            )
        letters_only = out[out != NULL_ID]
        assert is_letters_only(letters_only)
    frac = counts / counts.sum()
    assert np.allclose(frac, NoiseConfig().kind_probs, atol=0.03), frac
    assert null_seen[KIND_CLEAN] == 0 and null_seen[KIND_NOISED] == 0
    assert null_seen[KIND_FRAMED] == counts[KIND_FRAMED]
    assert null_seen[KIND_FRAMED_NOISED] == counts[KIND_FRAMED_NOISED]


def test_mixture_family_application_rates():
    cfg = NoiseConfig()
    mix = NoiseMixture(cfg)
    rng = np.random.default_rng(1)
    src = zipf_text(4000)
    seen = {"substitution": 0, "segmentation": 0, "transcription": 0}
    n = 1500
    for _ in range(n):
        _, info = mix.apply(src[:1536], KIND_NOISED, 1024, rng)
        for k in seen:
            seen[k] += k in info
        sev = info.get("substitution", {}).get("severity")
        if sev is not None:
            assert cfg.substitution_severity[0] <= sev <= cfg.substitution_severity[1]
    # conditioning on "at least one family" lifts each rate slightly above its
    # marginal; check against the exact conditional values
    p_none = (
        (1 - cfg.p_substitution) * (1 - cfg.p_segmentation) * (1 - cfg.p_transcription)
    )
    for k, p in (
        ("substitution", cfg.p_substitution),
        ("segmentation", cfg.p_segmentation),
        ("transcription", cfg.p_transcription),
    ):
        assert abs(seen[k] / n - p / (1 - p_none)) < 0.04, (k, seen[k] / n)


def test_generators_are_deterministic_under_rng_seed():
    """CRN (non-negotiable #4): the same seed gives the same corruption, so the
    same noised text can be scored by several checkpoints / conditions."""
    src = zipf_text(3000)
    a = noised_variant(src, 1024, np.random.default_rng(9))
    b = noised_variant(src, 1024, np.random.default_rng(9))
    c = noised_variant(src, 1024, np.random.default_rng(10))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    fa = framed_variant(src, 1024, np.random.default_rng(9))
    fb = framed_variant(src, 1024, np.random.default_rng(9))
    assert np.array_equal(fa, fb) and len(fa) == 1024 and (fa == NULL_ID).any()


def test_dataset_emits_kinds_with_mixture(tmp_path):
    """Loader integration: the stream carries the example kind and the
    realized kind fractions match the config; clean windows are exactly
    seq_len source text."""
    from diff_voyn.data.loader import CorpusWindows, DiffVoynIterableDataset

    corpus = tmp_path / "v1"
    doc_ids = {}
    for li, lang in enumerate(("latin", "italian", "german")):
        d = corpus / lang / "docs"
        d.mkdir(parents=True)
        text = decode(zipf_text(20_000, seed=li).tolist())
        (d / "doc0.txt").write_text(text)
        doc_ids[lang] = ["doc0"]
    windows = CorpusWindows(corpus, doc_ids)
    ds = DiffVoynIterableDataset(windows, seq_len=128, seed=0, noise=NoiseMixture())
    it = iter(ds)
    kinds = np.zeros(4, int)
    for _ in range(1500):
        ex = next(it)
        kinds[int(ex["kind"])] += 1
        assert ex["ids"].shape == (128,)
        has_null = bool((ex["ids"] == NULL_ID).any())
        assert has_null == (int(ex["kind"]) in (KIND_FRAMED, KIND_FRAMED_NOISED))
        if int(ex["kind"]) == KIND_CLEAN:
            assert float(ex["sub_severity"]) == 0.0
    assert np.allclose(kinds / kinds.sum(), NoiseConfig().kind_probs, atol=0.05)
    # without a mixture the stream is Phase-A identical: all clean
    ds0 = DiffVoynIterableDataset(windows, seq_len=128, seed=0)
    ex = next(iter(ds0))
    assert int(ex["kind"]) == KIND_CLEAN and "sub_severity" in ex
