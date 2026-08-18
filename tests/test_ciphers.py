"""Task 0.7 acceptance: pinned generators run; arithmetic round-trip exact;
Naibbe pairs stay aligned. Skipped when external repos are absent."""

import pytest

pytest.importorskip("pandas")

try:
    from diff_voyn.ciphers.external import naibbe_repo, voynich_attack_repo

    naibbe_repo()
    voynich_attack_repo()
    EXTERNALS = True
except Exception:  # noqa: BLE001 — any failure means externals unusable
    EXTERNALS = False

pytestmark = pytest.mark.skipif(
    not EXTERNALS, reason="external cipher repos not fetched/pinned"
)

SAMPLE = (
    "Gallia est omnis divisa in partes tres quarum unam incolunt Belgae "
    "aliam Aquitani tertiam qui ipsorum lingua Celtae nostra Galli appellantur"
)


def test_naibbe_pairs_align():
    from diff_voyn.ciphers.naibbe import NaibbeCipher

    cipher = NaibbeCipher(seed=0)
    tokens, segments = cipher.encipher(SAMPLE)
    assert len(tokens) == len(segments)
    assert all(len(s) in (1, 2) for s in segments)
    assert all(t and " " not in t for t in tokens)


def test_naibbe_seed_reproducible():
    from diff_voyn.ciphers.naibbe import NaibbeCipher

    cipher = NaibbeCipher(seed=0)
    a, _ = cipher.encipher(SAMPLE)
    cipher.seed(0)
    b, _ = cipher.encipher(SAMPLE)
    assert a == b


def test_naibbe_premap_covers_kw():
    from diff_voyn.ciphers.naibbe import naibbe_pre_map
    from diff_voyn.normalize import normalize

    assert naibbe_pre_map(normalize("Kraft Wasser")) == "craftuuasser"


def test_arithmetic_round_trip_exact():
    from diff_voyn.ciphers.arithmetic import ArithmeticCipher
    from diff_voyn.normalize import normalize

    arith = ArithmeticCipher()
    plain = normalize(SAMPLE)
    cipher = arith.encode(plain)
    assert arith.decode_text(cipher) == plain


def test_arithmetic_alphabet_is_frozen_vocab():
    from diff_voyn.ciphers.arithmetic import our_alphabet_values
    from diff_voyn.vocab import LETTERS

    values = our_alphabet_values()
    assert set(values) == set(LETTERS)
    assert values["a"] == 3 and values["z"] == 28 and "j" not in values


def test_arithmetic_save_load_identical(tmp_path):
    import random

    from diff_voyn.ciphers.arithmetic import ArithmeticCipher

    a = ArithmeticCipher()
    a.save(tmp_path / "table.csv")
    b = ArithmeticCipher(table_path=tmp_path / "table.csv")
    text = "galliaestomnis"
    assert a.enc.encode(text, rng=random.Random(1)) == b.enc.encode(
        text, rng=random.Random(1)
    )


def test_voynichesque_runs_and_differs_from_input():
    from diff_voyn.ciphers.controls import Voynichesque

    v = Voynichesque()
    out = v.generate(SAMPLE, seed=0)
    assert out and out != SAMPLE
    out2 = v.generate(SAMPLE, seed=0)
    assert out == out2  # seeded determinism
