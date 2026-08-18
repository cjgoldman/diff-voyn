"""Task 0.1 acceptance: frozen vocab spec, round-trip completeness."""

from diff_voyn import vocab


def test_vocab_is_exactly_32():
    assert vocab.VOCAB_SIZE == 32
    assert len(vocab.TOKENS) == 32
    assert len(set(vocab.TOKENS)) == 32


def test_naibbe_letters_are_the_23_attested():
    assert vocab.NAIBBE_LETTERS == "abcdefghilmnopqrstuvxyz"
    assert len(vocab.NAIBBE_LETTERS) == 23


def test_letters_are_naibbe_plus_kw_minus_nothing():
    assert set(vocab.LETTERS) == set(vocab.NAIBBE_LETTERS) | {"k", "w"}
    assert "j" not in vocab.LETTERS
    assert len(vocab.LETTERS) == 25


def test_u_v_distinct():
    assert vocab.TOKEN_TO_ID["u"] != vocab.TOKEN_TO_ID["v"]


def test_encode_decode_round_trip():
    s = "kwaequitasuvxyz"
    assert vocab.decode(vocab.encode(s)) == s


def test_encode_rejects_out_of_alphabet():
    import pytest

    with pytest.raises(KeyError):
        vocab.encode("j")
    with pytest.raises(KeyError):
        vocab.encode("a b")  # no space token by design


def test_spec_hash_stable():
    assert vocab.spec_hash() == vocab.spec_hash()
    assert len(vocab.spec_hash()) == 64
