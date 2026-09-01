"""PolygraphiaCipher wrapper (diff_voyn/ciphers/polygraphia.py)."""

import csv

import pytest

from diff_voyn.ciphers.polygraphia import (
    ALPHABET,
    PolygraphiaCipher,
    PolygraphiaTables,
    polygraphia_pre_map,
)


@pytest.fixture()
def tiny_csv(tmp_path):
    """Three columns; column 2 repeats a word under two letters (the real
    Book I col-22 'diuitias' situation)."""
    path = tmp_path / "tables.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["book", "column", "letter", "word", "raw"])
        for c in (1, 2, 3):
            for i, letter in enumerate(ALPHABET):
                word = f"w{c}x{i}"
                if c == 2 and letter == "y":
                    word = "w2x0"  # duplicate of the 'a' word
                w.writerow(["b1", c, letter, word, ""])
    return path


def test_pre_map_folds_u_only():
    assert polygraphia_pre_map("quum") == "qvvm"
    assert polygraphia_pre_map("wkyz") == "wkyz"


def test_alphabet_has_24_letters_w_last():
    assert len(ALPHABET) == 24
    assert ALPHABET[-1] == "w"
    assert "j" not in ALPHABET and "u" not in ALPHABET


def test_round_trip_with_wrap(tiny_csv):
    cipher = PolygraphiaCipher(PolygraphiaTables(tiny_csv))
    text = polygraphia_pre_map("gratiasagimustibi")  # 17 letters > 3 columns
    words = cipher.encipher(text, start_column=2)
    assert len(words) == len(text)
    assert cipher.decipher(words, start_column=2) == text


def test_ambiguous_word_candidates(tiny_csv):
    tables = PolygraphiaTables(tiny_csv)
    assert tables.ambiguous[1] == {"w2x0": ("a", "y")}
    cipher = PolygraphiaCipher(tables)
    # letter y in column 2 enciphers to the duplicated word; decipher takes
    # the first printed row (a) but candidates carry both
    words = cipher.encipher("y", start_column=2)
    assert cipher.decipher(words, start_column=2) == "a"
    assert cipher.decipher_candidates(words, start_column=2) == [("a", "y")]


def test_non_contiguous_columns_rejected(tiny_csv):
    with open(tiny_csv) as f:
        rows = list(csv.reader(f))
    with open(tiny_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(r for r in rows if r[1] != "2")
    with pytest.raises(ValueError, match="non-contiguous"):
        PolygraphiaTables(tiny_csv)


def test_n_columns_truncation(tiny_csv):
    tables = PolygraphiaTables(tiny_csv, n_columns=2)
    assert len(tables) == 2
