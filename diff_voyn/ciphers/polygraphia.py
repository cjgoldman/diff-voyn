"""Polygraphia I–II "Ave Maria" cipher wrapper
(docs/polygraphia_digitization_scope.md §3.3).

Trithemius (1518), Books I–II: each plaintext letter is replaced by the word
standing under that letter in the *current column* of the printed tables;
columns are consumed strictly in order, one word per column, so the
ciphertext reads as running Latin prayer prose. The printed instructions
(image 0029: "si de quolibet ordine una duntaxat capiatur"; 0228: Book II
keeps "formam eiusdem … & modum" with its 308 columns "simplici
progressione") give a deterministic cipher: the whole key is the start
column, columns advancing by one per letter, wrapping cyclically at the end
of the book's table sequence (the "reversion" of image 0030). No RNG.

Alphabet: 24 letters, ``a b c d e f g h i k l m n o p q r s t v x y z w``
("vicesima quarta litera w in fine alphabeti assumitur", image 0029 — w is
last; no j, no u). Cipher-side pre-map from the shared 25-letter normalized
alphabet: u→v only (j→i is already the shared normalizer's fold). Mirroring
the Naibbe wrapper convention, the plaintext half of a generated pair stays
in cipher-alphabet space so pairs align.

Tables: ``references/polygraphia_tables_v1.csv`` (book,column,letter,word,raw)
once frozen; until then a provisional CSV of the transcribed-so-far columns
can be passed explicitly. A cipher instance uses one book's contiguous
column range starting at column 1.
"""

from __future__ import annotations

import csv
import pathlib

ALPHABET = "abcdefghiklmnopqrstvxyz" + "w"  # printed row order, w last
_PRE_MAP = str.maketrans({"u": "v"})

_REPO = pathlib.Path(__file__).resolve().parent.parent.parent
TABLES_V1 = _REPO / "references" / "polygraphia_tables_v1.csv"


def polygraphia_pre_map(normalized_text: str) -> str:
    """Map the shared 25-letter normalized stream onto the 24-letter table
    alphabet (u→v; lossy, recorded here and nowhere else)."""
    out = normalized_text.translate(_PRE_MAP)
    assert set(out) <= set(ALPHABET), sorted(set(out) - set(ALPHABET))
    return out


class PolygraphiaTables:
    """The digitized word tables of one book, columns 1..n contiguous."""

    def __init__(
        self,
        csv_path: pathlib.Path | str = TABLES_V1,
        book: str = "b1",
        n_columns: int | None = None,
    ):
        self.book = book
        by_col: dict[int, dict[str, str]] = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["book"] != book:
                    continue
                col = int(row["column"])
                if n_columns is not None and col > n_columns:
                    continue
                by_col.setdefault(col, {})[row["letter"]] = row["word"]
        if not by_col:
            raise ValueError(f"no columns for book {book!r} in {csv_path}")
        n = max(by_col)
        missing = [c for c in range(1, n + 1) if c not in by_col]
        if missing:
            raise ValueError(f"non-contiguous columns, missing {missing[:5]}…")
        self.columns: list[dict[str, str]] = []
        self.inverse: list[dict[str, tuple[str, ...]]] = []
        self.ambiguous: list[dict[str, tuple[str, ...]]] = []
        for c in range(1, n + 1):
            col = by_col[c]
            if sorted(col) != sorted(ALPHABET):
                raise ValueError(f"column {c}: keys {sorted(col)} != alphabet")
            inv: dict[str, tuple[str, ...]] = {}
            for letter in ALPHABET:  # printed row order fixes tie order
                word = col[letter]
                inv[word] = inv.get(word, ()) + (letter,)
            # a word printed under two letters of the same column is REAL
            # (e.g. Book I col 22 "diuitias" under o and y, verified on the
            # 1518 scan) - deciphering it is ambiguous, as it was in 1518.
            self.columns.append(col)
            self.inverse.append(inv)
            self.ambiguous.append({w: ls for w, ls in inv.items() if len(ls) > 1})

    def __len__(self) -> int:
        return len(self.columns)


class PolygraphiaCipher:
    """Deterministic table cipher; the key is the start column."""

    def __init__(self, tables: PolygraphiaTables):
        self.tables = tables

    def encipher(self, text: str, start_column: int = 1) -> list[str]:
        """``text`` in cipher-alphabet space (use :func:`polygraphia_pre_map`
        first) → one table word per letter, columns advancing cyclically."""
        n = len(self.tables)
        cols = self.tables.columns
        return [cols[(start_column - 1 + i) % n][ch] for i, ch in enumerate(text)]

    def decipher(self, words: list[str], start_column: int = 1) -> str:
        """Truth alignment: invert each word in its position's column."""
        n = len(self.tables)
        inv = self.tables.inverse
        out = []
        for i, w in enumerate(words):
            col = inv[(start_column - 1 + i) % n]
            if w not in col:
                raise KeyError(f"position {i}: {w!r} not in its column")
            out.append(col[w][0])  # ambiguous words take the first printed row
        return "".join(out)

    def decipher_candidates(
        self, words: list[str], start_column: int = 1
    ) -> list[tuple[str, ...]]:
        """Per-position candidate letters (length > 1 where the printed
        column repeats a word)."""
        n = len(self.tables)
        inv = self.tables.inverse
        return [inv[(start_column - 1 + i) % n][w] for i, w in enumerate(words)]
