"""Naibbe token parsing + type vocabularies — the two stubs the inverse-note
skeleton left open (`parse_token`, vocabularies), task CH.6.

Built directly from the pinned glyph tables (greshko/naibbe-cipher @ df3d074,
``references/naibbe_tables.csv``) — with the actual tables available, the
Zattera-slot-grammar *approximation* of the inverse note is superseded by the
exact structural prior: a token is (a) a unigram glyph type, and/or (b) a
``prefix_glyph + suffix_glyph`` concatenation, enumerated exactly.

What is and is not "known" to the cryptanalyst here (structural prior, not
key material): the glyph *strings* of the 18 tables and which slot (unigram /
prefix / suffix) each belongs to — this is the published cipher apparatus.
What stays unknown and is learned by the head: WHICH LETTER each glyph type
encodes (the U / Pre / Suf soft inverse maps, ~9.5k logits). Greshko's
"unambiguous" bigram mode guarantees a true-bigram token has exactly one
(prefix, suffix) split and never equals a unigram glyph; true-unigram tokens
may still carry spurious bigram splits, which the semi-Markov DP marginalizes.

Ground-truth maps (validation only, never seen by the head): glyph type →
letter, read off the ``<state>_<table>_<letter>`` code names.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

from ..ciphers.external import naibbe_repo
from ..vocab import NAIBBE_LETTERS
from .ngram import LETTER_TO_IDX


@dataclass
class TokenParse:
    uni: int | None  # unigram type id, or None
    bi: list[tuple[int, int]]  # (prefix type id, suffix type id) splits


class NaibbeParser:
    def __init__(self):
        import csv

        rows = []
        with open(
            naibbe_repo() / "references" / "naibbe_tables.csv",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            rows = list(csv.DictReader(f))
        by_state: dict[str, dict[str, set[str]]] = {
            "unigram": {},
            "prefix": {},
            "suffix": {},
        }
        for r in rows:
            state, _table, letter = r["code"].split("_")
            by_state[state].setdefault(r["glyphs"], set()).add(letter)
        # A glyph string appearing in several tables for the same state is one
        # TYPE (the head can only map the string). In the pinned tables no
        # string maps to two different letters within a state — assert it.
        self.types: dict[str, list[str]] = {}
        self.truth: dict[str, np.ndarray] = {}
        for state, glyphmap in by_state.items():
            glyphs = sorted(glyphmap)
            self.types[state] = glyphs
            truth = np.empty(len(glyphs), dtype=np.int64)
            for i, g in enumerate(glyphs):
                letters = glyphmap[g]
                assert len(letters) == 1, f"{state} glyph {g!r} maps to {letters}"
                truth[i] = LETTER_TO_IDX[next(iter(letters))]
            self.truth[state] = truth
        self.uni_ids = {g: i for i, g in enumerate(self.types["unigram"])}
        self.pre_ids = {g: i for i, g in enumerate(self.types["prefix"])}
        self.suf_ids = {g: i for i, g in enumerate(self.types["suffix"])}
        # letters the Naibbe alphabet supports (23; no k/w) — head support set
        self.letter_support = np.array(
            [LETTER_TO_IDX[c] for c in NAIBBE_LETTERS], dtype=np.int64
        )

    @property
    def n_uni(self) -> int:
        return len(self.types["unigram"])

    @property
    def n_pre(self) -> int:
        return len(self.types["prefix"])

    @property
    def n_suf(self) -> int:
        return len(self.types["suffix"])

    # -- block (state x table) structure for the Sinkhorn head ---------------
    #
    # The published apparatus assigns every code to a (state, table) cell and
    # each block's key is a bijection between its 23 glyph codes and the 23
    # letters. Deck weights are published too. A glyph STRING may cover
    # several codes (only "dar": unigram e in beta2+beta3); its emission sums
    # its codes' contributions.

    TABLES = ("alpha", "beta1", "beta2", "beta3", "gamma1", "gamma2")
    # published deck weights (upstream CARD_WEIGHTS), keyed by use_78 flag
    CARD_WEIGHTS: typing.ClassVar[dict] = {
        False: {
            "alpha": 20,
            "beta1": 8,
            "beta2": 8,
            "beta3": 8,
            "gamma1": 4,
            "gamma2": 4,
        },
        True: {
            "alpha": 28,
            "beta1": 14,
            "beta2": 11,
            "beta3": 11,
            "gamma1": 7,
            "gamma2": 7,
        },
    }

    def build_blocks(self) -> None:
        """Populate code-level block structure (idempotent).

        - ``block_codes[(state, table)]``: list of 23 glyph strings (rows).
        - ``block_truth[(state, table)]``: row -> support-letter index (23-dim,
          index into NAIBBE_LETTERS order), validation only.
        - ``type_cells[state][type_id]``: list of (table, row) cells whose
          glyph equals that type's string.
        """
        import csv

        if hasattr(self, "block_codes"):
            return
        with open(
            naibbe_repo() / "references" / "naibbe_tables.csv",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            rows = list(csv.DictReader(f))
        support_idx = {c: i for i, c in enumerate(NAIBBE_LETTERS)}
        cells: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for r in rows:
            state, table, letter = r["code"].split("_")
            cells.setdefault((state, table), []).append((r["glyphs"], letter))
        self.block_codes: dict[tuple[str, str], list[str]] = {}
        self.block_truth: dict[tuple[str, str], np.ndarray] = {}
        for key, entries in cells.items():
            entries = sorted(entries)
            assert len(entries) == len(NAIBBE_LETTERS)
            self.block_codes[key] = [g for g, _ in entries]
            self.block_truth[key] = np.array(
                [support_idx[letter] for _, letter in entries], dtype=np.int64
            )
        self.type_cells: dict[str, list[list[tuple[str, int]]]] = {}
        for state in ("unigram", "prefix", "suffix"):
            per_type: list[list[tuple[str, int]]] = [[] for _ in self.types[state]]
            ids = {g: i for i, g in enumerate(self.types[state])}
            for table in self.TABLES:
                for row, g in enumerate(self.block_codes[(state, table)]):
                    per_type[ids[g]].append((table, row))
            self.type_cells[state] = per_type

    def parse_token(self, token: str) -> TokenParse:
        uni = self.uni_ids.get(token)
        bi = [
            (self.pre_ids[token[:i]], self.suf_ids[token[i:]])
            for i in range(1, len(token))
            if token[:i] in self.pre_ids and token[i:] in self.suf_ids
        ]
        return TokenParse(uni=uni, bi=bi)

    def parse_stream(self, tokens: list[str]) -> list[TokenParse]:
        parses = [self.parse_token(t) for t in tokens]
        bad = [t for t, p in zip(tokens, parses) if p.uni is None and not p.bi]
        if bad:
            raise ValueError(f"unparseable tokens (not Naibbe output?): {bad[:5]}")
        return parses
