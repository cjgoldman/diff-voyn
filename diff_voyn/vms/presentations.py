"""VMS ciphertext presentations for the cipher heads — task 6.1.

Phase 5 validated four heads on four ciphertext *forms*: a symbol stream
(rungs 1–2), a stream of word tokens over the Naibbe glyph alphabet (rung 3)
and an unsegmented 16-value character stream (rung 4). The manuscript is
transcribed, not enciphered by us, so each head needs the transcription
presented in its own form. Nothing in the design doc fixes these mappings
(design §9 specifies only per-dialect reporting), so they are frozen here
with their coverage numbers:

``eva``
    The EVA character stream of an IVTFF transcription (Takahashi ``IT2a``
    primary, Reference ``RF1b`` as the replicate transcription), per Currier
    dialect, exactly the model-facing stream of ``ingest.py`` (whitespace
    stripped). 20 characters in Takahashi, so it is inside the 25-letter
    support of the rung-1 bijection and the rung-2 homophonic head applies
    with ``n_symbols = 20``. RF1b carries IVTFF residue the Phase-0 ingest
    did not strip (``{ }`` ligature marks, ``'`` plumes, capital ``I``) —
    those characters are removed here; letters inside ``{}`` are kept.

``words``
    The word tokens (``.``/``,`` separated) of the same transcription, for
    the Naibbe head, which consumes a token stream and parses every token
    against the published tables. Tokens the tables cannot produce are
    dropped and counted — the head explains the *coverable* subsequence and
    the report carries the coverage (73% of Currier-A words and 82% of B
    words in Takahashi parse; the rest are the "unparseable" residue Greshko
    reports too).

``boxer``
    Boxer's independent glyph-level transcription (voynich-attack @
    ``e324bee``, ``transcription/vms.csv``: comma-separated glyph units per
    token, 66 glyph types). The rung-4 arithmetic head needs a 16-value
    stream, and Boxer's ``pseudo_vms`` generator — the pinned rung-4 cipher
    — was calibrated on this transcription, so it is the natural 16-symbol
    presentation: the 16 most frequent glyph types (97.9% of glyph
    occurrences) become the 16 symbols; tokens containing any other glyph
    (including the ``?`` uncertain marker) are dropped and counted. The
    same glyph stream, with the ``n_symbols`` most frequent types, is also
    the second symbol presentation for rungs 1–2 (glyph units instead of
    EVA characters — a different, coarser choice of "symbol"). Dialect is
    assigned per folio from the IVTFF ``$L`` headers (``102ra`` ↔
    ``f102r1``; composite foldout names take the dialect their sub-pages
    agree on, else ``unassigned``).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np

from ..ciphers.external import data_root
from .ingest import _clean_line, parse_ivtff

TRANSCRIPTIONS = {"IT2a": "IT2a-n.txt", "RF1b": "RF1b-e.txt"}
DIALECTS = ("A", "B")
_NON_LETTER = re.compile(r"[^a-z]")


@dataclass
class Presentation:
    transcription: str
    dialect: str
    kind: str  # eva | words | boxer | wordtypes<K>
    symbols: np.ndarray  # symbol ids (eva / boxer) — empty for words
    alphabet: list[str]  # symbol id -> glyph string
    tokens: list[str] = field(default_factory=list)  # words (kind == words)
    token_starts: np.ndarray | None = None  # boxer: token boundaries
    coverage: dict = field(default_factory=dict)

    @property
    def n_symbols(self) -> int:
        return len(self.alphabet)

    def as_dict(self) -> dict:
        return {
            "transcription": self.transcription,
            "dialect": self.dialect,
            "kind": self.kind,
            "n_symbols": self.n_symbols,
            "alphabet": self.alphabet,
            "n_stream": (
                len(self.symbols) if self.kind != "words" else len(self.tokens)
            ),
            "coverage": self.coverage,
        }


# -- IVTFF words / characters per dialect ----------------------------------


def ivtff_path(transcription: str, root: Path | None = None) -> Path:
    root = root or data_root()
    return root / "raw" / "vms" / TRANSCRIPTIONS[transcription]


def dialect_words(transcription: str, root: Path | None = None) -> dict[str, list[str]]:
    """{dialect: [word, ...]} in reading order; letters only (IVTFF residue
    stripped, uncertain glyphs / extended-EVA codes dropped as in ingest)."""
    pages = parse_ivtff(ivtff_path(transcription, root))
    out: dict[str, list[str]] = {"A": [], "B": [], "unassigned": []}
    dropped: Counter = Counter()
    for page in pages:
        key = page["currier"] if page["currier"] in DIALECTS else "unassigned"
        for line in page["lines"]:
            cleaned = _clean_line(line, dropped)
            for w in re.split(r"[.,\s]+", cleaned):
                w = _NON_LETTER.sub("", w.lower())
                if w:
                    out[key].append(w)
    return out


def page_dialects(
    transcription: str = "IT2a", root: Path | None = None
) -> dict[str, str | None]:
    return {
        p["page"]: p["currier"] for p in parse_ivtff(ivtff_path(transcription, root))
    }


def eva_presentation(
    transcription: str, dialect: str, root: Path | None = None
) -> Presentation:
    words = dialect_words(transcription, root)[dialect]
    stream = "".join(words)
    alphabet = sorted(set(stream))
    idx = {c: i for i, c in enumerate(alphabet)}
    sym = np.fromiter((idx[c] for c in stream), dtype=np.int64, count=len(stream))
    return Presentation(
        transcription,
        dialect,
        "eva",
        sym,
        alphabet,
        coverage={
            "n_words": len(words),
            "n_chars": len(stream),
            "covered_fraction": 1.0,
        },
    )


def words_presentation(
    transcription: str, dialect: str, parser, root: Path | None = None
) -> Presentation:
    """Naibbe-parseable word tokens (the head's input) + coverage."""
    words = dialect_words(transcription, root)[dialect]
    ok = []
    for w in words:
        p = parser.parse_token(w)
        if p.uni is not None or p.bi:
            ok.append(w)
    n_chars = sum(map(len, words))
    n_ok = sum(map(len, ok))
    return Presentation(
        transcription,
        dialect,
        "words",
        np.zeros(0, dtype=np.int64),
        [],
        tokens=ok,
        coverage={
            "n_words": len(words),
            "n_parseable_words": len(ok),
            "word_fraction": len(ok) / max(len(words), 1),
            "n_chars": n_chars,
            "n_parseable_chars": n_ok,
            "covered_fraction": n_ok / max(n_chars, 1),
        },
    )


# -- word-type presentation (word-level homophonic head) --------------------


def wordtypes_presentation(
    transcription: str,
    dialect: str,
    n_top: int | None = None,
    root: Path | None = None,
    words: list[str] | None = None,
    name: str | None = None,
) -> Presentation:
    """Word tokens as homophonic SYMBOLS: the ``n_top`` most frequent word
    types of the dialect (all types when ``None``) become symbol ids;
    other tokens are dropped and counted. ``token_starts`` carries each kept
    token's index in the original token stream so the head knows which
    kept tokens were adjacent (the repeat rule). Coverage is reported in
    words and in characters (the cross-head unit)."""
    words = words if words is not None else dialect_words(transcription, root)[dialect]
    counts = Counter(words)
    alphabet = [w for w, _ in counts.most_common(n_top)]
    idx = {w: i for i, w in enumerate(alphabet)}
    sym, pos = [], []
    for i, w in enumerate(words):
        if w in idx:
            sym.append(idx[w])
            pos.append(i)
    n_chars = sum(map(len, words))
    n_ok = sum(len(words[i]) for i in pos)
    kind = f"wordtypes{n_top if n_top is not None else 'all'}"
    pres = Presentation(
        transcription,
        dialect,
        kind,
        np.asarray(sym, dtype=np.int64),
        alphabet,
        tokens=list(words),
        token_starts=np.asarray(pos, dtype=np.int64),
        coverage={
            "n_words": len(words),
            "n_types": len(counts),
            "n_hapax": sum(1 for v in counts.values() if v == 1),
            "n_kept_words": len(sym),
            "word_fraction": len(sym) / max(len(words), 1),
            "n_chars": n_chars,
            "n_kept_chars": n_ok,
            "covered_fraction": n_ok / max(n_chars, 1),
            "adjacent_repeats_per_1000": 1000.0
            * sum(1 for a, b in pairwise(words) if a == b)
            / max(len(words) - 1, 1),
        },
    )
    if name is not None:
        pres.transcription = name
    return pres


def write_wordtypes_presentations(
    out_dir: Path, n_tops=(None,), root: Path | None = None
) -> dict:
    """Persist the word-type presentations per transcription × dialect."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for tr in TRANSCRIPTIONS:
        words_by_d = dialect_words(tr, root)
        for d in DIALECTS:
            for n_top in n_tops:
                pres = wordtypes_presentation(tr, d, n_top, root, words=words_by_d[d])
                rec = pres.as_dict()
                rec["name"] = f"{tr}/{d}"
                rec["symbols"] = pres.symbols.tolist()
                rec["token_pos"] = pres.token_starts.tolist()
                rec["all_tokens"] = pres.tokens
                (out_dir / f"{tr}_{d}_{pres.kind}.json").write_text(json.dumps(rec))
                summary[f"{tr}/{d}/{pres.kind}"] = pres.as_dict()
    (out_dir / "wordtypes_presentations.json").write_text(json.dumps(summary, indent=2))
    return summary


# -- Boxer glyph transcription ----------------------------------------------


def boxer_csv(root: Path | None = None) -> Path:
    root = root or data_root()
    return root / "external" / "voynich-attack" / "transcription" / "vms.csv"


def _boxer_folio_to_dialect(folio: str, pages: dict[str, str | None]) -> str:
    cands = [f"f{folio}"]
    m = re.match(r"^(\d+[rv])([ab])$", folio)
    if m:
        cands.append(f"f{m.group(1)}{'ab'.index(m.group(2)) + 1}")
    for c in cands:
        if c in pages:
            return pages[c] if pages[c] in DIALECTS else "unassigned"
    subs = {v for k, v in pages.items() if re.match(rf"^f{re.escape(folio)}\d+$", k)}
    if len(subs) == 1 and next(iter(subs)) in DIALECTS:
        return next(iter(subs))
    return "unassigned"


def boxer_tokens(root: Path | None = None) -> dict[str, list[list[str]]]:
    """{dialect: [[glyph, glyph, ...], ...]} from Boxer's csv, in reading
    order, dialect per folio from the Takahashi page headers."""
    pages = page_dialects("IT2a", root)
    out: dict[str, list[list[str]]] = {"A": [], "B": [], "unassigned": []}
    with open(boxer_csv(root), newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            d = _boxer_folio_to_dialect(row[0], pages)
            for cell in row[3:]:
                cell = cell.strip()
                if not cell or cell == "$":
                    continue
                toks = [t.strip() for t in cell.split(",") if t.strip()]
                if toks:
                    out[d].append(toks)
    return out


def boxer_presentation(
    dialect: str, n_symbols: int = 16, root: Path | None = None, tokens=None
) -> Presentation:
    """Top-``n_symbols`` glyph types → symbol ids; tokens containing any
    other glyph (incl. ``?``) dropped. Frequency ranking is over the WHOLE
    manuscript so both dialects share one alphabet."""
    all_tokens = tokens if tokens is not None else boxer_tokens(root)
    counts: Counter = Counter()
    for d in all_tokens.values():
        for t in d:
            counts.update(t)
    counts.pop("?", None)
    alphabet = [g for g, _ in counts.most_common(n_symbols)]
    idx = {g: i for i, g in enumerate(alphabet)}
    kept, starts, n_drop, n_glyph_drop = [], [], 0, 0
    n_glyphs = 0
    pos = 0
    for t in all_tokens[dialect]:
        n_glyphs += len(t)
        if all(g in idx for g in t):
            starts.append(pos)
            kept.extend(idx[g] for g in t)
            pos += len(t)
        else:
            n_drop += 1
            n_glyph_drop += len(t)
    return Presentation(
        "boxer",
        dialect,
        "boxer",
        np.asarray(kept, dtype=np.int64),
        alphabet,
        token_starts=np.asarray(starts, dtype=np.int64),
        coverage={
            "n_tokens": len(all_tokens[dialect]),
            "n_kept_tokens": len(all_tokens[dialect]) - n_drop,
            "token_fraction": (len(all_tokens[dialect]) - n_drop)
            / max(len(all_tokens[dialect]), 1),
            "n_glyphs": n_glyphs,
            "n_kept_glyphs": n_glyphs - n_glyph_drop,
            "covered_fraction": (n_glyphs - n_glyph_drop) / max(n_glyphs, 1),
            "glyph_share_of_alphabet": sum(counts[g] for g in alphabet)
            / max(sum(counts.values()), 1),
        },
    )


def write_presentations(out_dir: Path, parser, root: Path | None = None) -> dict:
    """Persist every presentation as JSON (streams + coverage) — the
    reproducible input record of the VMS run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    btok = boxer_tokens(root)
    for tr in TRANSCRIPTIONS:
        for d in DIALECTS:
            for pres in (
                eva_presentation(tr, d, root),
                words_presentation(tr, d, parser, root),
            ):
                rec = pres.as_dict()
                rec["name"] = f"{tr}/{d}"
                rec["symbols"] = pres.symbols.tolist()
                rec["tokens"] = pres.tokens
                (out_dir / f"{tr}_{d}_{pres.kind}.json").write_text(json.dumps(rec))
                summary[f"{tr}/{d}/{pres.kind}"] = pres.as_dict()
    for d in DIALECTS:
        for n in (16, 20):
            pres = boxer_presentation(d, n_symbols=n, root=root, tokens=btok)
            rec = pres.as_dict()
            rec["name"] = f"boxer{n}/{d}"
            rec["symbols"] = pres.symbols.tolist()
            rec["token_starts"] = pres.token_starts.tolist()
            (out_dir / f"boxer{n}_{d}_boxer.json").write_text(json.dumps(rec))
            summary[f"boxer{n}/{d}/boxer"] = pres.as_dict()
    (out_dir / "presentations.json").write_text(json.dumps(summary, indent=2))
    return summary
