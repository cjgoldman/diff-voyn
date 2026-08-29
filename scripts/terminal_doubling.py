"""Word-terminal doubling statistics — plaintext languages vs the Voynich MS.

Side-quest follow-up to ``scripts/doubling_rate.py`` (see
``docs/vms_doubling_rate.md`` §7).  Boxer's paper argues from an *inventory* of
paragraph-terminal doubled/tripled Voynich tokens and from cross-word
triple-letter rates (his Table 5); neither is the statistic the argument needs.
This script measures the load-bearing quantities on both sides:

Plaintext (per language, per word token)
    * word-FINAL double-letter rate (last two letters equal) per 1000 words,
      separated from the word-INITIAL double rate, with the distribution over
      which letter doubles and its diversity (modes at 90 % coverage, entropy);
    * word-final doubled *bigram* rate (``...xyxy``);
    * the same word-final double rate restricted to sentence-final and
      paragraph-final words (terminal position is not distributionally neutral);
    * paragraph-final *triple* letters (last three letters of the paragraph,
      whitespace stripped, equal — the only way a paragraph can "end in a triple");
    * cross-word triples per 10 000 letters decomposed into the two mechanisms
      Boxer's Table 5 conflates: ``..xx|x..`` (word ends in a double) vs
      ``..x|xx..`` (next word begins with a double).

Voynich (per transliteration × Currier language, tokens = EVA words)
    * number of paragraphs; paragraphs whose last two / last three tokens are
      identical (with uncertain-token handling), Wilson CIs;
    * the same for paragraph-initial position and for line-final position;
    * the running adjacent-pair rate for comparison and a binomial test of
      terminal enrichment;
    * the inventory of paragraph-terminal doubled tokens with counts.

Under Boxer's model (VMS word = homophone of one plaintext letter, same word
reused for a doubled letter with probability s) the VMS paragraph-terminal
token-doubling rate should be s × (plaintext paragraph-final word-final-double
rate), with s ≈ 0.2–0.35 from the running-text comparison in doubling_report.

Usage::

    uv run python scripts/terminal_doubling.py [--out DIR]

Writes ``terminal_doubling.md`` and ``terminal_doubling.json`` under
``DATA_ROOT/analysis/doubling``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doubling_rate import (
    _ALT_READING_RE,
    _EXT_EVA_RE,
    _LOCUS_RE,
    _MARKUP_RE,
    _PAGE_RE,
    _VAR_RE,
    BOXER_CSV,
    DATA_ROOT,
    IVTFF_FILES,
    parse_boxer_csv,
    rate_ci_wilson,
)

csv.field_size_limit(sys.maxsize)

EXT = DATA_ROOT / "external/voynich-attack/corpora"
ITALIAN_RAW = DATA_ROOT / "raw/italian"

# language -> list of (glob root, csv glob).  Schema A: Boxer's corpus_build
# CSVs (doc_id, block_type, para_id, sent_id, is_para_final, page_n,
# textstring_orig, ...).  Schema B: legacy single-column ``textstring`` rows
# (one sentence/line per row, no paragraph structure).
CORPORA = {
    "latin": [EXT / "latin/CorpusCorporum"],
    "german": [EXT / "german/DTA"],
    "dutch": [EXT / "dutch/DBNL"],
    "english": [EXT / "english/EEBO"],
    "french": [EXT / "french/ProjectGutenberg", EXT / "french/Wikisource"],
    "spanish": [EXT / "spanish/quixote"],
}
SPANISH_FILE = "quixote_parsed_edited.csv"

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Roman numerals (chapter numbers, dates) end in -ii/-xx/-vv and dominate the
# paragraph-final position of Latin/Italian editions; they are not words.
_ROMAN_RE = re.compile(r"^[ivxlcdm]+$")


# --------------------------------------------------------------------------- #
# Word normalisation
# --------------------------------------------------------------------------- #
def norm_word(w: str, keep_eszett: bool = False) -> str:
    """Lowercase, fold ligatures (ß→ss unless keep_eszett), strip diacritics,
    keep a–z only.  Mirrors the project normaliser's letter policy closely
    enough for a doubling count; ``keep_eszett`` maps ß to a private letter
    ``ß`` so German ``-ss`` vs ``-ß`` endings can be told apart."""
    out = []
    for ch in w:
        if keep_eszett and ch in "ßẞ":
            out.append("ß")
            continue
        for c in unicodedata.normalize("NFKD", ch.casefold()):
            if unicodedata.combining(c):
                continue
            if "a" <= c <= "z":
                out.append(c)
    return "".join(out)


def words_of(text: str, keep_eszett: bool) -> list[str]:
    return [
        w
        for w in (norm_word(t, keep_eszett) for t in text.split())
        if w and not _ROMAN_RE.match(w)
    ]


# --------------------------------------------------------------------------- #
# Plaintext corpus loading -> list of paragraphs, each a list of sentences,
# each a list of words
# --------------------------------------------------------------------------- #
def load_schema_a(path: Path, keep_eszett: bool) -> list[list[list[str]]]:
    paras: list[list[list[str]]] = []
    cur: list[list[str]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        rd = csv.DictReader(fh)
        if "textstring_orig" not in (rd.fieldnames or []):
            return []
        last_key = None
        for row in rd:
            if row.get("block_type", "body") != "body":
                continue
            key = (row.get("doc_id"), row.get("para_id"))
            if last_key is not None and key != last_key and cur:
                paras.append(cur)
                cur = []
            last_key = key
            ws = words_of(row["textstring_orig"], keep_eszett)
            if ws:
                cur.append(ws)
            if row.get("is_para_final", "").strip().lower() == "true" and cur:
                paras.append(cur)
                cur = []
                last_key = None
    if cur:
        paras.append(cur)
    return paras


def load_schema_b(path: Path, keep_eszett: bool) -> list[list[list[str]]]:
    """Legacy ``textstring`` rows: each row is a sentence; no paragraphs."""
    paras = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        rd = csv.DictReader(fh)
        col = "textstring" if "textstring" in (rd.fieldnames or []) else None
        if col is None:
            return []
        for row in rd:
            for sent in _SENT_SPLIT_RE.split(row[col]):
                ws = words_of(sent, keep_eszett)
                if ws:
                    paras.append([ws])
    return paras


def load_italian(keep_eszett: bool) -> dict[str, list[list[list[str]]]]:
    docs = {}
    for f in sorted(ITALIAN_RAW.glob("*.txt")):
        paras = []
        for block in re.split(
            r"\n\s*\n", f.read_text(encoding="utf-8", errors="replace")
        ):
            block = " ".join(block.split())
            if not block:
                continue
            sents = [
                ws
                for ws in (
                    words_of(s, keep_eszett) for s in _SENT_SPLIT_RE.split(block)
                )
                if ws
            ]
            if sents:
                paras.append(sents)
        docs[f.stem] = paras
    return docs


def load_language(lang: str, keep_eszett: bool) -> dict[str, list[list[list[str]]]]:
    if lang == "italian":
        return load_italian(keep_eszett)
    docs = {}
    for root in CORPORA[lang]:
        for f in sorted(root.rglob("*.csv")):
            if lang == "spanish" and f.name != SPANISH_FILE:
                continue
            if (
                "firstpass" in f.name
                or f.name.endswith("_lat0.csv")
                or f.name.endswith("_lat1.csv")
            ):
                continue
            paras = load_schema_a(f, keep_eszett) or load_schema_b(f, keep_eszett)
            if paras:
                docs[f.stem] = paras
    return docs


# --------------------------------------------------------------------------- #
# Plaintext statistics
# --------------------------------------------------------------------------- #
def ends_double(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2]


def starts_double(w: str) -> bool:
    return len(w) >= 2 and w[0] == w[1]


def ends_double_bigram(w: str) -> bool:
    return len(w) >= 4 and w[-4:-2] == w[-2:] and w[-1] != w[-2]


def ends_triple(s: str) -> bool:
    return len(s) >= 3 and s[-1] == s[-2] == s[-3]


def modes_at_coverage(counter: Counter, cov: float = 0.9) -> int:
    tot = sum(counter.values())
    if tot == 0:
        return 0
    acc, k = 0, 0
    for _, c in counter.most_common():
        acc += c
        k += 1
        if acc >= cov * tot:
            return k
    return k


def entropy_bits(counter: Counter) -> float:
    tot = sum(counter.values())
    return (
        -sum(c / tot * math.log2(c / tot) for c in counter.values() if c)
        if tot
        else 0.0
    )


def per_mille(k: int, n: int):
    lo, hi = rate_ci_wilson(k, n)
    return {
        "k": k,
        "n": n,
        "per_1000": 1000 * k / n if n else float("nan"),
        "ci95": [1000 * lo, 1000 * hi],
    }


def plaintext_stats(docs: dict[str, list[list[list[str]]]]) -> dict:
    n_words = n_final = n_initial = n_final_bigram = 0
    final_modes: Counter = Counter()
    initial_modes: Counter = Counter()
    final_bigram_modes: Counter = Counter()
    type_counter: Counter = Counter()
    # positional
    pos = {
        "all": [0, 0],
        "sentence_final": [0, 0],
        "sentence_nonfinal": [0, 0],
        "paragraph_final": [0, 0],
    }
    pos_modes = {k: Counter() for k in pos}
    # paragraph-level
    n_paras = 0
    para_final_triple = 0
    para_final_double_letter = 0  # last two letters of the paragraph string equal
    # cross-word triples
    n_letters = 0
    letter_pairs = letter_doubles = (
        0  # running adjacent-letter doubling, whitespace stripped
    )
    xw_triple_final_double = 0  # ..xx | x..
    xw_triple_initial_double = 0  # ..x | xx..
    xw_triple_both = 0  # ..xx | xx..  (quadruple; counted once)
    xw_modes: Counter = Counter()
    n_boundaries = 0

    for paras in docs.values():
        for para in paras:
            n_paras += 1
            flat: list[str] = []
            for si, sent in enumerate(para):
                for wi, w in enumerate(sent):
                    n_words += 1
                    type_counter[w] += 1
                    fd = ends_double(w)
                    if fd:
                        n_final += 1
                        final_modes[w[-1]] += 1
                    if starts_double(w):
                        n_initial += 1
                        initial_modes[w[0]] += 1
                    if ends_double_bigram(w):
                        n_final_bigram += 1
                        final_bigram_modes[w[-2:]] += 1
                    pos["all"][1] += 1
                    pos["all"][0] += fd
                    is_sf = wi == len(sent) - 1
                    key = "sentence_final" if is_sf else "sentence_nonfinal"
                    pos[key][1] += 1
                    pos[key][0] += fd
                    if fd:
                        pos_modes[key][w[-1]] += 1
                        pos_modes["all"][w[-1]] += 1
                    if is_sf and si == len(para) - 1:
                        pos["paragraph_final"][1] += 1
                        pos["paragraph_final"][0] += fd
                        if fd:
                            pos_modes["paragraph_final"][w[-1]] += 1
                flat.extend(sent)
            s = "".join(flat)
            n_letters += len(s)
            letter_pairs += max(0, len(s) - 1)
            letter_doubles += sum(1 for a, b in pairwise(s) if a == b)
            if ends_triple(s):
                para_final_triple += 1
            if len(s) >= 2 and s[-1] == s[-2]:
                para_final_double_letter += 1
            for a, b in pairwise(flat):
                n_boundaries += 1
                ad, bd = ends_double(a), starts_double(b)
                if a[-1] != b[0]:
                    continue
                if ad and bd:
                    xw_triple_both += 1
                    xw_modes[a[-1]] += 1
                elif ad:
                    xw_triple_final_double += 1
                    xw_modes[a[-1]] += 1
                elif bd:
                    xw_triple_initial_double += 1
                    xw_modes[a[-1]] += 1

    n_types = len(type_counter)
    types_final = sum(1 for t in type_counter if ends_double(t))
    xw_total = xw_triple_final_double + xw_triple_initial_double + xw_triple_both
    return {
        "n_docs": len(docs),
        "n_words": n_words,
        "n_types": n_types,
        "n_letters": n_letters,
        "n_paragraphs": n_paras,
        "word_final_double": per_mille(n_final, n_words),
        "word_initial_double": per_mille(n_initial, n_words),
        "word_final_double_bigram": per_mille(n_final_bigram, n_words),
        "type_final_double": per_mille(types_final, n_types),
        "final_modes": dict(final_modes.most_common()),
        "final_modes_at_90": modes_at_coverage(final_modes),
        "final_modes_entropy_bits": entropy_bits(final_modes),
        "initial_modes": dict(initial_modes.most_common()),
        "final_bigram_modes": dict(final_bigram_modes.most_common(10)),
        "running_letter_double": per_mille(letter_doubles, letter_pairs),
        "position": {k: per_mille(v[0], v[1]) for k, v in pos.items()},
        "position_modes": {k: dict(c.most_common(8)) for k, c in pos_modes.items()},
        "paragraph_final_triple": per_mille(para_final_triple, n_paras),
        "paragraph_final_double_letter": per_mille(para_final_double_letter, n_paras),
        "cross_word_triple_per_10k_letters": {
            "total": 1e4 * xw_total / n_letters,
            "final_double_then_same": 1e4 * xw_triple_final_double / n_letters,
            "single_then_initial_double": 1e4 * xw_triple_initial_double / n_letters,
            "double_double": 1e4 * xw_triple_both / n_letters,
            "share_word_final_mechanism": (
                (xw_triple_final_double + xw_triple_both) / xw_total
                if xw_total
                else float("nan")
            ),
            "modes": dict(xw_modes.most_common(6)),
            "n_boundaries": n_boundaries,
        },
    }


# --------------------------------------------------------------------------- #
# Voynich side
# --------------------------------------------------------------------------- #
def parse_ivtff_paragraphs(
    path: Path, end_marks: set[tuple[str, int]] | None = None
) -> list[dict]:
    """Paragraph records for locus type P: page, lang, hand, lines (list of
    token lists), plus whether the paragraph end was explicitly marked.

    ``end_marks``: extra (page, line) loci to treat as paragraph ends — used to
    transfer IT2a's ``<$>`` marks to RF1b, which has none (locus numbering is
    shared between the two files)."""
    end_marks = end_marks or set()
    paras: list[dict] = []
    cur: dict | None = None
    page_vars: dict[str, str] = {}
    page = None
    found_ends: set[tuple[str, int]] = set()

    def close(explicit: bool):
        nonlocal cur
        if cur and cur["lines"]:
            cur["explicit_end"] = explicit
            paras.append(cur)
        cur = None

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("#") or not raw.strip():
            continue
        m = _PAGE_RE.match(raw)
        if m and not _LOCUS_RE.match(raw):
            close(False)
            page = m.group(1)
            page_vars = dict(_VAR_RE.findall(m.group(3) or ""))
            continue
        lm = _LOCUS_RE.match(raw)
        if not lm or page is None:
            continue
        flags, text = lm.group(3), lm.group(4)
        ltype = next((c for c in flags if c.isalpha()), "?")
        if ltype != "P":
            close(False)
            continue
        first = flags[:1]
        if first in "@*":
            close(False)
        if cur is None:
            cur = {
                "page": page,
                "lang": page_vars.get("L"),
                "hand": page_vars.get("H"),
                "lines": [],
            }
        end_marked = (
            "<$>" in text or first == "=" or (page, int(lm.group(2))) in end_marks
        )
        if "<$>" in text or first == "=":
            found_ends.add((page, int(lm.group(2))))
        text = _ALT_READING_RE.sub(r"\1", text)
        text = _MARKUP_RE.sub("", text)
        text = text.replace("{", "").replace("}", "")
        toks = []
        for w in re.split(r"[.,\s]+", text):
            if not w:
                continue
            toks.append((w, ("?" in w) or bool(_EXT_EVA_RE.search(w))))
        if toks:
            cur["lines"].append(toks)
        if end_marked:
            close(True)
    close(False)
    parse_ivtff_paragraphs.last_end_marks = found_ends  # type: ignore[attr-defined]
    return paras


def boxer_paragraphs(records: list[dict]) -> list[dict]:
    """Group Boxer's csv rows into (folio, par) paragraphs; lang from IVTFF."""
    lang_by_page = {}
    for p in parse_ivtff_paragraphs(IVTFF_FILES["takahashi_IT2a"]):
        lang_by_page[p["page"]] = p["lang"]
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        groups[(r["page"], r["par"])].append(r)
    paras = []
    for (page, par), rows in groups.items():
        rows.sort(key=lambda r: r["line"])
        lines = [r["tokens"] for r in rows if r["tokens"]]
        if lines:
            paras.append(
                {
                    "page": page,
                    "lang": lang_by_page.get(page),
                    "hand": None,
                    "lines": lines,
                    "explicit_end": True,
                }
            )
    return paras


def vms_stats(paras: list[dict], *, drop_uncertain: bool) -> dict:
    def clean(toks):
        return [t for t, u in toks if not (drop_uncertain and u)]

    n_par = 0
    term_double = term_triple = init_double = 0
    term_double_tokens: Counter = Counter()
    term_triple_tokens: Counter = Counter()
    line_final_pairs = line_final_double = 0
    run_pairs = run_double = 0
    term_pairs_by_page: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in paras:
        lines = [clean(l) for l in p["lines"]]
        lines = [l for l in lines if l]
        if not lines:
            continue
        flat = [t for l in lines for t in l]
        for a, b in pairwise(flat):
            run_pairs += 1
            run_double += a == b
        for l in lines[:-1]:  # line-final pairs, excluding the paragraph-final line
            if len(l) >= 2:
                line_final_pairs += 1
                line_final_double += l[-1] == l[-2]
        if len(flat) < 2:
            continue
        n_par += 1
        d = flat[-1] == flat[-2]
        term_double += d
        term_pairs_by_page[p["page"]][1] += 1
        term_pairs_by_page[p["page"]][0] += d
        if d:
            term_double_tokens[flat[-1]] += 1
        if len(flat) >= 3 and d and flat[-2] == flat[-3]:
            term_triple += 1
            term_triple_tokens[flat[-1]] += 1
        init_double += flat[0] == flat[1]
    run_rate = run_double / run_pairs if run_pairs else float("nan")
    # binomial tail: P(X >= term_double | n_par, run_rate)
    p_enrich = (
        sum(
            math.comb(n_par, k) * run_rate**k * (1 - run_rate) ** (n_par - k)
            for k in range(term_double, n_par + 1)
        )
        if n_par
        else float("nan")
    )
    return {
        "n_paragraphs": n_par,
        "paragraph_terminal_double": per_mille(term_double, n_par),
        "paragraph_terminal_triple": per_mille(term_triple, n_par),
        "paragraph_initial_double": per_mille(init_double, n_par),
        "line_final_double": per_mille(line_final_double, line_final_pairs),
        "running_pair_double": per_mille(run_double, run_pairs),
        "terminal_enrichment_ratio": (
            (term_double / n_par) / run_rate if n_par and run_rate else float("nan")
        ),
        "terminal_binomial_p": p_enrich,
        "terminal_double_tokens": dict(term_double_tokens.most_common()),
        "terminal_triple_tokens": dict(term_triple_tokens.most_common()),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def f1(x):
    return "—" if math.isnan(x) else f"{x:.2f}"


def pm(d):
    return f"{d['per_1000']:.1f} ({d['ci95'][0]:.1f}–{d['ci95'][1]:.1f})"


def write_report(pt: dict, pt_eszett: dict | None, vms: dict, out: Path):
    L = []
    L.append("# Word-terminal doubling — plaintext languages vs the Voynich MS\n")
    L.append(
        "Generated by `scripts/terminal_doubling.py`. Rates per 1000 with Wilson 95 % CI. Words = whitespace tokens, lowercased, diacritics stripped, a–z only (ß→ss unless noted).\n"
    )

    L.append(
        "## A. Plaintext: word-final vs word-initial double letters (all word tokens)\n"
    )
    L.append(
        "| language | docs | words | word-FINAL double /1000 | word-INITIAL double /1000 | final doubled bigram /1000 | types ending in double /1000 types | final modes @90 % | H(final mode) bits | top final modes |"
    )
    L.append("|---|---:|---:|---|---|---|---|---:|---:|---|")
    for lang, s in pt.items():
        top = ", ".join(
            f"-{k}{k} {100*v/max(1,sum(s['final_modes'].values())):.0f}%"
            for k, v in list(s["final_modes"].items())[:6]
        )
        L.append(
            f"| {lang} | {s['n_docs']} | {s['n_words']:,} | {pm(s['word_final_double'])} | {pm(s['word_initial_double'])} | "
            f"{pm(s['word_final_double_bigram'])} | {pm(s['type_final_double'])} | {s['final_modes_at_90']} | {s['final_modes_entropy_bits']:.2f} | {top} |"
        )
    if pt_eszett:
        L.append(
            "\nGerman with ß kept as its own letter (so `-ß` endings are *not* counted as `-ss`):\n"
        )
        for lang, s in pt_eszett.items():
            top = ", ".join(
                f"-{k}{k} {100*v/max(1,sum(s['final_modes'].values())):.0f}%"
                for k, v in list(s["final_modes"].items())[:6]
            )
            L.append(
                f"- {lang}: word-final double {pm(s['word_final_double'])} /1000 words, modes @90 % {s['final_modes_at_90']}, top {top}"
            )

    L.append("\n## B. Plaintext: word-final double rate by position\n")
    L.append(
        "| language | all words | sentence-nonfinal | sentence-final | paragraph-final | paragraphs | para ends in TRIPLE letter /1000 paras | top paragraph-final modes |"
    )
    L.append("|---|---|---|---|---|---:|---|---|")
    for lang, s in pt.items():
        P = s["position"]
        top = ", ".join(
            f"-{k}{k}×{v}" for k, v in s["position_modes"]["paragraph_final"].items()
        )
        L.append(
            f"| {lang} | {pm(P['all'])} | {pm(P['sentence_nonfinal'])} | {pm(P['sentence_final'])} | {pm(P['paragraph_final'])} | "
            f"{s['n_paragraphs']:,} | {pm(s['paragraph_final_triple'])} | {top} |"
        )

    L.append(
        "\n## C. Plaintext: cross-word triple letters per 10 000 letters, decomposed (Boxer Table 5 mechanism split)\n"
    )
    L.append(
        "| language | letters | total | `..xx\\|x..` (word ends in double) | `..x\\|xx..` (next word starts with double) | `..xx\\|xx..` | share via word-final double | modes |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for lang, s in pt.items():
        X = s["cross_word_triple_per_10k_letters"]
        modes = ", ".join(f"{k}×{v}" for k, v in X["modes"].items())
        L.append(
            f"| {lang} | {s['n_letters']:,} | {X['total']:.2f} | {X['final_double_then_same']:.2f} | {X['single_then_initial_double']:.2f} | "
            f"{X['double_double']:.2f} | {f1(X['share_word_final_mechanism'])} | {modes} |"
        )

    L.append("\n## D. Voynich: token doubling by position (paragraph text only)\n")
    L.append(
        "| source | Currier | uncertain | paragraphs | para-TERMINAL double /1000 | para-terminal TRIPLE /1000 | para-INITIAL double /1000 | line-final double /1000 | running pair double /1000 | terminal/running | binomial p |"
    )
    L.append("|---|---|---|---:|---|---|---|---|---|---:|---:|")
    for key, s in vms.items():
        src, cur, unc = key.split("|")
        L.append(
            f"| {src} | {cur} | {unc} | {s['n_paragraphs']} | {pm(s['paragraph_terminal_double'])} | {pm(s['paragraph_terminal_triple'])} | "
            f"{pm(s['paragraph_initial_double'])} | {pm(s['line_final_double'])} | {pm(s['running_pair_double'])} | "
            f"{f1(s['terminal_enrichment_ratio'])} | {s['terminal_binomial_p']:.3f} |"
        )
    L.append("\nParagraph-terminal doubled tokens (inventory with counts):\n")
    for key, s in vms.items():
        if key.endswith("|all|dropped"):
            inv = ", ".join(f"{t}×{c}" for t, c in s["terminal_double_tokens"].items())
            tri = (
                ", ".join(f"{t}×{c}" for t, c in s["terminal_triple_tokens"].items())
                or "none"
            )
            L.append(f"- {key.split('|')[0]}: doubles: {inv or 'none'}; triples: {tri}")

    L.append("\n## E. Boxer-model consistency: implied homophone-reuse s\n")
    L.append(
        "Under Boxer's model (VMS word = homophone of one plaintext letter; the same word is reused for a doubled letter with probability s) the running VMS pair-doubling rate is s × (plaintext running letter-doubling rate) and the paragraph-terminal token-doubling rate is s × (plaintext paragraph-final word-final-double rate). s is estimated from running text and the terminal rate is then *predicted*; reference VMS = Takahashi IT2a, uncertain dropped.\n"
    )
    ref = vms["takahashi_IT2a|all|dropped"]
    v_run = ref["running_pair_double"]["per_1000"]
    v_term = ref["paragraph_terminal_double"]
    L.append(
        f"Observed VMS: running {v_run:.1f}/1000 pairs; paragraph-terminal {pm(v_term)}/1000 paragraphs (k={v_term['k']}, n={v_term['n']}).\n"
    )
    L.append(
        "| language | plaintext running letter-double /1000 | implied s (running) | plaintext paragraph-final word-double /1000 | predicted VMS terminal /1000 at that s | s needed to hit observed terminal | inside VMS terminal CI? |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---|")
    for lang, s_ in pt.items():
        r_run = s_["running_letter_double"]["per_1000"]
        r_par = s_["position"]["paragraph_final"]["per_1000"]
        s_run = v_run / r_run
        pred = s_run * r_par
        s_need = v_term["per_1000"] / r_par if r_par else float("nan")
        lo, hi = v_term["ci95"]
        L.append(
            f"| {lang} | {r_run:.1f} | {s_run:.2f} | {r_par:.1f} | {pred:.1f} | {s_need:.2f} | {'yes' if lo <= pred <= hi else 'no'} |"
        )
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "analysis/doubling")
    ap.add_argument(
        "--languages",
        nargs="*",
        default=["latin", "italian", "german", "dutch", "french", "english", "spanish"],
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pt = {}
    for lang in args.languages:
        docs = load_language(lang, keep_eszett=False)
        if not docs:
            print(f"[warn] no documents for {lang}", file=sys.stderr)
            continue
        pt[lang] = plaintext_stats(docs)
        print(
            f"{lang}: {pt[lang]['n_docs']} docs, {pt[lang]['n_words']:,} words, final double {pt[lang]['word_final_double']['per_1000']:.1f}/1000",
            file=sys.stderr,
        )
    pt_eszett = {}
    if "german" in pt:
        pt_eszett["german"] = plaintext_stats(load_language("german", keep_eszett=True))

    vms = {}
    sources = {"takahashi_IT2a": parse_ivtff_paragraphs(IVTFF_FILES["takahashi_IT2a"])}
    it2a_ends = parse_ivtff_paragraphs.last_end_marks  # type: ignore[attr-defined]
    sources["reference_RF1b"] = parse_ivtff_paragraphs(
        IVTFF_FILES["reference_RF1b"], end_marks=it2a_ends
    )
    if BOXER_CSV.exists():
        sources["boxer_csv"] = boxer_paragraphs(parse_boxer_csv(BOXER_CSV))
    for name, paras in sources.items():
        for cur in ("all", "A", "B"):
            sel = paras if cur == "all" else [p for p in paras if p["lang"] == cur]
            for unc, drop in (("kept", False), ("dropped", True)):
                vms[f"{name}|{cur}|{unc}"] = vms_stats(sel, drop_uncertain=drop)

    (args.out / "terminal_doubling.json").write_text(
        json.dumps(
            {"plaintext": pt, "plaintext_eszett": pt_eszett, "vms": vms}, indent=1
        ),
        encoding="utf-8",
    )
    write_report(pt, pt_eszett, vms, args.out / "terminal_doubling.md")
    print((args.out / "terminal_doubling.md").read_text())


if __name__ == "__main__":
    main()
