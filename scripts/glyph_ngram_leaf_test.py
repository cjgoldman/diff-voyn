"""Glyph n-gram version of the nested-vs-stacked leaf test (2026-09-02).

Rare character n-grams (2..kmax occurrences) over the space-stripped text of
each page, both EVA transcriptions; leaf-pair categories and within-quire-and-
language null as in doubleton_leaf_affinity.py.  Variant 'cross': only n-grams
that straddle a word boundary (not fragments of a single rare word).  Controls:
known texts, spaces stripped, laid on the IT2a page slots by symbol count,
written nested or stacked.
Outputs DATA_ROOT/analysis/doubleton_gaps/glyph_ngrams.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from doubleton_gaps import DATA_ROOT, OUT, load_csv_text, load_raw_text, load_vms, stacked_bifolia_rank  # noqa: E402
from rare_type_clustering import leaf_test_pooled  # noqa: E402

_SYM = re.compile(r"<\d+>|[a-z]")


def page_symbols(toks, n_pages):
    """Per page: list of symbols and set of boundary positions (index of the first symbol of each word after the first)."""
    syms = [[] for _ in range(n_pages)]
    bounds = [set() for _ in range(n_pages)]
    for t in toks:
        p = t["page_idx"]
        if syms[p]:
            bounds[p].add(len(syms[p]))
        syms[p].extend(_SYM.findall(t["w"]))
    return syms, bounds


def ngram_stream(syms, bounds, n, cross_only):
    words, page_of = [], []
    for p, (s, b) in enumerate(zip(syms, bounds)):
        for i in range(len(s) - n + 1):
            if cross_only and not any(i < j < i + n for j in b):
                continue
            words.append("".join(s[i : i + n]))
            page_of.append(p)
    return words, np.array(page_of)


def fmt(t):
    return f"pairs {t['n_pairs']:6d}  conj {t['conjugate']['obs']:5d}/{t['conjugate']['null_mean']:7.1f} z={t['conjugate']['z']:+.1f}  nested {t['nested_adjacent']['obs']:5d}/{t['nested_adjacent']['null_mean']:7.1f} z={t['nested_adjacent']['z']:+.1f}  conj-nested z={t['conj_minus_nested']['z']:+.1f}"


def main():
    n_shuf = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rng = np.random.default_rng(6)
    res = {"vms": {}, "controls": {}}
    ns = (4, 5, 6, 7, 8)
    for tr, fname in (("IT2a", "IT2a-n.txt"), ("RF1b", "RF1b-e.txt")):
        toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / fname)
        syms, bounds = page_symbols(toks, len(pages))
        res["vms"][tr] = {}
        print(f"== VMS {tr}")
        for n in ns:
            for variant in ("all", "cross"):
                words, page_of = ngram_stream(syms, bounds, n, variant == "cross")
                for kmax in (2, 10):
                    t = leaf_test_pooled(words, page_of, pages, kmax, n_shuf, rng)
                    res["vms"][tr][f"n{n}_{variant}_k2to{kmax}"] = t
                    print(f"  n={n} {variant:5s} k2to{kmax:2d}: {fmt(t)}")
    # controls on IT2a page slots (symbol counts)
    toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / "IT2a-n.txt")
    syms, _ = page_symbols(toks, len(pages))
    page_len = np.array([len(s) for s in syms])
    cc = DATA_ROOT / "external/voynich-attack/corpora/latin/CorpusCorporum/auctores_scientiarum_varii"
    dta = DATA_ROOT / "external/voynich-attack/corpora/german/DTA"
    texts = {
        "la_isidorus_etym": load_csv_text(cc / "isidorus_hispalensis/etymologiae/etymologiae.csv"),
        "la_seneca_nq": load_csv_text(cc / "seneca/naturales_quaestiones/naturales_quaestiones.csv"),
        "de_bullinger": load_csv_text(dta / "1558_bullinger_haussbuoch/1558_bullinger_haussbuoch.csv"),
        "it_decameron": load_raw_text(DATA_ROOT / "raw/italian/boccaccio_decameron.txt"),
        "it_commedia": load_raw_text(DATA_ROOT / "raw/italian/dante_divina_commedia.txt"),
        "it_orlando_furioso": load_raw_text(DATA_ROOT / "raw/italian/ariosto_orlando_furioso.txt"),
    }
    for name, src_words in texts.items():
        # symbol stream with word boundaries
        chars, bset = [], set()
        for w in src_words:
            if chars:
                bset.add(len(chars))
            chars.extend(w)
            if len(chars) > page_len.sum() + 10:
                break
        for mode in ("nested", "stacked"):
            rank = np.arange(len(pages)) if mode == "nested" else stacked_bifolia_rank(pages)
            order = np.argsort(rank)
            acc = 0
            psyms = [None] * len(pages)
            pb = [None] * len(pages)
            for slot in order:
                psyms[slot] = chars[acc : acc + page_len[slot]]
                pb[slot] = {j - acc for j in bset if acc < j < acc + page_len[slot]}
                acc += page_len[slot]
            r = {}
            line = []
            for n in (5, 6, 7):
                for variant in ("all", "cross"):
                    words, page_of = ngram_stream(psyms, pb, n, variant == "cross")
                    t = leaf_test_pooled(words, page_of, pages, 10, n_shuf, rng)
                    r[f"n{n}_{variant}_k2to10"] = t
                    line.append(f"n{n} {variant} z={t['conj_minus_nested']['z']:+.1f}")
            res["controls"][f"{name}/{mode}"] = r
            print(f"== control {name} {mode}: " + "  ".join(line))
    (OUT / "glyph_ngrams.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
