"""Burst front-loading: does a rare type's burst carry a time arrow? (2026-09-03)

For every type with k >= 3 occurrences inside one segment (a manuscript sheet read
a-r, a-v, b-r, b-v, where the direction is known; a window of sheet length in known
prose), two direction statistics that are symmetric under reversal when there is
no arrow:
  first-gap < last-gap   fraction of types whose first inter-occurrence gap is
                         shorter than the last (front-loaded burst -> > 0.5)
  skew                   (mean position - midpoint of first/last) / span, per type;
                         front-loaded -> negative
Reported with the sign-test / t-test against the symmetric null, by k class.
Units: words; glyph n-grams n 5-8 (spaces stripped, as in the pipeline).
Outputs DATA_ROOT/analysis/doubleton_gaps/burst_frontloading.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from doubleton_gaps import DATA_ROOT, OUT, load_csv_text, load_raw_text, load_vms  # noqa: E402
from glyph_ngram_leaf_test import page_symbols  # noqa: E402
from order_optimize import build_sheets  # noqa: E402

KMAX = 10


def burst_stats(segments):
    """segments: list of lists of items. Returns per-type (k, first_gap, last_gap, skew)."""
    out = []
    for items in segments:
        pos = defaultdict(list)
        for i, w in enumerate(items):
            pos[w].append(i)
        for p in pos.values():
            k = len(p)
            if 3 <= k <= KMAX:
                span = p[-1] - p[0]
                if span == 0:
                    continue
                out.append((k, p[1] - p[0], p[-1] - p[-2], (np.mean(p) - (p[0] + p[-1]) / 2) / span))
    return np.array(out, dtype=float)


def summarize(arr):
    if len(arr) == 0:
        return {"n": 0}
    k, g1, gl, sk = arr.T
    res = {}
    for name, sel in (("all", np.ones(len(k), bool)), ("k3", k == 3), ("k4", k == 4), ("k5+", k >= 5)):
        if sel.sum() < 5:
            continue
        a, b, s = g1[sel], gl[sel], sk[sel]
        lt, gt = int((a < b).sum()), int((a > b).sum())
        frac = lt / max(lt + gt, 1)
        p_sign = float(stats.binomtest(lt, lt + gt, 0.5).pvalue) if lt + gt > 0 else 1.0
        t = stats.ttest_1samp(s, 0.0)
        res[name] = {"n": int(sel.sum()), "frac_first_lt_last": frac, "p_sign": p_sign, "skew_mean": float(s.mean()), "skew_se": float(s.std(ddof=1) / np.sqrt(len(s))), "p_skew": float(t.pvalue)}
    return res


def fmt(name, r):
    a = r.get("all", {})
    if not a:
        return f"  {name:34s} n<5"
    line = f"  {name:34s} n {a['n']:5d} | first<last {a['frac_first_lt_last']:.3f} (p {a['p_sign']:.3f}) | skew {a['skew_mean']:+.4f} ± {a['skew_se']:.4f} (p {a['p_skew']:.3f})"
    for kk in ("k3", "k4", "k5+"):
        if kk in r:
            line += f" | {kk}: {r[kk]['frac_first_lt_last']:.2f}/{r[kk]['skew_mean']:+.3f} (n {r[kk]['n']})"
    return line


def ngrams(sym_segments, n):
    return [["".join(s[i:i + n]) for i in range(len(s) - n + 1)] for s in sym_segments]


def main():
    res = {"corpus": {}, "vms": {}}
    L = 1500
    cc = DATA_ROOT / "external/voynich-attack/corpora/latin/CorpusCorporum/auctores_scientiarum_varii"
    dta = DATA_ROOT / "external/voynich-attack/corpora/german/DTA"
    texts = {
        "la_isidorus": load_csv_text(cc / "isidorus_hispalensis/etymologiae/etymologiae.csv"),
        "la_seneca_nq": load_csv_text(cc / "seneca/naturales_quaestiones/naturales_quaestiones.csv"),
        "de_bullinger": load_csv_text(dta / "1558_bullinger_haussbuoch/1558_bullinger_haussbuoch.csv"),
        "it_decameron": load_raw_text(DATA_ROOT / "raw/italian/boccaccio_decameron.txt"),
        "it_commedia": load_raw_text(DATA_ROOT / "raw/italian/dante_divina_commedia.txt"),
        "it_orlando_furioso": load_raw_text(DATA_ROOT / "raw/italian/ariosto_orlando_furioso.txt"),
    }
    print(f"== known texts, segments of {L} tokens (first 60 000 tokens)", flush=True)
    pooled = {"words": [], "n5": [], "n6": [], "n7": [], "n8": []}
    for name, words in texts.items():
        words = words[:60000]
        segs = [words[i:i + L] for i in range(0, len(words) - L + 1, L)]
        symsegs = [list("".join(s)) for s in segs]
        r = {}
        for unit, seg in (("words", segs), *((f"n{n}", ngrams(symsegs, n)) for n in (5, 6, 7, 8))):
            arr = burst_stats(seg); pooled[unit].append(arr)
            r[unit] = summarize(arr)
            print(fmt(f"{name} {unit}", r[unit]), flush=True)
        res["corpus"][name] = r
    res["corpus"]["pooled"] = {}
    for unit, arrs in pooled.items():
        r = summarize(np.concatenate(arrs)); res["corpus"]["pooled"][unit] = r
        print(fmt(f"POOLED prose {unit}", r), flush=True)
    # manuscript: sheets (direction known within a sheet); also pages
    for tr, fname in (("IT2a", "IT2a-n.txt"), ("RF1b", "RF1b-e.txt")):
        toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / fname)
        units = build_sheets(pages)
        page_words = [[] for _ in pages]
        for t in toks:
            page_words[t["page_idx"]].append(t["w"])
        syms, _ = page_symbols(toks, len(pages))
        sheet_w = [sum((page_words[i] for i in u["a"] + u["b"]), []) for u in units if len(u["a"]) + len(u["b"]) >= 3]
        sheet_s = [sum((syms[i] for i in u["a"] + u["b"]), []) for u in units if len(u["a"]) + len(u["b"]) >= 3]
        # by language
        lang_of_unit = [u["lang"] for u in units if len(u["a"]) + len(u["b"]) >= 3]
        res["vms"][tr] = {}
        print(f"== VMS {tr}: {len(sheet_w)} sheets with >= 3 pages", flush=True)
        for scope, sel in (("all sheets", [True] * len(sheet_w)), ("Currier A sheets", [l == "A" for l in lang_of_unit]), ("Currier B sheets", [l == "B" for l in lang_of_unit])):
            sw = [s for s, k in zip(sheet_w, sel) if k]; ss = [s for s, k in zip(sheet_s, sel) if k]
            r = {}
            for unit, seg in (("words", sw), *((f"n{n}", ngrams(ss, n)) for n in (5, 6, 7, 8))):
                r[unit] = summarize(burst_stats(seg))
                print(fmt(f"{tr} {scope} {unit}", r[unit]), flush=True)
            res["vms"][tr][scope] = r
        r = {}
        for unit, seg in (("words", page_words), *((f"n{n}", ngrams(syms, n)) for n in (5, 6, 7, 8))):
            r[unit] = summarize(burst_stats(seg))
            print(fmt(f"{tr} pages {unit}", r[unit]), flush=True)
        res["vms"][tr]["pages"] = r
    (OUT / "burst_frontloading.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
