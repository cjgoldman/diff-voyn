"""Control for the Currier-A sheet asymmetry seen in burst_frontloading.py (2026-09-03).

(1) Mean tokens per page by position within the sheet (a-r, a-v, b-r, b-v), by language.
(2) Null: the four pages of every sheet are put in a random order (independently per
    sheet, 200 draws) and the first<last fraction / skew recomputed - keeps every
    page-level effect (within-page bursts, page length) and destroys only the
    a-r,a-v,b-r,b-v reading order.  Reports the real order's position in that null.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from burst_frontloading import burst_stats, ngrams, summarize  # noqa: E402
from doubleton_gaps import DATA_ROOT, load_vms  # noqa: E402
from glyph_ngram_leaf_test import page_symbols  # noqa: E402
from order_optimize import build_sheets  # noqa: E402


def main():
    rng = np.random.default_rng(21)
    n_shuf = 200
    for tr, fname in (("IT2a", "IT2a-n.txt"), ("RF1b", "RF1b-e.txt")):
        toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / fname)
        units = [u for u in build_sheets(pages) if len(u["a"]) + len(u["b"]) == 4]
        page_words = [[] for _ in pages]
        for t in toks:
            page_words[t["page_idx"]].append(t["w"])
        syms, _ = page_symbols(toks, len(pages))
        print(f"== {tr}: {len(units)} four-page sheets", flush=True)
        for lang in ("A", "B"):
            us = [u for u in units if u["lang"] == lang]
            pl = np.array([[len(page_words[i]) for i in u["a"] + u["b"]] for u in us])
            print(f"  Currier {lang}: {len(us)} sheets; mean tokens by position a-r/a-v/b-r/b-v = {pl.mean(axis=0).round(0).tolist()}; median {np.median(pl, axis=0).round(0).tolist()}", flush=True)
            for unit, n in (("words", None), ("n6", 6), ("n7", 7), ("n8", 8)):
                def segs(orders):
                    out = []
                    for u, o in zip(us, orders):
                        pidx = [(u["a"] + u["b"])[j] for j in o]
                        out.append(sum((page_words[i] for i in pidx), []) if n is None else sum((syms[i] for i in pidx), []))
                    return out if n is None else ngrams(out, n)
                real = summarize(burst_stats(segs([[0, 1, 2, 3]] * len(us))))["all"]
                fr, sk = [], []
                for _ in range(n_shuf):
                    s = summarize(burst_stats(segs([rng.permutation(4) for _ in us])))["all"]
                    fr.append(s["frac_first_lt_last"]); sk.append(s["skew_mean"])
                fr, sk = np.array(fr), np.array(sk)
                p_fr = ((fr <= real["frac_first_lt_last"]).sum() + 1) / (n_shuf + 1)
                p_sk = ((sk >= real["skew_mean"]).sum() + 1) / (n_shuf + 1)
                print(f"    {unit:5s} n {real['n']:5d} | real first<last {real['frac_first_lt_last']:.3f}, null {fr.mean():.3f} ± {fr.std():.3f} (min {fr.min():.3f}) p {p_fr:.3f} | real skew {real['skew_mean']:+.4f}, null {sk.mean():+.4f} ± {sk.std():.4f} (max {sk.max():+.4f}) p {p_sk:.3f}", flush=True)


if __name__ == "__main__":
    main()
