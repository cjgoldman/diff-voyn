"""Empirical tail probabilities for the nested-vs-stacked leaf test (2026-09-03).

Question: if rare material were evenly spread over the pages, how often would the
conj - nested_adjacent excess of docs/doubleton_gaps.md §6-7 arise by chance?
Two nulls:
  perm    - page contents permuted within (quire, lang[, hand]); the §6/§7 null,
            keeps each page's own rare-word load.
  uniform - every occurrence of a rare type lands independently on a page of the
            type's language with probability proportional to page length
            (literal "evenly spread"; Poisson variance added).
Reports one-sided p = (#null >= obs + 1) / (n + 1).
Outputs DATA_ROOT/analysis/doubleton_gaps/leaf_test_pvalue.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from doubleton_gaps import DATA_ROOT, OUT, load_vms  # noqa: E402
from glyph_ngram_leaf_test import ngram_stream, page_symbols  # noqa: E402
from rare_type_clustering import leaf_of  # noqa: E402


def category_matrix(pages):
    for p in pages:
        p["leaf"] = leaf_of(p["page"])
    quire_pages = defaultdict(list)
    for p in pages:
        quire_pages[p["quire"]].append(p)
    P = len(pages)
    cat_idx = np.full((P, P), 4, dtype=np.int8)
    for plist in quire_pages.values():
        leaves = list(dict.fromkeys(p["leaf"] for p in plist))
        lr = {lf: k for k, lf in enumerate(leaves)}
        lb = {p["leaf"]: p["bifolio"] for p in plist}
        for a in plist:
            for b in plist:
                if a["page_idx"] >= b["page_idx"] or a["leaf"] == b["leaf"]:
                    continue
                conj = lb[a["leaf"]] == lb[b["leaf"]]
                adj = abs(lr[a["leaf"]] - lr[b["leaf"]]) == 1
                c = 2 if (conj and adj) else 0 if conj else 1 if adj else 3
                cat_idx[a["page_idx"], b["page_idx"]] = c
                cat_idx[b["page_idx"], a["page_idx"]] = c
    return cat_idx


def rare_pairs(words, page_of, kmax):
    pos = defaultdict(list)
    for i, w in enumerate(words):
        pos[w].append(i)
    occ, pi, pj, tid = [], [], [], []
    t = 0
    for p in pos.values():
        if 2 <= len(p) <= kmax:
            pp = page_of[p]
            a, b = np.triu_indices(len(p), 1)
            base = len(occ)
            occ.extend(pp.tolist())
            pi.append(base + a); pj.append(base + b); tid.append(np.full(len(a), t)); t += 1
    return np.array(occ), np.concatenate(pi), np.concatenate(pj), np.concatenate(tid)


def tails(words, page_of, pages, kmax, n_shuf, rng, cat_idx, group_keys, page_len):
    occ, a, b, tid = rare_pairs(words, page_of, kmax)
    P = len(pages)

    def diff(page_of_occ):
        c = np.bincount(cat_idx[page_of_occ[a], page_of_occ[b]], minlength=5)
        return int(c[0] - c[1])

    obs = diff(occ)
    # perm null
    g = defaultdict(list)
    for p in pages:
        g[tuple(p[k] for k in group_keys)].append(p["page_idx"])
    groups = [np.array(v) for v in g.values()]
    ident = np.arange(P)
    perm_null = np.zeros(n_shuf, dtype=np.int64)
    for s in range(n_shuf):
        perm = ident.copy()
        for m in groups:
            perm[m] = rng.permutation(perm[m])
        perm_null[s] = diff(perm[occ])
    # uniform null: each occurrence of a type -> page of the type's language, prob ∝ page length
    lang_of = np.array([p["lang"] for p in pages])
    n_types = tid.max() + 1
    # type language = language of its first occurrence's page (types are almost never mixed)
    first_occ = np.zeros(n_types, dtype=np.int64)
    first_occ[tid[::-1]] = a[::-1]
    type_lang = lang_of[occ[first_occ]]
    occ_lang = np.empty(len(occ), dtype=object)
    # map each occurrence to its type via pair arrays
    occ_type = np.zeros(len(occ), dtype=np.int64)
    occ_type[a] = tid; occ_type[b] = tid
    occ_lang = type_lang[occ_type]
    uni_null = np.zeros(n_shuf, dtype=np.int64)
    langs = np.unique(lang_of)
    for s in range(n_shuf):
        new = np.zeros(len(occ), dtype=np.int64)
        for L in langs:
            idx = np.where(lang_of == L)[0]
            w = page_len[idx].astype(float); w /= w.sum()
            sel = occ_lang == L
            new[sel] = rng.choice(idx, size=sel.sum(), p=w)
        uni_null[s] = diff(new)
    out = {"n_pairs": int(len(a)), "obs": obs}
    for name, arr in (("perm", perm_null), ("uniform", uni_null)):
        ge = int((arr >= obs).sum())
        out[name] = {"null_mean": float(arr.mean()), "null_sd": float(arr.std()), "z": float((obs - arr.mean()) / arr.std()),
                     "n_ge": ge, "n": int(n_shuf), "p": (ge + 1) / (n_shuf + 1), "max": int(arr.max())}
    return out


def main():
    n_shuf = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rng = np.random.default_rng(11)
    res = {}
    for tr, fname in (("IT2a", "IT2a-n.txt"), ("RF1b", "RF1b-e.txt")):
        toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / fname)
        cat_idx = category_matrix(pages)
        res[tr] = {}
        words = [t["w"] for t in toks]
        page_of = np.array([t["page_idx"] for t in toks])
        page_len = np.bincount(page_of, minlength=len(pages))
        syms, bounds = page_symbols(toks, len(pages))
        sym_len = np.array([len(s) for s in syms])
        print(f"== {tr}", flush=True)
        for gk in (("quire", "lang"), ("quire", "lang", "hand")):
            gname = "+".join(gk)
            r = tails(words, page_of, pages, 10, n_shuf, rng, cat_idx, gk, page_len)
            res[tr][f"words_k2to10/{gname}"] = r
            print(f"  words k2-10 null {gname}: obs {r['obs']:+d}  perm z={r['perm']['z']:+.2f} p={r['perm']['p']:.2e} ({r['perm']['n_ge']}/{n_shuf}, max {r['perm']['max']:+d})  uniform z={r['uniform']['z']:+.2f} p={r['uniform']['p']:.2e} ({r['uniform']['n_ge']}/{n_shuf})", flush=True)
            for n in (5, 6, 7, 8):
                w, po = ngram_stream(syms, bounds, n, False)
                r = tails(w, po, pages, 10, n_shuf, rng, cat_idx, gk, sym_len)
                res[tr][f"n{n}_all_k2to10/{gname}"] = r
                print(f"  n={n} all k2-10 null {gname}: obs {r['obs']:+d}  perm z={r['perm']['z']:+.2f} p={r['perm']['p']:.2e} ({r['perm']['n_ge']}/{n_shuf}, max {r['perm']['max']:+d})  uniform z={r['uniform']['z']:+.2f} p={r['uniform']['p']:.2e} ({r['uniform']['n_ge']}/{n_shuf})", flush=True)
    (OUT / "leaf_test_pvalue.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
