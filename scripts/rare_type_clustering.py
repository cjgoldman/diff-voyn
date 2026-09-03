"""Clustering of rare word types (2–5 occurrences) — extension of the
doubleton study (2026-09-02).  Consecutive-occurrence gaps and span per
frequency class against a uniform-placement null (Monte Carlo, same N and k),
for the manuscript (both transcriptions, A/B) and known texts; plus the
leaf-pair nested-vs-stacked test pooled over all rare types.
Outputs DATA_ROOT/analysis/doubleton_gaps/rare_types.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from doubleton_gaps import DATA_ROOT, OUT, load_csv_text, load_raw_text, load_vms  # noqa: E402
from doubleton_leaf_affinity import leaf_of  # noqa: E402

KS = (2, 3, 4, 5)


def positions_by_k(words):
    pos = defaultdict(list)
    for i, w in enumerate(words):
        pos[w].append(i)
    return {k: np.array([p for p in pos.values() if len(p) == k]) for k in KS}


def class_stats(n, P, rng, page_of=None, n_mc=200000):
    """P: (m, k) sorted positions. Returns observed vs uniform-null stats."""
    if len(P) == 0:
        return {"n_types": 0}
    k = P.shape[1]
    gaps = np.diff(P, axis=1).ravel()
    span = P[:, -1] - P[:, 0]
    # null: k uniform positions without replacement
    U = np.sort(rng.choice(n, size=(n_mc, k), replace=True), axis=1)
    U = U[(np.diff(U, axis=1) > 0).all(axis=1)]
    ug = np.diff(U, axis=1).ravel()
    us = U[:, -1] - U[:, 0]
    out = {"n_types": int(len(P)), "n_gaps": int(len(gaps))}
    for g in (30, 100, 300, 1000):
        o, e = float(np.mean(gaps <= g)), float(np.mean(ug <= g))
        out[f"gap_le_{g}"] = o
        out[f"gap_le_{g}_ratio"] = o / e if e > 0 else None
    out["median_gap"] = float(np.median(gaps))
    out["median_gap_null"] = float(np.median(ug))
    out["median_span_over_n"] = float(np.median(span) / n)
    out["median_span_over_n_null"] = float(np.median(us) / n)
    # fraction of types whose occurrences all lie within 1000 tokens
    out["span_le_1000"] = float(np.mean(span <= 1000))
    out["span_le_1000_ratio"] = float(np.mean(span <= 1000) / max(np.mean(us <= 1000), 1e-9))
    if page_of is not None:
        pages = page_of[P]
        out["mean_distinct_pages"] = float(np.mean([len(set(r)) for r in pages]))
        out["frac_all_same_page"] = float(np.mean([len(set(r)) == 1 for r in pages]))
        upages = page_of[np.minimum(U, n - 1)]
        out["mean_distinct_pages_null"] = float(np.mean([len(set(r)) for r in upages]))
        out["frac_all_same_page_null"] = float(np.mean([len(set(r)) == 1 for r in upages]))
        # adjacent-page consecutive pairs (cross-page only)
        pi, pj = pages[:, :-1].ravel(), pages[:, 1:].ravel()
        cross = pi != pj
        out["frac_cross_pairs_adjacent_page"] = float(np.mean(np.abs(pi - pj)[cross] == 1)) if cross.any() else None
    return out


def leaf_test_pooled(words, page_of, pages, kmax, n_shuf, rng, lang=None, group_keys=("quire", "lang")):
    """Leaf-pair category counts over all occurrence pairs of types with 2..kmax occurrences (vectorized)."""
    pos = defaultdict(list)
    for i, w in enumerate(words):
        pos[w].append(i)
    pi, pj = [], []
    for p in pos.values():
        if 2 <= len(p) <= kmax:
            pp = page_of[p]
            a, b = np.triu_indices(len(p), 1)
            pi.append(pp[a]); pj.append(pp[b])
    pi = np.concatenate(pi) if pi else np.zeros(0, int)
    pj = np.concatenate(pj) if pj else np.zeros(0, int)
    if lang is not None:
        lang_of = np.array([p["lang"] for p in pages])
        keep = (lang_of[pi] == lang) & (lang_of[pj] == lang)
        pi, pj = pi[keep], pj[keep]
    for p in pages:
        p["leaf"] = leaf_of(p["page"])
    quire_pages = defaultdict(list)
    for p in pages:
        quire_pages[p["quire"]].append(p)
    cats = ("conjugate", "nested_adjacent", "both", "other")
    P = len(pages)
    cat_idx = np.full((P, P), 4, dtype=np.int8)  # 4 = not a same-quire cross-leaf pair
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

    def count(slot):
        return np.bincount(cat_idx[slot[pi], slot[pj]], minlength=5)[:4]

    ident = np.arange(P)
    obs = count(ident)
    groups = [np.array(v) for v in defaultdict(list, {}).values()]
    g = defaultdict(list)
    for p in pages:
        g[tuple(p[k] for k in group_keys)].append(p["page_idx"])
    groups = [np.array(v) for v in g.values()]
    null = np.zeros((n_shuf, 4))
    for s_ in range(n_shuf):
        perm = ident.copy()
        for m in groups:
            perm[m] = rng.permutation(perm[m])
        null[s_] = count(perm)
    out = {"n_pairs": int(len(pi))}
    for k, name in enumerate(cats):
        arr = null[:, k]; sd = arr.std()
        out[name] = {"obs": int(obs[k]), "null_mean": float(arr.mean()), "null_sd": float(sd), "z": float((obs[k] - arr.mean()) / sd) if sd > 0 else None}
    d = null[:, 0] - null[:, 1]; sd = d.std(); od = int(obs[0] - obs[1])
    out["conj_minus_nested"] = {"obs": od, "null_mean": float(d.mean()), "null_sd": float(sd), "z": float((od - d.mean()) / sd) if sd > 0 else None}
    return out


def main():
    n_shuf = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rng = np.random.default_rng(3)
    res = {"vms": {}, "known": {}}
    for tr, fname in (("IT2a", "IT2a-n.txt"), ("RF1b", "RF1b-e.txt")):
        toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / fname)
        words = [t["w"] for t in toks]
        page_of = np.array([t["page_idx"] for t in toks])
        r = {"n": len(words), "classes": {}}
        byk = positions_by_k(words)
        for k in KS:
            r["classes"][k] = class_stats(len(words), byk[k], rng, page_of)
        for lang in ("A", "B"):
            keep = np.array([pages[p]["lang"] == lang for p in page_of])
            sw = [w for w, kp in zip(words, keep) if kp]
            sub = positions_by_k(sw)
            r[f"currier_{lang}"] = {k: class_stats(len(sw), sub[k], rng) for k in KS}
        r["leaf_test_k2"] = leaf_test_pooled(words, page_of, pages, 2, n_shuf, rng)
        r["leaf_test_k2to5"] = leaf_test_pooled(words, page_of, pages, 5, n_shuf, rng)
        r["leaf_test_k2to10"] = leaf_test_pooled(words, page_of, pages, 10, n_shuf, rng)
        for lang in ("A", "B"):
            r[f"leaf_test_k2to5_{lang}"] = leaf_test_pooled(words, page_of, pages, 5, n_shuf, rng, lang=lang)
            r[f"leaf_test_k2to10_{lang}"] = leaf_test_pooled(words, page_of, pages, 10, n_shuf, rng, lang=lang)
        res["vms"][tr] = r
        print(f"== VMS {tr} n={len(words)}")
        for k in KS:
            c = r["classes"][k]
            print(f"  k={k}: types {c['n_types']:4d}  gap<=100 ratio {c['gap_le_100_ratio']:5.1f}  gap<=1000 ratio {c['gap_le_1000_ratio']:4.2f}  median gap {c['median_gap']:.0f}/{c['median_gap_null']:.0f}  span/N {c['median_span_over_n']:.3f}/{c['median_span_over_n_null']:.3f}  span<=1000 {c['span_le_1000']:.3f} (x{c['span_le_1000_ratio']:.1f})  all-same-page {c['frac_all_same_page']:.3f}/{c['frac_all_same_page_null']:.3f}  distinct pages {c['mean_distinct_pages']:.2f}/{c['mean_distinct_pages_null']:.2f}")
            for lang in ("A", "B"):
                c = r[f"currier_{lang}"][k]
                print(f"        {lang}: types {c['n_types']:4d} gap<=100 ratio {c['gap_le_100_ratio']:5.1f} gap<=1000 ratio {c['gap_le_1000_ratio']:4.2f} span/N {c['median_span_over_n']:.3f}/{c['median_span_over_n_null']:.3f}")
        for key in ("leaf_test_k2", "leaf_test_k2to5", "leaf_test_k2to10", "leaf_test_k2to5_A", "leaf_test_k2to10_A", "leaf_test_k2to5_B", "leaf_test_k2to10_B"):
            t = r[key]
            print(f"  {key}: pairs {t['n_pairs']}  conj {t['conjugate']['obs']}/{t['conjugate']['null_mean']:.1f} z={t['conjugate']['z']:+.1f}  nested {t['nested_adjacent']['obs']}/{t['nested_adjacent']['null_mean']:.1f} z={t['nested_adjacent']['z']:+.1f}  other z={t['other']['z']:+.1f}  conj-nested z={t['conj_minus_nested']['z']:+.1f}")
    n_window = res["vms"]["IT2a"]["n"]
    cc = DATA_ROOT / "external/voynich-attack/corpora/latin/CorpusCorporum/auctores_scientiarum_varii"
    dta = DATA_ROOT / "external/voynich-attack/corpora/german/DTA"
    known = {
        "la_isidorus_etym": load_csv_text(cc / "isidorus_hispalensis/etymologiae/etymologiae.csv"),
        "la_seneca_nq": load_csv_text(cc / "seneca/naturales_quaestiones/naturales_quaestiones.csv"),
        "la_plinius_nh": load_csv_text(cc / "plinius_maior/naturalis_historia/naturalis_historia.csv"),
        "de_bullinger": load_csv_text(dta / "1558_bullinger_haussbuoch/1558_bullinger_haussbuoch.csv"),
        "de_staden": load_csv_text(dta / "1557_staden_landschafft/1557_staden_landschafft.csv"),
        "it_decameron": load_raw_text(DATA_ROOT / "raw/italian/boccaccio_decameron.txt"),
        "it_commedia": load_raw_text(DATA_ROOT / "raw/italian/dante_divina_commedia.txt"),
        "it_orlando_furioso": load_raw_text(DATA_ROOT / "raw/italian/ariosto_orlando_furioso.txt"),
    }
    page_len = 168
    for name, words in known.items():
        w = words[:n_window]
        page_of = np.arange(len(w)) // page_len
        byk = positions_by_k(w)
        r = {k: class_stats(len(w), byk[k], rng, page_of) for k in KS}
        res["known"][name] = r
        print(f"== {name} n={len(w)}")
        for k in KS:
            c = r[k]
            print(f"  k={k}: types {c['n_types']:4d}  gap<=100 ratio {c['gap_le_100_ratio']:5.1f}  gap<=1000 ratio {c['gap_le_1000_ratio']:4.2f}  span/N {c['median_span_over_n']:.3f}/{c['median_span_over_n_null']:.3f}  span<=1000 {c['span_le_1000']:.3f} (x{c['span_le_1000_ratio']:.1f})  all-same-page {c['frac_all_same_page']:.3f}/{c['frac_all_same_page_null']:.3f}  distinct pages {c['mean_distinct_pages']:.2f}/{c['mean_distinct_pages_null']:.2f}")
    (OUT / "rare_types.json").write_text(json.dumps(res, indent=1, default=str))


def controls(n_shuf):
    """Known texts laid on the IT2a page slots, written nested or stacked, pooled leaf test."""
    from doubleton_gaps import stacked_bifolia_rank
    rng = np.random.default_rng(4)
    toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / "IT2a-n.txt")
    page_of0 = np.array([t["page_idx"] for t in toks])
    page_len = np.bincount(page_of0, minlength=len(pages))
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
    res = {}
    for name, src in texts.items():
        for mode in ("nested", "stacked"):
            rank = np.arange(len(pages)) if mode == "nested" else stacked_bifolia_rank(pages)
            order = np.argsort(rank)
            acc = 0; slot_words = {}
            for slot in order:
                slot_words[int(slot)] = src[acc: acc + page_len[slot]]; acc += page_len[slot]
            words, page_of = [], []
            for slot in range(len(pages)):
                words.extend(slot_words[slot]); page_of.extend([slot] * len(slot_words[slot]))
            page_of = np.array(page_of)
            r = {}
            for kmax in (2, 5, 10):
                r[f"k2to{kmax}"] = leaf_test_pooled(words, page_of, pages, kmax, n_shuf, rng)
            res[f"{name}/{mode}"] = r
            print(f"== control {name} {mode}: " + "  ".join(f"k2to{k}: conj z={r[f'k2to{k}']['conjugate']['z']:+.1f} nested z={r[f'k2to{k}']['nested_adjacent']['z']:+.1f} diff z={r[f'k2to{k}']['conj_minus_nested']['z']:+.1f}" for k in (2, 5, 10)))
    (OUT / "rare_types_controls.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    if "--control" in sys.argv:
        controls(int(sys.argv[1]))
    else:
        main()
