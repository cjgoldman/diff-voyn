"""Proof of concept (2026-09-03): can a rare-material clustering metric point to an
intended stacking order of the bifolia inside one quire?

One quire (default M: 20 pages, 5 bifolia, Currier B, hand 2, section B, ~6 900
tokens).  Units: word types and glyph n-grams (n 5-8) with 2..10 occurrences in the
quire.  Metric per reading order: over occurrence pairs of a rare type, the mean
distance in tokens/symbols (L1; "total distance of all members" per pair) and the
mean exp(-d/300) kernel (K; 300 ~ one page).
Two questions, two pair sets:
  (a) which stacking?  Only pairs whose members sit on different sheets count
      (within-sheet pairs are contiguous in every stacked order and would only add
      the §7 sheet effect as a constant).  Candidates: every order of the S sheets
      read stacked a-r,a-v,b-r,b-v (S! = 120 for S = 5); reversal gives the same L1.
      Null: page contents permuted among the quire's slots, best-of-120 recomputed.
      Consistency: correlation of the 120-vector between units vs the same
      correlation on shuffled contents.
  (b) stacked at all vs as bound?  All cross-page pairs, the nested binding as the
      121st candidate.
Extension: inverted folding per sheet (b-v,b-r,a-r,a-v; x2^S) on all cross-page pairs.
Power: known texts written onto the same slots in the current stacked order, a random
stacked order, or nested - does the metric rank the writing order first?
Outputs DATA_ROOT/analysis/doubleton_gaps/quire_order_poc_<Q>.json
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from doubleton_gaps import DATA_ROOT, OUT, load_csv_text, load_raw_text, load_vms  # noqa: E402
from glyph_ngram_leaf_test import page_symbols  # noqa: E402
from order_optimize import build_sheets  # noqa: E402

KMAX = 10
L_KERNEL = 300.0
UNITS = ("words", "n5", "n6", "n7", "n8")


def rare_pairs(items_per_page, sheet_of, kmax=KMAX):
    pos = defaultdict(list)
    for p, items in enumerate(items_per_page):
        for i, w in enumerate(items):
            pos[w].append((p, i))
    pa, ia, pb, ib = [], [], [], []
    n_types = 0
    for occ in pos.values():
        if 2 <= len(occ) <= kmax:
            n_types += 1
            for (p1, i1), (p2, i2) in itertools.combinations(occ, 2):
                if p1 != p2:
                    pa.append(p1); ia.append(i1); pb.append(p2); ib.append(i2)
    pa, ia, pb, ib = map(np.array, (pa, ia, pb, ib))
    return {"pa": pa, "ia": ia, "pb": pb, "ib": ib, "n_types": n_types, "between": sheet_of[pa] != sheet_of[pb],
            "page_len": np.array([len(x) for x in items_per_page])}


def metrics(pairs, orders, mask=None):
    pl = pairs["page_len"]
    off = np.zeros_like(orders, dtype=np.int64)
    for o in range(orders.shape[0]):
        seq = orders[o]
        off[o, seq] = np.concatenate([[0], np.cumsum(pl[seq])[:-1]])
    m = slice(None) if mask is None else mask
    d = np.abs(off[:, pairs["pa"][m]] + pairs["ia"][m] - off[:, pairs["pb"][m]] - pairs["ib"][m])
    return d.mean(axis=1), np.exp(-d / L_KERNEL).mean(axis=1)


def stacked_orders(units, inverted=False):
    S = len(units)
    out, labels = [], []
    for perm in itertools.permutations(range(S)):
        for fl in (itertools.product((0, 1), repeat=S) if inverted else [tuple([0] * S)]):
            seq = []
            for s in perm:
                u = units[s]
                seq.extend((list(reversed(u["b"])) + u["a"]) if fl[s] else (u["a"] + u["b"]))
            out.append(seq)
            labels.append("".join(str(s + 1) + ("'" if fl[s] else "") for s in perm))
    return np.array(out), labels


def canon(label):
    """orders and their reversals give the same L1; report the one starting with the smaller digit."""
    r = label[::-1]
    return min(label, r)


def sheet_affinity(pairs, sheet_of, S, perms):
    def mat(perm):
        inv = np.argsort(perm)
        sa, sb = sheet_of[inv[pairs["pa"]]], sheet_of[inv[pairs["pb"]]]
        m = np.zeros((S, S))
        np.add.at(m, (np.minimum(sa, sb), np.maximum(sa, sb)), 1)
        return m
    obs = mat(np.arange(len(sheet_of)))
    exp = np.mean([mat(p) for p in perms], axis=0)
    return obs, exp


def units_of(pages_q, toks_q):
    n_pages = len(pages_q)
    words = [[] for _ in range(n_pages)]
    for t in toks_q:
        words[t["page_idx"]].append(t["w"])
    syms, _ = page_symbols(toks_q, n_pages)
    out = {"words": words}
    for n in (5, 6, 7, 8):
        out[f"n{n}"] = [["".join(s[i:i + n]) for i in range(len(s) - n + 1)] for s in syms]
    return out, np.array([len(w) for w in words])


def lay_text(src_words, page_len, order):
    items = [None] * len(page_len)
    acc = 0
    for slot in order:
        items[slot] = src_words[acc:acc + page_len[slot]]
        acc += page_len[slot]
    return items


def analyse_quire(pq, tq, n_shuf, rng, tag):
    units = build_sheets(pq)
    S, P = len(units), len(pq)
    sheet_of = np.zeros(P, dtype=int)
    for s, u in enumerate(units):
        for p in u["a"] + u["b"]:
            sheet_of[p] = s
    cand, labels = stacked_orders(units)
    cur_label = "".join(str(s + 1) for s in range(S))
    cur_i = labels.index(cur_label)
    nested = np.arange(P)
    items, wlen = units_of(pq, tq)
    perms = [rng.permutation(P) for _ in range(n_shuf)]
    res = {"pages": [p["page"] for p in pq], "n_tokens": len(tq), "sheets": [[pq[i]["page"] for i in u["a"] + u["b"]] for u in units], "units": {}}
    vec, nvec = {}, {}
    print(f"== {tag}: {P} pages, {len(tq)} tokens, sheets " + "; ".join(f"{s+1}:{'/'.join(pq[i]['page'] for i in u['a']+u['b'])}" for s, u in enumerate(units)), flush=True)
    for name in UNITS:
        pairs = rare_pairs(items[name], sheet_of)
        bt = pairs["between"]
        # (a) which stacking: between-sheet pairs
        L1b, Kb = metrics(pairs, cand, bt)
        vec[name] = L1b
        order = np.argsort(L1b)
        z_within = (L1b[order[0]] - L1b.mean()) / L1b.std()
        nb, nvec[name], ncur = [], [], []
        for perm in perms:
            inv = np.argsort(perm)  # content page pa sits in slot inv[pa]
            bt_p = sheet_of[inv[pairs["pa"]]] != sheet_of[inv[pairs["pb"]]]  # between-sheet under the permuted contents
            v, _ = metrics(pairs, perm[cand], bt_p)
            nvec[name].append(v); nb.append(v.min()); ncur.append(v[cur_i])
        nb, ncur = np.array(nb), np.array(ncur)
        # (b) stacked vs nested: all cross-page pairs
        L1a, Ka = metrics(pairs, cand)
        L1n, Kn = metrics(pairs, nested[None, :])
        nn = np.array([metrics(pairs, perm[nested][None, :])[0][0] for perm in perms])
        na = np.array([metrics(pairs, perm[cand])[0].mean() for perm in perms])
        r = {"n_types": pairs["n_types"], "n_pairs": int(len(pairs["pa"])), "n_between": int(bt.sum()),
             "between": {"best": canon(labels[order[0]]), "best_val": float(L1b[order[0]]), "top5": [canon(labels[i]) for i in order[:5]],
                         "cand_mean": float(L1b.mean()), "cand_sd": float(L1b.std()), "z_best_within_candidates": float(z_within),
                         "current_val": float(L1b[cur_i]), "current_rank": int((L1b < L1b[cur_i]).sum() + 1),
                         "null_best_mean": float(nb.mean()), "null_best_sd": float(nb.std()), "p_best": float(((nb <= L1b[order[0]]).sum() + 1) / (n_shuf + 1)),
                         "z_current_vs_null": float(-(L1b[cur_i] - ncur.mean()) / ncur.std()),
                         "K_best": canon(labels[int(Kb.argmax())]), "K_current_rank": int((Kb > Kb[cur_i]).sum() + 1)},
             "all": {"nested": float(L1n[0]), "nested_rank_among_121": int((L1a < L1n[0]).sum() + 1), "stacked_mean": float(L1a.mean()), "stacked_min": float(L1a.min()), "stacked_max": float(L1a.max()),
                     "current": float(L1a[cur_i]), "z_nested_vs_null": float(-(L1n[0] - nn.mean()) / nn.std()), "z_stackedmean_vs_null": float(-(L1a.mean() - na.mean()) / na.std()),
                     "K_nested_rank": int((Ka > Kn[0]).sum() + 1)}}
        res["units"][name] = r
        b, a = r["between"], r["all"]
        print(f"  {name:5s} types {r['n_types']:5d} pairs {r['n_pairs']:6d} (between-sheet {r['n_between']:6d})", flush=True)
        print(f"        which stacking: best {b['best']} {b['best_val']:.1f} | cand {b['cand_mean']:.1f} ± {b['cand_sd']:.1f} → best z within 120 = {b['z_best_within_candidates']:+.2f} (iid min of 120 ≈ −2.5) | null best {b['null_best_mean']:.1f} ± {b['null_best_sd']:.1f} → p {b['p_best']:.3f} | current 12345: rank {b['current_rank']:3d}/120, vs null z {b['z_current_vs_null']:+.2f} | K best {b['K_best']} current K rank {b['K_current_rank']}", flush=True)
        print(f"        stacked vs nested (all pairs): nested {a['nested']:.1f} rank {a['nested_rank_among_121']:3d}/121 (z vs null {a['z_nested_vs_null']:+.2f}) | stacked {a['stacked_min']:.1f}…{a['stacked_max']:.1f} mean {a['stacked_mean']:.1f} (z vs null {a['z_stackedmean_vs_null']:+.2f}) | K: nested rank {a['K_nested_rank']}", flush=True)
        if name in ("words", "n7"):
            obs, exp = sheet_affinity(pairs, sheet_of, S, perms[:200])
            r["sheet_affinity_obs_over_exp"] = [[float(obs[i, j] / exp[i, j]) if j > i else None for j in range(S)] for i in range(S)]
            print(f"        sheet affinity obs/exp ({name}), rows/cols = sheets 1..{S} (outer→inner):")
            for i in range(S):
                print("          " + " ".join(f"{obs[i,j]/exp[i,j]:5.2f}" if j > i else "  .  " for j in range(S)), flush=True)
    # consistency across units
    cons = {}
    for u1, u2 in itertools.combinations(UNITS, 2):
        c = float(np.corrcoef(vec[u1], vec[u2])[0, 1])
        cn = np.array([np.corrcoef(nvec[u1][s], nvec[u2][s])[0, 1] for s in range(n_shuf)])
        cons[f"{u1}-{u2}"] = {"corr": c, "null_mean": float(cn.mean()), "null_sd": float(cn.std()), "z": float((c - cn.mean()) / cn.std()), "p": float(((cn >= c).sum() + 1) / (n_shuf + 1))}
    res["consistency"] = cons
    print("  consistency of the 120-vector across units (corr; null = same shuffled contents): " + "  ".join(f"{k} {v['corr']:+.2f} (null {v['null_mean']:+.2f}±{v['null_sd']:.2f}, z {v['z']:+.1f})" for k, v in cons.items()), flush=True)
    # inverted extension (all cross-page pairs, since within-sheet order changes)
    cand_i, labels_i = stacked_orders(units, inverted=True)
    res["inverted"] = {}
    for name in ("words", "n7", "n8"):
        pairs = rare_pairs(items[name], sheet_of)
        L1, _ = metrics(pairs, cand_i)
        o = np.argsort(L1)
        res["inverted"][name] = {"best": labels_i[o[0]], "best_val": float(L1[o[0]]), "top5": [labels_i[i] for i in o[:5]], "n_candidates": len(labels_i), "cand_mean": float(L1.mean()), "cand_sd": float(L1.std()), "z_best_within": float((L1[o[0]] - L1.mean()) / L1.std())}
        print(f"  inverted-folding extension ({name}, {len(labels_i)} candidates, all pairs): best {labels_i[o[0]]} {L1[o[0]]:.1f}, top5 {[labels_i[i] for i in o[:5]]}, z within {(L1[o[0]] - L1.mean()) / L1.std():+.2f} (iid min of 3840 ≈ −3.5)", flush=True)
    return res, units, cand, labels, cur_label, nested, wlen, sheet_of


def controls(units, cand, labels, cur_label, nested, wlen, sheet_of, rng):
    cc = DATA_ROOT / "external/voynich-attack/corpora/latin/CorpusCorporum/auctores_scientiarum_varii"
    dta = DATA_ROOT / "external/voynich-attack/corpora/german/DTA"
    texts = {
        "la_isidorus": load_csv_text(cc / "isidorus_hispalensis/etymologiae/etymologiae.csv"),
        "de_bullinger": load_csv_text(dta / "1558_bullinger_haussbuoch/1558_bullinger_haussbuoch.csv"),
        "it_decameron": load_raw_text(DATA_ROOT / "raw/italian/boccaccio_decameron.txt"),
    }
    out = {}
    rand_perm = rng.permutation(len(units))
    rand_label = "".join(str(s + 1) for s in rand_perm)
    rand_seq = np.array([i for s in rand_perm for i in units[s]["a"] + units[s]["b"]])
    for tname, src in texts.items():
        for start in (0, 1, 2):
            src_w = src[start * 20000: start * 20000 + int(wlen.sum()) + 5]
            line = []
            for mode, order, true_label in (("stacked_cur", cand[labels.index(cur_label)], cur_label), ("stacked_rand", rand_seq, rand_label), ("nested", nested, "nested")):
                pairs = rare_pairs(lay_text(src_w, wlen, order), sheet_of)
                if mode == "nested":
                    L1a, _ = metrics(pairs, cand); L1n, _ = metrics(pairs, nested[None, :])
                    rk = int((L1a < L1n[0]).sum() + 1); best = "nested" if rk == 1 else canon(labels[int(L1a.argmin())])
                    out[f"{tname}/w{start}/{mode}"] = {"rank_all_121": rk, "best": best}
                    line.append(f"nested-written: nested rank {rk}/121")
                else:
                    L1b, _ = metrics(pairs, cand, pairs["between"])
                    rk = int((L1b < L1b[labels.index(true_label)]).sum() + 1)
                    zt = (L1b[labels.index(true_label)] - L1b.mean()) / L1b.std()
                    out[f"{tname}/w{start}/{mode}"] = {"true": canon(true_label), "rank_between_120": rk, "best": canon(labels[int(L1b.argmin())]), "z_true_within": float(zt), "n_between": int(pairs["between"].sum())}
                    line.append(f"{mode} (true {canon(true_label)}): rank {rk:3d}/120 best {canon(labels[int(L1b.argmin())])} z {zt:+.1f}")
            print(f"  control {tname:12s} w{start}: " + " | ".join(line), flush=True)
    return out


def main():
    Q = sys.argv[1] if len(sys.argv) > 1 else "M"
    n_shuf = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    rng = np.random.default_rng(3)
    res = {}
    for tr, fname in (("IT2a", "IT2a-n.txt"), ("RF1b", "RF1b-e.txt")):
        toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / fname)
        pages_q = [p for p in pages if p["quire"] == Q]
        gidx = {p["page_idx"]: k for k, p in enumerate(pages_q)}
        pq = []
        for p in pages_q:
            q = dict(p); q["page_idx"] = gidx[p["page_idx"]]; pq.append(q)
        tq = [{"w": t["w"], "page_idx": gidx[t["page_idx"]]} for t in toks if t["page_idx"] in gidx]
        r, units, cand, labels, cur_label, nested, wlen, sheet_of = analyse_quire(pq, tq, n_shuf, rng, f"{tr} quire {Q}")
        res[tr] = r
        if tr == "IT2a":
            res["controls"] = controls(units, cand, labels, cur_label, nested, wlen, sheet_of, rng)
    (OUT / f"quire_order_poc_{Q}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
