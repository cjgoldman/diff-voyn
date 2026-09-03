"""Burst-scaled seriation metric for the one-quire stacking test (2026-09-03).

Follows quire_order_poc.py (same quire, units, 120 stacked candidates, nulls,
controls) but replaces the plain mean pair distance by metrics that weight a rare
type by how tightly it clusters where it lives:
  burst   sum over types t and occurrences outside t's home sheet of d/lambda_t,
          d = distance (under the candidate order) to the nearest home occurrence,
          lambda_t = shrunk mean gap between consecutive home-sheet occurrences
          (order-invariant across stacked orders) -> log-likelihood of an
          exponential-tail burst model.
  blog    sum of log(1 + d/lambda_t)  (caps a single far outlier).
  modal   sum of w_t * d, w_t = fraction of home occurrences on the modal page
          (the discrete version of the same intuition).
  L1      plain mean between-sheet pair distance (quire_order_poc.py) for reference.
Home sheet = sheet with most occurrences; types with a tie (e.g. doubletons split
1-1) carry no intra-sheet information and are skipped.
Outputs DATA_ROOT/analysis/doubleton_gaps/quire_order_burst_<Q>.json
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
from order_optimize import build_sheets  # noqa: E402
from quire_order_poc import UNITS, canon, lay_text, metrics, rare_pairs, stacked_orders, units_of  # noqa: E402

KMAX = 10
FLOOR = 30.0
METRICS = ("L1", "burst", "blog", "modal")


def occurrences(items_per_page, kmax=KMAX):
    pos = defaultdict(list)
    for p, items in enumerate(items_per_page):
        for i, w in enumerate(items):
            pos[w].append((p, i))
    ot, op, oi = [], [], []
    t = 0
    for occ in pos.values():
        if 2 <= len(occ) <= kmax:
            for p, i in occ:
                ot.append(t); op.append(p); oi.append(i)
            t += 1
    return np.array(ot), np.array(op), np.array(oi), t, np.array([len(x) for x in items_per_page])


class Burst:
    """Per-type burst scales and outlier->home pairs for one content assignment (inv: content page -> slot)."""

    def __init__(self, ot, op, oi, n_types, page_len, inv, sheet_of, S, ref_order):
        slot = inv[op]
        sheet = sheet_of[slot]
        cnt = np.zeros((n_types, S), dtype=np.int64)
        np.add.at(cnt, (ot, sheet), 1)
        srt = np.sort(cnt, axis=1)
        home = cnt.argmax(axis=1)
        valid = srt[:, -1] > srt[:, -2]  # unique home sheet
        is_home = (sheet == home[ot]) & valid[ot]
        # positions under a reference stacked order (within-sheet gaps are the same in all stacked orders)
        off = np.zeros(len(page_len), dtype=np.int64)
        off[ref_order] = np.concatenate([[0], np.cumsum(page_len[ref_order])[:-1]])
        pos = off[op] + oi
        # home gaps per type
        hi = np.where(is_home)[0]
        o = np.lexsort((pos[hi], ot[hi]))
        hi = hi[o]
        same = ot[hi][1:] == ot[hi][:-1]
        gaps = (pos[hi][1:] - pos[hi][:-1])[same]
        gt = ot[hi][1:][same]
        gsum = np.bincount(gt, weights=gaps, minlength=n_types)
        gn = np.bincount(gt, minlength=n_types)
        raw = np.where(gn > 0, gsum / np.maximum(gn, 1), np.nan)
        lam0 = float(np.nanmedian(raw)) if np.isfinite(raw).any() else 100.0
        lam = np.maximum((gsum + lam0) / (gn + 1), FLOOR)
        # modal-page weight per type (fraction of home occurrences on the modal page)
        key = ot[hi] * len(page_len) + op[hi]
        pc = np.bincount(key, minlength=n_types * len(page_len)).reshape(n_types, len(page_len))
        nh = np.bincount(ot[hi], minlength=n_types)
        w_modal = np.where(nh > 0, pc.max(axis=1) / np.maximum(nh, 1), 0.0)
        # outliers and their home candidates
        out = np.where(valid[ot] & ~is_home)[0]
        out = out[np.argsort(ot[out], kind="stable")]
        # home occurrences grouped by type
        hstart = np.searchsorted(ot[hi], np.arange(n_types + 1))
        nhome = hstart[1:] - hstart[:-1]
        rep = nhome[ot[out]]
        pair_out = np.repeat(out, rep)
        pair_home = np.concatenate([hi[hstart[t]:hstart[t + 1]] for t in ot[out]]) if len(out) else np.zeros(0, int)
        self.n_types_valid = int(valid.sum()); self.n_out = len(out)
        self.pair_out_page, self.pair_out_idx = op[pair_out], oi[pair_out]
        self.pair_home_page, self.pair_home_idx = op[pair_home], oi[pair_home]
        self.seg = np.concatenate([[0], np.cumsum(rep)[:-1]]) if len(out) else np.zeros(0, int)
        self.lam = lam[ot[out]]; self.w_modal = w_modal[ot[out]]
        self.page_len = page_len
        self.lam0 = lam0

    def costs(self, orders):
        pl = self.page_len
        off = np.zeros_like(orders, dtype=np.int64)
        for o in range(orders.shape[0]):
            seq = orders[o]
            off[o, seq] = np.concatenate([[0], np.cumsum(pl[seq])[:-1]])
        if self.n_out == 0:
            z = np.zeros(orders.shape[0])
            return {"burst": z, "blog": z, "modal": z}
        d = np.abs(off[:, self.pair_out_page] + self.pair_out_idx - off[:, self.pair_home_page] - self.pair_home_idx)
        dmin = np.minimum.reduceat(d, self.seg, axis=1)
        return {"burst": (dmin / self.lam).sum(axis=1) / self.n_out, "blog": np.log1p(dmin / self.lam).sum(axis=1) / self.n_out,
                "modal": (dmin * self.w_modal).sum(axis=1) / max(self.w_modal.sum(), 1e-9)}


def uniform_costs(ot, op, oi, n_types, page_len, sheet_of, S, cand, rng):
    """Geometry null: real page lengths stay in their slots; every occurrence is re-placed uniformly over the tokens."""
    tot = int(page_len.sum())
    start = np.concatenate([[0], np.cumsum(page_len)[:-1]])
    pos = rng.integers(0, tot, size=len(ot))
    op2 = np.searchsorted(start, pos, side="right") - 1
    oi2 = pos - start[op2]
    b = Burst(ot, op2, oi2, n_types, page_len, np.arange(len(page_len)), sheet_of, S, cand[0])
    c = b.costs(cand)
    # L1 between-sheet on the re-placed occurrences
    o = np.lexsort((oi2, op2, ot))
    pa, ia, pb, ib = [], [], [], []
    starts = np.searchsorted(ot[o], np.arange(n_types + 1))
    for t in range(n_types):
        idx = o[starts[t]:starts[t + 1]]
        if len(idx) < 2:
            continue
        a_, b_ = np.triu_indices(len(idx), 1)
        pa.append(op2[idx][a_]); ia.append(oi2[idx][a_]); pb.append(op2[idx][b_]); ib.append(oi2[idx][b_])
    pairs = {"pa": np.concatenate(pa), "ia": np.concatenate(ia), "pb": np.concatenate(pb), "ib": np.concatenate(ib), "page_len": page_len}
    keep = pairs["pa"] != pairs["pb"]
    bt = keep & (sheet_of[pairs["pa"]] != sheet_of[pairs["pb"]])
    c["L1"] = metrics(pairs, cand, bt)[0]
    return c


def all_costs(items, sheet_of, S, cand, inv=None):
    ot, op, oi, n_types, page_len = occurrences(items)
    P = len(page_len)
    inv = np.arange(P) if inv is None else inv
    b = Burst(ot, op, oi, n_types, page_len, inv, sheet_of, S, cand[0] if inv is None else inv_apply(cand[0], inv))
    orders = cand if inv is None else perm_apply(cand, inv)
    c = b.costs(orders)
    pairs = rare_pairs(items, sheet_of) if inv is None else None
    if pairs is not None:
        c["L1"] = metrics(pairs, cand, pairs["between"])[0]
    else:
        pairs = rare_pairs(items, sheet_of)
        bt_p = sheet_of[inv[pairs["pa"]]] != sheet_of[inv[pairs["pb"]]]
        c["L1"] = metrics(pairs, orders, bt_p)[0]
    return c, b


def perm_apply(cand, inv):
    """content permuted: slot s holds content page perm[s]; reading slots in order cand gives content pages perm[cand]."""
    perm = np.argsort(inv)
    return perm[cand]


def inv_apply(order, inv):
    return np.argsort(inv)[order]


def analyse(tag, pq, tq, n_shuf, rng, res):
    units = build_sheets(pq)
    S, P = len(units), len(pq)
    sheet_of = np.zeros(P, dtype=int)
    for s, u in enumerate(units):
        for p in u["a"] + u["b"]:
            sheet_of[p] = s
    cand, labels = stacked_orders(units)
    cur_label = "".join(str(s + 1) for s in range(S)); cur_i = labels.index(cur_label)
    items, wlen = units_of(pq, tq)
    perms = [rng.permutation(P) for _ in range(n_shuf)]
    invs = [np.argsort(p) for p in perms]
    print(f"== {tag}", flush=True)
    vec = {m: {} for m in METRICS}; nvec = {m: {u: [] for u in UNITS} for m in METRICS}
    for name in UNITS:
        c, b = all_costs(items[name], sheet_of, S, cand)
        nulls = {m: [] for m in METRICS}
        for inv in invs:
            cn, _ = all_costs(items[name], sheet_of, S, cand, inv)
            for m in METRICS:
                nulls[m].append(cn[m]); nvec[m][name].append(cn[m])
        ot, op, oi, n_types, page_len = occurrences(items[name])
        unif = {m: [] for m in METRICS}
        for _ in range(n_shuf):
            cu = uniform_costs(ot, op, oi, n_types, page_len, sheet_of, S, cand, rng)
            for m in METRICS:
                unif[m].append(cu[m])
        r = {"n_types_valid": b.n_types_valid, "n_outliers": b.n_out, "lambda0": b.lam0}
        for m in METRICS:
            v = c[m]; vec[m][name] = v
            o = np.argsort(v); nb = np.array([x.min() for x in nulls[m]]); ncur = np.array([x[cur_i] for x in nulls[m]])
            ub = np.array([x.min() for x in unif[m]]); uarg = np.array([labels[int(x.argmin())] for x in unif[m]])
            geo_best = max(set(canon(l) for l in uarg), key=lambda l: sum(canon(x) == l for x in uarg))
            r[m] = {"p_best_uniform": float(((ub <= v[o[0]]).sum() + 1) / (n_shuf + 1)), "geometry_favoured": geo_best, "geometry_favoured_frac": float(np.mean([canon(x) == geo_best for x in uarg])),"best": canon(labels[o[0]]), "top3": [canon(labels[i]) for i in o[:3]], "best_val": float(v[o[0]]), "cand_mean": float(v.mean()), "cand_sd": float(v.std()),
                    "z_best_within": float((v[o[0]] - v.mean()) / v.std()), "p_best": float(((nb <= v[o[0]]).sum() + 1) / (n_shuf + 1)),
                    "current_rank": int((v < v[cur_i]).sum() + 1), "z_current_vs_null": float(-(v[cur_i] - ncur.mean()) / ncur.std())}
        res[name] = r
        print(f"  {name:5s} valid types {b.n_types_valid:5d} outliers {b.n_out:5d} lambda0 {b.lam0:6.0f} | " + " | ".join(f"{m}: best {r[m]['best']} z_in {r[m]['z_best_within']:+.2f} p {r[m]['p_best']:.3f} p_uni {r[m]['p_best_uniform']:.3f} (geom {r[m]['geometry_favoured']} {r[m]['geometry_favoured_frac']:.2f}) cur rank {r[m]['current_rank']:3d}" for m in METRICS), flush=True)
    cons = {}
    for m in METRICS:
        cons[m] = {}
        for u1, u2 in itertools.combinations(UNITS, 2):
            cc = float(np.corrcoef(vec[m][u1], vec[m][u2])[0, 1])
            cn = np.array([np.corrcoef(nvec[m][u1][s], nvec[m][u2][s])[0, 1] for s in range(n_shuf)])
            cons[m][f"{u1}-{u2}"] = {"corr": cc, "null_mean": float(cn.mean()), "null_sd": float(cn.std()), "z": float((cc - cn.mean()) / cn.std())}
        zs = [v["z"] for v in cons[m].values()]
        print(f"  consistency {m:5s}: words-vs-ngram z " + " ".join(f"{cons[m][f'words-{u}']['z']:+.1f}" for u in UNITS[1:]) + f" | ngram-ngram z mean {np.mean([cons[m][k]['z'] for k in cons[m] if not k.startswith('words')]):+.1f} | all pairs corr " + " ".join(f"{v['corr']:+.2f}" for v in cons[m].values()), flush=True)
    res["consistency"] = cons
    return units, cand, labels, cur_label, wlen, sheet_of, S


def controls(units, cand, labels, cur_label, wlen, sheet_of, S, rng):
    cc = DATA_ROOT / "external/voynich-attack/corpora/latin/CorpusCorporum/auctores_scientiarum_varii"
    dta = DATA_ROOT / "external/voynich-attack/corpora/german/DTA"
    texts = {
        "la_isidorus": load_csv_text(cc / "isidorus_hispalensis/etymologiae/etymologiae.csv"),
        "de_bullinger": load_csv_text(dta / "1558_bullinger_haussbuoch/1558_bullinger_haussbuoch.csv"),
        "it_decameron": load_raw_text(DATA_ROOT / "raw/italian/boccaccio_decameron.txt"),
    }
    out = {}
    rand_perm = rng.permutation(S)
    rand_label = "".join(str(s + 1) for s in rand_perm)
    rand_seq = np.array([i for s in rand_perm for i in units[s]["a"] + units[s]["b"]])
    tally = {m: [] for m in METRICS}
    for tname, src in texts.items():
        for start in (0, 1, 2):
            src_w = src[start * 20000: start * 20000 + int(wlen.sum()) + 5]
            for mode, order, true_label in (("stacked_cur", cand[labels.index(cur_label)], cur_label), ("stacked_rand", rand_seq, rand_label), ("nested", np.arange(len(wlen)), None)):
                items = lay_text(src_w, wlen, order)
                c, b = all_costs(items, sheet_of, S, cand)
                r = {}
                if true_label is None:
                    for m in METRICS:
                        v = c[m]; r[m] = {"best_stacked": canon(labels[int(v.argmin())]), "z_best": float((v.min() - v.mean()) / v.std())}
                    out[f"{tname}/w{start}/{mode}"] = r
                    print(f"  control {tname:12s} w{start} nested-written : best stacked order " + " | ".join(f"{m} {r[m]['best_stacked']} z {r[m]['z_best']:+.1f}" for m in METRICS), flush=True)
                    continue
                ti = labels.index(true_label)
                for m in METRICS:
                    v = c[m]; rk = int((v < v[ti]).sum() + 1); tally[m].append(rk)
                    r[m] = {"rank": rk, "z_true": float((v[ti] - v.mean()) / v.std()), "best": canon(labels[int(v.argmin())])}
                out[f"{tname}/w{start}/{mode}"] = r
                print(f"  control {tname:12s} w{start} {mode:12s} (true {canon(true_label)}): " + " | ".join(f"{m} rank {r[m]['rank']:3d} z {r[m]['z_true']:+.1f}" for m in METRICS), flush=True)
    summ = {m: {"rank1": int(sum(r == 1 for r in tally[m])), "top10": int(sum(r <= 10 for r in tally[m])), "median_rank": float(np.median(tally[m])), "n": len(tally[m])} for m in METRICS}
    print("  control summary: " + " | ".join(f"{m}: rank1 {s['rank1']}/{s['n']}, top10 {s['top10']}, median rank {s['median_rank']:.0f}" for m, s in summ.items()), flush=True)
    out["summary"] = summ
    return out


def main():
    Q = sys.argv[1] if len(sys.argv) > 1 else "M"
    n_shuf = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    rng = np.random.default_rng(5)
    res = {}
    for tr, fname in (("IT2a", "IT2a-n.txt"), ("RF1b", "RF1b-e.txt")):
        toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / fname)
        pages_q = [p for p in pages if p["quire"] == Q]
        gidx = {p["page_idx"]: k for k, p in enumerate(pages_q)}
        pq = []
        for p in pages_q:
            q = dict(p); q["page_idx"] = gidx[p["page_idx"]]; pq.append(q)
        tq = [{"w": t["w"], "page_idx": gidx[t["page_idx"]]} for t in toks if t["page_idx"] in gidx]
        res[tr] = {}
        units, cand, labels, cur_label, wlen, sheet_of, S = analyse(f"{tr} quire {Q}", pq, tq, n_shuf, rng, res[tr])
        if tr == "IT2a":
            res["controls"] = controls(units, cand, labels, cur_label, wlen, sheet_of, S, rng)
    (OUT / f"quire_order_burst_{Q}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
