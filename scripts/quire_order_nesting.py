"""Alternative nesting patterns inside one quire (2026-09-03).

§13-16 compared the S! fully-stacked sheet orders (a-r, a-v, b-r, b-v per sheet) with
the single as-bound nested order.  This script asks whether any *other* nesting
pattern does as well as or better than the stacked optimum: every order of the S
sheets, cut into consecutive sub-gatherings, each sub-gathering nested (a-leaves of its
sheets outer->inner, then b-leaves inner->outer).  S! * 2^(S-1) page orders
(4 sheets 192, 5 sheets 1 920, 6 sheets 23 040); the fully stacked orders (S blocks)
and the fully nested ones (1 block; the binding is one of them) are the two extremes.
Labels: blocks separated by '|', sheets outer->inner inside a block, e.g. '16|5|4|23'.

Metrics as in quire_order_burst.py, evaluated on pair sets that vary with the
candidate (a nested block separates a sheet's two leaves): L1 over all cross-page
pairs; burst/blog/modal over the outlier->nearest-home term of §14 PLUS a home-cluster
term (each home occurrence -> nearest other home occurrence), which is constant across
the stacked orders (so the stacked ranking is §15's) and charges a pattern that splits a
home sheet.  Null: page contents permuted among the quire's slots, best-of-all
recomputed.  Statistics: best pattern and its block count; best stacked / best nested /
as-bound values and ranks; gain of the full space over the stacked optimum in candidate
sd against the same gain on shuffled contents.  Controls: prose written on the quire's
slots as bound, nested in a random sheet order, in a random two-block pattern, or stacked
in a random order -> rank of the writing pattern among all patterns.
Usage: quire_order_nesting.py Q n_shuf n_workers
Outputs DATA_ROOT/analysis/doubleton_gaps/quire_order_nesting_<Q>.json
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from doubleton_gaps import DATA_ROOT, OUT, load_csv_text, load_raw_text, load_vms  # noqa: E402
from order_optimize import build_sheets  # noqa: E402
from quire_order_burst import FLOOR, KMAX, occurrences  # noqa: E402
from quire_order_poc import UNITS, lay_text, units_of  # noqa: E402

METRICS = ("L1", "burst", "blog", "modal")
CHUNK = 256


def nesting_orders(units):
    S = len(units)
    out, labels, nblocks = [], [], []
    for perm in itertools.permutations(range(S)):
        for cuts in itertools.product((0, 1), repeat=S - 1):
            blocks, cur = [], [perm[0]]
            for s, c in zip(perm[1:], cuts):
                if c:
                    blocks.append(cur); cur = [s]
                else:
                    cur.append(s)
            blocks.append(cur)
            seq = []
            for blk in blocks:
                for s in blk:
                    seq.extend(units[s]["a"])
                for s in reversed(blk):
                    seq.extend(units[s]["b"])
            out.append(seq)
            labels.append("|".join("".join(str(s + 1) for s in blk) for blk in blocks))
            nblocks.append(len(blocks))
    cand = np.array(out); nblocks = np.array(nblocks)
    # a single-leaf "sheet" (e.g. quire B's f13, f12 lost) reads the same as outermost of a block or stacked
    # before it: keep one page order per distinct sequence, labelled by its reading with the most blocks
    _, first, inverse = np.unique(cand, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    keep = np.full(len(first), -1)
    for i in np.argsort(nblocks):  # ascending -> last write has the most blocks
        keep[inverse[i]] = i
    keep = np.sort(keep)
    return cand[keep], [labels[i] for i in keep], nblocks[keep]


def order_from_label(label, units):
    seq = []
    for blk in label.split("|"):
        sheets = [int(ch) - 1 for ch in blk]
        for s in sheets:
            seq.extend(units[s]["a"])
        for s in reversed(sheets):
            seq.extend(units[s]["b"])
    return np.array(seq)


def offsets(orders, page_len):
    """token offset of every slot under each order: (n_orders, P)."""
    n, P = orders.shape
    starts = np.zeros((n, P), dtype=np.int64)
    starts[:, 1:] = np.cumsum(page_len[orders], axis=1)[:, :-1]
    off = np.empty((n, P), dtype=np.int64)
    off[np.arange(n)[:, None], orders] = starts
    return off


class NestBurst:
    """§14 burst terms + home-cluster term, for one content assignment (inv: content page -> slot)."""

    def __init__(self, ot, op, oi, n_types, page_len, inv, sheet_of, S, ref_order):
        slot = inv[op]
        sheet = sheet_of[slot]
        cnt = np.zeros((n_types, S), dtype=np.int64)
        np.add.at(cnt, (ot, sheet), 1)
        srt = np.sort(cnt, axis=1)
        home = cnt.argmax(axis=1)
        valid = srt[:, -1] > srt[:, -2]
        is_home = (sheet == home[ot]) & valid[ot]
        off = np.zeros(len(page_len), dtype=np.int64)
        off[ref_order] = np.concatenate([[0], np.cumsum(page_len[ref_order])[:-1]])
        pos = off[op] + oi
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
        key = ot[hi] * len(page_len) + op[hi]
        pc = np.bincount(key, minlength=n_types * len(page_len)).reshape(n_types, len(page_len))
        nh = np.bincount(ot[hi], minlength=n_types)
        w_modal = np.where(nh > 0, pc.max(axis=1) / np.maximum(nh, 1), 0.0)
        hstart = np.searchsorted(ot[hi], np.arange(n_types + 1))
        nhome = hstart[1:] - hstart[:-1]
        # outlier -> home pairs
        out = np.where(valid[ot] & ~is_home)[0]
        out = out[np.argsort(ot[out], kind="stable")]
        rep = nhome[ot[out]]
        pair_a = np.repeat(out, rep)
        pair_b = np.concatenate([hi[hstart[t]:hstart[t + 1]] for t in ot[out]]) if len(out) else np.zeros(0, int)
        seg = np.concatenate([[0], np.cumsum(rep)[:-1]]) if len(out) else np.zeros(0, int)
        # home -> other home pairs (types with >= 2 home occurrences)
        ha, hb, hseg, hrep = [], [], [], []
        acc = len(pair_a)
        for t in range(n_types):
            idx = hi[hstart[t]:hstart[t + 1]]
            k = len(idx)
            if k < 2:
                continue
            for j in range(k):
                ha.append(np.full(k - 1, idx[j])); hb.append(np.delete(idx, j)); hseg.append(acc); acc += k - 1
                hrep.append(idx[j])
        if ha:
            ha = np.concatenate(ha); hb = np.concatenate(hb); hseg = np.array(hseg); hrep = np.array(hrep)
        else:
            ha = hb = hseg = hrep = np.zeros(0, int)
        self.pa_page = np.concatenate([op[pair_a], op[ha]]); self.pa_idx = np.concatenate([oi[pair_a], oi[ha]])
        self.pb_page = np.concatenate([op[pair_b], op[hb]]); self.pb_idx = np.concatenate([oi[pair_b], oi[hb]])
        self.seg = np.concatenate([seg, hseg]).astype(int)
        own = np.concatenate([out, hrep]).astype(int)
        self.lam = lam[ot[own]]; self.w = w_modal[ot[own]]
        self.n_terms = len(own); self.n_out = len(out); self.n_home_terms = len(hrep)
        self.n_types_valid = int(valid.sum()); self.lam0 = lam0
        self.page_len = page_len

    def costs(self, orders):
        n = orders.shape[0]
        res = {"burst": np.zeros(n), "blog": np.zeros(n), "modal": np.zeros(n)}
        if self.n_terms == 0:
            return res
        wsum = max(self.w.sum(), 1e-9)
        for c0 in range(0, n, CHUNK):
            off = offsets(orders[c0:c0 + CHUNK], self.page_len)
            d = np.abs(off[:, self.pa_page] + self.pa_idx - off[:, self.pb_page] - self.pb_idx)
            dmin = np.minimum.reduceat(d, self.seg, axis=1)
            res["burst"][c0:c0 + CHUNK] = (dmin / self.lam).sum(axis=1) / self.n_terms
            res["blog"][c0:c0 + CHUNK] = np.log1p(dmin / self.lam).sum(axis=1) / self.n_terms
            res["modal"][c0:c0 + CHUNK] = (dmin * self.w).sum(axis=1) / wsum
        return res


def cross_pairs(items_per_page, kmax=KMAX):
    pos = defaultdict(list)
    for p, items in enumerate(items_per_page):
        for i, w in enumerate(items):
            pos[w].append((p, i))
    pa, ia, pb, ib = [], [], [], []
    for occ in pos.values():
        if 2 <= len(occ) <= kmax:
            for (p1, i1), (p2, i2) in itertools.combinations(occ, 2):
                if p1 != p2:
                    pa.append(p1); ia.append(i1); pb.append(p2); ib.append(i2)
    return tuple(np.array(x) for x in (pa, ia, pb, ib))


def l1_costs(pairs, orders, page_len):
    pa, ia, pb, ib = pairs
    n = orders.shape[0]
    out = np.zeros(n)
    if len(pa) == 0:
        return out
    for c0 in range(0, n, CHUNK):
        off = offsets(orders[c0:c0 + CHUNK], page_len)
        out[c0:c0 + CHUNK] = np.abs(off[:, pa] + ia - off[:, pb] - ib).mean(axis=1)
    return out


def all_costs(items, sheet_of, S, cand, ref_stacked, inv=None):
    """cost vectors over cand for the content assignment inv (None = real)."""
    ot, op, oi, n_types, page_len = occurrences(items)
    P = len(page_len)
    if inv is None:
        inv = np.arange(P)
    perm = np.argsort(inv)
    orders = perm[cand]  # content pages read in slot order cand
    b = NestBurst(ot, op, oi, n_types, page_len, inv, sheet_of, S, perm[ref_stacked])
    c = b.costs(orders)
    c["L1"] = l1_costs(cross_pairs(items), orders, page_len)
    return c, b


G = {}


def _stats(v, flags):
    """reduced statistics of one cost vector over the full candidate set."""
    sd = float(v.std()); mu = float(v.mean())
    ib = int(v.argmin()); ist = int(np.where(flags["stacked"])[0][v[flags["stacked"]].argmin()]); ine = int(np.where(flags["nested"])[0][v[flags["nested"]].argmin()])
    ibnd = flags["bound_i"]
    return {"best_i": ib, "best_val": float(v[ib]), "z_best": (v[ib] - mu) / sd, "best_nblocks": int(flags["nblocks"][ib]),
            "stacked_i": ist, "stacked_val": float(v[ist]), "z_stacked": (v[ist] - mu) / sd, "stacked_rank": int((v < v[ist]).sum() + 1),
            "nested_i": ine, "nested_val": float(v[ine]), "z_nested": (v[ine] - mu) / sd, "nested_rank": int((v < v[ine]).sum() + 1),
            "bound_val": float(v[ibnd]), "z_bound": (v[ibnd] - mu) / sd, "bound_rank": int((v < v[ibnd]).sum() + 1),
            "gain_over_stacked_sd": (v[ist] - v[ib]) / sd, "nested_minus_stacked_sd": (v[ine] - v[ist]) / sd,
            "top10_nblocks": [int(x) for x in flags["nblocks"][np.argsort(v)[:10]]],
            "best_by_nblocks": {int(k): float(v[flags["nblocks"] == k].min()) for k in np.unique(flags["nblocks"])},
            "mean": mu, "sd": sd}


def _shuffle_job(seed):
    rng = np.random.default_rng(seed)
    P = len(G["sheet_of"])
    inv = np.argsort(rng.permutation(P))
    out = {}
    for name in UNITS:
        c, _ = all_costs(G["items"][name], G["sheet_of"], G["S"], G["cand"], G["ref_stacked"], inv)
        out[name] = {m: _stats(c[m], G["flags"]) for m in METRICS}
    return out


def _control_job(args):
    tname, start, mode, src_w, order, true_label = args
    items = lay_text(src_w, G["wlen"], order)  # word stream only (the prose has no glyph n-grams)
    ti = G["labels"].index(true_label)
    out = {}
    for name in ("words",):
        c, _ = all_costs(items, G["sheet_of"], G["S"], G["cand"], G["ref_stacked"])
        out[name] = {}
        for m in METRICS:
            v = c[m]; o = np.argsort(v)
            out[name][m] = {"rank": int((v < v[ti]).sum() + 1), "z_true": float((v[ti] - v.mean()) / v.std()), "best": G["labels"][int(o[0])],
                            "best_nblocks": int(G["nblocks"][o[0]]), "stacked_rank_of_best_stacked": int((v < v[G["flags"]["stacked"]].min()).sum() + 1),
                            "same_partition": _partition(G["labels"][int(o[0])]) == _partition(true_label)}
    return f"{tname}/w{start}/{mode}", out


def _partition(label):
    return tuple(sorted(tuple(sorted(b)) for b in label.split("|")))


def analyse(tag, pq, tq, n_shuf, n_workers, rng, res):
    units = build_sheets(pq)
    S, P = len(units), len(pq)
    sheet_of = np.zeros(P, dtype=int)
    for s, u in enumerate(units):
        for p in u["a"] + u["b"]:
            sheet_of[p] = s
    cand, labels, nblocks = nesting_orders(units)
    bound_label = "".join(str(s + 1) for s in range(S)); bound_i = labels.index(bound_label)
    assert (cand[bound_i] == np.arange(P)).all(), "as-bound order is not file order"
    flags = {"stacked": nblocks == S, "nested": nblocks == 1, "nblocks": nblocks, "bound_i": bound_i}
    ref_stacked = cand[np.where(flags["stacked"])[0][0]]
    items, wlen = units_of(pq, tq)
    G.update(items=items, sheet_of=sheet_of, S=S, cand=cand, ref_stacked=ref_stacked, flags=flags, labels=labels, nblocks=nblocks, wlen=wlen)
    print(f"== {tag}: {S} sheets, {P} pages, {len(labels)} distinct nesting patterns ({int(flags['stacked'].sum())} stacked, {int(flags['nested'].sum())} nested)", flush=True)
    t0 = time.time()
    real = {}
    for name in UNITS:
        c, b = all_costs(items[name], sheet_of, S, cand, ref_stacked)
        real[name] = {m: _stats(c[m], flags) for m in METRICS}
        real[name]["n_types_valid"] = b.n_types_valid; real[name]["n_outliers"] = b.n_out; real[name]["n_home_terms"] = b.n_home_terms
        real[name]["vec"] = {m: c[m] for m in METRICS}
    print(f"  real costs in {time.time() - t0:.0f}s", flush=True)
    seeds = [int(rng.integers(1 << 30)) for _ in range(n_shuf)]
    t0 = time.time()
    with Pool(n_workers) as pool:
        shuf = pool.map(_shuffle_job, seeds, chunksize=1)
    print(f"  {n_shuf} shuffles in {time.time() - t0:.0f}s", flush=True)
    out = {"S": S, "P": P, "n_patterns": len(labels)}
    for name in UNITS:
        r = {"n_types_valid": real[name]["n_types_valid"], "n_outliers": real[name]["n_outliers"], "n_home_terms": real[name]["n_home_terms"]}
        for m in METRICS:
            s = real[name][m]; v = real[name]["vec"][m]
            nb = np.array([x[name][m]["best_val"] for x in shuf]); ng = np.array([x[name][m]["gain_over_stacked_sd"] for x in shuf])
            nns = np.array([x[name][m]["nested_minus_stacked_sd"] for x in shuf]); nzb = np.array([x[name][m]["z_best"] for x in shuf])
            nbl = np.array([x[name][m]["best_nblocks"] for x in shuf]); nst = np.array([x[name][m]["stacked_val"] for x in shuf])
            nne = np.array([x[name][m]["nested_val"] for x in shuf])
            r[m] = {k: (float(val) if isinstance(val, (np.floating, float)) else val) for k, val in s.items()}
            r[m].update({"best": labels[s["best_i"]], "best_stacked": labels[s["stacked_i"]], "best_nested": labels[s["nested_i"]], "top10": [labels[i] for i in np.argsort(v)[:10]],
                         "p_best": float(((nb <= s["best_val"]).sum() + 1) / (n_shuf + 1)), "p_best_stacked": float(((nst <= s["stacked_val"]).sum() + 1) / (n_shuf + 1)),
                         "p_best_nested": float(((nne <= s["nested_val"]).sum() + 1) / (n_shuf + 1)),
                         "p_gain": float(((ng >= s["gain_over_stacked_sd"]).sum() + 1) / (n_shuf + 1)), "null_gain_mean": float(ng.mean()), "null_gain_q": [float(q) for q in np.quantile(ng, [0.05, 0.5, 0.95])],
                         "p_nested_minus_stacked": float(((nns <= s["nested_minus_stacked_sd"]).sum() + 1) / (n_shuf + 1)), "null_nms_q": [float(q) for q in np.quantile(nns, [0.05, 0.5, 0.95])],
                         "null_z_best_q": [float(q) for q in np.quantile(nzb, [0.05, 0.5, 0.95])], "p_z_best": float(((nzb <= s["z_best"]).sum() + 1) / (n_shuf + 1)),
                         "null_best_nblocks_hist": {int(k): int((nbl == k).sum()) for k in range(1, S + 1)}})
            r[m].pop("best_i"); r[m].pop("stacked_i"); r[m].pop("nested_i")
        out[name] = r
        print(f"  {name:5s} valid {r['n_types_valid']:5d} out {r['n_outliers']:5d} home-terms {r['n_home_terms']:5d}", flush=True)
        for m in METRICS:
            x = r[m]
            print(f"      {m:5s} best {x['best']:>14s} ({x['best_nblocks']} blk) z {x['z_best']:+.2f} p {x['p_best']:.3f} | best stacked {x['best_stacked']:>12s} rank {x['stacked_rank']:5d} z {x['z_stacked']:+.2f} | best nested {x['best_nested']:>8s} rank {x['nested_rank']:5d} z {x['z_nested']:+.2f} | bound rank {x['bound_rank']:5d} z {x['z_bound']:+.2f} | gain full-vs-stacked {x['gain_over_stacked_sd']:.2f} sd (null {x['null_gain_q'][1]:.2f} [{x['null_gain_q'][0]:.2f},{x['null_gain_q'][2]:.2f}] p {x['p_gain']:.3f}) | nested-stacked {x['nested_minus_stacked_sd']:+.2f} sd (null {x['null_nms_q'][1]:+.2f} p {x['p_nested_minus_stacked']:.3f}) | top10 blocks {x['top10_nblocks']} | null best-blocks {x['null_best_nblocks_hist']}", flush=True)
    res.update(out)
    return units, cand, labels, nblocks, flags, wlen, sheet_of, S


def controls(units, cand, labels, nblocks, flags, wlen, sheet_of, S, n_workers, rng):
    cc = DATA_ROOT / "external/voynich-attack/corpora/latin/CorpusCorporum/auctores_scientiarum_varii"
    dta = DATA_ROOT / "external/voynich-attack/corpora/german/DTA"
    texts = {
        "la_isidorus": load_csv_text(cc / "isidorus_hispalensis/etymologiae/etymologiae.csv"),
        "de_bullinger": load_csv_text(dta / "1558_bullinger_haussbuoch/1558_bullinger_haussbuoch.csv"),
        "it_decameron": load_raw_text(DATA_ROOT / "raw/italian/boccaccio_decameron.txt"),
    }
    bound_label = "".join(str(s + 1) for s in range(S))
    jobs = []
    for tname, src in texts.items():
        for start in (0, 1, 2):
            src_w = src[start * 20000: start * 20000 + int(wlen.sum()) + 5]
            perm = rng.permutation(S) + 1
            nested_rand = "".join(str(s) for s in perm)
            cut = int(rng.integers(1, S))
            two_block = "".join(str(s) for s in perm[:cut]) + "|" + "".join(str(s) for s in perm[cut:])
            stacked_rand = "|".join(str(s) for s in perm)
            for mode, lab in (("nested_bound", bound_label), ("nested_rand", nested_rand), ("two_block", two_block), ("stacked_rand", stacked_rand)):
                seq = order_from_label(lab, units)  # look up by page sequence: a label may have been renamed by the dedup
                idx = int(np.where((cand == seq).all(axis=1))[0][0])
                jobs.append((tname, start, mode, src_w, cand[idx], labels[idx]))
    with Pool(n_workers) as pool:
        results = pool.map(_control_job, jobs, chunksize=1)
    out = {}
    tally = {mode: {m: {"rank": [], "same_partition": [], "best_nblocks": []} for m in METRICS} for mode in ("nested_bound", "nested_rand", "two_block", "stacked_rand")}
    for (key, r), job in zip(results, jobs):
        out[key] = {"true": job[5], **r}
        mode = job[2]
        for name in ("words",):
            for m in METRICS:
                x = r[name][m]
                tally[mode][m]["rank"].append(x["rank"]); tally[mode][m]["same_partition"].append(x["same_partition"]); tally[mode][m]["best_nblocks"].append(x["best_nblocks"])
        print(f"  control {key:34s} true {job[5]:>14s}: " + " | ".join(f"{name} " + " ".join(f"{m} r{r[name][m]['rank']}{'*' if r[name][m]['same_partition'] else ''}" for m in METRICS) for name in ("words",)), flush=True)
    summ = {}
    for mode, t in tally.items():
        summ[mode] = {}
        for m in METRICS:
            rk = np.array(t[m]["rank"]); sp = np.array(t[m]["same_partition"]); nb = np.array(t[m]["best_nblocks"])
            summ[mode][m] = {"n": int(len(rk)), "rank1": int((rk == 1).sum()), "top10": int((rk <= 10).sum()), "median_rank": float(np.median(rk)), "same_partition": int(sp.sum()), "best_nblocks_median": float(np.median(nb))}
        print(f"  control summary {mode:13s} (words): " + " | ".join(f"{m}: rank1 {s['rank1']}/{s['n']} top10 {s['top10']} median {s['median_rank']:.0f} same-partition {s['same_partition']} best-blocks median {s['best_nblocks_median']:.0f}" for m, s in summ[mode].items()), flush=True)
    out["summary"] = summ
    return out


def main():
    Q = sys.argv[1] if len(sys.argv) > 1 else "M"
    n_shuf = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    rng = np.random.default_rng(7)
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
        units, cand, labels, nblocks, flags, wlen, sheet_of, S = analyse(f"{tr} quire {Q}", pq, tq, n_shuf, n_workers, rng, res[tr])
        if tr == "IT2a":
            res["controls"] = controls(units, cand, labels, nblocks, flags, wlen, sheet_of, S, n_workers, rng)
    (OUT / f"quire_order_nesting_{Q}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
