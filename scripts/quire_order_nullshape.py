"""Shape of the null distributions behind §13-16 (2026-09-03).

For one quire: (a) the candidate cost vector over all stacked orders - skewness,
excess kurtosis, Shapiro-Wilk p - on the real contents and on shuffled contents;
(b) the null distribution of the winner's within-candidate z (best - mean)/sd on
shuffled contents: quantiles and the empirical p of the real z (replaces the
Gaussian 'random minimum ~ -2.5 sd' heuristic); (c) the null distribution of the
direction gap delta = cost(reversal) - cost(best): quantiles, share at < 0.1 sd,
and the empirical p of the real delta.
Outputs DATA_ROOT/analysis/doubleton_gaps/quire_order_nullshape_<Q>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from doubleton_gaps import DATA_ROOT, OUT, load_vms  # noqa: E402
from order_optimize import build_sheets  # noqa: E402
from quire_order_burst import UNITS, all_costs  # noqa: E402
from quire_order_poc import stacked_orders, units_of  # noqa: E402

METRICS = ("L1", "modal")


def q(a):
    return {k: float(v) for k, v in zip(("min", "q05", "q25", "q50", "q75", "q95", "max"), np.quantile(a, [0, .05, .25, .5, .75, .95, 1]))}


def main():
    Q = sys.argv[1] if len(sys.argv) > 1 else "M"
    n_shuf = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    rng = np.random.default_rng(9)
    res = {}
    for tr, fname in (("IT2a", "IT2a-n.txt"),):
        toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / fname)
        pages_q = [p for p in pages if p["quire"] == Q]
        gidx = {p["page_idx"]: k for k, p in enumerate(pages_q)}
        pq = []
        for p in pages_q:
            x = dict(p); x["page_idx"] = gidx[p["page_idx"]]; pq.append(x)
        tq = [{"w": t["w"], "page_idx": gidx[t["page_idx"]]} for t in toks if t["page_idx"] in gidx]
        units = build_sheets(pq)
        S, P = len(units), len(pq)
        sheet_of = np.zeros(P, dtype=int)
        for s, u in enumerate(units):
            for p in u["a"] + u["b"]:
                sheet_of[p] = s
        cand, labels = stacked_orders(units)
        rev_i = np.array([labels.index(l[::-1]) for l in labels])
        items, _ = units_of(pq, tq)
        invs = [np.argsort(rng.permutation(P)) for _ in range(n_shuf)]
        print(f"== {tr} quire {Q}: {len(labels)} candidate orders", flush=True)
        for name in UNITS:
            c, _ = all_costs(items[name], sheet_of, S, cand)
            null = {m: {"z": [], "delta": [], "skew": [], "kurt": [], "shapiro": []} for m in METRICS}
            for k, inv in enumerate(invs):
                cn, _ = all_costs(items[name], sheet_of, S, cand, inv)
                for m in METRICS:
                    v = cn[m]; i = int(v.argmin()); sd = v.std()
                    null[m]["z"].append((v[i] - v.mean()) / sd)
                    null[m]["delta"].append((v[rev_i[i]] - v[i]) / sd)
                    if k < 50:
                        null[m]["skew"].append(stats.skew(v)); null[m]["kurt"].append(stats.kurtosis(v))
                        null[m]["shapiro"].append(stats.shapiro(v if len(v) <= 5000 else v[:5000]).pvalue)
            r = {}
            for m in METRICS:
                v = c[m]; i = int(v.argmin()); sd = v.std()
                z_real = (v[i] - v.mean()) / sd; d_real = (v[rev_i[i]] - v[i]) / sd
                nz = np.array(null[m]["z"]); nd = np.array(null[m]["delta"])
                r[m] = {"real": {"skew": float(stats.skew(v)), "kurt": float(stats.kurtosis(v)), "shapiro_p": float(stats.shapiro(v).pvalue), "z_best": float(z_real), "delta": float(d_real)},
                        "null_candidates": {"skew_median": float(np.median(null[m]["skew"])), "kurt_median": float(np.median(null[m]["kurt"])), "shapiro_p_median": float(np.median(null[m]["shapiro"])), "shapiro_frac_lt_05": float(np.mean(np.array(null[m]["shapiro"]) < 0.05))},
                        "null_zbest": {**q(nz), "p_real": float(((nz <= z_real).sum() + 1) / (n_shuf + 1)), "gauss_min_expect": float(stats.norm.ppf(1 / (len(labels) + 1)))},
                        "null_delta": {**q(nd), "frac_lt_0.1": float(np.mean(nd < 0.1)), "skew": float(stats.skew(nd)), "p_real": float(((nd >= d_real).sum() + 1) / (n_shuf + 1)), "mean_pm_sd": [float(nd.mean()), float(nd.std())]}}
                a = r[m]
                print(f"  {name:5s} {m:5s} | candidates real: skew {a['real']['skew']:+.2f} kurt {a['real']['kurt']:+.2f} shapiro p {a['real']['shapiro_p']:.3f}; null: skew {a['null_candidates']['skew_median']:+.2f} kurt {a['null_candidates']['kurt_median']:+.2f} shapiro p<.05 in {a['null_candidates']['shapiro_frac_lt_05']:.0%} | winner z: real {a['real']['z_best']:+.2f}; null q05/q50/q95 {a['null_zbest']['q05']:+.2f}/{a['null_zbest']['q50']:+.2f}/{a['null_zbest']['q95']:+.2f} (gauss {a['null_zbest']['gauss_min_expect']:+.2f}) p {a['null_zbest']['p_real']:.3f} | delta: real {a['real']['delta']:+.2f}; null min/q25/q50/q75/max {a['null_delta']['min']:.2f}/{a['null_delta']['q25']:.2f}/{a['null_delta']['q50']:.2f}/{a['null_delta']['q75']:.2f}/{a['null_delta']['max']:.2f} skew {a['null_delta']['skew']:+.1f} p {a['null_delta']['p_real']:.3f}", flush=True)
            res[name] = r
    (OUT / f"quire_order_nullshape_{Q}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
