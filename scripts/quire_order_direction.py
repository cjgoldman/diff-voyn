"""Direction of the best stacked order (2026-09-03).

quire_order_burst.py reported orders up to reversal.  Reversing the sheet order
does NOT reverse the page sequence (each sheet still reads a-r, a-v, b-r, b-v), so
an order and its reversal have different costs: for a type at within-sheet
position x in sheet A and y in sheet B, AB costs (L_A - x) + y and BA costs
(L_B - y) + x.  Here: the un-folded argmin, the cost of its reversal, the gap
delta = cost(reversed) - cost(best) in candidate sd, and a null for delta from
content shuffles (each shuffle's own best order and its reversal, same selection).
Outputs DATA_ROOT/analysis/doubleton_gaps/quire_order_direction_<Q>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from doubleton_gaps import DATA_ROOT, OUT, load_vms  # noqa: E402
from order_optimize import build_sheets  # noqa: E402
from quire_order_burst import METRICS, UNITS, all_costs  # noqa: E402
from quire_order_poc import stacked_orders, units_of  # noqa: E402


def main():
    Q = sys.argv[1] if len(sys.argv) > 1 else "T"
    n_shuf = int(sys.argv[2]) if len(sys.argv) > 2 else 200
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
        units = build_sheets(pq)
        S, P = len(units), len(pq)
        sheet_of = np.zeros(P, dtype=int)
        for s, u in enumerate(units):
            for p in u["a"] + u["b"]:
                sheet_of[p] = s
        cand, labels = stacked_orders(units)
        rev_i = {i: labels.index(l[::-1]) for i, l in enumerate(labels)}
        items, _ = units_of(pq, tq)
        invs = [np.argsort(rng.permutation(P)) for _ in range(n_shuf)]
        res[tr] = {}
        print(f"== {tr} quire {Q}", flush=True)
        for name in UNITS:
            c, _ = all_costs(items[name], sheet_of, S, cand)
            nulls = {m: [] for m in METRICS}
            for inv in invs:
                cn, _ = all_costs(items[name], sheet_of, S, cand, inv)
                for m in METRICS:
                    v = cn[m]; i = int(v.argmin()); nulls[m].append((v[rev_i[i]] - v[i]) / v.std())
            r = {}
            for m in METRICS:
                v = c[m]; i = int(v.argmin()); j = rev_i[i]
                delta = float((v[j] - v[i]) / v.std())
                rank_rev = int((v < v[j]).sum() + 1)
                nd = np.array(nulls[m])
                r[m] = {"best": labels[i], "reversed": labels[j], "delta_sd": delta, "reversed_rank": rank_rev, "null_delta_mean": float(nd.mean()), "null_delta_sd": float(nd.std()), "p_delta": float(((nd >= delta).sum() + 1) / (n_shuf + 1))}
            res[tr][name] = r
            print(f"  {name:5s} " + " | ".join(f"{m}: {r[m]['best']} (rev {r[m]['reversed']} rank {r[m]['reversed_rank']:3d}) Δ {r[m]['delta_sd']:+.2f} sd, null {r[m]['null_delta_mean']:.2f}±{r[m]['null_delta_sd']:.2f}, p {r[m]['p_delta']:.3f}" for m in METRICS), flush=True)
    (OUT / f"quire_order_direction_{Q}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
