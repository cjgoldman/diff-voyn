"""Derive quire-T quantities not stored in the recorded JSON (candidate cost vectors,
null cost vectors, sheet affinity) by re-running the recorded analysis functions from
scripts/quire_order_poc.py on the raw transcriptions.  Output:
../data/derived_quire_T_costs.json.  Real-contents values are deterministic; the null
uses 200 content shuffles with a fresh seed, so null quantiles differ slightly from the
recorded quire_order_nullshape_T.json (compare there)."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/scripts")
from doubleton_gaps import DATA_ROOT, load_vms  # noqa
from quire_order_poc import build_sheets, rare_pairs, metrics, stacked_orders, units_of, sheet_affinity, UNITS  # noqa

Q = "T"; N_SHUF = 200
out = {}
rng = np.random.default_rng(11)
for tr, fname in (("IT2a", "IT2a-n.txt"), ("RF1b", "RF1b-e.txt")):
    toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / fname)
    pages_q = [p for p in pages if p["quire"] == Q]
    gidx = {p["page_idx"]: k for k, p in enumerate(pages_q)}
    pq = []
    for p in pages_q:
        q = dict(p); q["page_idx"] = gidx[p["page_idx"]]; pq.append(q)
    tq = [{"w": t["w"], "page_idx": gidx[t["page_idx"]]} for t in toks if t["page_idx"] in gidx]
    units = build_sheets(pq); S, P = len(units), len(pq)
    sheet_of = np.zeros(P, dtype=int)
    for s, u in enumerate(units):
        for p in u["a"] + u["b"]:
            sheet_of[p] = s
    cand, labels = stacked_orders(units)
    items, wlen = units_of(pq, tq)
    perms = [rng.permutation(P) for _ in range(N_SHUF)]
    r = {"sheets": [[pq[i]["page"] for i in u["a"] + u["b"]] for u in units], "page_len": wlen.tolist(), "labels": labels, "units": {}}
    for name in UNITS:
        t0 = time.time()
        pairs = rare_pairs(items[name], sheet_of)
        bt = pairs["between"]
        L1b, _ = metrics(pairs, cand, bt)
        nulls = []
        for perm in perms:
            inv = np.argsort(perm)
            bt_p = sheet_of[inv[pairs["pa"]]] != sheet_of[inv[pairs["pb"]]]
            v, _ = metrics(pairs, perm[cand], bt_p)
            nulls.append(v)
        nulls = np.array(nulls)
        zb = (nulls.min(axis=1) - nulls.mean(axis=1)) / nulls.std(axis=1)
        obs, exp = sheet_affinity(pairs, sheet_of, S, perms)
        r["units"][name] = {
            "n_types": int(pairs["n_types"]), "n_between": int(bt.sum()),
            "L1_between": L1b.tolist(),
            "null_best_z": zb.tolist(),
            "null_cand_z_sample": ((nulls[:20] - nulls[:20].mean(axis=1, keepdims=True)) / nulls[:20].std(axis=1, keepdims=True)).tolist(),
            "affinity_obs": obs.tolist(), "affinity_exp": exp.tolist(),
        }
        print(tr, name, f"{time.time()-t0:.1f}s best {labels[int(np.argmin(L1b))]} z {(L1b.min()-L1b.mean())/L1b.std():+.2f}", flush=True)
    out[tr] = r
Path("../data/derived_quire_T_costs.json").write_text(json.dumps(out))
print("done")
