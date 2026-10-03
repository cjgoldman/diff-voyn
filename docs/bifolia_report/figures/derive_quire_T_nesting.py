"""Derive the full per-pattern L1 cost vectors over every nesting pattern of quire T's
six sheets (record §19), which the recorded quire_order_nesting_T.json reduces to ranks
and best-per-class values.  Re-runs the recorded functions of scripts/quire_order_nesting.py
on the raw transcriptions (deterministic, no null).  Output:
../data/derived_quire_T_nesting.json — per transcription: nblocks per pattern, the index of
the binding, and per unit the L1 cost of every pattern; asserts that the stacked / nested /
bound ranks reproduce the recorded ones.

    cd docs/bifolia_report/figures && uv run python derive_quire_T_nesting.py
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/scripts")
from doubleton_gaps import DATA_ROOT, load_vms  # noqa
from order_optimize import build_sheets  # noqa
from quire_order_nesting import nesting_orders, cross_pairs, l1_costs  # noqa
from quire_order_burst import occurrences  # noqa
from quire_order_poc import UNITS, units_of  # noqa

HERE = Path(__file__).resolve().parent
Q = "T"
REC = json.load(open(HERE / ".." / "data" / "quire_order_nesting_T.json"))
out = {}
for tr, fname in (("IT2a", "IT2a-n.txt"), ("RF1b", "RF1b-e.txt")):
    toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / fname)
    pages_q = [p for p in pages if p["quire"] == Q]
    gidx = {p["page_idx"]: k for k, p in enumerate(pages_q)}
    pq = []
    for p in pages_q:
        q = dict(p); q["page_idx"] = gidx[p["page_idx"]]; pq.append(q)
    tq = [{"w": t["w"], "page_idx": gidx[t["page_idx"]]} for t in toks if t["page_idx"] in gidx]
    units = build_sheets(pq); S, P = len(units), len(pq)
    cand, labels, nblocks = nesting_orders(units)
    bound_i = labels.index("".join(str(s + 1) for s in range(S)))
    assert (cand[bound_i] == np.arange(P)).all()
    items, wlen = units_of(pq, tq)
    r = {"S": S, "P": P, "n_patterns": len(labels), "nblocks": nblocks.tolist(), "bound_i": bound_i, "units": {}}
    for name in UNITS:
        t0 = time.time()
        _, _, _, _, page_len = occurrences(items[name])
        v = l1_costs(cross_pairs(items[name]), cand, page_len)
        rec = REC[tr][name]["L1"]
        st = nblocks == S; ne = nblocks == 1
        ranks = {"stacked_rank": int((v < v[st].min()).sum() + 1), "nested_rank": int((v < v[ne].min()).sum() + 1),
                 "bound_rank": int((v < v[bound_i]).sum() + 1)}
        for k, val in ranks.items():
            assert val == rec[k], (tr, name, k, val, rec[k])
        assert abs(v.mean() - rec["mean"]) < 1e-6 * rec["mean"] and abs(v.std() - rec["sd"]) < 1e-6 * rec["sd"]
        r["units"][name] = {"L1": [round(float(x), 3) for x in v], "best_label": labels[int(v.argmin())],
                            "best_stacked_label": labels[int(np.where(st)[0][v[st].argmin()])]}
        print(tr, name, f"{time.time()-t0:.1f}s", ranks, "ok", flush=True)
    out[tr] = r
(HERE / ".." / "data" / "derived_quire_T_nesting.json").write_text(json.dumps(out))
print("done")
