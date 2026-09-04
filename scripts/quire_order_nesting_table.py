"""Markdown table for docs/doubleton_gaps.md §19 from quire_order_nesting_<Q>.json (L1 metric)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from doubleton_gaps import OUT

UNITS = ("words", "n5", "n6", "n7", "n8")
rows = []
for Q in sys.argv[1:] or ["A", "B", "C", "M", "T"]:
    f = OUT / f"quire_order_nesting_{Q}.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    for tr in ("IT2a", "RF1b"):
        d = r[tr]; N = d["n_patterns"]; S = d["S"]
        cells = [d[u]["L1"] for u in UNITS]
        best = " / ".join(f"{x['best']}" for x in cells)
        blk = "/".join(str(x["best_nblocks"]) for x in cells)
        gain = "–".join(f"{v:.2f}" for v in sorted([min(x["gain_over_stacked_sd"] for x in cells), max(x["gain_over_stacked_sd"] for x in cells)]))
        pg = "–".join(f"{v:.2f}" for v in sorted([min(x["p_gain"] for x in cells), max(x["p_gain"] for x in cells)]))
        st = "–".join(str(v) for v in sorted({min(x["stacked_rank"] for x in cells), max(x["stacked_rank"] for x in cells)}))
        ne = "–".join(str(v) for v in sorted({min(x["nested_rank"] for x in cells), max(x["nested_rank"] for x in cells)}))
        bd = "–".join(str(v) for v in sorted({min(x["bound_rank"] for x in cells), max(x["bound_rank"] for x in cells)}))
        nms = "–".join(f"{v:+.1f}" for v in sorted([min(x["nested_minus_stacked_sd"] for x in cells), max(x["nested_minus_stacked_sd"] for x in cells)]))
        ntot = sum(cells[0]["null_best_nblocks_hist"].values())
        pc = "–".join(f"{v:.2f}" for v in sorted([min(1 - x["p_nested_minus_stacked"] + 1 / (ntot + 1) for x in cells), max(1 - x["p_nested_minus_stacked"] + 1 / (ntot + 1) for x in cells)]))
        rows.append(f"| {Q} ({S} sheets, {N}) | {tr} | {blk} | {gain} ({pg}) | {st} | {ne} | {bd} | {nms} ({pc}) |")
print("| quire (sheets, patterns) | transcr. | blocks in winner, words/n5/n6/n7/n8 | gain of full space over best stacked, sd (p) | rank of best stacked | rank of best fully nested | rank of binding | best nested − best stacked, sd (P chance) |")
print("|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
