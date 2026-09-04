"""Summarise quire_order_nesting_<Q>.json (2026-09-03): per quire x transcription x unit, the
winning nesting pattern, the ranks of the best stacked / best nested / as-bound orders, the
selection-corrected p-values, and the prose controls (words only)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from doubleton_gaps import OUT  # noqa: E402

UNITS = ("words", "n5", "n6", "n7", "n8")
MODES = ("nested_bound", "nested_rand", "two_block", "stacked_rand")


def main(metric="L1"):
    quires = sys.argv[1:] or ["T", "M", "C", "A", "B"]
    print(f"metric {metric}: best pattern (blocks) p_best | rank of best stacked / best nested / as-bound among N | P(null nested class trails stacked >= observed) | null share of fully-stacked winners")
    for Q in quires:
        f = OUT / f"quire_order_nesting_{Q}.json"
        if not f.exists():
            print(f"-- {Q}: missing"); continue
        r = json.loads(f.read_text())
        for tr in ("IT2a", "RF1b"):
            d = r[tr]; N = d["n_patterns"]; S = d["S"]
            for u in UNITS:
                x = d[u][metric]
                nst = x["null_best_nblocks_hist"].get(str(S), 0); ntot = sum(x["null_best_nblocks_hist"].values())
                print(f"  {Q} {tr} {u:5s} best {x['best']:>14s} ({x['best_nblocks']}/{S} blk) p {x['p_best']:.3f} | stacked {x['stacked_rank']:5d} nested {x['nested_rank']:5d} bound {x['bound_rank']:5d} /{N} | gain vs stacked {x['gain_over_stacked_sd']:.2f} sd (p {x['p_gain']:.2f}) | nested-stacked {x['nested_minus_stacked_sd']:+.2f} sd, P(chance) {1 - x['p_nested_minus_stacked'] + 1 / (ntot + 1):.3f} | null stacked-winner share {nst / ntot:.2f}")
        c = r["controls"]["summary"]
        line = []
        for mode in MODES:
            s = c[mode][metric]
            n = s["n"]; k = 5 if n == 45 else 1  # first T run tallied the word stream under all five unit labels
            line.append(f"{mode} rank1 {s['rank1'] // k}/{n // k} top10 {s['top10'] // k} median {s['median_rank']:.0f} same-partition {s['same_partition'] // k}")
        print(f"  {Q} controls ({metric}, words): " + " | ".join(line))


if __name__ == "__main__":
    main()
    print()
    main("blog")
