"""Re-derive the two real quire-T words behind the executive summary's metric figure
(fig_es_metric) into data/derived_burst_example.json.

The burst-scaled seriation cost (record §14; report eq. burst / lambda) is illustrated on the
word unit of quire T (IT2a): for every word used 2-10 times in the quire, its home sheet is
the sheet holding most of its uses, its burst scale is the mean gap between consecutive
home-sheet uses in the stacked reading of that sheet (pages in folio order), shrunk toward the
median over types by one pseudo-gap and floored at 30, exactly as scripts/quire_order_burst.py
does; each use outside the home sheet costs its distance to the nearest home use divided by the
burst scale.  Two words with three home uses and one stray are stored: the one with the
smallest burst scale (qokaly) and the one with the largest (chckhedy), with sheet lengths so
that the figure can draw the stray's sheet read directly after the home sheet, or with one
more sheet in between.

    cd docs/bifolia_report/figures && uv run python derive_burst_example.py

Needs DATA_ROOT (/workspace/data); seconds of CPU.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scripts"))
from doubleton_gaps import DATA_ROOT, load_vms  # noqa: E402

OUT = os.path.join(HERE, "..", "data", "derived_burst_example.json")
QUIRE = "T"
FLOOR = 30.0
KMAX = 10


def main():
    toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / "IT2a-n.txt")
    pq = [p for p in pages if p["quire"] == QUIRE]
    ids = {p["page_idx"]: p for p in pq}
    sheet_of = {p["page_idx"]: p["bifolio"] for p in pq}
    sheets = sorted(set(sheet_of.values()))
    per = defaultdict(list)
    for t in toks:
        if t["page_idx"] in ids:
            per[t["page_idx"]].append(t["w"])
    plen = {p: len(per[p]) for p in ids}
    # stacked reading of each sheet: its pages in folio order
    sheet_pages = {s: [p for p in sorted(ids) if sheet_of[p] == s] for s in sheets}
    sheet_len = {s: sum(plen[p] for p in sheet_pages[s]) for s in sheets}
    sheet_folios = {s: f"f{ids[sheet_pages[s][0]]['page'][1:-1]}/f{ids[sheet_pages[s][-1]]['page'][1:-1]}" for s in sheets}
    # position of each token within its own sheet's stacked reading
    off_in_sheet = {}
    for s in sheets:
        c = 0
        for p in sheet_pages[s]:
            off_in_sheet[p] = c
            c += plen[p]
    pos = defaultdict(list)
    for p in sorted(ids):
        for i, w in enumerate(per[p]):
            pos[w].append((p, i))

    rows, raw = [], []
    for w, occ in pos.items():
        if not 2 <= len(occ) <= KMAX:
            continue
        cnt = defaultdict(int)
        for p, _ in occ:
            cnt[sheet_of[p]] += 1
        srt = sorted(cnt.values())
        if len(srt) > 1 and srt[-1] == srt[-2]:
            continue  # tied home sheet: skipped by the metric
        home = max(cnt, key=cnt.get)
        hp = sorted(off_in_sheet[p] + i for p, i in occ if sheet_of[p] == home)
        g = np.diff(hp)
        if len(g):
            raw.append(float(g.mean()))
        rows.append((w, home, hp, g, [(p, i) for p, i in occ if sheet_of[p] != home]))
    lam0 = float(np.median(raw))

    examples = []
    for w, home, hp, g, strays in rows:
        if len(hp) != 3 or len(strays) != 1:
            continue
        lam = max(FLOOR, (float(g.sum()) + lam0) / (len(g) + 1))
        p, i = strays[0]
        s = sheet_of[p]
        stray_off = off_in_sheet[p] + i
        d_after = sheet_len[home] - hp[-1] + stray_off  # stray's sheet read directly after home
        examples.append({
            "word": w, "home_sheet": home, "home_folios": sheet_folios[home], "home_len": sheet_len[home],
            "home_positions": [int(x) for x in hp], "home_gaps": [int(x) for x in g],
            "lambda": lam, "stray_sheet": s, "stray_folios": sheet_folios[s], "stray_len": sheet_len[s],
            "stray_page": ids[p]["page"], "stray_offset": int(stray_off), "d_after": int(d_after),
            "score_after": d_after / lam,
        })
    examples.sort(key=lambda e: e["lambda"])
    burst, spread = examples[0], examples[-1]
    between = 2  # the sheet drawn between home and stray sheet in the third row
    assert between not in (burst["home_sheet"], burst["stray_sheet"])
    for e in (burst, spread):
        print(f"{e['word']:10s} home sheet {e['home_sheet']} ({e['home_folios']}, {e['home_len']} words) gaps {e['home_gaps']} "
              f"lambda {e['lambda']:.0f} | stray {e['stray_page']} sheet {e['stray_sheet']}: d {e['d_after']} = {e['score_after']:.1f} bursts")
    d_between = burst["d_after"] + sheet_len[between]
    print(f"with sheet {between} ({sheet_folios[between]}, {sheet_len[between]} words) between: d {d_between} = {d_between / burst['lambda']:.1f} bursts")
    out = {
        "note": "quire T, IT2a, word unit; burst scale per scripts/quire_order_burst.py (shrunk mean home gap, floor 30); "
        "derive_burst_example.py",
        "quire": QUIRE, "lambda0": lam0, "n_types": len(rows), "n_candidates": len(examples),
        "sheet_len": {str(s): sheet_len[s] for s in sheets}, "sheet_folios": {str(s): sheet_folios[s] for s in sheets},
        "burst": burst, "spread": spread, "between_sheet": between,
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", os.path.relpath(OUT, HERE))


if __name__ == "__main__":
    main()
