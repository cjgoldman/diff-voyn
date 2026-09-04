"""Figure: the leaf-pair test on the manuscript.

(a) D = (conjugate-leaf pairs) − (nested-adjacent-leaf pairs), as z against the
    within-quire × language page-content permutation null, per unit: words k = 2,
    2–5, 2–10 and glyph n-grams n = 4…8 (all, and boundary-straddling "cross").
    Hollow markers: null additionally restricted to the same scribal hand (§7.3).
    Washes: range of the six known texts written nested (red) and written stacked
    (aqua) on the same page slots (§6.2, §7.2).
(b) The two components: conjugate z (aqua) and nested-adjacent z (red).

Sources: data/rare_types.json (leaf_test_*), data/leaf_affinity.json,
data/rare_types_controls.json, data/glyph_ngrams.json, data/hand_control.json.
"""

import json
import os

import numpy as np

from style import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
J = lambda n: json.load(open(os.path.join(DATA, n)))  # noqa: E731

rare = J("rare_types.json")
rc = J("rare_types_controls.json")
gl = J("glyph_ngrams.json")
hand = J("hand_control.json")

# unit list: (label, vms getter(tr) -> dict with conjugate/nested_adjacent/conj_minus_nested,
#             control getter(ctrlkey) -> same dict or None, hand key or None)
UNITS = []
for k, key, ckey in (("2", "leaf_test_k2", "k2to2"), ("2–5", "leaf_test_k2to5", "k2to5"), ("2–10", "leaf_test_k2to10", "k2to10")):
    UNITS.append((f"words\n$k$={k}", (lambda tr, key=key: rare["vms"][tr][key]),
                  (lambda c, ckey=ckey: rc[c][ckey]), "words k2to10" if k == "2–10" else None))
for n in (4, 5, 6, 7, 8):
    UNITS.append((f"$n$={n}\nall", (lambda tr, n=n: gl["vms"][tr][f"n{n}_all_k2to10"]),
                  (lambda c, n=n: gl["controls"][c].get(f"n{n}_all_k2to10")), f"n{n} all" if n in (6, 7) else None))
for n in (5, 6, 7, 8):
    UNITS.append((f"$n$={n}\ncross", (lambda tr, n=n: gl["vms"][tr][f"n{n}_cross_k2to10"]),
                  (lambda c, n=n: gl["controls"][c].get(f"n{n}_cross_k2to10")), f"n{n} cross" if n in (6, 7) else None))

CTRL_TEXTS = ["la_isidorus_etym", "la_seneca_nq", "de_bullinger", "it_decameron", "it_commedia", "it_orlando_furioso"]
X = np.arange(len(UNITS))
TR = (("IT2a", C_IT2A, MARK_IT2A, -0.16), ("RF1b", C_RF1B, MARK_RF1B, 0.16))

fig, (ax, bx) = plt.subplots(2, 1, figsize=(W_FULL, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.12})

# ---- (a) D statistic ------------------------------------------------------------
for i, (lab, vget, cget, hkey) in enumerate(UNITS):
    for mode, col, wash in (("nested", C_NESTED, C_NESTED_WASH), ("stacked", C_STACKED, C_STACKED_WASH)):
        zs = []
        for t in CTRL_TEXTS:
            d = cget(f"{t}/{mode}")
            if d is not None:
                zs.append(d["conj_minus_nested"]["z"])
        if zs:
            ax.add_patch(mpl.patches.Rectangle((i - 0.42, min(zs)), 0.84, max(zs) - min(zs), color=wash, lw=0, zorder=0))
    for tr, col, mk, dx in TR:
        z = vget(tr)["conj_minus_nested"]["z"]
        ax.plot(i + dx, z, marker=mk, color=col, ms=5.5, zorder=4, ls="none")
        if hkey is not None:
            zh = hand[tr][f"{hkey}|quire+lang+hand"]["conj_minus_nested"]["z"]
            ax.plot(i + dx, zh, marker=mk, mfc="white", mec=col, ms=5.5, zorder=5, ls="none")
ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
ax.set_ylim(-11.0, 6.4)
ax.set_ylabel("$D$ = conjugate − nested-adjacent   (z)")
ax.text(5.0, 5.9, "aqua wash: six known texts written stacked", color=C_STACKED, fontsize=6.8, ha="center", va="top")
ax.text(5.0, -8.05, "red wash: the same texts written nested", color=C_NESTED, fontsize=6.8, ha="center", va="top")
ax.text(-0.45, 6.0, "no controls run\nfor $n$ = 4, 8", color=MUTED, fontsize=6.3, va="top")
h = [
    mpl.lines.Line2D([], [], marker=MARK_IT2A, color=C_IT2A, ls="none", ms=5.5, label="manuscript IT2a"),
    mpl.lines.Line2D([], [], marker=MARK_RF1B, color=C_RF1B, ls="none", ms=5.5, label="manuscript RF1b"),
    mpl.lines.Line2D([], [], marker="o", mfc="white", mec=INK2, ls="none", ms=5.5, label="null also within hand"),
]
ax.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), columnspacing=1.2)
panel_label(ax, "(a)")

# ---- (b) components ----------------------------------------------------------------
for i, (lab, vget, cget, hkey) in enumerate(UNITS):
    for tr, col, mk, dx in TR:
        d = vget(tr)
        bx.plot(i + dx, d["conjugate"]["z"], marker=mk, color=C_STACKED, ms=5, ls="none", zorder=4)
        bx.plot(i + dx, d["nested_adjacent"]["z"], marker=mk, color=C_NESTED, ms=5, ls="none", zorder=4)
        bx.plot([i + dx, i + dx], [d["nested_adjacent"]["z"], d["conjugate"]["z"]], color=AXIS, lw=0.7, zorder=2)
bx.axhline(0, color=AXIS, lw=0.8, zorder=1)
bx.set_ylim(-5.0, 5.2)
bx.set_ylabel("component z")
h2 = [
    mpl.lines.Line2D([], [], marker="o", color=C_STACKED, ls="none", ms=5, label="conjugate-leaf pairs (same sheet)"),
    mpl.lines.Line2D([], [], marker="o", color=C_NESTED, ls="none", ms=5, label="nested-adjacent-leaf pairs (different sheets)"),
]
bx.legend(handles=h2, loc="upper left", ncol=2, columnspacing=1.2)
bx.set_xticks(X)
bx.set_xticklabels([u[0] for u in UNITS], fontsize=7)
bx.set_xlim(-0.6, len(UNITS) - 0.4)
bx.text(1.0, -4.7, "words: types with $k$ occurrences", color=INK2, fontsize=6.8, ha="center")
bx.text(5.0, -4.7, "glyph $n$-grams, $k$ = 2–10", color=INK2, fontsize=6.8, ha="center")
bx.text(9.4, -4.7, "$n$-grams straddling a word boundary", color=INK2, fontsize=6.6, ha="center")
for xsep in (2.5, 7.5):
    for a in (ax, bx):
        a.axvline(xsep, color=GRID, lw=0.6, zorder=0)
panel_label(bx, "(b)")

save(fig, "fig_leaf_test")
