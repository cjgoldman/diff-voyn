"""Figure: the leaf-pair test on six known texts laid onto the manuscript's own page
slots, written in the nested order or in the stacked order and then bound nested.

One panel per text; x = unit (words k 2–5, words k 2–10, n-grams n 5, 6, 7 all);
red = written nested, aqua = written stacked; blue band = the manuscript's IT2a/RF1b
range for the same unit. Shows the 36/36 separation of §6.2 / §7.2.

Sources: data/rare_types_controls.json, data/glyph_ngrams.json (controls and vms).
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

TEXTS = [
    ("la_isidorus_etym", "Isidore (Latin prose)"),
    ("la_seneca_nq", "Seneca NQ (Latin prose)"),
    ("de_bullinger", "Bullinger (German prose)"),
    ("it_decameron", "Decameron (Italian prose)"),
    ("it_commedia", "Commedia (Italian verse)"),
    ("it_orlando_furioso", "Orlando Furioso (Italian verse)"),
]
UNITS = [
    ("words\n2–5", lambda c: rc[c]["k2to5"], lambda tr: rare["vms"][tr]["leaf_test_k2to5"]),
    ("words\n2–10", lambda c: rc[c]["k2to10"], lambda tr: rare["vms"][tr]["leaf_test_k2to10"]),
    ("$n$5", lambda c: gl["controls"][c]["n5_all_k2to10"], lambda tr: gl["vms"][tr]["n5_all_k2to10"]),
    ("$n$6", lambda c: gl["controls"][c]["n6_all_k2to10"], lambda tr: gl["vms"][tr]["n6_all_k2to10"]),
    ("$n$7", lambda c: gl["controls"][c]["n7_all_k2to10"], lambda tr: gl["vms"][tr]["n7_all_k2to10"]),
]
X = np.arange(len(UNITS))

fig, axes = plt.subplots(2, 3, figsize=(W_FULL, 3.9), sharex=True, sharey=True, gridspec_kw={"hspace": 0.35, "wspace": 0.12})
for ax, (key, title) in zip(axes.ravel(), TEXTS):
    # manuscript band per unit
    for i, (lab, cget, vget) in enumerate(UNITS):
        zs = [vget(tr)["conj_minus_nested"]["z"] for tr in ("IT2a", "RF1b")]
        ax.add_patch(mpl.patches.Rectangle((i - 0.36, min(zs) - 0.12), 0.72, max(zs) - min(zs) + 0.24, color=C_BLUE_WASH, lw=0, zorder=0))
    zn = [cget(f"{key}/nested")["conj_minus_nested"]["z"] for _, cget, _ in UNITS]
    zs_ = [cget(f"{key}/stacked")["conj_minus_nested"]["z"] for _, cget, _ in UNITS]
    for i in X:
        ax.plot([i, i], [zn[i], zs_[i]], color=AXIS, lw=0.7, zorder=1)
    ax.plot(X, zs_, ls="none", marker="o", color=C_STACKED, ms=5, zorder=3, label="written stacked")
    ax.plot(X, zn, ls="none", marker="o", color=C_NESTED, ms=5, zorder=3, label="written nested")
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.set_title(title, loc="left", fontsize=7.8, pad=3)
    ax.set_xticks(X)
    ax.set_xticklabels([u[0] for u in UNITS], fontsize=6.8)
    ax.set_xlim(-0.6, len(UNITS) - 0.4)
axes[0, 0].set_ylim(-8.6, 5.6)
for ax in axes[:, 0]:
    ax.set_ylabel("$D$  (z)")
h = [
    mpl.lines.Line2D([], [], marker="o", color=C_STACKED, ls="none", ms=5, label="written sheet by sheet (stacked), bound nested"),
    mpl.lines.Line2D([], [], marker="o", color=C_NESTED, ls="none", ms=5, label="written in the bound (nested) order"),
    mpl.patches.Patch(color=C_BLUE_WASH, label="manuscript, IT2a–RF1b range"),
]
fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), columnspacing=1.0, handlelength=1.2, fontsize=6.6)
save(fig, "fig_leaf_test_controls")
