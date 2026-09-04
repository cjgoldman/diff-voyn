"""Figure: how often does even spreading of the rare material reproduce the observed
D = conjugate − nested-adjacent excess?  Empirical tail from 5 000 permutation draws
per cell (§12; data/leaf_test_pvalue.json).

y = empirical p = (#draws ≥ observed + 1)/(n + 1), log scale; a cell with 0/5 000
draws sits on the floor 1/5001 and is labelled "0".  Filled: null permutes page
contents within quire × language; hollow: also within scribal hand.  Small crosses:
normal-tail p for the recorded z (for reference only).
"""

import json
import os

import numpy as np
from scipy.stats import norm

from style import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
P = json.load(open(os.path.join(DATA, "leaf_test_pvalue.json")))

UNITS = [("words_k2to10", "words\n$k$ 2–10"), ("n5_all_k2to10", "$n$ = 5"), ("n6_all_k2to10", "$n$ = 6"),
         ("n7_all_k2to10", "$n$ = 7"), ("n8_all_k2to10", "$n$ = 8")]
NULLS = [("quire+lang", False, -0.09), ("quire+lang+hand", True, 0.09)]
TR = (("IT2a", C_IT2A, MARK_IT2A, -0.2), ("RF1b", C_RF1B, MARK_RF1B, 0.2))
N = 5000
FLOOR = 1 / (N + 1)

fig, ax = plt.subplots(figsize=(W_TWO_THIRDS, 3.0))
for i, (ukey, ulab) in enumerate(UNITS):
    for nkey, hollow, dn in NULLS:
        for tr, col, mk, dt in TR:
            cell = P[tr][f"{ukey}/{nkey}"]["perm"]
            n_ge, z = cell["n_ge"], cell["z"]
            p = (n_ge + 1) / (N + 1)
            x = i + dt + dn
            ax.plot(x, p, marker=mk, color=col, mfc="white" if hollow else col, mec=col, ms=5.5, ls="none", zorder=4)
            ax.plot(x, norm.sf(z), marker="x", color=col, ms=3.5, mew=0.7, ls="none", zorder=3, alpha=0.8)
            if n_ge == 0:
                ax.annotate("0", (x, p), xytext=(0, -7), textcoords="offset points", ha="center", va="top", fontsize=5.8, color=col)
            elif n_ge <= 5:
                ax.annotate(str(n_ge), (x, p), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=5.8, color=col)

ax.set_yscale("log")
ax.set_ylim(2e-7, 0.25)
ax.axhline(FLOOR, color=AXIS, lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.text(-0.5, FLOOR * 0.62, "floor: 0 / 5 000 draws", fontsize=6.3, color=MUTED, ha="left", va="top")
for pv, lab in ((0.05, "0.05"), (0.01, "0.01"), (0.001, "1/1000")):
    ax.axhline(pv, color=GRID, lw=0.6, zorder=0)
ax.set_yticks([0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6])
ax.set_yticklabels(["0.1", "0.01", "10$^{-3}$", "10$^{-4}$", "10$^{-5}$", "10$^{-6}$"])
ax.set_xticks(np.arange(len(UNITS)))
ax.set_xticklabels([u[1] for u in UNITS])
ax.set_xlim(-0.6, len(UNITS) - 0.4)
ax.set_ylabel("one-sided $p$ under the even-spread null")
h = [
    mpl.lines.Line2D([], [], marker=MARK_IT2A, color=C_IT2A, ls="none", ms=5.5, label="IT2a"),
    mpl.lines.Line2D([], [], marker=MARK_RF1B, color=C_RF1B, ls="none", ms=5.5, label="RF1b"),
    mpl.lines.Line2D([], [], marker="o", mfc="white", mec=INK2, ls="none", ms=5.5, label="null also within hand"),
    mpl.lines.Line2D([], [], marker="x", color=INK2, ls="none", ms=3.5, mew=0.7, label="normal tail of recorded z"),
]
ax.legend(handles=h, loc="upper right", ncol=2, columnspacing=1.0, handletextpad=0.4)
save(fig, "fig_leaf_test_tail")
