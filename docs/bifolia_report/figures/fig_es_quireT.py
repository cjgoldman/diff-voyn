"""Executive-summary figure: the order of the six sheets of quire T (record §15, §17).

(a) All 720 possible orders of the six sheets (each sheet read whole), scored by how
    close together the recurrences of rare 7-letter glyph strings fall (IT2a), shown as
    standard deviations from the average order.  The best real order is marked; the grey
    histogram shows where the *best of 720* lands when the pages' contents are shuffled
    200 times (the "best of many" effect the winner has to beat).
(b) The sheets as bound (1 = outermost) and in the order the text prefers.  No arrow:
    the direction of reading along the chain is not established.

Source: data/derived_quire_T_costs.json (re-derived from the recorded code; the
recorded z* = −3.96 for 7-grams under L1 is reproduced), data/quire_order_burst_T.json.
"""

import json
import os

import numpy as np

from style import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "data", "derived_quire_T_costs.json")))
FOLIOS = {1: "f103/f116", 2: "f104/f115", 3: "f105/f114", 4: "f106/f113", 5: "f107/f112", 6: "f108/f111"}
SHEET_COL = {1: "#CDE2FB", 2: "#86B6EF", 3: "#3987E5", 4: "#1C5CAB", 5: "#104281", 6: "#0D366B"}
CHAIN = [1, 6, 5, 4, 2, 3]

fig = plt.figure(figsize=(W_FULL, 2.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.25, left=0.09, right=0.99, top=0.93, bottom=0.2)

# ---- (a) histogram --------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
u = D["IT2a"]["units"]["n7"]
labels = D["IT2a"]["labels"]
c = np.array(u["L1_between"])
z = (c - c.mean()) / c.std()
nb = np.array(u["null_best_z"])
bins = np.linspace(-4.4, 3.4, 40)
ax.hist(z, bins=bins, color=C_BLUE_WASH, edgecolor=C_IT2A, lw=0.6, label="the 720 real orders", zorder=3)
wbins = np.linspace(-4.4, -1.6, 15)
h, _ = np.histogram(nb, bins=wbins)
ax.bar((wbins[:-1] + wbins[1:]) / 2, h / h.max() * 22, width=wbins[1] - wbins[0], color=C_NULL, lw=0,
       label="best of 720 when the pages'\ncontents are shuffled (200 trials)", zorder=2)
wi = int(np.argmin(z))
ax.axvline(z[wi], color=C_ACCENT, lw=1.6, zorder=5)
lab = "-".join(labels[wi][::-1]) if labels[wi][::-1] == "".join(map(str, CHAIN)) else "-".join(labels[wi])
ax.annotate(f"the winning order\n{'-'.join(map(str, CHAIN))}\n({z[wi]:+.1f} sd)", (z[wi], 30), xytext=(-2.75, 46),
            textcoords="data", fontsize=7, color=C_ACCENT, ha="center", va="center", fontweight="semibold",
            arrowprops=dict(arrowstyle="-", color=C_ACCENT, lw=0.7))
ax.set_xlim(-4.4, 3.4)
ax.set_ylim(0, 78)
ax.set_xlabel("fit of the order (sd from the average order; lower = better)", fontsize=7.2)
ax.set_ylabel("number of orders", fontsize=7.2)
ax.legend(loc="upper right", fontsize=6.4, handlelength=1.4)
ax.text(0.99, 0.6, "7-letter glyph strings,\nIT2a transcription", transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color=INK2)
panel_label(ax, "(a)", x=-0.14)

# ---- (b) as bound vs the chain --------------------------------------------------
bx = fig.add_subplot(gs[0, 1])
bx.set_xlim(-0.3, 6.6)
bx.set_ylim(-1.0, 3.2)
bx.axis("off")


def row(y, order, title):
    bx.text(-0.2, y + 0.78, title, fontsize=7.4, color=INK, va="bottom", ha="left", fontweight="semibold")
    for i, s in enumerate(order):
        dark = s >= 4
        bx.add_patch(mpl.patches.FancyBboxPatch((i * 1.05 + 0.02, y), 0.92, 0.62, boxstyle="round,pad=0.02",
                                                fc=SHEET_COL[s], ec="white", lw=0.8))
        bx.text(i * 1.05 + 0.48, y + 0.31, str(s), ha="center", va="center", fontsize=9,
                color="white" if dark else INK, fontweight="bold")
        bx.text(i * 1.05 + 0.48, y - 0.1, FOLIOS[s].replace("/", "/\n"), ha="center", va="top", fontsize=5.6, color=INK2, linespacing=1.0)


row(1.85, [1, 2, 3, 4, 5, 6], "sheets as bound (1 = outermost, 6 = innermost)")
row(0.2, CHAIN, "the order the text prefers  (31 of 32 tests agree)")
bx.text(-0.2, -0.62, "no arrow: which end is the beginning is not established", fontsize=6.6, color=C_NESTED, va="top")
panel_label(bx, "(b)", x=-0.06, y=0.98)
save(fig, "fig_es_quireT")
print("ok argmin", labels[wi], round(float(z[wi]), 2), "null best median", round(float(np.median(nb)), 2))
