"""Alternative nesting patterns (record §19): does any other way of gathering a
quire's sheets do as well as the stacked optimum?  Source: data/quire_order_nesting_{Q}.json
(L1 cost only -- the burst family is biased toward stacked patterns in this space).

(a) best L1 cost per number of nested blocks (1 = fully nested ... S = fully stacked),
    standardised within the quire's candidate set, n = 7 glyph n-grams, both transcriptions;
    the binding marked.
(b) margin by which the best fully nested order trails the best stacked order (sd) per
    quire x unit, against the 5-95 % band of the same margin on shuffled contents.
(c) rank of the best fully nested order and of the binding among all patterns (log scale).
"""
import json
import os

import numpy as np
from style import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
QUIRES = ["T", "M", "C", "A", "B"]
UNITS = ["words", "n5", "n6", "n7", "n8"]
UNIT_LABEL = {"words": "words", "n5": "n5", "n6": "n6", "n7": "n7", "n8": "n8"}
TR = [("IT2a", C_IT2A, MARK_IT2A), ("RF1b", C_RF1B, MARK_RF1B)]

R = {q: json.load(open(os.path.join(DATA, f"quire_order_nesting_{q}.json"))) for q in QUIRES}

fig = plt.figure(figsize=(W_FULL, 7.4))
gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.6], hspace=0.66, wspace=0.30)
axa = fig.add_subplot(gs[0, :])
axb = fig.add_subplot(gs[1, 0])
axc = fig.add_subplot(gs[1, 1])

# ---- (a) best z by number of blocks, n7 ------------------------------------------
offs = {"T": 0, "M": 1, "C": 2, "A": 3, "B": 4}
xpos = {}
x0 = 0.0
for q in QUIRES:
    S = R[q]["IT2a"]["S"]
    xpos[q] = x0 + np.arange(1, S + 1)
    x0 += S + 1.2
for q in QUIRES:
    S = R[q]["IT2a"]["S"]
    for tr, col, mk in TR:
        L = R[q][tr]["n7"]["L1"]
        z = np.array([(L["best_by_nblocks"][str(b)] - L["mean"]) / L["sd"] for b in range(1, S + 1)])
        axa.plot(xpos[q], z, marker=mk, color=col, ms=4.2, lw=1.2,
                 label=tr if q == "T" else None, zorder=3)
        axa.plot(xpos[q][0], L["z_bound"], marker="_", color=C_NESTED, ms=9, mew=1.6, lw=0,
                 label="binding (as bound)" if (q == "T" and tr == "IT2a") else None, zorder=4)
    axa.text(xpos[q].mean(), -0.13, f"quire {q}\n{R[q]['IT2a']['n_patterns']:,} patterns",
             transform=axa.get_xaxis_transform(), ha="center", va="top", fontsize=6.4, color=INK2)
axa.axhline(0, color=AXIS, lw=0.6, zorder=1)
axa.set_xticks(np.concatenate([xpos[q] for q in QUIRES]))
axa.set_xticklabels([str(b) for q in QUIRES for b in range(1, R[q]["IT2a"]["S"] + 1)])
axa.set_xlabel("number of nested blocks (1 = fully nested, S = fully stacked)", labelpad=22)
axa.set_ylabel("best L1 cost in class, z (sd)")
axa.set_ylim(-5.4, 1.6)
axa.legend(loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.0), handletextpad=0.4)
axa.text(0.995, 0.97, "n = 7 glyph n-grams, L1 cost; lower is better",
         transform=axa.transAxes, ha="right", va="top", fontsize=7, color=INK2)
panel_label(axa, "(a)", x=-0.06, y=1.02)

# ---- (b) nested - stacked margin vs null band ------------------------------------
ypos = []
ylab = []
y = 0
for q in QUIRES:
    for u in UNITS:
        ypos.append(y)
        ylab.append(f"{q} {UNIT_LABEL[u]}")
        for k, (tr, col, mk) in enumerate(TR):
            L = R[q][tr][u]["L1"]
            lo, med, hi = L["null_nms_q"]
            dy = -0.22 if k == 0 else 0.22
            axb.plot([lo, hi], [y + dy, y + dy], color=C_NULL, lw=2.2, solid_capstyle="butt", zorder=1,
                     label="shuffled contents, 5–95 %" if (q == "T" and u == "words" and k == 0) else None)
            axb.plot(L["nested_minus_stacked_sd"], y + dy, marker=mk, color=col, ms=3.8, lw=0, zorder=3,
                     label=tr if (q == "T" and u == "words") else None)
        y += 1
    y += 1.0
axb.axvline(0, color=AXIS, lw=0.6)
axb.set_yticks(ypos)
axb.set_yticklabels(ylab, fontsize=6.4)
axb.invert_yaxis()
axb.set_xlabel("best fully nested − best stacked (sd)")
axb.grid(False)
axb.grid(True, axis="x")
axb.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=6.4, handlelength=1.6, columnspacing=1.0)
panel_label(axb, "(b)", x=-0.30, y=1.13)

# ---- (c) ranks of best nested and binding among all patterns ----------------------
y = 0
ypos_c = []
for q in QUIRES:
    N = R[q]["IT2a"]["n_patterns"]
    for u in UNITS:
        ypos_c.append(y)
        for k, (tr, col, mk) in enumerate(TR):
            L = R[q][tr][u]["L1"]
            dy = -0.22 if k == 0 else 0.22
            axc.plot(L["nested_rank"] / N, y + dy, marker=mk, color=col, ms=3.8, lw=0, zorder=3,
                     label=f"best fully nested, {tr}" if (q == "T" and u == "words") else None)
            axc.plot(L["bound_rank"] / N, y + dy, marker=mk, mfc="none", color=C_NESTED, ms=3.8, lw=0,
                     zorder=3, label=f"binding, {tr}" if (q == "T" and u == "words") else None)
            axc.plot(L["stacked_rank"] / N, y + dy, marker="|", color=C_STACKED, ms=6, mew=1.2, lw=0,
                     zorder=4, label="best stacked" if (q == "T" and u == "words" and k == 0) else None)
        y += 1
    y += 1.0
axc.set_xscale("log")
axc.set_xlim(2e-4, 1.3)
axc.set_xlabel("rank / number of patterns")
axc.set_yticks(ypos_c)
axc.set_yticklabels([])
axc.invert_yaxis()
axc.grid(False)
axc.grid(True, axis="x")
h, l = axc.get_legend_handles_labels()
order = [0, 3, 1, 4, 2]
axc.legend([h[i] for i in order], [l[i] for i in order], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=6.2, handlelength=1.2, columnspacing=1.0)
panel_label(axc, "(c)", x=-0.06, y=1.13)

save(fig, "fig_nesting")
