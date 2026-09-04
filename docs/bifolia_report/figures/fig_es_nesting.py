"""Executive-summary figure: every way of gathering quire T's six sheets (record §19).

(a) The best achievable fit (mean distance between the recurrences of rare glyph
    strings, L1 cost, 7-letter strings) when the six sheets are gathered as 1 nested
    block (fully nested, the binding among them) up to 6 blocks (every sheet read whole,
    i.e. stacked), standardised over all 23 040 patterns; lower is better.  The binding's
    own value is the red tick.
(b) Where three particular arrangements rank among the 23 040: the best stacked order,
    the best of all fully nested orders, and the binding as it is — range over the five
    text units (words, 5–8-letter strings), both transcriptions.

Source: data/quire_order_nesting_T.json (L1).
"""

import json
import os

import numpy as np

from style import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "..", "data", "quire_order_nesting_T.json")))
UNITS = ["words", "n5", "n6", "n7", "n8"]
TR = [("IT2a", C_IT2A, MARK_IT2A), ("RF1b", C_RF1B, MARK_RF1B)]
S = R["IT2a"]["S"]
NP = R["IT2a"]["n_patterns"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(W_FULL, 2.55), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.75})

# ---- (a) staircase --------------------------------------------------------------
for tr, col, mk in TR:
    L = R[tr]["n7"]["L1"]
    z = [(L["best_by_nblocks"][str(b)] - L["mean"]) / L["sd"] for b in range(1, S + 1)]
    ax.plot(range(1, S + 1), z, marker=mk, color=col, ms=5, lw=1.4, label=f"{tr} transcription", zorder=3)
    ax.plot(1, L["z_bound"], marker="_", color=C_NESTED, ms=11, mew=2.0, lw=0,
            label="the binding as it is" if tr == "IT2a" else None, zorder=4)
ax.axhline(0, color=AXIS, lw=0.7, zorder=1)
ax.set_xticks(range(1, S + 1))
ax.set_xticklabels(["1\nfully\nnested", "2", "3", "4", "5", "6\nevery sheet\nread whole"], fontsize=6.8)
ax.set_xlabel("number of nested groups the six sheets are gathered into", fontsize=7.2)
ax.set_ylabel("best fit in the class (sd)\nlower = rare strings closer together", fontsize=7.2)
ax.set_ylim(-5.3, 1.9)
ax.text(0.98, 0.03, "7-letter glyph strings", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6, color=INK2)
ax.legend(loc="upper right", fontsize=6.6, handlelength=1.6)
panel_label(ax, "(a)", x=-0.27)

# ---- (b) ranks ------------------------------------------------------------------
cats = [("best order with every\nsheet read whole", "stacked_rank", C_STACKED),
        ("best of all fully\nnested orders", "nested_rank", C_NESTED),
        ("the binding as it is", "bound_rank", C_NESTED)]
for j, (lab, key, col) in enumerate(cats):
    y = len(cats) - 1 - j
    for tr, tcol, mk in TR:
        ranks = np.array([R[tr][u]["L1"][key] for u in UNITS])
        dy = 0.16 if tr == "IT2a" else -0.16
        bx.plot([ranks.min(), ranks.max()], [y + dy, y + dy], color=col, lw=2.2, alpha=0.45, solid_capstyle="round", zorder=2)
        bx.plot(ranks, np.full(len(ranks), y + dy), marker=mk, color=tcol, ms=4.2, ls="none", zorder=3)
bx.set_yticks(range(len(cats)))
bx.set_yticklabels([c[0] for c in cats][::-1], fontsize=7)
bx.tick_params(axis="y", length=0)
bx.set_xscale("log")
bx.set_xlim(0.8, NP * 1.3)
bx.set_xticks([1, 10, 100, 1000, 10000])
bx.set_xticklabels(["1st", "10th", "100th", "1 000th", "10 000th"], fontsize=6.4)
bx.axvline(NP, color=AXIS, lw=0.7, ls=(0, (3, 2)), zorder=1)
bx.text(NP, 2.62, "last\n(23 040th)", fontsize=5.8, color=MUTED, ha="center", va="bottom")
bx.set_xlabel(f"rank among all {NP:,} ways of gathering the sheets (1st = best fit)".replace(",", " "), fontsize=7.2)
bx.grid(axis="x", color=GRID, lw=0.5)
bx.grid(axis="y", visible=False)
bx.spines["left"].set_visible(False)
bx.set_ylim(-0.6, 2.95)
panel_label(bx, "(b)", x=-0.62)
fig.subplots_adjust(left=0.11, right=0.99, top=0.95, bottom=0.3)
save(fig, "fig_es_nesting")
print("ok", {k: [R[t][u]["L1"][k] for t in ("IT2a", "RF1b") for u in UNITS] for k in ("stacked_rank", "nested_rank", "bound_rank")})
