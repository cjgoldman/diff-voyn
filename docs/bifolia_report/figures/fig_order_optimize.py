"""Figure: can re-ordering the 52 sheets raise the manuscript's rare-word locality
toward the known-text corpus?  Simulated annealing over sheet orders under four
constraint levels (§10; data/order_optimize_none.json, orientation fixed).

(a) P(gap ≤ 1000)/uniform (r1000) after optimisation: the manuscript from its
    stacked order, from three random sheet orders, and three page-content-shuffled
    manuscripts (no order information — the noise ceiling); corpus percentiles.
(b) The same for P(gap ≤ 100)/uniform (r100), which sheet order barely moves.
(c) Known texts written sheet by sheet, sheets shuffled, then optimised: recovered
    value minus the true-order value (overfitting excess).
"""

import json
import os

import numpy as np

from style import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
O = json.load(open(os.path.join(DATA, "order_optimize_none.json")))

LEVELS = ["init", "strict (lang+section+hand)", "topic (lang+section)", "language only", "free"]
LABELS = ["start", "strict", "topic", "language", "free"]
X = np.arange(len(LEVELS))
KNOWN = [("la_isidorus_etym", "Isidore", "la"), ("la_seneca_nq", "Seneca NQ", "la"), ("de_bullinger", "Bullinger", "de"),
         ("it_decameron", "Decameron", "it"), ("it_orlando_furioso", "Orlando Furioso", "it_verse")]


def series(d, stat):
    return [d[l][stat] for l in LEVELS]


fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 3.3), gridspec_kw={"width_ratios": [1.15, 1.0, 1.05], "wspace": 0.5})

for ax, stat, title in ((axes[0], "r1000", "(a)  $r_{1000}$ after re-ordering"), (axes[1], "r100", "(b)  $r_{100}$ after re-ordering")):
    # noise ceiling band from page-content-shuffled manuscripts
    sh = np.array([series(d, stat) for d in O["vms_pages_shuffled"]])
    ax.fill_between(X, sh.min(0), sh.max(0), color=C_NULL, alpha=0.55, lw=0, zorder=1, label="page contents shuffled (no order information)")
    for d in O["vms_stacked_from_shuffled_sheets"]:
        ax.plot(X, series(d, stat), color=C_IT2A, lw=0.8, alpha=0.5, marker=MARK_IT2A, ms=2.8, mfc="white", zorder=3)
    ax.plot(X, series(O["vms_stacked"], stat), color=C_IT2A, lw=1.8, marker=MARK_IT2A, ms=5, zorder=4, label="manuscript from stacked order")
    ax.plot([], [], color=C_IT2A, lw=0.8, alpha=0.5, marker=MARK_IT2A, ms=2.8, mfc="white", label="manuscript from random sheet orders (×3)")
    ax.set_xticks(X)
    ax.set_xticklabels(LABELS, fontsize=7, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlabel("sheets may swap within …")
    ax.set_title(title, loc="left", fontsize=7.8, pad=4)
    ax.set_ylabel("P(gap ≤ 1000) / uniform" if stat == "r1000" else "P(gap ≤ 100) / uniform")
    ax.set_xlim(-0.3, len(LEVELS) - 0.7)

# corpus percentiles (source: data/corpus_sweep.log, k 2..5 pooled)
for v, lab in ((1.65, "5th"), (2.74, "25th"), (3.35, "median")):
    axes[0].axhline(v, color=AXIS, lw=0.7, ls=(0, (3, 2)), zorder=0)
    axes[0].text(len(LEVELS) - 0.72, v, f"corpus {lab}", fontsize=6.2, color=MUTED, ha="right", va="bottom")
for v, lab in ((4.66, "5th"), (12.87, "median")):
    axes[1].axhline(v, color=AXIS, lw=0.7, ls=(0, (3, 2)), zorder=0)
    axes[1].text(len(LEVELS) - 0.72, v, f"corpus {lab}", fontsize=6.2, color=MUTED, ha="right", va="bottom")
axes[0].set_ylim(1.6, 3.6)
axes[1].set_ylim(3.5, 14)
axes[1].set_yscale("log")
axes[1].set_yticks([4, 5, 6, 8, 10, 12])
axes[1].set_yticklabels(["4", "5", "6", "8", "10", "12"])
axes[1].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

# (c) known texts recovery: optimised − true, r1000
cx = axes[2]
ends = []
for j, (key, lab, lang) in enumerate(KNOWN):
    true = O["known"][key]["true"]["init"]["r1000"]
    runs = O["known"][key]["from_shuffled"]
    for r in runs:
        y = [r[l]["r1000"] - true for l in LEVELS[1:]]
        cx.plot(X[1:], y, color=C_KNOWN, lw=0.8, marker=MARK_LANG[lang], ms=3.5, mfc="white" if lang == "it_verse" else C_KNOWN, mec=C_KNOWN, zorder=3, alpha=0.9)
    ends.append((np.mean([r["free"]["r1000"] - true for r in runs]), lab))
ends.sort()
pos = [v for v, _ in ends]
for i in range(1, len(pos)):
    pos[i] = max(pos[i], pos[i - 1] + 0.045)
for (v, lab), yp in zip(ends, pos):
    cx.annotate(lab, (X[-1], v), xytext=(X[-1] + 0.12, yp), textcoords="data", fontsize=6.2, color=INK2, va="center",
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.5, shrinkA=0, shrinkB=1))
# manuscript's own gain on the same axis, for scale
ms = [O["vms_stacked"][l]["r1000"] - O["vms_stacked"]["init"]["r1000"] for l in LEVELS[1:]]
cx.plot(X[1:], ms, color=C_IT2A, lw=1.8, marker=MARK_IT2A, ms=5, zorder=4)
cx.annotate("manuscript gain\nover its start", (X[-1], ms[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.2, color=C_IT2A, va="center")
cx.axhline(0, color=AXIS, lw=0.8, zorder=1)
cx.set_xticks(X[1:])
cx.set_xticklabels(LABELS[1:], fontsize=7, rotation=30, ha="right", rotation_mode="anchor")
cx.set_xlim(0.7, len(LEVELS) - 0.7 + 1.3)
cx.set_ylim(-0.1, 0.65)
cx.set_xlabel("sheets may swap within …")
cx.set_title("(c)  known texts: recovered − true $r_{1000}$", loc="left", fontsize=7.8, pad=4)

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17), columnspacing=1.2, handlelength=1.8, fontsize=6.8)
save(fig, "fig_order_optimize")
