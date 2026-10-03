"""Main-text figure (§7): every way of gathering quire T's six sheets, by class (record §19).

Moved out of the executive summary 2026-09-06 (the summary version, fig_es_nesting, shows only
the fully nested and fully stacked classes).

Vertical axis = the fit itself: the mean distance between the recurrences of rare text
(L1 cost, n = 7 glyph strings), standardised over all 23,040 gatherings, with the better
fit (shorter distance) UPWARD.  One box per class of gathering — by the number of nested
blocks, from "one nested block" (the 720 fully nested orders; the binding is one of them)
to "six blocks" (the 720 fully stacked orders, every sheet read whole) — showing mean
± 1 sd (box), mean (line) and min–max (whiskers) of every pattern in the class.  Marked:
the binding as it is, and the best stacked order, which is the best of all 23,040 (dashed
line across the panel).  One panel per transcription.

The two-fold story: (1) every way of assembling the sheets was scored, not just the
binding and the stacked orders — and the nested class is uniformly mediocre (tight, all
on the worse side of the mean; its best member is only average) while a random stacked
order beats a random nested one about nine times in ten; (2) the best stacked order is
not merely better than the nested bulk, it is the best of the whole space by a wide
margin, ≈ 4.5 sd better than the average gathering.  (Replaced the rank ladder of
2026-09-06, which showed three arrangements only.)

Source: data/derived_quire_T_nesting.json (derive_quire_T_nesting.py, from
scripts/quire_order_nesting.py's recorded functions; ranks checked against
data/quire_order_nesting_T.json).
"""

import json
import os

import numpy as np

from style import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "data", "derived_quire_T_nesting.json")))
UNIT = "n7"
TR = [("IT2a", C_IT2A, MARK_IT2A), ("RF1b", C_RF1B, MARK_RF1B)]

fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.1), sharey=True)
summary = {}
for ax, (tr, tcol, mk) in zip(axes, TR):
    d = D[tr]
    nb = np.array(d["nblocks"])
    S = d["S"]
    v = np.array(d["units"][UNIT]["L1"])
    z = (v - v.mean()) / v.std()
    bound_z = z[d["bound_i"]]
    st, ne = nb == S, nb == 1
    best_st = z[st].min()
    # pairwise: how often a random stacked order beats a random nested one
    zs, zn = np.sort(z[st]), np.sort(z[ne])
    p_pair = float((len(zn) - np.searchsorted(zn, zs, side="right")).sum() / (len(zs) * len(zn)))
    summary[tr] = {"bound_z": float(bound_z), "best_stacked_z": float(best_st), "best_nested_z": float(z[ne].min()),
                   "p_stacked_beats_nested": p_pair, "best_label": d["units"][UNIT]["best_label"],
                   "raw_best": float(v.min()), "raw_bound": float(v[d["bound_i"]]), "raw_mean": float(v.mean()),
                   "class_stats": {int(k): (float(z[nb == k].mean()), float(z[nb == k].std()), float(z[nb == k].min()), float(z[nb == k].max()), int((nb == k).sum())) for k in range(1, S + 1)}}

    for k in range(1, S + 1):
        m = nb == k
        zk = z[m]
        mu, sd = zk.mean(), zk.std()
        col = C_NESTED if k == 1 else (C_STACKED if k == S else C_KNOWN)
        wash = C_NESTED_WASH if k == 1 else (C_STACKED_WASH if k == S else "#eceae4")
        x = k
        ax.plot([x, x], [zk.min(), zk.max()], color=col, lw=0.9, zorder=2)          # min–max
        for yv in (zk.min(), zk.max()):
            ax.plot([x - 0.12, x + 0.12], [yv, yv], color=col, lw=0.9, zorder=2)
        ax.add_patch(mpl.patches.Rectangle((x - 0.3, mu - sd), 0.6, 2 * sd, facecolor=wash, edgecolor=col, lw=0.9, zorder=3))
        ax.plot([x - 0.3, x + 0.3], [mu, mu], color=col, lw=1.6, zorder=4)             # mean

    # the two marked arrangements
    ax.axhline(best_st, color=C_STACKED, lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.plot([S], [best_st], marker=mk, color=C_STACKED, ms=6, markeredgecolor=INK, markeredgewidth=0.7, zorder=6)
    ax.plot([1], [bound_z], marker=mk, color=C_NESTED, ms=6, markeredgecolor=INK, markeredgewidth=0.7, zorder=6)
    ax.annotate("the binding\nas it is", (1, bound_z), xytext=(1.55, bound_z + 0.35), fontsize=6.4, color=INK2,
                ha="left", va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkB=3), zorder=7)
    ax.annotate("best stacked order:\nbest of all 23,040", (S, best_st), xytext=(S - 0.55, best_st - 0.55), fontsize=6.4,
                color=INK2, ha="right", va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkB=3), zorder=7)

    ax.set_title(f"{tr} transcription", fontsize=7.2, fontweight="normal", loc="left", color=INK2, pad=4)
    ax.set_xlim(0.4, S + 0.6)
    ax.set_xticks(range(1, S + 1))
    counts = [int((nb == k).sum()) for k in range(1, S + 1)]
    names = ["fully\nnested", "2 blocks", "3", "4", "5", "fully\nstacked"]
    ax.set_xticklabels([f"{n}\n" + f"{c:,}" for n, c in zip(names, counts)], fontsize=6.4)
    for t in ax.get_xticklabels():
        t.set_linespacing(1.25)
    ax.tick_params(axis="x", length=0)
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(2.6, -5.2)   # better (shorter distance) at the top

axes[0].set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3])
axes[0].set_yticklabels(["−5", "−4", "−3", "−2", "−1", "mean", "+1", "+2", "+3"], fontsize=6.6)
axes[0].tick_params(axis="y", length=0)
axes[0].set_ylabel("distance between recurrences of rare text,\nsd from the mean of all gatherings  (closer fit ↑)", fontsize=6.8)
fig.text(0.55, 0.035, "how the six sheets are gathered: number of nested blocks (6 = every sheet read whole), and how many such patterns there are", ha="center", va="bottom", fontsize=6.8, color=INK2)
fig.subplots_adjust(left=0.115, right=0.985, top=0.93, bottom=0.2, wspace=0.08)
save(fig, "fig_nesting_classes")
for tr, s in summary.items():
    print(tr, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in s.items() if k != "class_stats"})
    for k, (mu, sd, lo, hi, n) in s["class_stats"].items():
        print(f"   {k} blocks n {n:5d}  mean {mu:+.2f} sd {sd:.2f} min {lo:+.2f} max {hi:+.2f}")
