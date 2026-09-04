"""fig_seriation_power: can the seriation metrics recover a known writing order?

Known prose (Isidore, Bullinger, Decameron; three
windows each) written onto each quire's page slots in the current or a random stacked
sheet order (18 cases per quire), then scored: rank of the true writing order among all
S! stacked candidates, per metric.  Nested-written control (9 cases): which stacked
order does text written as bound prefer?  Source: quire_order_burst_{T,M,C,A,B}.json
['controls'] (docs/doubleton_gaps.md §13-15).
"""
import json
import numpy as np
from style import *  # noqa

QUIRES = ["T", "M", "C", "A", "B"]
NCAND = {"T": 720, "M": 120, "C": 24, "A": 24, "B": 24}
NSHEETS = {"T": 6, "M": 5, "C": 4, "A": 4, "B": 4}
TOKENS = {"T": "10 673", "M": "6 911", "C": "1 401", "A": "1 495", "B": "1 019"}
METRICS = ["L1", "burst", "blog", "modal"]
LANG = {"la": ("Latin (Isidore)", MARK_LANG["la"]), "de": ("German (Bullinger)", MARK_LANG["de"]),
        "it": ("Italian (Decameron)", MARK_LANG["it"])}

fig, axes = plt.subplots(1, 5, figsize=(W_FULL, 3.2), sharey=False,
                         gridspec_kw=dict(wspace=0.38, left=0.1, right=0.99, top=0.87, bottom=0.27))
rng = np.random.default_rng(0)
NESTED_NOTE = []
for ax, Q in zip(axes, QUIRES):
    d = json.load(open(f"../data/quire_order_burst_{Q}.json"))["controls"]
    N = NCAND[Q]
    cur = "".join(str(s + 1) for s in range(NSHEETS[Q]))
    ax.axhspan(0.7, 1.45, color=C_STACKED_WASH, lw=0, zorder=0)
    ax.axhline(10.5, color=AXIS, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for mi, m in enumerate(METRICS):
        ranks = {}
        for key, v in d.items():
            if key == "summary" or not key.endswith(("stacked_cur", "stacked_rand")):
                continue
            lang = key[:2]
            ranks.setdefault(lang, []).append(v[m]["rank"])
        for li, (lang, rr) in enumerate(ranks.items()):
            x = mi + (li - 1) * 0.22 + rng.uniform(-0.05, 0.05, len(rr))
            ax.scatter(x, rr, marker=LANG[lang][1], s=16, color=C_KNOWN, edgecolor="white",
                       linewidth=0.4, zorder=3, label=LANG[lang][0] if Q == "T" and mi == 0 else None)
        allr = np.array(sum(ranks.values(), []))
        med = np.median(allr)
        ax.plot([mi - 0.32, mi + 0.32], [med, med], color=C_ACCENT, lw=1.6, zorder=4,
                label="median rank" if Q == "T" and mi == 0 else None)
        # rank-1 count (top of the panel; y axis is inverted so small y is up)
        ax.text(mi, 0.6, f"{(allr == 1).sum()}", ha="center", va="bottom", fontsize=6.8, color=INK2)
    # nested-written control (L1): does text written as bound prefer the bound order?
    bs = [v["L1"]["best_stacked"] for k, v in d.items() if k.endswith("nested")]
    exact = sum(b == cur for b in bs)
    kendall = lambda b: sum(1 for i in range(len(b)) for j in range(i + 1, len(b)) if b[i] > b[j])  # inversions vs 1..S
    near = sum(kendall(b) <= 3 for b in bs)
    S = NSHEETS[Q]
    outer_inner = sum(abs(b.index("1") - b.index(str(S))) == 1 for b in bs)  # outermost next to innermost
    NESTED_NOTE.append((Q, exact, near, outer_inner, len(bs)))
    ax.set_yscale("log")
    ax.set_ylim(N * 1.6, 0.42)
    ax.set_yticks([1, 10, 100, 720] if N == 720 else ([1, 10, 120] if N == 120 else [1, 10, 24]))
    ax.set_yticklabels([str(t) for t in ax.get_yticks()])
    ax.set_xticks(range(4))
    ax.set_xticklabels(METRICS, fontsize=6.8, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, 3.6)
    ax.set_title(f"quire {Q}\n{NSHEETS[Q]}! = {N} orders", fontsize=7.2, pad=12)
    if Q == "T":
        ax.text(-0.5, 0.6, "rank 1:", ha="right", va="bottom", fontsize=6.8, color=INK2, clip_on=False)
    ax.text(3.55, 10.5, "top 10", ha="right", va="bottom", fontsize=6.0, color=MUTED)
    ax.grid(axis="y", which="major")
    ax.grid(axis="y", which="minor", alpha=0.35)
    if Q == "T":
        ax.set_ylabel("rank of the true writing order\n(log scale; 1 = recovered)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, fontsize=6.8,
           handletextpad=0.3, columnspacing=1.2, frameon=False)
note = ("18 stacked-written prose cases per quire (3 texts × 3 windows × {current, random} sheet order); "
        "numbers above each column = cases recovered at rank 1")
print("as-bound control (exact / <=3 inversions / 1-next-to-S):", NESTED_NOTE)
fig.text(0.5, 0.09, note, ha="center", fontsize=6.6, color=INK2, va="bottom")
save(fig, "fig_seriation_power")
print("ok")
