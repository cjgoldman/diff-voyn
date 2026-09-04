"""fig_quires_all: one-quire seriation across quires T, M, C, A, B.

Per quire, per unit (words, glyph n-grams n5..n8) and metric (L1, burst, blog, modal), the
best stacked sheet order and its content-shuffle p (best-of-candidates on the real
contents vs best-of-candidates on 200 draws with page contents permuted within the
quire; floor 1/201).  Cell colour = -log10 p (single-hue sequential); cell text = the
best order, bold when it belongs to the quire's winning family.  Right-hand strip: rank
of the current (as-bound, outer→inner) order among the S! candidates.
Source: quire_order_burst_{T,M,C,A,B}.json (docs/doubleton_gaps.md §14-15).
Sheet labels 1..S = outermost..innermost; the digit string is the stacked reading order.
"""
import json
import numpy as np
from matplotlib.colors import Normalize
from style import *  # noqa

QUIRES = ["T", "M", "C", "A", "B"]
NCAND = {"T": 720, "M": 120, "C": 24, "A": 24, "B": 24}
UNITS = ["words", "n5", "n6", "n7", "n8"]
UNIT_LBL = {"words": "words", "n5": "5-gram", "n6": "6-gram", "n7": "7-gram", "n8": "8-gram"}
METRICS = ["L1", "burst", "blog", "modal"]
TRS = ["IT2a", "RF1b"]
# the winning family per quire (docs §15): exact best or one adjacent transposition away
FAMILY = {"T": ["165423"], "M": ["23514", "32514", "41235"], "C": ["1432"], "A": [], "B": []}


def near(lbl, fam):
    """lbl is in the family, or one adjacent transposition away; orders are compared up to
    reversal because the recorded 'best' is the canonical (smaller) member of each pair."""
    cands = {lbl, lbl[::-1]}
    for f in fam:
        variants = {f, f[::-1]}
        l = list(f)
        for i in range(len(l) - 1):
            m = l.copy(); m[i], m[i + 1] = m[i + 1], m[i]
            variants.add("".join(m)); variants.add("".join(m)[::-1])
        if cands & variants:
            return True
    return False


def dash(lbl):
    return "-".join(lbl)


norm = Normalize(vmin=0, vmax=np.log10(201))
fig = plt.figure(figsize=(W_FULL, 7.4))
gs = fig.add_gridspec(len(QUIRES), 3, width_ratios=[1, 1, 0.42], hspace=0.55, wspace=0.12,
                      left=0.09, right=0.98, top=0.93, bottom=0.075)
for qi, Q in enumerate(QUIRES):
    d = json.load(open(f"../data/quire_order_burst_{Q}.json"))
    for ti, tr in enumerate(TRS):
        ax = fig.add_subplot(gs[qi, ti])
        P = np.zeros((len(UNITS), len(METRICS)))
        for ui, u in enumerate(UNITS):
            for mi, m in enumerate(METRICS):
                c = d[tr][u][m]
                P[ui, mi] = -np.log10(c["p_best"])
                lbl = c["best"]
                fam = near(lbl, FAMILY[Q])
                ax.text(mi, ui, dash(lbl), ha="center", va="center", fontsize=5.6 if len(lbl) > 5 else 6.2,
                        color="white" if P[ui, mi] > 1.25 else INK,
                        fontweight="bold" if fam else "normal")
        ax.imshow(P, cmap=CMAP_SEQ, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(METRICS)))
        ax.set_xticklabels(METRICS if qi == len(QUIRES) - 1 else [""] * 4, fontsize=7)
        ax.set_yticks(range(len(UNITS)))
        ax.set_yticklabels([UNIT_LBL[u] for u in UNITS] if ti == 0 else [""] * 5, fontsize=6.8)
        ax.tick_params(length=0)
        ax.grid(False)
        despine_all(ax)
        ax.set_title(f"quire {Q} · {tr}", fontsize=7.6, loc="left", pad=3)
        # thin white gaps between cells
        for k in range(1, len(METRICS)):
            ax.axvline(k - 0.5, color="white", lw=1.2)
        for k in range(1, len(UNITS)):
            ax.axhline(k - 0.5, color="white", lw=1.2)
    # current-order rank strip
    ax = fig.add_subplot(gs[qi, 2])
    N = NCAND[Q]
    for ti, tr in enumerate(TRS):
        for ui, u in enumerate(UNITS):
            rk = [d[tr][u][m]["current_rank"] for m in METRICS]
            ax.scatter(np.array(rk) / N, [ui + (ti - 0.5) * 0.3] * 4, s=9,
                       color=C_IT2A if tr == "IT2a" else C_RF1B,
                       marker=MARK_IT2A if tr == "IT2a" else MARK_RF1B, edgecolor="white", linewidth=0.3, zorder=3,
                       label=tr if (qi == 0 and ui == 0) else None)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(len(UNITS) - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["1", f"{N//2}", f"{N}"], fontsize=6.2)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.axvline(1 / N, color=C_STACKED, lw=0.8, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"rank of bound order\namong {N}", fontsize=6.6, loc="left", pad=3)
    if qi == 0:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.32), ncol=2, fontsize=6.2, handletextpad=0.1,
                  borderaxespad=0.0, markerscale=1.2, columnspacing=0.8)

# colourbar
cax = fig.add_axes([0.09, 0.02, 0.42, 0.012])
sm = plt.cm.ScalarMappable(cmap=CMAP_SEQ, norm=norm)
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_ticks([0, 1, np.log10(20), np.log10(201)])
cb.set_ticklabels(["p = 1", "0.1", "0.05", "0.005 (floor)"])
cb.ax.tick_params(labelsize=6.5, length=2)
cb.outline.set_visible(False)
cb.set_label("content-shuffle p of the best order (200 draws)", fontsize=6.8, labelpad=2)
fig.text(0.58, 0.02, "cell text = best stacked order, shown up to reversal\n"
         "(sheet 1 = outermost … S = innermost); bold = the quire's winning\n"
         "family (T 1-6-5-4-2-3, C 1-4-3-2, M 2-3-5-1-4 / 3-2-5-1-4 / 4-1-2-3-5)\n"
         "or one adjacent transposition away from it",
         fontsize=6.3, color=INK2, va="center")
save(fig, "fig_quires_all")
print("ok")
