"""fig_direction: is the reading direction along the sheet chain established?  No.

(a) Direction gap Δ = cost(reversal of best) − cost(best), in candidate-sd units, for the
    real contents (marker) against the null distribution of the same quantity on 200
    content shuffles (each shuffle's own best vs its reversal, so the selection effect is
    in the null).  Quires T, M, C; units words, 5..8-grams; metrics L1 and modal
    (nullshape run).  Source: quire_order_nullshape_{T,M,C}.json (null quantiles) and
    quire_order_direction_{T,M,C}.json (real Δ, p; IT2a).  docs/doubleton_gaps.md §16-17.
(b) Burst front-loading: fraction of rare types (3-10 occurrences inside one segment)
    whose first inter-occurrence gap is shorter than the last.  0.5 = symmetric burst.
    Prose: six known texts (1 500-token segments) and pooled; manuscript: sheets read
    a-r, a-v, b-r, b-v (all / Currier A / Currier B) and single pages, both
    transcriptions.  Source: burst_frontloading.json (§18).
"""
import json
import numpy as np
from style import *  # noqa

# ---------------------------------------------------------------- panel (a)
QUIRES = ["T", "M", "C"]
UNITS = ["words", "n5", "n6", "n7", "n8"]
UNIT_LBL = {"words": "words", "n5": "5-gram", "n6": "6-gram", "n7": "7-gram", "n8": "8-gram"}
METRICS = ["L1", "modal"]

fig = plt.figure(figsize=(W_FULL, 5.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.62, wspace=0.28, left=0.09, right=0.98, top=0.93, bottom=0.08)
for qi, Q in enumerate(QUIRES):
    ax = fig.add_subplot(gs[0, qi])
    ns = json.load(open(f"../data/quire_order_nullshape_{Q}.json"))
    dr = json.load(open(f"../data/quire_order_direction_{Q}.json"))["IT2a"]
    for ui, u in enumerate(UNITS):
        for mi, m in enumerate(METRICS):
            y = ui + (mi - 0.5) * 0.36
            q = ns[u][m]["null_delta"]
            # null: min–max whisker, interquartile box, median tick
            ax.plot([q["min"], q["max"]], [y, y], color=C_NULL, lw=0.8, solid_capstyle="butt", zorder=1)
            ax.add_patch(plt.Rectangle((q["q25"], y - 0.13), q["q75"] - q["q25"], 0.26, color=GRID, lw=0, zorder=2))
            ax.plot([q["q50"], q["q50"]], [y - 0.13, y + 0.13], color=MUTED, lw=0.9, zorder=3)
            real = dr[u][m]["delta_sd"]; p = dr[u][m]["p_delta"]
            ax.scatter([real], [y], s=22 if m == "L1" else 18, marker="o" if m == "L1" else "D",
                       color=C_IT2A, edgecolor="white", linewidth=0.5, zorder=4,
                       label=(f"real Δ, {m}" if (qi == 0 and ui == 0) else None))
            ax.text(4.05, y, f"{p:.2f}", fontsize=5.9, color=INK2, va="center", ha="left")
    ax.set_ylim(len(UNITS) - 0.5, -0.6)
    ax.set_yticks(range(len(UNITS)))
    ax.set_yticklabels([UNIT_LBL[u] for u in UNITS] if qi == 0 else [""] * 5, fontsize=6.8)
    ax.set_xlim(-0.1, 4.6)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"quire {Q}", fontsize=8, loc="left")
    ax.text(4.05, -0.62, "p", fontsize=6.2, color=MUTED, ha="left", va="bottom")
    if qi == 1:
        ax.set_xlabel("direction gap Δ = cost(reversal) − cost(best)   [candidate sd]", fontsize=7.2)
    if qi == 0:
        panel_label(ax, "(a)", x=-0.34, y=1.02)
        h, l = ax.get_legend_handles_labels()
        from matplotlib.patches import Patch
        h += [Patch(color=GRID, label="null Δ: interquartile range (whisker = range, tick = median)")]
        l += ["null Δ: interquartile (whisker = range, tick = median)"]
        ax.legend(h, l, loc="upper left", bbox_to_anchor=(0.0, -0.32), ncol=3, fontsize=6.4, frameon=False,
                  handletextpad=0.4, columnspacing=1.0)

# ---------------------------------------------------------------- panel (b)
f = json.load(open("../data/burst_frontloading.json"))
ax = fig.add_subplot(gs[1, :])
units = ["words", "n5", "n6", "n7", "n8"]
texts = ["la_isidorus", "la_seneca_nq", "de_bullinger", "it_decameron", "it_commedia", "it_orlando_furioso"]
tlbl = {"la_isidorus": "Isidore", "la_seneca_nq": "Seneca", "de_bullinger": "Bullinger", "it_decameron": "Decameron",
        "it_commedia": "Commedia", "it_orlando_furioso": "Orlando F."}
groups = [("prose\npooled", None), ("prose\nby text", None), ("VMS\nall sheets", "all sheets"),
          ("VMS\nCurrier A sheets", "Currier A sheets"), ("VMS\nCurrier B sheets", "Currier B sheets"), ("VMS\nsingle pages", "pages")]
xs = np.arange(len(groups))
ax.axhline(0.5, color=AXIS, lw=0.8, zorder=1)
pooled_vals = [f["corpus"]["pooled"][u]["all"]["frac_first_lt_last"] for u in units]
bytext_vals = [f["corpus"][t][u]["all"]["frac_first_lt_last"] for t in texts for u in units]
ax.axhspan(min(bytext_vals), max(bytext_vals), color=GRID, alpha=0.45, lw=0, zorder=0)
ax.axhspan(min(pooled_vals), max(pooled_vals), color=C_NULL, alpha=0.8, lw=0, zorder=0)
off = np.linspace(-0.3, 0.3, len(units))
umark = {"words": "o", "n5": "v", "n6": "s", "n7": "D", "n8": "^"}
for ui, u in enumerate(units):
    # pooled prose
    v = f["corpus"]["pooled"][u]["all"]["frac_first_lt_last"]
    ax.scatter(xs[0] + off[ui], v, marker=umark[u], s=20, color=C_KNOWN, edgecolor="white", linewidth=0.4, zorder=4,
               label=UNIT_LBL[u])
    # by text
    for t in texts:
        v = f["corpus"][t][u]["all"]["frac_first_lt_last"]
        ax.scatter(xs[1] + off[ui] + np.random.default_rng(ui).uniform(-0.03, 0.03), v, marker=umark[u], s=11,
                   color=C_KNOWN, alpha=0.75, edgecolor="white", linewidth=0.3, zorder=3)
    # manuscript
    for gi, (glbl, key) in enumerate(groups[2:], start=2):
        for tr, col, dx in (("IT2a", C_IT2A, -0.02), ("RF1b", C_RF1B, 0.02)):
            v = f["vms"][tr][key][u]["all"]["frac_first_lt_last"]
            ax.scatter(xs[gi] + off[ui] + dx, v, marker=umark[u], s=20, color=col, edgecolor="white", linewidth=0.4, zorder=4)
# transcription legend proxies
from matplotlib.lines import Line2D
h, l = ax.get_legend_handles_labels()
from matplotlib.patches import Patch
h += [Patch(color=C_NULL, alpha=0.8), Patch(color=GRID, alpha=0.45)]
l += ["range of pooled prose (5 units)", "range of the six prose texts"]
h += [Line2D([], [], marker="o", ls="", color=C_KNOWN, label="known text"),
      Line2D([], [], marker="o", ls="", color=C_IT2A, label="VMS IT2a"),
      Line2D([], [], marker="o", ls="", color=C_RF1B, label="VMS RF1b")]
l += ["known text", "VMS IT2a", "VMS RF1b"]
ax.legend(h, l, loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=5, fontsize=6.4, frameon=False, handletextpad=0.3, columnspacing=0.9)
ax.set_xticks(xs)
ax.set_xticklabels([g[0] for g in groups], fontsize=7)
ax.set_ylabel("fraction of bursts with first gap < last gap\n(0.5 = time-symmetric burst)", fontsize=7.2)
ax.set_ylim(0.40, 0.58)
ax.annotate("Currier A sheets, glyph n-grams: back-loaded\n(p 0.005–0.10 vs within-sheet page permutation,\ndriven by k = 3 types; words show nothing)",
            xy=(xs[3] + 0.1, 0.462), xytext=(xs[3] + 0.55, 0.425), fontsize=6.0, color=INK2, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
panel_label(ax, "(b)", x=-0.1, y=1.02)
save(fig, "fig_direction")
print("ok")
