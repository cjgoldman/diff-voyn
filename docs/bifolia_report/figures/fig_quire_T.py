"""fig_quire_T: quire T (f103–f116; 6 sheets, 23 pages, Currier B, hand 3, 10 673 tokens).

(a) All 720 stacked sheet orders scored on the real contents (between-sheet L1, 7-grams,
    IT2a), as within-candidate z; the same on 20 content shuffles (grey outline) and the
    best-of-720 on each of 200 shuffles (grey rug/hist).  The real winner 1-6-5-4-2-3 sits
    in a left tail no shuffle produces.
(b) Winner's z within the candidate set, per unit and transcription, against the null
    distribution of the shuffled winner (200 draws, L1).
(c) Between-sheet affinity: shared rare pairs observed / expected under content
    permutation (words and 7-grams, IT2a); sheets 1 = outermost (f103/f116) … 6 = innermost
    (f108/f111).
Source: ../data/derived_quire_T_costs.json (re-derived from the recorded analysis code,
scripts/quire_order_poc.py, see derive_quire_T.py) — the recorded quire_order_burst_T.json
stores only summaries; recorded values cross-checked: best 1-6-5-4-2-3, z −3.96 (n7 L1).
docs/doubleton_gaps.md §15, §17.
"""
import json
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from style import *  # noqa

D = json.load(open("../data/derived_quire_T_costs.json"))
REC = json.load(open("../data/quire_order_burst_T.json"))
UNITS = ["words", "n5", "n6", "n7", "n8"]
UNIT_LBL = {"words": "words", "n5": "5-gram", "n6": "6-gram", "n7": "7-gram", "n8": "8-gram"}
SHEETS = ["1 f103/f116", "2 f104/f115", "3 f105/f114", "4 f106/f113", "5 f107/f112", "6 f108/f111"]
WIN = "165423"


def dash(l):
    return "-".join(l)


fig = plt.figure(figsize=(W_FULL, 6.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1, 1.05], hspace=0.8, wspace=0.3,
                      left=0.09, right=0.98, top=0.95, bottom=0.12)

# ------------------------------------------------------------------ (a)
ax = fig.add_subplot(gs[0, 0])
u = D["IT2a"]["units"]["n7"]
labels = D["IT2a"]["labels"]
c = np.array(u["L1_between"])
z = (c - c.mean()) / c.std()
nz = np.array(u["null_cand_z_sample"]).ravel()
nb = np.array(u["null_best_z"])
bins = np.linspace(-4.3, 3.3, 39)
ax.hist(nz, bins=bins, density=True, histtype="step", color=MUTED, lw=1.0, label="shuffled contents, 20 draws", zorder=2)
ax.hist(z, bins=bins, density=True, color=C_BLUE_WASH, edgecolor=C_IT2A, lw=0.6, label="real contents", zorder=3)
# null best-of-720
ax.hist(nb, bins=np.linspace(-4.3, -2.0, 24), weights=np.full(len(nb), 0.06 / (len(nb) * (2.3 / 24))) * 1.0,
        color=C_NULL, lw=0, label="shuffled best-of-720 (200)", zorder=1)
wi = int(np.argmin(z)); wz = z[wi]
rz = z[labels.index(labels[wi][::-1])]
ax.axvline(wz, color=C_ACCENT, lw=1.4, zorder=5)
ax.axvline(rz, color=C_ACCENT, lw=0.8, ls=(0, (2, 2)), zorder=5)
ax.text(0.02, 0.97, f"solid: argmin {dash(labels[wi])} (z {wz:+.2f})\ndashed: its reversal {dash(labels[wi][::-1])} (z {rz:+.2f}, rank {int((z < rz).sum()) + 1})",
        transform=ax.transAxes, fontsize=5.9, color=C_ACCENT, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_ACCENT, lw=0.5))
# the family: label the 5 lowest
order = np.argsort(z)
fam = [labels[i] for i in order[:6]]
ax.text(1.5, 0.45, "next best:\n" + "\n".join(f"{dash(labels[i])} {z[i]:+.2f}" for i in order[1:6]),
        fontsize=5.6, color=INK2, va="top", ha="left", family="monospace")
ax.set_xlim(-4.3, 3.3)
ax.set_ylim(0, 0.82)
ax.set_xlabel("within-candidate z of the order's cost (between-sheet L1, 7-grams, IT2a)", fontsize=7)
ax.set_ylabel("density", fontsize=7)
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.8), fontsize=6.0, handlelength=1.4, borderaxespad=0.0)
panel_label(ax, "(a)", x=-0.1)

# ------------------------------------------------------------------ (b)
ax = fig.add_subplot(gs[0, 1])
for ui, un in enumerate(UNITS):
    for ti, (tr, col, mk) in enumerate((("IT2a", C_IT2A, MARK_IT2A), ("RF1b", C_RF1B, MARK_RF1B))):
        uu = D[tr]["units"][un]
        cc = np.array(uu["L1_between"]); zz = (cc - cc.mean()) / cc.std()
        nbz = np.array(uu["null_best_z"])
        y = ui + (ti - 0.5) * 0.36
        q = np.percentile(nbz, [0, 25, 50, 75, 100])
        ax.plot([q[0], q[4]], [y, y], color=C_NULL, lw=0.8, zorder=1)
        ax.add_patch(plt.Rectangle((q[1], y - 0.13), q[3] - q[1], 0.26, color=GRID, lw=0, zorder=2))
        ax.plot([q[2], q[2]], [y - 0.13, y + 0.13], color=MUTED, lw=0.9, zorder=3)
        best_i = int(np.argmin(zz))  # the un-folded argmin (for T: 3-2-4-5-6-1, reversal of 1-6-5-4-2-3)
        ax.scatter([zz[best_i]], [y], s=22, color=col, marker=mk, edgecolor="white", linewidth=0.5, zorder=4,
                   label=f"real winner, {tr}" if ui == 0 else None)
        p = (np.sum(nbz <= zz[best_i]) + 1) / (len(nbz) + 1)
        ax.text(-1.55, y, f"{p:.3f}" if p < 0.01 else f"{p:.2f}", fontsize=5.9, color=INK2, va="center")
        if labels[best_i] != WIN and labels[best_i][::-1] != WIN:
            ax.text(zz[best_i], y - 0.2, dash(labels[best_i]), fontsize=5.4, color=col, ha="center", va="bottom")
ax.set_ylim(len(UNITS) - 0.5, -0.7)
ax.set_yticks(range(len(UNITS)))
ax.set_yticklabels([UNIT_LBL[x] for x in UNITS], fontsize=6.8)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(-4.4, -1.1)
ax.set_xticks([-4, -3, -2])
ax.grid(axis="x", color=GRID, lw=0.5)
ax.grid(axis="y", visible=False)
ax.text(-1.55, -0.72, "p", fontsize=6.2, color=MUTED, va="bottom")
ax.set_xlabel("winner's z within the 720 candidates (L1)", fontsize=7)
from matplotlib.patches import Patch
h, l = ax.get_legend_handles_labels()
h.append(Patch(color=GRID)); l.append("shuffled winners (200): IQR, range, median")
ax.legend(h, l, loc="upper left", bbox_to_anchor=(-0.02, -0.2), fontsize=6.2, ncol=2, frameon=False, columnspacing=0.8, handletextpad=0.4)
panel_label(ax, "(b)", x=-0.2)

# ------------------------------------------------------------------ (c)
norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.45, vmax=0.45)
for k, un in enumerate(["words", "n7"]):
    ax = fig.add_subplot(gs[1, k])
    uu = D["IT2a"]["units"][un]
    obs, exp = np.array(uu["affinity_obs"]), np.array(uu["affinity_exp"])
    S = obs.shape[0]
    R = np.full((S, S), np.nan)
    for i in range(S):
        for j in range(i + 1, S):
            R[i, j] = R[j, i] = np.log2(obs[i, j] / exp[i, j])
    im = ax.imshow(R, cmap=CMAP_DIV, norm=norm, interpolation="nearest")
    for i in range(S):
        for j in range(S):
            if i != j:
                ax.text(j, i, f"{2 ** R[i, j]:.2f}", ha="center", va="center", fontsize=6.2,
                        color="white" if abs(R[i, j]) > 0.3 else INK)
            else:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=GRID, lw=0))
    # draw the winning chain as arcs on the upper side: pairs adjacent in 1-6-5-4-2-3
    chain = [int(ch) - 1 for ch in WIN]
    for a, b in zip(chain[:-1], chain[1:]):
        i, j = min(a, b), max(a, b)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=C_ACCENT, lw=1.6, zorder=5))
        ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, edgecolor=C_ACCENT, lw=1.6, zorder=5))
    ax.set_xticks(range(S)); ax.set_yticks(range(S))
    ax.set_xticklabels([s.split()[0] for s in SHEETS], fontsize=7)
    ax.set_yticklabels(SHEETS if k == 0 else [s.split()[0] for s in SHEETS], fontsize=6.6)
    ax.tick_params(length=0)
    ax.grid(False)
    despine_all(ax)
    ax.set_title(f"{UNIT_LBL[un]}s, IT2a" if un != "words" else "words, IT2a", fontsize=7.2, loc="left")
fig.text(0.09, 0.50, f"(c) shared rare pairs between sheets, observed / expected under content permutation; orange frame = sheet pair adjacent in {dash(WIN)}",
         fontsize=7, color=INK, va="bottom")
cax = fig.add_axes([0.3, 0.055, 0.4, 0.012])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_ticks([-0.4, -0.2, 0, 0.2, 0.4]); cb.set_ticklabels(["×0.76", "×0.87", "×1", "×1.15", "×1.32"])
cb.ax.tick_params(labelsize=6.2, length=2); cb.outline.set_visible(False)
cb.set_label("observed / expected (log₂ colour scale)", fontsize=6.6, labelpad=2)
save(fig, "fig_quire_T")
print("ok; argmin", labels[wi], "z", round(wz, 3), "recorded (canonical label)", REC["IT2a"]["n7"]["L1"]["best"], REC["IT2a"]["n7"]["L1"]["z_best_within"])
