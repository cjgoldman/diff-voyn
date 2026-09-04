"""Figure: how local is the manuscript's rare vocabulary compared with known texts?

(a) P(consecutive gap <= 100 tokens) / uniform-placement expectation, by frequency
    class k = 2..5 (docs/doubleton_gaps.md §6.1; data/rare_types.json).
(b) Corpus sweep: the same ratio, pooled over k = 2..5, for 148 known 37 759-token
    windows (§9; data/corpus_sweep.json), with the manuscript marked.
"""

import json
import os

import numpy as np

from style import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

rare = json.load(open(os.path.join(DATA, "rare_types.json")))
sweep = json.load(open(os.path.join(DATA, "corpus_sweep.json")))

KS = [2, 3, 4, 5]
KNOWN = {
    # key: (label, language marker key, is_verse)
    "la_isidorus_etym": ("Isidore", "la", False),
    "la_plinius_nh": ("Pliny", "la", False),
    "la_seneca_nq": ("Seneca NQ", "la", False),
    "de_bullinger": ("Bullinger", "de", False),
    "de_staden": ("Staden", "de", False),
    "it_decameron": ("Decameron", "it", False),
    "it_commedia": ("Commedia", "it_verse", True),
    "it_orlando_furioso": ("Orlando Furioso", "it_verse", True),
}


def ratios(d):
    return [d[str(k)]["gap_le_100_ratio"] for k in KS]


fig, (ax, bx) = plt.subplots(1, 2, figsize=(W_FULL, 3.5), gridspec_kw={"width_ratios": [1.05, 1.25], "wspace": 0.32})

# ---- (a) by frequency class -------------------------------------------------
ends = []
for key, (label, lang, verse) in KNOWN.items():
    y = ratios(rare["known"][key])
    ax.plot(KS, y, color=C_KNOWN, lw=0.9, ls="--" if verse else "-", marker=MARK_LANG[lang],
            ms=3.6, mfc="white" if verse else C_KNOWN, mec=C_KNOWN, zorder=2)
    ends.append((y[-1], label))
# spread the end labels in log space so they do not collide (min 0.075 dex apart)
ends.sort()
pos = [np.log10(v) for v, _ in ends]
for i in range(1, len(pos)):
    pos[i] = max(pos[i], pos[i - 1] + 0.075)
for (v, label), lp in zip(ends, pos):
    ax.annotate(label, (KS[-1], v), xytext=(5.12, 10 ** lp), textcoords="data",
                fontsize=6.3, color=INK2, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.5, shrinkA=0, shrinkB=1))

for tr, col, mk in (("IT2a", C_IT2A, MARK_IT2A), ("RF1b", C_RF1B, MARK_RF1B)):
    y = ratios(rare["vms"][tr]["classes"])
    ax.plot(KS, y, color=col, lw=1.6, marker=mk, ms=4.5, zorder=4, label=f"manuscript {tr}")
for tr, sub, lab in (("IT2a", "currier_A", "Currier A alone"), ("IT2a", "currier_B", "Currier B alone")):
    y = ratios(rare["vms"][tr][sub])
    ax.plot(KS, y, color=C_IT2A, lw=0.9, ls=":", marker=MARK_IT2A, ms=3, mfc="white", zorder=3, label=lab)

ax.axhline(1, color=AXIS, lw=0.8, zorder=1)
ax.text(5.05, 1, "uniform", fontsize=6.3, color=MUTED, va="center", ha="left")
ax.set_yscale("log")
ax.set_ylim(0.8, 80)
ax.set_yticks([1, 2, 5, 10, 20, 50])
ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
ax.set_xticks(KS)
ax.set_xlim(1.8, 6.4)
ax.set_xlabel("occurrences of the type, $k$")
ax.set_ylabel("P(consecutive gap ≤ 100) / uniform")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=2.2, columnspacing=1.0)
ax.text(1.9, 62, "known texts (grey; verse dashed, hollow)", fontsize=6.5, color=INK2)
panel_label(ax, "(a)")

# ---- (b) corpus sweep -------------------------------------------------------
LANG_ROW = {"latin": (2, "Latin (25)", MARK_LANG["la"]), "german": (1, "German (113)", MARK_LANG["de"]),
            "italian": (0, "Italian (10)", MARK_LANG["it"])}
rng = np.random.default_rng(0)
for lang, (row, label, mk) in LANG_ROW.items():
    vals = np.array([e[2]["pooled"]["r100"] for e in sweep if e[0] == lang])
    yy = row + rng.uniform(-0.28, 0.28, size=len(vals))
    bx.scatter(vals, yy, s=13, marker=mk, color=C_KNOWN, edgecolor="white", linewidth=0.4, zorder=3, label=label)
known = np.array([e[2]["pooled"]["r100"] for e in sweep if e[0] in LANG_ROW])
# source: data/corpus_sweep.log (known 5th pct 4.66, median 12.87)
for v, lab in ((4.66, "5th pct 4.7"), (12.87, "median 12.9")):
    bx.axvline(v, color=AXIS, lw=0.7, ls=(0, (3, 2)), zorder=1)
    bx.text(v * (1.06 if v < 6 else 1.0), -0.95 if v > 6 else -0.32, lab, fontsize=6.3, color=MUTED,
            ha="left" if v < 6 else "center", va="bottom")

vms = {e[1]: e[2]["pooled"]["r100"] for e in sweep if e[0] == "VMS"}
for tr, col in (("IT2a", C_IT2A), ("RF1b", C_RF1B)):
    bx.axvline(vms[tr], color=col, lw=1.8, zorder=4)
# percentile among the 148 de-duplicated known windows drawn here (the record's 2.5 % / 3.8 %
# were computed on the sweep log's 158 windows, which count the ten Italian raw texts twice; §9 footnote)
pct = {tr: 100.0 * np.mean(known < vms[tr]) for tr in ("IT2a", "RF1b")}
bx.annotate(f"IT2a {vms['IT2a']:.2f}\n({pct['IT2a']:.1f}th pct)", (vms["IT2a"], 2.75), xytext=(-4, 0),
            textcoords="offset points", ha="right", va="center", fontsize=6.5, color=C_IT2A)
bx.annotate(f"RF1b {vms['RF1b']:.2f}\n({pct['RF1b']:.1f}th pct)", (vms["RF1b"], 2.75), xytext=(4, 0),
            textcoords="offset points", ha="left", va="center", fontsize=6.5, color=C_RF1B)
print("percentiles on 148 windows:", {k: round(v, 1) for k, v in pct.items()})
for sub in ("IT2a-A", "IT2a-B", "RF1b-A", "RF1b-B"):
    bx.axvline(vms[sub], color=C_BLUE_WASH, lw=1.2, zorder=2)
bx.annotate("Currier\nA / B\nalone", (min(vms[s] for s in ("IT2a-A", "IT2a-B", "RF1b-A", "RF1b-B")), -0.75),
            xytext=(1.9, -0.98), textcoords="data", fontsize=6.3, color=C_IT2A, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", color=C_BLUE_WASH, lw=0.8, shrinkA=0, shrinkB=0))

bx.set_xscale("log")
bx.set_xlim(1.85, 40)
bx.set_xticks([2, 3, 5, 10, 20, 30])
bx.set_xticklabels(["2", "3", "5", "10", "20", "30"])
bx.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
bx.set_ylim(-1.05, 3.15)
bx.set_yticks([0, 1, 2])
bx.set_yticklabels(["Italian", "German", "Latin"])
bx.grid(False)
bx.grid(True, axis="x")
ax.set_xticks(KS)
bx.set_xlabel("P(gap ≤ 100) / uniform, types with 2–5 occurrences pooled")
bx.set_title("148 known windows of 37 759 tokens", loc="left", pad=10)
panel_label(bx, "(b)", x=-0.2)

save(fig, "fig_locality_baseline")
