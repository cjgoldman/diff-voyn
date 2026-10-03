"""Executive-summary figure: how much closer together than chance a text's rare
words sit (record §2, §6.1, §9, §10).

One bar per text: the probability that two consecutive uses of a word used two to
five times fall within 100 words of each other, divided by the same probability
under random placement (r_100, pooled k = 2..5, the corpus-sweep statistic).
1 = no clustering at all.  Known prose and verse are windows cut to the
manuscript's length; the manuscript is shown twice, as bound (light blue) and in
the best stacked sheet order the annealer finds under no constraint (dark blue).

Bars are sorted by size across all texts.

Sources: data/corpus_sweep.json (known windows [0] and VMS IT2a pooled r100 — the
as-bound value; the optimiser's own objective gives 3.88 for the same order) and
data/order_optimize_none.json (vms_stacked['free'].r100 — orientation fixed, any
sheet may swap with any other).
"""

import json
import os

import numpy as np

from style import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
sweep = json.load(open(os.path.join(DATA, "corpus_sweep.json")))
opt = json.load(open(os.path.join(DATA, "order_optimize_none.json")))


def sweep_r100(lang, key):
    for e in sweep:
        if e[0] == lang and e[1] == key:
            return e[2]["pooled"]["r100"]
    raise KeyError((lang, key))


KNOWN = [
    ("Isidore, Etymologiae (Latin prose)", "latin", "corpuscorporum_auctores_scientiarum_varii_isidorus[0]", "prose"),
    ("Staden (German prose)", "german", "dta_1557_staden_landschafft_1557_staden_landschaff[0]", "prose"),
    ("Pliny, Natural History (Latin prose)", "latin", "corpuscorporum_auctores_scientiarum_varii_plinius_[0]", "prose"),
    ("Bullinger (German prose)", "german", "dta_1558_bullinger_haussbuoch_1558_bullinger_hauss[0]", "prose"),
    ("Boccaccio, Decameron (Italian prose)", "italian", "boccaccio_decameron[0]", "prose"),
    ("Seneca, Natural Questions (Latin prose)", "latin", "corpuscorporum_auctores_scientiarum_varii_seneca_n[0]", "prose"),
    ("Ariosto, Orlando Furioso (Italian verse)", "italian", "ariosto_orlando_furioso[0]", "verse"),
    ("Dante, Commedia (Italian verse)", "italian", "dante_divina_commedia[0]", "verse"),
]
rows = [(lab, sweep_r100(lang, key), kind) for lab, lang, key, kind in KNOWN]
v_bound = sweep_r100("VMS", "IT2a")
v_best = opt["vms_stacked"]["free"]["r100"]
rows += [
    ("Voynich MS, as bound", v_bound, "bound"),
    ("Voynich MS, bifolia hypothesis", v_best, "best"),
]
rows.sort(key=lambda r: -r[1])  # every bar by size; the manuscript interleaves with the verse
n_prose = sum(r[2] == "prose" for r in rows)

C_BOUND, C_BEST = "#86b6ef", C_RF1B
fig, ax = plt.subplots(figsize=(W_FULL, 2.4))
y = np.arange(len(rows))[::-1]
for yi, (lab, v, kind) in zip(y, rows):
    col = {"prose": C_KNOWN, "verse": C_KNOWN, "bound": C_BOUND, "best": C_BEST}[kind]
    ec = {"prose": C_KNOWN, "verse": C_KNOWN, "bound": C_IT2A, "best": C_BEST}[kind]
    hatch = "////" if kind == "verse" else None
    ax.barh(yi, v - 1, left=1, height=0.62, color=col, edgecolor=ec, linewidth=0.6, hatch=hatch, zorder=3)
    ax.text(v * 1.06, yi, f"×{v:.0f}" if v >= 10 else f"×{v:.1f}", va="center", ha="left", fontsize=7.2,
            color=INK if kind in ("prose", "verse") else ec)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=7.4)
for t, (_, _, kind) in zip(ax.get_yticklabels(), rows):
    if kind == "bound":
        t.set_color(C_IT2A)
    elif kind == "best":
        t.set_color(C_BEST)
ax.tick_params(axis="y", length=0)
ax.set_xscale("log")
ax.set_xlim(1, 90)
ax.set_xticks([1, 2, 5, 10, 20, 50])
ax.set_xticklabels(["×1", "×2", "×5", "×10", "×20", "×50"])
ax.set_xlabel("how much more often than chance two consecutive uses of a rare word fall within 100 words of each other")
ax.axvline(1, color=AXIS, lw=0.8, zorder=1)
ax.text(1.03, len(rows) - 0.35, "×1 = no clustering (random placement)", fontsize=6.6, color=MUTED, va="bottom")
ax.grid(axis="x", color=GRID, lw=0.5)
ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
# one separator: prose above, verse and the manuscript (interleaved by size) below
ax.axhline(y[n_prose - 1] - 0.5, color=AXIS, lw=0.8)
ax.text(84, y[n_prose // 2] - 0.5, "known prose", fontsize=7, color=INK2, ha="right", va="center", style="italic")
ax.text(84, y[n_prose + (len(rows) - n_prose) // 2] - 0.5, "known verse\nand the manuscript", fontsize=7, color=INK2,
        ha="right", va="center", style="italic", linespacing=1.15)
fig.subplots_adjust(left=0.30, right=0.98, top=0.97, bottom=0.18)
save(fig, "fig_es_locality")
print("ok", [(r[0][:24], round(r[1], 2)) for r in rows])
