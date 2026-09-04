"""Executive-summary figure: how much closer together than chance a text's rare
words sit (record §2, §6.1).

One bar per text: the probability that two consecutive uses of a word used exactly
twice fall within 100 words of each other, divided by the same probability under
random placement (r_100 at k = 2).  1 = no clustering at all.  Known prose, known
verse, and the manuscript (both transcriptions, and each Currier language alone).

Source: data/rare_types.json (vms[tr].classes['2'], vms[tr].currier_A/B['2'],
known[text]['2'] -> gap_le_100_ratio).
"""

import json
import os

import numpy as np

from style import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
rare = json.load(open(os.path.join(HERE, "..", "data", "rare_types.json")))

KNOWN = [
    ("Isidore, Etymologiae (Latin prose)", "la_isidorus_etym", "prose"),
    ("Bullinger (German prose)", "de_bullinger", "prose"),
    ("Staden (German prose)", "de_staden", "prose"),
    ("Pliny, Natural History (Latin prose)", "la_plinius_nh", "prose"),
    ("Seneca, Natural Questions (Latin prose)", "la_seneca_nq", "prose"),
    ("Boccaccio, Decameron (Italian prose)", "it_decameron", "prose"),
    ("Ariosto, Orlando Furioso (Italian verse)", "it_orlando_furioso", "verse"),
    ("Dante, Commedia (Italian verse)", "it_commedia", "verse"),
]
rows = [(lab, rare["known"][k]["2"]["gap_le_100_ratio"], kind) for lab, k, kind in KNOWN]
rows.sort(key=lambda r: -r[1])
vms_rows = [
    ("Voynich MS, whole text (IT2a transcription)", rare["vms"]["IT2a"]["classes"]["2"]["gap_le_100_ratio"], "IT2a"),
    ("Voynich MS, whole text (RF1b transcription)", rare["vms"]["RF1b"]["classes"]["2"]["gap_le_100_ratio"], "RF1b"),
    ("Voynich MS, Currier A pages alone (IT2a)", rare["vms"]["IT2a"]["currier_A"]["2"]["gap_le_100_ratio"], "sub"),
    ("Voynich MS, Currier B pages alone (IT2a)", rare["vms"]["IT2a"]["currier_B"]["2"]["gap_le_100_ratio"], "sub"),
]
rows = rows + vms_rows

fig, ax = plt.subplots(figsize=(W_FULL, 2.55))
y = np.arange(len(rows))[::-1]
for yi, (lab, v, kind) in zip(y, rows):
    col = {"prose": C_KNOWN, "verse": C_KNOWN, "IT2a": C_IT2A, "RF1b": C_RF1B, "sub": C_BLUE_WASH}[kind]
    ec = C_IT2A if kind == "sub" else col
    hatch = "////" if kind == "verse" else None
    ax.barh(yi, v - 1, left=1, height=0.62, color=col, edgecolor=ec, linewidth=0.6, hatch=hatch, zorder=3)
    ax.text(v * 1.06, yi, f"×{v:.0f}" if v >= 10 else f"×{v:.1f}", va="center", ha="left", fontsize=7.2,
            color=INK if kind in ("prose", "verse") else ec)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=7.4)
ax.tick_params(axis="y", length=0)
ax.set_xscale("log")
ax.set_xlim(1, 90)
ax.set_xticks([1, 2, 5, 10, 20, 50])
ax.set_xticklabels(["×1", "×2", "×5", "×10", "×20", "×50"])
ax.set_xlabel("how much more often than chance the two uses of a twice-used word fall within 100 words of each other")
ax.axvline(1, color=AXIS, lw=0.8, zorder=1)
ax.text(1.03, len(rows) - 0.35, "×1 = no clustering (random placement)", fontsize=6.6, color=MUTED, va="bottom")
ax.grid(axis="x", color=GRID, lw=0.5)
ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
# group separators
ax.axhline(y[6] + 0.5, color=GRID, lw=0.6)
ax.axhline(y[8] + 0.5, color=AXIS, lw=0.8)
ax.text(84, y[3] - 0.1, "known prose", fontsize=7, color=INK2, ha="right", va="center", style="italic")
ax.text(84, y[7] - 0.1, "known verse", fontsize=7, color=INK2, ha="right", va="center", style="italic")
ax.text(84, y[9] - 0.6, "the manuscript", fontsize=7, color=C_IT2A, ha="right", va="center", style="italic")
fig.subplots_adjust(left=0.42, right=0.98, top=0.97, bottom=0.17)
save(fig, "fig_es_locality")
print("ok", [(r[0][:20], round(r[1], 2)) for r in rows])
