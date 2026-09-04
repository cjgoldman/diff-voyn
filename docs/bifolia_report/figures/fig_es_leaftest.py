"""Executive-summary figure: do the two halves of a sheet share more rare material
than leaves that are neighbours only in the binding?  (record §6.2, §7, §12)

The statistic D = (shared rare material between the two leaves of one sheet)
− (shared between leaves consecutive in the binding but on different sheets), as
standard deviations above what random placement of page contents produces.  Three
units: words used 2–10 times, 7-letter glyph strings within words, 7-letter strings
straddling a word boundary.  Filled markers: manuscript, two transcriptions.  Hollow:
the same with the scribal hand also held fixed.  Washes: what six real texts give on the
manuscript's own pages when written sheet by sheet (aqua) or in the binding order (red).

Sources: data/rare_types.json, rare_types_controls.json, glyph_ngrams.json,
hand_control.json.
"""

import json
import os

import numpy as np

from style import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
J = lambda n: json.load(open(os.path.join(DATA, n)))  # noqa: E731
rare, rc, gl, hand = J("rare_types.json"), J("rare_types_controls.json"), J("glyph_ngrams.json"), J("hand_control.json")
CTRL = ["la_isidorus_etym", "la_seneca_nq", "de_bullinger", "it_decameron", "it_commedia", "it_orlando_furioso"]

UNITS = [
    ("words used 2–10 times", lambda tr: rare["vms"][tr]["leaf_test_k2to10"], lambda c: rc[c]["k2to10"], "words k2to10"),
    ("7-letter glyph strings\ninside words", lambda tr: gl["vms"][tr]["n7_all_k2to10"], lambda c: gl["controls"][c].get("n7_all_k2to10"), "n7 all"),
    ("7-letter glyph strings\nacross a word boundary", lambda tr: gl["vms"][tr]["n7_cross_k2to10"], lambda c: gl["controls"][c].get("n7_cross_k2to10"), "n7 cross"),
]
TR = (("IT2a", C_IT2A, MARK_IT2A, -0.13), ("RF1b", C_RF1B, MARK_RF1B, 0.13))

fig, ax = plt.subplots(figsize=(W_FULL, 2.6))
for i, (lab, vget, cget, hkey) in enumerate(UNITS):
    for mode, wash in (("nested", C_NESTED_WASH), ("stacked", C_STACKED_WASH)):
        zs = [cget(f"{t}/{mode}")["conj_minus_nested"]["z"] for t in CTRL if cget(f"{t}/{mode}") is not None]
        ax.add_patch(mpl.patches.Rectangle((i - 0.38, min(zs)), 0.76, max(zs) - min(zs), color=wash, lw=0, zorder=0))
    for tr, col, mk, dx in TR:
        z = vget(tr)["conj_minus_nested"]["z"]
        ax.plot(i + dx, z, marker=mk, color=col, ms=7, ls="none", zorder=4)
        zh = hand[tr][f"{hkey}|quire+lang+hand"]["conj_minus_nested"]["z"]
        ax.plot(i + dx, zh, marker=mk, mfc="white", mec=col, ms=7, ls="none", zorder=5)
ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
ax.text(2.42, 0.12, "0 = no difference (chance)", fontsize=6.6, color=MUTED, ha="right", va="bottom")
ax.set_xticks(range(len(UNITS)))
ax.set_xticklabels([u[0] for u in UNITS], fontsize=7.4)
ax.set_xlim(-0.6, len(UNITS) - 0.4)
ax.set_ylim(-8.6, 6.4)
ax.set_ylabel("binding-neighbours share more  ←  →  sheet-halves share more\n(standard deviations from chance)", fontsize=7.2)
ax.text(1.0, 5.2, "six real texts written sheet by sheet land here", color=C_STACKED, fontsize=7, ha="center", va="center", fontweight="semibold")
ax.text(1.0, -7.9, "the same six texts written in the binding order land here", color=C_NESTED, fontsize=7, ha="center", va="center", fontweight="semibold")
h = [
    mpl.lines.Line2D([], [], marker=MARK_IT2A, color=C_IT2A, ls="none", ms=6.5, label="manuscript, IT2a transcription"),
    mpl.lines.Line2D([], [], marker=MARK_RF1B, color=C_RF1B, ls="none", ms=6.5, label="manuscript, RF1b transcription"),
    mpl.lines.Line2D([], [], marker="o", mfc="white", mec=INK2, ls="none", ms=6.5, label="same, scribal hand also held fixed"),
]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=1.2, handletextpad=0.4)
fig.subplots_adjust(left=0.13, right=0.99, top=0.97, bottom=0.27)
save(fig, "fig_es_leaftest")
print("ok")
