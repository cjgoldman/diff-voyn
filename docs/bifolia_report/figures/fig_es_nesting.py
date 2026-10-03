"""Executive-summary figure: the fully nested and the fully stacked ways of gathering quire
T's six sheets (record §19), one chart, two units.

Vertical axes = the metric itself, the mean distance between the recurrences of rare text,
closer fit UPWARD.  Grouped by UNIT: the left half is rare words on the left axis (words), the
right half rare seven-glyph strings on the right axis (glyphs); the two axes are tied by the
quire's glyphs per word, so a box's height means the same physical distance in either half.
Inside each half, the 720 fully nested orders (grey; the binding is one of them) then the 720
fully stacked orders (blue; every sheet read whole), one box per transcription (lighter IT2a,
darker RF1b), showing mean ± 1 sd (box), mean (line), worst and best order (whiskers).
Marked with a dot: the binding as it is (nested boxes) and the best stacked order, labelled
"best stacked hypothesis" (stacked boxes; on glyph strings the best of all 23,040 gatherings,
on words 5th / 2nd).  The mixed patterns
(2–5 nested blocks) are in the main text (fig_nesting_classes, fig_nesting).

Colour rule (user, 2026-09-06): grey = fully nested, blue = fully stacked; shade = transcription;
unit only by position and axis.  Earlier same-day forms are listed in SPEC.md.

Source: data/derived_quire_T_nesting.json (derive_quire_T_nesting.py).
"""

import json
import os

import numpy as np

from style import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "data", "derived_quire_T_nesting.json")))
GLYPHS_PER_WORD = 56889 / 10673  # quire T, IT2a (23 pages): sets the right-hand axis only

# colour comes from the class (grey = nested, blue = stacked); the lighter shade is IT2a, the darker RF1b
SHADES = {"nested": {"IT2a": ("#a8a69f", "#e6e5df"), "RF1b": ("#5f5e59", "#cfcec6")},
          "stacked": {"IT2a": (C_IT2A, "#cde2fb"), "RF1b": (C_RF1B, "#9ec5f4")}}
YL = (20.6, 16.6)  # thousands of glyphs, closer at the top

fig, ax = plt.subplots(1, 1, figsize=(W_FULL * 0.78, 3.0))
axw = ax.twinx()
# groups by UNIT: words (left half, left axis) and glyph strings (right half, right axis);
# inside each, the nested pair (grey) then the stacked pair (blue), IT2a lighter / RF1b darker
GROUPS = [("words", 0, "words"), ("n7", 1, "seven-glyph strings")]
PAIRS = [("nested", 1, -0.2), ("stacked", "S", 0.2)]
TRS = [("IT2a", -0.1), ("RF1b", 0.1)]
ticks, ticklabels = [], []
for unit, g, gname in GROUPS:
    a = ax if unit == "n7" else axw
    for cname, nbk, dxp in PAIRS:
        for tr, dxt in TRS:
            d = D[tr]
            nb = np.array(d["nblocks"]); S = d["S"]
            k = S if nbk == "S" else nbk
            v = np.array(d["units"][unit]["L1"])
            v = v / 1000.0 if unit == "n7" else v
            col, wash = SHADES[cname][tr]
            vk = v[nb == k]; mu, sd = vk.mean(), vk.std()
            x = g + dxp + dxt
            a.plot([x, x], [vk.min(), vk.max()], color=col, lw=0.9, zorder=2)
            for yv in (vk.min(), vk.max()):
                a.plot([x - 0.06, x + 0.06], [yv, yv], color=col, lw=0.9, zorder=2)
            a.add_patch(mpl.patches.Rectangle((x - 0.085, mu - sd), 0.17, 2 * sd, facecolor=wash, edgecolor=col, lw=0.9, zorder=3))
            a.plot([x - 0.085, x + 0.085], [mu, mu], color=col, lw=1.6, zorder=4)
            ticks.append(x); ticklabels.append(tr)
            # one dot per box, the same shape everywhere (shape carries no information):
            # the binding in each nested box, the best stacked order in each stacked box
            yb = v[d["bound_i"]] if k == 1 else vk.min()
            a.plot([x], [yb], marker="o", color=col, ms=4.6, markeredgecolor=INK, markeredgewidth=0.6, zorder=6)
            if unit == "words" and k == 1 and tr == "IT2a":   # words half, between the 3,200 and 3,400 lines
                axw.annotate("the binding\nas it is", (x, yb), xytext=(x + 0.08, 3300), fontsize=6.3, color=INK2, ha="center", va="center",
                             arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkB=3), zorder=7)
            if unit == "n7" and k != 1 and tr == "IT2a":     # glyph half, just below the 17,000 (= 3,200) line
                ax.annotate("best stacked\nhypothesis", (x, yb), xytext=(g - 0.2, 17.45), fontsize=6.3, color=INK2, ha="center", va="center",
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkB=3), zorder=7)
        ax.text(g + dxp, -0.075, f"fully {cname}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.2, color=INK2)
    ax.text(g, 1.0, f"rare {gname}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=INK)

ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels, fontsize=5.4, color=MUTED)
ax.tick_params(axis="x", length=0, pad=2)
ax.set_xlim(-0.55, 1.55)

ax.set_ylim(*YL)
ax.set_yticks([17, 18, 19, 20])
ax.set_yticklabels(["17,000", "18,000", "19,000", "20,000"], fontsize=6.6)
ax.tick_params(axis="y", length=0)
ax.set_ylabel("seven-glyph strings (glyphs)", fontsize=6.8)
ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
ax.grid(axis="y", color=GRID, lw=0.5)

axw.set_ylim(YL[0] * 1000 / GLYPHS_PER_WORD, YL[1] * 1000 / GLYPHS_PER_WORD)
axw.set_yticks([3200, 3400, 3600, 3800])
axw.set_yticklabels(["3,200", "3,400", "3,600", "3,800"], fontsize=6.6)
axw.tick_params(axis="y", length=0)
axw.set_ylabel("mean rare recurrence distance:\nwords (words)", fontsize=6.8)
axw.yaxis.tick_left(); axw.yaxis.set_label_position("left")
axw.grid(False)
for a in (ax, axw):
    for sname in ("left", "bottom", "right", "top"):
        a.spines[sname].set_visible(False)
fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.17)
save(fig, "fig_es_nesting")
for unit in ("n7", "words"):
    for tr in ("IT2a", "RF1b"):
        d = D[tr]; nb = np.array(d["nblocks"]); v = np.array(d["units"][unit]["L1"]); st = nb == d["S"]
        print(unit, tr, "best stacked rank", int((v < v[st].min()).sum() + 1), "nested mean %.0f sd %.0f | stacked mean %.0f sd %.0f" % (v[nb == 1].mean(), v[nb == 1].std(), v[st].mean(), v[st].std()))
