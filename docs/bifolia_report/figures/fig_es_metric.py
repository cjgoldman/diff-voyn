"""Executive-summary figure: the cluster-scaled score (record §14) on two real words of quire T.

Three rows on one scale in words.  A word's home sheet (the sheet holding most of its uses)
is drawn as a band with the uses as ticks; the gaps between them set the home gap.  The
sheet holding the word's one stray use is drawn read directly after the home sheet, and the
stray's distance to the nearest home use is given in words and in home gaps (distance / home
scale), which is what the stray costs.
  row 1  qokaly:   three uses 51 and 115 words apart on sheet 1 (home gap 173), stray on
                   sheet 5 at 1,651 words = 9.6 home gaps
  row 2  chckhedy: three uses 1,480 and 437 apart on sheet 6 (home gap 756), stray on
                   sheet 4 at 1,857 words = 2.5 home gaps.  Almost the same distance, a quarter
                   of the cost: the plain average would count these two strays alike.
  row 3  qokaly with sheet 2 read between: 3,390 words = 19.6 home gaps.  The order moves the
                   distance, never the home gap.

Source: data/derived_burst_example.json (from derive_burst_example.py; IT2a, word unit,
home gap per scripts/quire_order_burst.py).
"""

import json
import os

from style import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "data", "derived_burst_example.json")))
B, S = D["burst"], D["spread"]
SL, SF = {int(k): v for k, v in D["sheet_len"].items()}, {int(k): v for k, v in D["sheet_folios"].items()}
BETWEEN = D["between_sheet"]

C_HOME_WASH = "#fbe0d2"  # light accent wash for the home sheet
C_STRAY_WASH = C_BLUE_WASH
C_MID_WASH = "#eceae2"

fig, ax = plt.subplots(figsize=(W_FULL, 3.3))
despine_all(ax)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
X0 = -2950
ax.set_xlim(X0, 5700)
ax.set_ylim(-0.85, 3.55)
H = 0.34  # band height


def band(y, x0, L, color, label):
    ax.add_patch(plt.Rectangle((x0, y - H / 2), L, H, facecolor=color, edgecolor="none", zorder=1))
    ax.text(x0 + L / 2, y + H / 2 + 0.03, label, fontsize=6.4, color=INK2, ha="center", va="bottom")


def uses(y, xs, color):
    for x in xs:
        ax.plot([x, x], [y - H / 2, y + H / 2], color=color, lw=1.5, solid_capstyle="butt", zorder=3)


def gap_arrow(y, x0, x1, text, color=INK2, dy=-0.36, fs=6.4, weight="normal"):
    ax.annotate("", xy=(x1, y + dy), xytext=(x0, y + dy),
                arrowprops=dict(arrowstyle="<->,head_length=0.3,head_width=0.15", color=color, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y + dy - 0.05, text, fontsize=fs, color=color, ha="center", va="top", fontweight=weight)


def row(y, title, sub, ex, between=None):
    ax.text(X0, y + 0.2, title, fontsize=7.6, color=INK, ha="left", va="center", fontweight="semibold")
    for k, line in enumerate(sub):
        ax.text(X0, y - 0.02 - 0.2 * k, line, fontsize=6.4, color=INK2, ha="left", va="center")
    hl, hp = ex["home_len"], ex["home_positions"]
    band(y, 0, hl, C_HOME_WASH, f"home sheet {ex['home_sheet']} ({ex['home_folios']})")
    uses(y, hp, C_ACCENT)
    x = hl
    if between is not None:
        band(y, x, SL[between], C_MID_WASH, f"sheet {between} ({SF[between]})")
        x += SL[between]
    band(y, x, ex["stray_len"], C_STRAY_WASH, f"sheet {ex['stray_sheet']} ({ex['stray_folios']})")
    xs = x + ex["stray_offset"]
    uses(y, [xs], C_IT2A)
    d = xs - hp[-1]
    lam = ex["lambda"]
    gap_arrow(y, hp[-1], xs, f"{d:,} words = {d / lam:.1f} home gaps", color=C_IT2A, weight="semibold")
    return d


# row 1: the clustering word
row(3.0, f"{B['word']}, a clustering word",
    (f"{len(B['home_positions'])} uses, {B['home_gaps'][0]} and {B['home_gaps'][1]} words apart", f"home gap {B['lambda']:.0f} words"), B)
# row 2: the spread word
row(1.85, f"{S['word']}, a spread word",
    (f"{len(S['home_positions'])} uses, {S['home_gaps'][0]:,} and {S['home_gaps'][1]} words apart", f"home gap {S['lambda']:.0f} words"), S)
# row 3: the clustering word under an order with one more sheet between
row(0.7, f"{B['word']} again, another order", (f"sheet {BETWEEN} read between", f"same home gap, {B['lambda']:.0f} words"), B, between=BETWEEN)

# legend-like notes
ax.plot([X0 + 40, X0 + 40], [-0.28 - 0.08, -0.28 + 0.08], color=C_ACCENT, lw=1.5)
ax.text(X0 + 120, -0.28, "use on the home sheet", fontsize=6.4, color=INK2, ha="left", va="center")
ax.plot([X0 + 2700, X0 + 2700], [-0.28 - 0.08, -0.28 + 0.08], color=C_IT2A, lw=1.5)
ax.text(X0 + 2780, -0.28, "stray use, on another sheet", fontsize=6.4, color=INK2, ha="left", va="center")
ax.text(X0, -0.6, "score of a candidate order = each stray use's distance in home gaps, summed over the quire's rare text; the lowest score wins",
        fontsize=6.8, color=INK, ha="left", va="center", style="italic")
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
save(fig, "fig_es_metric")
print(f"ok: {B['word']} {B['score_after']:.1f} home gaps, {S['word']} {S['score_after']:.1f} home gaps, "
      f"between: {(B['d_after'] + SL[BETWEEN]) / B['lambda']:.1f}")
