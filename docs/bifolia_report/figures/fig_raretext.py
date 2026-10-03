"""Main-text figure (next to the definition of rare types): one line of the manuscript, every
window classified, and where its rare text recurs on the conjugate leaf.  Figure S2
(fig_es_raretext.py) is the crisp version for the executive summary; the data and glyph
helpers are shared in raretext_data.py.

(a) One line of f116r (Takahashi IT2a transcription, line 47) set in Voynich glyphs, with the
    EVA transliteration and, under each word, how often that word type occurs in the whole
    manuscript.  A word used 2-10 times is rare text (blue).  Below it the same line with the
    spaces removed: every position starts a 7-letter window, and every window is a "glyph
    string" type counted over the whole manuscript in the same way.  Most windows straddle a
    word gap, so a rare glyph string is usually a fragment of a word *pair*.  The strip under
    the stream classifies all 44 windows of the line.
(b) Where the rare text of that line recurs: three of its rare glyph strings and its rare word
    also occur on f103r, the other half of the same physical sheet (sheet 1 of quire T),
    25 pages away in the binding.  Such an occurrence pair is one "conjugate-leaf" pair in the
    leaf test of Finding 2.

Glyphs: fonts/VoynichUnicode.ttf (UCSUR block U+FF400; see fonts/README.md), laid out letter
by letter from the font's own advance widths so that windows can be bracketed exactly.

Counts: IT2a stream as loaded by scripts/doubleton_gaps.py (37,759 tokens, 225 pages,
uncertain tokens dropped), 7-grams over the space-stripped page text with extended-EVA codes
as single symbols, exactly as scripts/glyph_ngram_leaf_test.py builds them.  Hard-coded below;
`uv run python fig_raretext.py --recount` re-derives them from DATA_ROOT and fails if they
differ.  Source: record §7 (units), Table S1 (quire T sheet 1 = f103/f116).
"""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch, Rectangle

from raretext_data import (
    BOUNDS,
    CALLOUTS,
    LINE_NO,
    PAGE_A,
    PAGE_B,
    RECUR,
    STREAM,
    WIN_K,
    WORD_K,
    WORDS,
    N,
    bracket,  # fmt: skip
    draw_eva,
    is_cross,
    kind,
    maybe_recount,
)
from style import *

COL = {"rare": C_IT2A, "once": "white", "common": C_NULL}
EDGE = {"rare": C_IT2A, "once": AXIS, "common": C_NULL}
TXT = {"rare": C_IT2A, "once": MUTED, "common": INK2}

maybe_recount()

# ---- figure ------------------------------------------------------------------------
W_PT, H_PT = W_FULL * 72, 5.7 * 72
fig = plt.figure(figsize=(W_FULL, H_PT / 72))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_PT)
ax.set_ylim(0, H_PT)
ax.axis("off")

GS = 11.5  # glyph size (pt)
X0 = 78  # left edge of the text column (row labels live to the left)
LABX = 4
GAP = 6.5  # word gap (pt) in the spaced lines


def rowlabel(y, text):
    ax.text(LABX, y, text, fontsize=6.4, color=INK2, ha="left", va="top", linespacing=1.15)


def leaf_tag(x, y, text, w=70):
    ax.add_patch(FancyBboxPatch((x, y - 4), w, 11, boxstyle="round,pad=0.6", fc=C_STACKED_WASH, ec=C_STACKED, lw=0.6, zorder=2))
    ax.text(x + w / 2, y, text, fontsize=6.4, color=C_STACKED, ha="center", va="baseline", fontweight="semibold", zorder=3)


# (a) ---------------------------------------------------------------------------------
yA = H_PT - 12
ax.text(LABX, yA, "(a)", fontsize=9, fontweight="bold", color=INK, va="top", ha="left")
ax.text(LABX + 16, yA, "one line of the manuscript, cut into words and into glyph strings",
        fontsize=7.6, color=INK, va="top", ha="left", fontweight="semibold")  # fmt: skip

# row 1: the line as written, words spaced
y1 = yA - 38
words = WORDS
x = X0
for w in words:
    sp = draw_eva(ax, x, y1, w, GS, color=INK)
    x0, x1 = sp[0][0], sp[-1][1]
    kk = kind(WORD_K[w])
    if kk == "rare":
        ax.add_patch(FancyBboxPatch((x0 - 1.5, y1 - 4), x1 - x0 + 3, GS * 1.45, boxstyle="round,pad=0.8",
                                    fc=C_BLUE_WASH, ec="none", zorder=1))  # fmt: skip
    ax.text((x0 + x1) / 2, y1 - 9.5, w, fontsize=6.0, color=INK2, ha="center", va="top")
    ax.text((x0 + x1) / 2, y1 - 18.5, f"{WORD_K[w]}", fontsize=6.4, color=TXT[kk], ha="center", va="top",
            fontweight="bold" if kk == "rare" else "normal")  # fmt: skip
    x = x1 + GAP
leaf_tag(X0 + 2, y1 + 20, f"{PAGE_A}, line {LINE_NO}")
ax.text(X0 + 78, y1 + 20, "sheet 1 of quire T, leaf b", fontsize=6.2, color=INK2, ha="left", va="baseline")
rowlabel(y1 + 8, "as written")
rowlabel(y1 - 9.5, "EVA letters")
rowlabel(y1 - 18.5, "uses in the whole\nmanuscript")
ax.text(X0 + 2, y1 - 32, "a word used 2–10 times is rare text", fontsize=6.2, color=C_IT2A, ha="left", va="top", fontweight="semibold")
ax.text(X0 + 2 + 132, y1 - 32, "· used once, or more than ten times: not counted", fontsize=6.2, color=MUTED, ha="left", va="top")

# row 2: the stream, spaces removed
y2 = y1 - 104
stream = STREAM
bounds = BOUNDS
gspans = draw_eva(ax, X0, y2, stream, GS, color=INK)
for i, ch in enumerate(stream):
    x0, x1 = gspans[i]
    ax.text((x0 + x1) / 2, y2 - 9.5, ch, fontsize=5.4, color=INK2, ha="center", va="top")
for b in bounds:  # tick marks where the word gaps were
    xb = gspans[b][0]
    ax.plot([xb, xb], [y2 - 3.5, y2 - 1], color=MUTED, lw=0.5, zorder=3)
rowlabel(y2 + 8, "the same line,\nspaces removed")
rowlabel(y2 - 9.5, "EVA letters; ticks\nmark the old gaps")

# strip of all windows
ys = y2 - 33
SH = 6.5
for i, k in enumerate(WIN_K):
    x0 = gspans[i][0]
    x1 = gspans[i + 1][0]
    kk = kind(k)
    ax.add_patch(Rectangle((x0 + 0.4, ys - SH), x1 - x0 - 0.8, SH, fc=COL[kk], ec=EDGE[kk], lw=0.45, zorder=3))
rowlabel(ys + 1, "every 7-letter\nwindow, by its uses")
xl = X0 + 2
yl = ys - SH - 10
for kk, lab in (("rare", "2–10 uses: rare text"), ("once", "used once"), ("common", "more than 10")):
    ax.add_patch(Rectangle((xl, yl - 4.5), 6.5, 6, fc=COL[kk], ec=EDGE[kk], lw=0.45, zorder=3))
    ax.text(xl + 9, yl - 3.7, lab, fontsize=6.0, color=TXT[kk] if kk == "rare" else INK2, ha="left", va="baseline",
            fontweight="semibold" if kk == "rare" else "normal")  # fmt: skip
    xl += 9 + len(lab) * 3.35 + 10
n_rare = sum(1 for k in WIN_K if kind(k) == "rare")
ax.text(xl + 6, yl - 3.7, f"{n_rare} of the line's {len(WIN_K)} windows are rare text", fontsize=6.0, color=INK2, ha="left", va="baseline")

# call-outs above the stream
for start, tier, _others in CALLOUTS:
    k = WIN_K[start]
    kk = kind(k)
    x0 = gspans[start][0] + 0.5
    x1 = gspans[start + N - 1][1] - 0.5
    cross = is_cross(start)
    yb = y2 + GS * 1.5 + 13 * tier
    col = C_IT2A if kk == "rare" else MUTED
    bracket(ax, x0, x1, yb, 2.5, col, lw=0.9 if kk == "rare" else 0.7)
    s = stream[start : start + N]
    where = "across a gap" if cross else "inside one word"
    if kk == "rare":
        ax.text((x0 + x1) / 2, yb + 4, f"{s} · {k} uses · {where}", fontsize=5.8, color=col, ha="center", va="baseline", fontweight="semibold")
    else:
        why = "used once" if kk == "once" else f"{k} uses, too common"
        ax.text((x0 + x1) / 2, yb + 4, f"{s} · {why} · {where}", fontsize=5.8, color=col, ha="center", va="baseline")

# (b) ---------------------------------------------------------------------------------
yB = y2 - 66
ax.text(LABX, yB, "(b)", fontsize=9, fontweight="bold", color=INK, va="top", ha="left")
ax.text(LABX + 16, yB, "where this line's rare text recurs: on the other half of the same sheet",
        fontsize=7.6, color=INK, va="top", ha="left", fontweight="semibold")  # fmt: skip
leaf_tag(X0 + 2, yB - 24, f"{PAGE_B}")
ax.text(X0 + 78, yB - 24, "sheet 1 of quire T, leaf a — joined to f116r at the fold; 25 pages away as bound",
        fontsize=6.2, color=INK2, ha="left", va="baseline")  # fmt: skip

yr = yB - 58
for where, frag, target, note in RECUR:
    rowlabel(yr, where)
    fstream = frag.replace(".", "")
    x = X0
    spans_all = []
    for w in frag.split("."):
        sp = draw_eva(ax, x, yr, w, GS, color=INK)
        spans_all.extend(sp)
        ax.text((sp[0][0] + sp[-1][1]) / 2, yr - 9.5, w, fontsize=6.0, color=INK2, ha="center", va="top")
        x = sp[-1][1] + GAP
    j = fstream.index(target)
    x0 = spans_all[j][0] - 0.5
    x1 = spans_all[j + N - 1][1] + 0.5
    # a window that spans a gap covers the gap too: shade the whole span
    ax.add_patch(FancyBboxPatch((x0 - 1, yr - 4), x1 - x0 + 2, GS * 1.45, boxstyle="round,pad=0.8", fc=C_BLUE_WASH, ec="none", zorder=1))
    bracket(ax, x0, x1, yr + GS * 1.5, 2.5, C_IT2A, lw=0.9)
    ax.text((x0 + x1) / 2, yr + GS * 1.5 + 4, target, fontsize=5.8, color=C_IT2A, ha="center", va="baseline", fontweight="semibold")
    tail = f"…  {target} again" + (f", {note}" if note else "")
    ax.text(x + 4, yr, tail, fontsize=6.0, color=INK2, ha="left", va="baseline")
    yr -= 46

# the conjugate-leaf bracket at the far right, joining the two page tags
xr = W_PT - 8
ya_top, yb_bot = y1 + 27, yB - 30
ax.plot([xr, xr + 3, xr + 3, xr], [ya_top, ya_top, yb_bot, yb_bot], color=C_STACKED, lw=0.9, zorder=2)
ax.text(xr - 2, (ya_top + yb_bot) / 2, "the two leaves of one sheet", fontsize=6.0, color=C_STACKED,
        rotation=90, ha="right", va="center", fontweight="semibold")  # fmt: skip

# closing line
ax.text(LABX, 18,
        "Each such shared occurrence is one conjugate-leaf pair in the leaf-pair test. The same strings recur on other pages too;\n"
        "the test asks only whether pairs of occurrences fall on the two halves of one sheet more often than random placement gives.",
        fontsize=6.0, color=INK2, ha="left", va="top", linespacing=1.25)  # fmt: skip

save(fig, "fig_raretext")
