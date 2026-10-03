"""Executive-summary Figure S2: the two units the report counts, on real text of the manuscript.

Top ("words"): a fragment of f116r line 47 (Takahashi IT2a) in Voynich glyphs, cut at the
spaces, EVA and manuscript count under each word; the one rare word in view, qorain (4 uses),
is washed blue and a blue elbow connector joins it to an earlier use (f103r line 11, the other
leaf of the same sheet, shown first, in manuscript order) with the same wash.
Bottom ("glyph strings"): a different sheet (quire C, sheet 3), spaces kept for legibility, with
the 7-letter window "shorshy" washed blue on both leaves: on f19r it is the two words shor shy,
on f22v the same seven letters start inside the longer word fshor, so the unit ignores the word
division.  The match is maximal (the letters before and after differ on the two leaves), so the
box is the whole of what recurs.

Data, glyph layout and `--recount` live in raretext_data.py; fig_raretext.py is the detailed
version for the main text.
"""

from __future__ import annotations

from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

from raretext_data import (
    PAGE_A,
    PAGE_B,
    WIN2,
    WIN2_K,
    WIN2_OCC,
    WORD_K,
    N,
    draw_eva,
    kind,
    maybe_recount,
)
from style import *

maybe_recount()

# fragments: (page label, words, leading ellipsis, trailing ellipsis)
FRAG_A = (f"{PAGE_A}, line 47", ["qorain", "chckhey", "qokey", "lkechy", "okeey"], False, True)
FRAG_B1 = (f"{PAGE_B}, line 11", ["chey", "qorain", "shey"], True, False)
FRAG_C1 = (f"{WIN2_OCC[0][0]}, line {WIN2_OCC[0][1]}", WIN2_OCC[0][2], False, False)
FRAG_C2 = (f"{WIN2_OCC[1][0]}, line {WIN2_OCC[1][1]}", WIN2_OCC[1][2], False, True)
WORD = "qorain"
WIN = WIN2  # shorshy: shor + shy on f19r; inside fshor shy on f22v
K_WIN = WIN2_K
assert kind(WORD_K[WORD]) == "rare" and kind(K_WIN) == "rare"

W_PT, H_PT = W_FULL * 72, 2.35 * 72
fig = plt.figure(figsize=(W_FULL, H_PT / 72))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_PT)
ax.set_ylim(0, H_PT)
ax.axis("off")

GS = 11.5
X0 = 90
LABX = 4
GAP = 6.5
ELL = 16  # width of an ellipsis


def label(y, head, sub):
    ax.text(LABX, y, head, fontsize=7.4, color=INK, ha="left", va="baseline", fontweight="semibold")
    ax.text(LABX, y - 9, sub, fontsize=5.9, color=MUTED, ha="left", va="top", linespacing=1.15)


def wash(x0, x1, y):
    ax.add_patch(FancyBboxPatch((x0 - 0.8, y - 4), x1 - x0 + 1.6, GS * 1.45, boxstyle="round,pad=0.6",
                                fc=C_BLUE_WASH, ec="none", zorder=1))  # fmt: skip


_renderer = fig.canvas.get_renderer()


def _tw(text, size, weight="normal"):
    w, _h, _d = _renderer.get_text_width_height_descent(text, fm.FontProperties(size=size, weight=weight), ismath=False)
    return w * 72 / fig.dpi


def eva_label(xc, y, word, hl=None, size=6.0):
    """EVA word centred at xc; the slice `hl` (start, stop) set bold blue, the rest INK2."""
    if hl is None:
        ax.text(xc, y, word, fontsize=size, color=INK2, ha="center", va="top")
        return
    a, b = hl
    parts = [(word[:a], INK2, "normal"), (word[a:b], C_IT2A, "bold"), (word[b:], INK2, "normal")]
    widths = [_tw(t, size, wt) if t else 0.0 for t, _c, wt in parts]
    x = xc - sum(widths) / 2
    for (t, c, wt), w in zip(parts, widths):
        if t:
            ax.text(x, y, t, fontsize=size, color=c, ha="left", va="top", fontweight=wt)
        x += w


def ellipsis(x, y):
    ax.text(x + ELL / 2, y, "…", fontsize=9, color=MUTED, ha="center", va="baseline")
    return x + ELL


def fragment(x, y, frag, counts, highlight):
    """Draw one spaced fragment; return (x after it, highlighted span)."""
    page, words, lead, trail = frag
    ax.text(x + (ELL if lead else 0), y + 27, page, fontsize=5.9, color=MUTED, ha="left", va="baseline")
    if lead:
        x = ellipsis(x, y)
    joined = "".join(words)
    if highlight == "word":
        off = sum(len(w) for w in words[: words.index(WORD)])  # locate by word, not by substring
        hl_letters = (off, off + len(WORD))
    else:
        j = joined.index(WIN)
        hl_letters = (j, j + N)
    lspans, wspans = [], []
    off = 0
    for w in words:
        sp = draw_eva(ax, x, y, w, GS, color=INK)
        lspans.extend(sp)
        x0, x1 = sp[0][0], sp[-1][1]
        wspans.append((sp[0].ink0, sp[-1].ink1))
        # which letters of this word fall inside the highlight
        lo, hi = max(hl_letters[0] - off, 0), min(hl_letters[1] - off, len(w))
        eva_label((x0 + x1) / 2, y - 9.5, w, (lo, hi) if lo < hi else None)
        if counts:
            kk = kind(WORD_K[w])
            ax.text((x0 + x1) / 2, y - 18.5, f"{WORD_K[w]}", fontsize=6.4, color=C_IT2A if kk == "rare" else INK2,
                    ha="center", va="top", fontweight="bold" if kk == "rare" else "normal")  # fmt: skip
        off += len(w)
        x = x1 + GAP
    x -= GAP
    if trail:
        x = ellipsis(x + 3, y)
    hl = (lspans[hl_letters[0]].ink0, lspans[hl_letters[1] - 1].ink1)
    wash(*hl, y)
    return x, hl


def connector(hl_a, hl_b, y):
    """Blue elbow line above the text joining the two highlighted spans."""
    xa, xb = sum(hl_a) / 2, sum(hl_b) / 2
    y0, y1 = y + GS * 1.45 - 3, y + 22
    ax.plot([xa, xa, xb, xb], [y0, y1, y1, y0], color=C_IT2A, lw=0.9, solid_joinstyle="miter", zorder=3)


# ---- row 1: words ---------------------------------------------------------------------
y1 = H_PT - 42
label(y1, "words", "cut at the spaces;\nunder each, its uses in\nthe whole manuscript")
x, hl_b = fragment(X0, y1, FRAG_B1, True, "word")
x = ellipsis(x + 10, y1)
x, hl_a = fragment(x + 10, y1, FRAG_A, True, "word")
connector(hl_b, hl_a, y1)
ax.text(sum(hl_b) / 2, y1 - 27, "a rare word …", fontsize=6.0, color=INK,
        ha="center", va="top", fontweight="semibold")  # fmt: skip
ax.text(sum(hl_a) / 2, y1 - 27, f"… the same word again: {WORD_K[WORD]} uses = rare text", fontsize=6.0, color=INK,
        ha="center", va="top", fontweight="semibold")  # fmt: skip

# ---- row 2: glyph strings ------------------------------------------------------------
y2 = y1 - 82
label(y2, "glyph strings", "windows of 7 letters, one\nstarting at every letter;\nthe spaces are kept here\nfor legibility but ignored")
X2 = X0 + 34  # row 2 has no leading ellipsis and is shorter: start it further from the row label
x, hl_b = fragment(X2, y2, FRAG_C1, False, "window")
x = ellipsis(x + 22, y2)
x, hl_a = fragment(x + 22, y2, FRAG_C2, False, "window")
connector(hl_b, hl_a, y2)
ax.text(sum(hl_b) / 2, y2 - 27, f"a rare glyph string, {WIN} …", fontsize=6.0, color=INK, ha="center", va="top",
        fontweight="semibold")  # fmt: skip
ax.text(sum(hl_a) / 2, y2 - 27, f"… the same string again: {K_WIN} uses = rare text", fontsize=6.0,
        color=INK, ha="center", va="top", fontweight="semibold")  # fmt: skip
save(fig, "fig_es_raretext")
