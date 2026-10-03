"""Shared data and glyph-layout helpers for the two rare-text figures.

fig_es_raretext.py (executive summary, Figure S2): the two units only.
fig_raretext.py (main text, next to the definition of rare types): the full example with
every window of the line classified and the recurrences on the conjugate leaf.

The example is f116r line 47 (Takahashi IT2a transcription), sheet 1 of quire T, whose
conjugate leaf is f103r.  Counts are over the IT2a stream as loaded by
scripts/doubleton_gaps.py (37,759 tokens, 225 pages, uncertain tokens dropped); 7-grams over
the space-stripped page text with extended-EVA codes as single symbols, exactly as
scripts/glyph_ngram_leaf_test.py builds them.  Hard-coded here; `recount()` re-derives them
from DATA_ROOT and both figure scripts accept `--recount`.

Glyphs: fonts/VoynichUnicode.ttf (UCSUR block U+FF400; see fonts/README.md), laid out letter
by letter from the font's own advance widths so that windows can be bracketed exactly.
"""

from __future__ import annotations

import os
import sys

from fontTools.ttLib import TTFont
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "fonts", "VoynichUnicode.ttf")
MAP = os.path.join(HERE, "fonts", "EVA.TXT")

# ---- the example (IT2a) ------------------------------------------------------------
PAGE_A = "f116r"  # sheet 1 of quire T, leaf b (as bound: page 27 of the quire)
PAGE_B = "f103r"  # sheet 1 of quire T, leaf a (as bound: page 1 of the quire)
LINE_NO = 47
LINE = "osain.shky.qorain.chckhey.qokey.lkechy.okeey.okal.chedkaly"  # f116r.47
WORD_K = {"osain": 3, "shky": 13, "qorain": 4, "chckhey": 30, "qokey": 107,
          "lkechy": 1, "okeey": 177, "okal": 138, "chedkaly": 1,
          "chey": 344, "shey": 283}  # fmt: skip  (last two: f103r line 11, Figure S2)
N = 7
# every 7-letter window of the space-stripped line, by start position: its count in the manuscript
WIN_K = [1, 2, 3, 1, 1, 1, 1, 1, 4, 3, 16, 6, 40, 97, 20, 30, 12, 28, 19, 109, 36, 12,
         5, 3, 1, 3, 1, 1, 3, 6, 11, 13, 16, 41, 12, 4, 7, 17, 50, 31, 1, 1, 1, 1]  # fmt: skip
# windows called out in the detailed figure: start, label tier, other pages of the string
CALLOUTS = [
    (8, 0, ["f76r", "f77r", "f103r"]),  # yqorain   k=4  straddles shky|qorain
    (15, 1, None),  # chckhey   k=30 one word, too common
    (22, 2, ["f84v", "f103r", "f108r", "f115v"]),  # qokeylk k=5 straddles qokey|lkechy
    (28, 0, ["f103r", "f108r"]),  # kechyok   k=3  straddles lkechy|okeey
    (42, 1, None),  # chedkal   k=1  one word, used once
]
# f103r fragments carrying the recurrences (IT2a line numbers), with the matching window
RECUR = [
    ("f103r, line 11", "shedy.okain.chey.qorain.shey.otoy", "yqorain", "and the rare word qorain"),
    ("f103r, line 5", "shedy.oteey.qokey.lkar.sheeky", "qokeylk", None),
    ("f103r, line 27", "qokechy.okeey.qokeey.lkeeody", "kechyok", None),
]
RARE_LO, RARE_HI = 2, 10

# second example for Figure S2's glyph-string row: a rare 7-gram that is a *maximal* match on the
# two leaves of one sheet (the letters before and after it differ), crossing a word gap 4|3
WIN2 = "shorshy"
WIN2_K = 4
WIN2_OCC = [("f19r", 6, ["y", "shor", "shy", "daiin"]), ("f22v", 6, ["fshor", "shy", "tchor"])]  # quire C, sheet 3

STREAM = LINE.replace(".", "")
WORDS = LINE.split(".")
BOUNDS: set[int] = set()  # index of the first letter of each word after the first
_n = 0
for _w in WORDS[:-1]:
    _n += len(_w)
    BOUNDS.add(_n)

# ---- glyph layout helpers ----------------------------------------------------------
class Span(tuple):
    """(x0, x1) advance span of one drawn letter, with .ink0/.ink1 for its ink extent."""

    def __new__(cls, x0, x1, ink0, ink1):
        obj = super().__new__(cls, (x0, x1))
        obj.ink0, obj.ink1 = ink0, ink1
        return obj


_tt = TTFont(FONT)
_cmap = _tt.getBestCmap()
_hmtx = _tt["hmtx"]
_upm = _tt["head"].unitsPerEm
_map = {}
with open(MAP) as _fh:
    for _ln in _fh:
        if _ln.startswith("#") or not _ln.strip():
            continue
        _a, _b = _ln.split()[:2]
        _map[chr(int(_a, 16))] = int(_b, 16)
FP = fm.FontProperties(fname=FONT)


_glyphset = _tt.getGlyphSet()


def ink(ch: str, size: float) -> tuple[float, float]:
    """Horizontal ink extent (xMin, xMax) of one EVA letter in points, relative to its origin."""
    from fontTools.pens.boundsPen import BoundsPen

    pen = BoundsPen(_glyphset)
    _glyphset[_cmap[_map[ch]]].draw(pen)
    if pen.bounds is None:
        return 0.0, adv(ch, size)
    return pen.bounds[0] / _upm * size, pen.bounds[2] / _upm * size


def adv(ch: str, size: float) -> float:
    """Advance width of one EVA letter in points at the given size."""
    return _hmtx[_cmap[_map[ch]]][0] / _upm * size


def draw_eva(ax, x, y, s, size, color="black", zorder=4):
    """Set an EVA string letter by letter at (x, y) [points, baseline]; return per-letter x-spans."""
    spans = []
    for ch in s:
        w = adv(ch, size)
        ax.text(x, y, chr(_map[ch]), fontproperties=FP, fontsize=size, color=color,
                ha="left", va="baseline", zorder=zorder)  # fmt: skip
        spans.append(Span(x, x + w, *(x + v for v in ink(ch, size))))
        x += w
    return spans


def kind(k: int) -> str:
    return "rare" if RARE_LO <= k <= RARE_HI else ("once" if k == 1 else "common")


def is_cross(start: int) -> bool:
    """Does the window starting at `start` straddle a former word gap?"""
    return any(start < j < start + N for j in BOUNDS)


def bracket(ax, x0, x1, y, h, color, lw=0.8, zorder=5):
    """Square bracket from x0 to x1 with the bar at y+h and legs of height h (h<0: bar below)."""
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color=color, lw=lw, solid_capstyle="butt", zorder=zorder)


# ---- optional re-derivation of the hard-coded counts --------------------------------
def recount() -> bool:
    import re
    from collections import Counter

    sys.path.insert(0, "/workspace/scripts")
    from doubleton_gaps import DATA_ROOT, load_vms  # type: ignore

    toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / "IT2a-n.txt")
    sym = re.compile(r"<\d+>|[a-z]")
    syms = [[] for _ in pages]
    for t in toks:
        syms[t["page_idx"]].extend(sym.findall(t["w"]))
    cnt = Counter()
    for s in syms:
        for i in range(len(s) - N + 1):
            cnt["".join(s[i : i + N])] += 1
    wcnt = Counter(t["w"] for t in toks)
    ok = True
    for w, k in WORD_K.items():
        if wcnt[w] != k:
            print("WORD", w, k, "->", wcnt[w])
            ok = False
    for i, k in enumerate(WIN_K):
        if cnt[STREAM[i : i + N]] != k:
            print("WIN", i, STREAM[i : i + N], k, "->", cnt[STREAM[i : i + N]])
            ok = False
    if cnt[WIN2] != WIN2_K:
        print("WIN2", WIN2, WIN2_K, "->", cnt[WIN2])
        ok = False
    for page, _line, words in WIN2_OCC:
        pi = next(i for i, pg in enumerate(pages) if pg["page"] == page)
        if WIN2 not in "".join(syms[pi]) or " ".join(words) not in " ".join(t["w"] for t in toks if t["page_idx"] == pi):
            print("WIN2 occurrence not found on", page)
            ok = False
    print("recount", "OK" if ok else "MISMATCH")
    return ok


def maybe_recount() -> None:
    if "--recount" in sys.argv:
        sys.exit(0 if recount() else 1)
