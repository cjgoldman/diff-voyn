"""Shared matplotlib style for the bifolia-ordering report figures.

Every figure script in this directory does ``from style import *`` and uses the
names below, so the report's figures read as one system.  Palette follows the
dataviz skill's validated reference palette (fixed categorical order, single-hue
sequential ramp, blue/red diverging with a grey midpoint, recessive chrome).

Semantic colour assignment for THIS report (keep it fixed across figures):

    C_IT2A     manuscript, Takahashi IT2a transcription   (blue, slot 1)
    C_RF1B     manuscript, Reference RF1b transcription    (dark blue step 650)
    C_STACKED  "written stacked" controls / the stacked hypothesis (aqua, slot 3)
    C_NESTED   "written nested" controls / the as-bound order     (red, slot 8)
    C_KNOWN    known texts (prose / verse) as neutral reference   (muted grey)
    C_NULL     null distributions, uniform references              (baseline grey)
    C_ACCENT   a highlighted winner / the single thing to look at  (orange, slot 2)

Marker convention: IT2a = filled circle 'o', RF1b = filled square 's'.
Latin controls = '^', German = 'D', Italian prose = 'v', Italian verse = 'P'.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# ---- palette ---------------------------------------------------------------
C_IT2A = "#2a78d6"
C_RF1B = "#104281"
C_STACKED = "#1baf7a"
C_NESTED = "#e34948"
C_KNOWN = "#898781"
C_NULL = "#c3c2b7"
C_ACCENT = "#eb6834"
C_YELLOW = "#eda100"
C_VIOLET = "#4a3aa7"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# fill versions (light washes for bands)
C_STACKED_WASH = "#d3f1e5"
C_NESTED_WASH = "#fadcdb"
C_BLUE_WASH = "#cde2fb"

SEQ_BLUES = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
CMAP_SEQ = LinearSegmentedColormap.from_list("report_blues", SEQ_BLUES)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "report_div", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f0a4a3", "#e34948", "#9c1f1f"]
)
CMAP_SEQ_R = CMAP_SEQ.reversed()

MARK_IT2A = "o"
MARK_RF1B = "s"
MARK_LANG = {"la": "^", "de": "D", "it": "v", "it_verse": "P"}

# ---- sizes (inches). Text width in the report is 6.3 in. ---------------------
W_FULL = 6.3
W_HALF = 3.05
W_TWO_THIRDS = 4.2

# ---- rcParams ----------------------------------------------------------------
mpl.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "legend.frameon": False,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.6,
        "patch.linewidth": 0.6,
        "text.color": INK,
        "mathtext.fontset": "dejavusans",
    }
)


def panel_label(ax, s, x=-0.08, y=1.04):
    """Bold (a), (b) … label at the top-left of an axes, outside the plot."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left", color=INK)


def despine_all(ax):
    for s in ax.spines.values():
        s.set_visible(False)


def band(ax, lo, hi, color, label=None, alpha=1.0, orient="h", zorder=0):
    """A light wash marking a control band (horizontal by default)."""
    if orient == "h":
        ax.axhspan(lo, hi, color=color, alpha=alpha, lw=0, zorder=zorder, label=label)
    else:
        ax.axvspan(lo, hi, color=color, alpha=alpha, lw=0, zorder=zorder, label=label)


def save(fig, name, outdir=None):
    """Save as PDF (for LaTeX) and PNG (for quick review) next to this file."""
    import os

    outdir = outdir or os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=160)
    plt.close(fig)


__all__ = [n for n in dir() if n.isupper() or n in ("panel_label", "despine_all", "band", "save", "plt", "mpl")]
