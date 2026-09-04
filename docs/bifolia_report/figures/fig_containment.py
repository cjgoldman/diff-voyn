"""Figure: the sheet effect is a small excess, not a containment of vocabulary by sheet.

(a) Where the occurrences of a rare type sit (percentage of types with all
    occurrences inside the unit), k = 2..5, both transcriptions; the within-quire
    part is zoomed in (a′) with the permutation-null value as a tick.
    Source: data/containment.log (§11) — parsed from the log.
(b) Tokens per type in non-overlapping solver windows, nested vs stacked reading
    order, Currier A and B, both transcriptions (§8).
    Source: data/window_tokens_per_type.log — parsed from the log.
"""

import os
import re

import numpy as np

from style import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# ---- parse containment.log ---------------------------------------------------------
CATS = ["same page", "same leaf", "same sheet\n(both leaves)", "2+ sheets,\nsame quire", "2+ quires"]
cont = {}
tr = None
for line in open(os.path.join(DATA, "containment.log")):
    m = re.match(r"== (\w+)", line)
    if m:
        tr = m.group(1)
        cont[tr] = {}
        continue
    m = re.match(r"\s+(\d)\s+(\d+)\s+\|(.*)", line)
    if m and tr:
        k, ntypes, rest = int(m.group(1)), int(m.group(2)), m.group(3)
        cells = re.findall(r"([\d.]+)% \(null\s+([\d.]+)±([\d.]+)\)", rest)
        cont[tr][k] = {"n": ntypes, "obs": [float(c[0]) for c in cells], "null": [float(c[1]) for c in cells]}
assert cont["IT2a"][2]["obs"][0] == 2.9 and cont["RF1b"][5]["obs"][4] == 94.9

# ---- parse window_tokens_per_type.log ----------------------------------------------
win = {}
tr = lang = None
for line in open(os.path.join(DATA, "window_tokens_per_type.log")):
    m = re.match(r"== (\w+)", line)
    if m:
        tr = m.group(1)
        continue
    m = re.match(r"\s+Currier (\w):", line)
    if m:
        lang = m.group(1)
        continue
    m = re.match(r"\s+W=\s*(\d+): tokens/type nested ([\d.]+)\s+stacked ([\d.]+)\s+null ([\d.]+)±([\d.]+)", line)
    if m and tr and lang:
        win.setdefault((tr, lang), []).append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))))
assert win[("IT2a", "A")][0][:3] == (512, 1.607, 1.620)

fig = plt.figure(figsize=(W_FULL, 5.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[1.0, 1.05], wspace=0.25, hspace=0.62)
ax = fig.add_subplot(gs[0, 0])
az = fig.add_subplot(gs[0, 1])
bx = fig.add_subplot(gs[1, :])

SEQ5 = [SEQ_BLUES[i] for i in (0, 3, 6, 9, 12)]
KS = [2, 3, 4, 5]
rows = []  # (y, tr, k)
y = 0
for k in KS:
    for tr, dy in (("IT2a", 0.0), ("RF1b", 0.5)):
        rows.append((y + dy, tr, k))
    y += 1.3
for yy, tr, k in rows:
    left = 0
    for c, col in zip(range(5), SEQ5):
        v = cont[tr][k]["obs"][c]
        ax.barh(yy, v, left=left, height=0.42, color=col, lw=0.5, edgecolor="white")
        left += v
ax.set_yticks([r[0] + 0.25 for r in rows[::2]])
ax.set_yticklabels([f"$k$ = {k}" for k in KS])
ax.set_ylim(-0.4, y - 0.4)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("% of types with all occurrences inside the unit")
ax.grid(False)
ax.grid(True, axis="x")
for yy, tr, k in rows[:2]:
    ax.text(101, yy, tr, fontsize=6.3, color=INK2, va="center")
ax.set_title("where a rare type's occurrences sit", loc="left", fontsize=7.8, pad=8)
panel_label(ax, "(a)", x=-0.2)

# (a') zoom: first four categories, with null ticks
for yy, tr, k in rows:
    left = 0
    for c, col in zip(range(4), SEQ5):
        v = cont[tr][k]["obs"][c]
        az.barh(yy, v, left=left, height=0.42, color=col, lw=0.5, edgecolor="white")
        left += v
    # null cumulative ticks for each boundary
    nl = 0
    for c in range(4):
        nl += cont[tr][k]["null"][c]
        az.plot([nl, nl], [yy - 0.26, yy + 0.26], color=INK, lw=0.8, zorder=5)
az.set_yticks([])
az.set_ylim(-0.4, y - 0.4)
az.invert_yaxis()
az.set_xlim(0, 25)
az.set_xlabel("% of types")
az.grid(False)
az.grid(True, axis="x")
az.set_title("within-quire part, zoomed (ticks: null)", loc="left", fontsize=7.8, pad=8)
h = [mpl.patches.Patch(color=c, label=l.replace("\n", " ")) for c, l in zip(SEQ5, CATS)]
fig.legend(handles=h, loc="center", ncol=5, bbox_to_anchor=(0.5, 0.505), columnspacing=1.0, handlelength=1.2, fontsize=6.8)
panel_label(az, "(a′)", x=-0.13)

# ---- (b) tokens/type in windows ---------------------------------------------------------
for (tr, lang), vals in win.items():
    W = [v[0] for v in vals]
    col = C_IT2A if tr == "IT2a" else C_RF1B
    mk = MARK_IT2A if tr == "IT2a" else MARK_RF1B
    bx.plot(W, [v[1] for v in vals], color=col, marker=mk, ms=3.8, lw=1.0, ls="-" if lang == "B" else (0, (2, 1.2)), zorder=3)
    bx.plot(W, [v[2] for v in vals], color=C_STACKED, marker=mk, ms=3.8, mfc="white", mec=C_STACKED, lw=0.9, ls="-" if lang == "B" else (0, (2, 1.2)), zorder=4)
    if tr == "IT2a":
        bx.annotate(f"Currier {lang}", (W[-1], vals[-1][1]), xytext=(4, 0), textcoords="offset points", fontsize=6.3, color=col, va="center")
bx.set_xscale("log", base=2)
bx.set_xticks([512, 1024, 2048, 4096])
bx.set_xticklabels(["512", "1024", "2048", "4096"])
bx.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
bx.set_xlim(450, 6200)
bx.set_ylim(1.4, 3.4)
bx.set_xlabel("window length $W$ (tokens)")
bx.set_ylabel("tokens per type in the window")
h2 = [
    mpl.lines.Line2D([], [], color=C_IT2A, marker=MARK_IT2A, ms=3.8, lw=1.0, label="nested order, IT2a"),
    mpl.lines.Line2D([], [], color=C_RF1B, marker=MARK_RF1B, ms=3.8, lw=1.0, label="nested order, RF1b"),
    mpl.lines.Line2D([], [], color=C_STACKED, marker="o", mfc="white", ms=3.8, lw=0.9, label="stacked order"),
    mpl.lines.Line2D([], [], color=INK2, lw=1.0, ls="-", label="Currier B"),
    mpl.lines.Line2D([], [], color=INK2, lw=1.0, ls=(0, (2, 1.2)), label="Currier A"),
]
bx.legend(handles=h2, loc="upper left", fontsize=6.5, handlelength=2.4, ncol=5, columnspacing=1.1)
bx.set_title("tokens per type in non-overlapping windows: reading order moves it by ≤ 0.01", loc="left", fontsize=7.8, pad=6)
panel_label(bx, "(b)", x=-0.09)

save(fig, "fig_containment")
