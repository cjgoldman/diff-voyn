"""In-word vs cross-word letter doubling per language (follow-up to §8: can a
"reuse only for in-word doubles" rule give s = 1?).  Rates per 1000 adjacent
letter pairs of the whitespace-stripped text; VMS reference 8.6/1000 (IT2a)."""

import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terminal_doubling import DATA_ROOT, load_language

VMS = (8.6, 6.6, 10.4)
rows = []
for lang in ["latin", "italian", "german", "dutch", "french", "english", "spanish"]:
    for esz in [False, True] if lang == "german" else [False]:
        docs = load_language(lang, keep_eszett=esz)
        pairs = inw = xw = 0
        modes = Counter()
        for paras in docs.values():
            for para in paras:
                flat = [w for s in para for w in s]
                pairs += max(0, len("".join(flat)) - 1)
                for w in flat:
                    for a, b in pairwise(w):
                        if a == b:
                            inw += 1
                            modes[a] += 1
                for a, b in pairwise(flat):
                    xw += a[-1] == b[0]
        top = ", ".join(f"{k}{k} {100*v/inw:.0f}%" for k, v in modes.most_common(6))
        # geminate collapse: how many top modes must be written single to reach VMS rate
        acc, n_collapse = inw, 0
        for _, v in modes.most_common():
            if 1000 * acc / pairs <= VMS[0]:
                break
            acc -= v
            n_collapse += 1
        rows.append(
            (
                lang + (" (ß kept)" if esz else ""),
                pairs,
                1000 * inw / pairs,
                1000 * xw / pairs,
                VMS[0] / (1000 * inw / pairs),
                n_collapse,
                top,
            )
        )
print(
    "| language | letter pairs | in-word doubles /1000 | cross-word doubles /1000 | s needed (in-word only) | top modes to collapse for s=1 | in-word modes |"
)
print("|---|---:|---:|---:|---:|---:|---|")
for r in rows:
    print(
        f"| {r[0]} | {r[1]:,} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.2f} | {r[5]} | {r[6]} |"
    )
out = DATA_ROOT / "analysis/doubling/inword_doubling.md"
