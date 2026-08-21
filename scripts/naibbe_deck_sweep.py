"""What does making Naibbe's table choice deterministic do to its other metrics?

Three ways to raise Naibbe's doubling rate toward the VMS's ~9/1000 are run on
the same plaintext and scored on Greshko's metric set (type count / TTR,
hapax share, Zipf slope, mean token length, token-length histogram, glyph
entropy h1 and conditional entropy h2, word entropy) against the VMS
(Takahashi IT2a paragraph text) computed with the same code on the same
token budget:

* ``deck p``   — concentrate the card deck on the alpha table (alpha share p,
                 other tables scaled down proportionally); p = 1 is a single
                 table, i.e. fully deterministic choice.
* ``sticky s`` — keep the stock deck but, when the respaced unit equals the
                 previous unit, reuse the previous token with probability s
                 (Boxer's ``doubling_strength``, applied to Naibbe).

Usage: uv run python scripts/naibbe_deck_sweep.py [--chars 200000] [--sample 30000]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doubling_rate import IVTFF_FILES, parse_ivtff_lines
from naibbe_doubling import OUT, load_plaintext

from diff_voyn.ciphers.naibbe import NaibbeCipher

STOCK = {"alpha": 20, "beta1": 8, "beta2": 8, "beta3": 8, "gamma1": 4, "gamma2": 4}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def entropy_bits(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values())


def metrics(tokens):
    """Greshko-style summary on a token list (EVA strings)."""
    n = len(tokens)
    types = Counter(tokens)
    pairs = sum(a == b for a, b in pairwise(tokens))
    text = " ".join(tokens)
    c1 = Counter(text)
    c2 = Counter(pairwise(text))
    h1 = entropy_bits(c1)
    h2 = entropy_bits(c2) - h1  # H(c_{i+1} | c_i)
    freqs = np.array(sorted(types.values(), reverse=True), float)
    r = np.arange(1, min(1000, len(freqs)) + 1)
    slope = np.polyfit(np.log(r), np.log(freqs[: len(r)]), 1)[0]
    lens = np.array([len(t) for t in tokens])
    hist = {L: 100 * float((lens == L).mean()) for L in range(1, 11)}
    return {
        "tokens": n,
        "doubling_per_1000": 1000 * pairs / (n - 1),
        "types": len(types),
        "ttr": len(types) / n,
        "hapax_share_of_types": sum(v == 1 for v in types.values()) / len(types),
        "zipf_slope_top1000": float(slope),
        "word_entropy_bits": entropy_bits(types),
        "h1_char_bits": h1,
        "h2_cond_char_bits": h2,
        "mean_len": float(lens.mean()),
        "len_hist_pct": hist,
    }


def vms_tokens(sample):
    recs = parse_ivtff_lines(IVTFF_FILES["takahashi_IT2a"])
    toks = [w for r in recs if r["ltype"] == "P" for w, unc in r["tokens"] if not unc]
    return toks[:sample]


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #
def deck_weights(p_alpha):
    others = {k: v for k, v in STOCK.items() if k != "alpha"}
    tot_o = sum(others.values())
    w = {"alpha": round(1000 * p_alpha)}
    for k, v in others.items():
        w[k] = round(1000 * (1 - p_alpha) * v / tot_o)
    return {k: v for k, v in w.items() if v > 0}


def encipher(text, weights, seed=0):
    c = NaibbeCipher(seed=seed)
    c._mod.CARD_WEIGHTS[False] = dict(weights)
    try:
        return c.encipher(text)
    finally:
        c._mod.CARD_WEIGHTS[False] = dict(STOCK)


def sticky(tokens, segments, s, seed=0):
    rng = random.Random(seed)
    out = list(tokens)
    for i in range(1, len(out)):
        if segments[i] == segments[i - 1] and rng.random() < s:
            out[i] = out[i - 1]
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=200_000)
    ap.add_argument("--sample", type=int, default=30_000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = {}
    rows["VMS (IT2a, paragraph)"] = metrics(vms_tokens(args.sample))
    for lang in ("latin", "italian", "german"):
        text = load_plaintext(lang, args.chars)
        toks, segs = encipher(text, STOCK)
        rows[f"{lang} / stock deck"] = metrics(toks[: args.sample])
        for p in (0.5, 0.7, 0.9, 1.0):
            t, _ = encipher(text, deck_weights(p))
            rows[f"{lang} / deck alpha={p:.1f}"] = metrics(t[: args.sample])
        for s in (0.5, 1.0):
            rows[f"{lang} / sticky s={s:.1f}"] = metrics(
                sticky(toks, segs, s)[: args.sample]
            )

    cols = [
        ("doubling_per_1000", "dbl /1000", "{:.1f}"),
        ("types", "types", "{:d}"),
        ("ttr", "TTR", "{:.3f}"),
        ("hapax_share_of_types", "hapax", "{:.2f}"),
        ("zipf_slope_top1000", "Zipf", "{:.2f}"),
        ("word_entropy_bits", "H(word)", "{:.2f}"),
        ("h1_char_bits", "h1", "{:.2f}"),
        ("h2_cond_char_bits", "h2", "{:.2f}"),
        ("mean_len", "mean len", "{:.2f}"),
    ]
    lines = [
        "# Naibbe: deterministic table choice vs Greshko's metrics",
        "",
        (
            f"All rows: first {args.sample} tokens. VMS = Takahashi IT2a paragraph text, certain tokens, "
            "same functions. h1/h2 over EVA characters of the space-joined token stream; Zipf = log-log "
            "slope over ranks 1–1000; hapax = share of types occurring once."
        ),
        "",
        "| variant | "
        + " | ".join(c[1] for c in cols)
        + " | L2 | L3 | L4 | L5 | L6 | L7 |",
        "|---|" + "---:|" * (len(cols) + 6),
    ]
    for name, m in rows.items():
        cells = [fmt.format(m[k]) for k, _, fmt in cols]
        cells += [f"{m['len_hist_pct'][L]:.0f}" for L in range(2, 8)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (OUT / "naibbe_deck_sweep.md").write_text("\n".join(lines))
    (OUT / "naibbe_deck_sweep.json").write_text(json.dumps(rows, indent=2, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
