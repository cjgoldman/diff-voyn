"""Positive control: is scribal hand visible in the text at all, where length isn't?

Within Herbal-section Currier-B pages (hands 2, 3, 5 — same section, same
dialect) and, as sensitivity, within all of Currier B, compare a battery of
glyph-level statistics between hands with the same page-permutation machinery
used for token length.  Each statistic is a categorical distribution pooled
over a hand's tokens; the test statistic is the (token-weighted, generalised)
Jensen-Shannon divergence between hands in bits, with pages permuted across
hands.  Unit token length (i-runs/e-runs collapsed) is included in the same
table so the contrast is direct: if hand shows in glyph choice, vocabulary or
line structure but not in length, length is not where scribal freedom lives.

Usage: uv run python scripts/hand_positive_control.py [--perms N] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doubling_rate import DATA_ROOT, IVTFF_FILES, parse_ivtff_lines
from token_length_by_hand import _EVA_GLYPH_RE, build_tokens, eva_collapsed_len

TOP_BIGRAMS = 60
TOP_WORDS = 60
GALLOWS = {"k", "t", "p", "f", "ckh", "cth", "cph", "cfh"}


# --------------------------------------------------------------------------- #
# Per-page feature counts
# --------------------------------------------------------------------------- #
def glyphs(w):
    return _EVA_GLYPH_RE.findall(w)


def page_features(tokens, vocab):
    """{page: {feature: Counter}} ; vocab fixes the category sets."""
    lines = defaultdict(list)
    for t in tokens:
        lines[t["line_id"]].append(t)
    feats = defaultdict(lambda: defaultdict(Counter))
    for ts in lines.values():
        ts.sort(key=lambda t: t["idx"])
        page = ts[0]["page"]
        F = feats[page]
        F["tokens_per_line"][min(len(ts), 15)] += 1
        g0 = glyphs(ts[0]["word"])
        if g0:
            F["line_initial_glyph"][g0[0]] += 1
        for t in ts:
            w = t["word"]
            gs = glyphs(w)
            if not gs:
                continue
            F["len_units"][min(eva_collapsed_len(w), 7)] += 1
            F["glyph_unigram"].update(gs)
            F["word_initial_glyph"][gs[0]] += 1
            F["word_final_glyph"][gs[-1]] += 1
            F["gallows_rate"].update("G" if g in GALLOWS else "-" for g in gs)
            for a, b in pairwise(["^"] + gs + ["$"]):
                bg = a + b
                F["glyph_bigram"][bg if bg in vocab["glyph_bigram"] else "OTHER"] += 1
            F["word_types"][w if w in vocab["word_types"] else "OTHER"] += 1
    return feats


def build_vocab(tokens):
    bg = Counter()
    wt = Counter()
    for t in tokens:
        gs = glyphs(t["word"])
        bg.update(a + b for a, b in pairwise(["^"] + gs + ["$"]))
        wt[t["word"]] += 1
    return {
        "glyph_bigram": {k for k, _ in bg.most_common(TOP_BIGRAMS)},
        "word_types": {k for k, _ in wt.most_common(TOP_WORDS)},
    }


# --------------------------------------------------------------------------- #
# JSD permutation test
# --------------------------------------------------------------------------- #
def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def gjsd(mats):
    """Generalised JSD of rows (count vectors), token-weighted."""
    tot = mats.sum(axis=1)
    w = tot / tot.sum()
    P = mats / tot[:, None]
    return entropy((w[:, None] * P).sum(0)) - float(
        (w * np.array([entropy(p) for p in P])).sum()
    )


def jsd_test(page_vecs, page_hands, hands, n_perm, seed=0):
    """page_vecs: (pages, cats) counts; page_hands: list of hand labels."""
    labels = np.array(page_hands)
    X = np.asarray(page_vecs, float)

    def stat(lbl):
        return gjsd(np.stack([X[lbl == h].sum(0) for h in hands]))

    obs = stat(labels)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(labels)) for _ in range(n_perm)])
    return {
        "jsd_bits": obs,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "z": (
            float((obs - null.mean()) / null.std()) if null.std() > 0 else float("nan")
        ),
        "p": float(((null >= obs).sum() + 1) / (n_perm + 1)),
        "n_pages": [int((labels == h).sum()) for h in hands],
        "n_items": [int(X[labels == h].sum()) for h in hands],
    }


def top_diffs(page_vecs, page_hands, hands, cats, k=6):
    """Largest |Δ share| categories between the first two hands."""
    X = np.asarray(page_vecs, float)
    lab = np.array(page_hands)
    P = [X[lab == h].sum(0) for h in hands[:2]]
    P = [p / p.sum() for p in P]
    d = P[0] - P[1]
    idx = np.argsort(-np.abs(d))[:k]
    return [(cats[i], round(100 * P[0][i], 2), round(100 * P[1][i], 2)) for i in idx]


# --------------------------------------------------------------------------- #
def run_battery(feats, hands, n_perm, title, rep, res):
    pages = [p for p in feats if feats[p]["_hand"] in hands]
    page_hands = [feats[p]["_hand"] for p in pages]
    features = [
        "len_units",
        "glyph_unigram",
        "gallows_rate",
        "word_initial_glyph",
        "word_final_glyph",
        "glyph_bigram",
        "line_initial_glyph",
        "tokens_per_line",
        "word_types",
    ]
    rep.append(f"\n### {title} — hands {', '.join('H' + h for h in hands)}")
    rep.append("")
    rep.append(
        f"| statistic | categories | items per hand | JSD bits | perm-null mean ± sd | z | page-perm p | largest differences (H{hands[0]} % vs H{hands[1]} %) |"
    )
    rep.append("|---|---:|---|---:|---|---:|---:|---|")
    out = {}
    for f in features:
        cats = sorted({c for p in pages for c in feats[p][f]}, key=str)
        X = np.array([[feats[p][f].get(c, 0) for c in cats] for p in pages], float)
        r = jsd_test(X, page_hands, hands, n_perm)
        r["top_diffs"] = top_diffs(X, page_hands, hands, cats)
        out[f] = r
        td = ", ".join(f"{c}: {a} vs {b}" for c, a, b in r["top_diffs"][:4])
        rep.append(
            f"| {f} | {len(cats)} | {'/'.join(map(str, r['n_items']))} | {r['jsd_bits']:.4f} | "
            f"{r['null_mean']:.4f} ± {r['null_sd']:.4f} | {r['z']:+.1f} | **{r['p']:.3f}** | {td} |"
        )
    res[title] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "analysis/token_length")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    recs = parse_ivtff_lines(IVTFF_FILES["takahashi_IT2a"])
    toks = [t for t in build_tokens(recs, "eva_glyphs") if t["hand"] in "12345"]
    vocab = build_vocab(toks)
    feats = page_features(toks, vocab)
    meta = {}
    for t in toks:
        meta[t["page"]] = (t["hand"], t["lang"], t["illus"])
    for p in feats:
        feats[p]["_hand"], feats[p]["_lang"], feats[p]["_illus"] = meta[p]

    rep = ["# Positive control: is hand visible in non-length text statistics?", ""]
    rep.append(
        "Takahashi IT2a, paragraph text, certain tokens, EVA glyphs (ch/sh/benched gallows merged). "
        "Each statistic is a categorical distribution pooled over a hand's tokens; the test statistic is the "
        "token-weighted generalised Jensen-Shannon divergence between hands (bits), p-values from "
        f"{args.perms} page-label permutations. `len_units` is the collapsed unit length (1..7+) for contrast. "
        f"`glyph_bigram` uses the top {TOP_BIGRAMS} bigrams (+OTHER), `word_types` the top {TOP_WORDS} words (+OTHER), "
        "both chosen on the whole manuscript."
    )
    res = {}

    herbalB = {
        p: f for p, f in feats.items() if f["_illus"] == "H" and f["_lang"] == "B"
    }
    rep.append("\n## A. Herbal section, Currier B only (same section, same dialect)")
    run_battery(herbalB, ["2", "3", "5"], args.perms, "Herbal-B, three hands", rep, res)
    run_battery(herbalB, ["2", "3"], args.perms, "Herbal-B, H2 vs H3", rep, res)
    run_battery(herbalB, ["2", "5"], args.perms, "Herbal-B, H2 vs H5", rep, res)
    run_battery(herbalB, ["3", "5"], args.perms, "Herbal-B, H3 vs H5", rep, res)

    allB = {p: f for p, f in feats.items() if f["_lang"] == "B"}
    rep.append(
        "\n## B. All Currier B (section confounded: H2 = biological/herbal, H3 = stars/herbal)"
    )
    run_battery(allB, ["2", "3"], args.perms, "All-B, H2 vs H3", rep, res)

    herbal = {
        p: f
        for p, f in feats.items()
        if f["_illus"] == "H" and f["_lang"] in ("A", "B")
    }
    rep.append(
        "\n## C. Reference effect size: Herbal A (H1) vs Herbal B (H2) — a known different text"
    )
    run_battery(herbal, ["1", "2"], args.perms, "Herbal, H1/A vs H2/B", rep, res)

    (args.out / "hand_positive_control.md").write_text("\n".join(rep))
    (args.out / "hand_positive_control.json").write_text(
        json.dumps(res, indent=2, default=str)
    )
    print("\n".join(rep))
    print(f"\nwritten: {args.out}/hand_positive_control.*")


if __name__ == "__main__":
    main()
