"""Side-quest: token doubling rate of Naibbe ciphertext vs the VMS (~9.2/1000).

Enciphers Phase-0 plaintext (Latin / Italian / German, corpora v1) with the
pinned Naibbe v2 wrapper (``diff_voyn.ciphers.naibbe``, greshko/naibbe-cipher
@ df3d074) and counts Boxer's statistic — #{i : tok[i] == tok[i+1]} / (N-1)
over the token stream — together with its decomposition:

    doubling ≈ P(adjacent plaintext units equal) × P(same token | same unit)

where a "unit" is the unigram/bigram segment Naibbe's respacing step produced
and P(same token | same unit) is set by the card deck (same table drawn
twice for a unigram; same prefix *and* suffix tables for a bigram).

Usage: uv run python scripts/naibbe_doubling.py [--chars 300000] [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from diff_voyn.ciphers.naibbe import NaibbeCipher

CORPORA = Path("/workspace/data/corpora/v1")
OUT = Path("/workspace/data/analysis/doubling")


def load_plaintext(lang: str, n_chars: int) -> str:
    """Concatenate normalized docs (alphabetical order) up to n_chars."""
    buf = []
    total = 0
    for f in sorted((CORPORA / lang / "docs").glob("*.txt")):
        t = f.read_text(encoding="utf-8").strip()
        buf.append(t)
        total += len(t)
        if total >= n_chars:
            break
    return "".join(buf)[:n_chars]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def measure(tokens, segments):
    pairs = len(tokens) - 1
    dbl = Counter()
    same_seg = 0
    same_seg_same_tok = 0
    same_seg_uni = same_seg_uni_dbl = 0
    same_seg_bi = same_seg_bi_dbl = 0
    diff_seg_same_tok = 0
    for (t0, s0), (t1, s1) in zip(zip(tokens, segments), zip(tokens[1:], segments[1:])):
        if t0 == t1:
            dbl[t0] += 1
        if s0 == s1:
            same_seg += 1
            if t0 == t1:
                same_seg_same_tok += 1
            if len(s0) == 1:
                same_seg_uni += 1
                same_seg_uni_dbl += t0 == t1
            else:
                same_seg_bi += 1
                same_seg_bi_dbl += t0 == t1
        elif t0 == t1:
            diff_seg_same_tok += 1
    d = sum(dbl.values())
    lo, hi = wilson(d, pairs)
    return {
        "tokens": len(tokens),
        "pairs": pairs,
        "doubles": d,
        "per_1000": 1000 * d / pairs,
        "wilson_95": [1000 * lo, 1000 * hi],
        "p_adjacent_same_unit_per_1000": 1000 * same_seg / pairs,
        "p_same_token_given_same_unit": (
            same_seg_same_tok / same_seg if same_seg else float("nan")
        ),
        "unigram_units: same-unit pairs / doubled": [same_seg_uni, same_seg_uni_dbl],
        "bigram_units: same-unit pairs / doubled": [same_seg_bi, same_seg_bi_dbl],
        "doubles_from_different_units": diff_seg_same_tok,
        "unigram_share_of_units": sum(len(s) == 1 for s in segments) / len(segments),
        "top_doubled": dbl.most_common(5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=300_000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cipher = NaibbeCipher(seed=0)
    weights = cipher._mod.CARD_WEIGHTS
    results = {"card_weights": {str(k): v for k, v in weights.items()}, "runs": {}}
    lines = ["# Naibbe ciphertext doubling rate", ""]
    lines.append(f"Card weights (table -> cards): {weights}")
    for use78 in (False, True):
        w = weights[use78]
        tot = sum(w.values())
        p_same_table = sum((c / tot) ** 2 for c in w.values())
        lines.append(
            f"- deck use_78={use78}: P(same table twice) ≈ {p_same_table:.3f} "
            f"(ignoring without-replacement depletion within a deck)"
        )
    lines.append("")
    lines.append(
        "| language | deck | seed | tokens | doubles | per 1000 (95%) | adjacent same unit /1000 | "
        "P(same token \\| same unit) | unigram share | doubles from different units | top doubled |"
    )
    lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|")
    for lang in ("latin", "italian", "german"):
        text = load_plaintext(lang, args.chars)
        for use78 in (False, True):
            agg = Counter()
            for seed in range(args.seeds):
                c = NaibbeCipher(seed=seed, use_78_card_deck=use78)
                toks, segs = c.encipher(text)
                r = measure(toks, segs)
                results["runs"][f"{lang}/use78={use78}/seed={seed}"] = r
                agg["pairs"] += r["pairs"]
                agg["doubles"] += r["doubles"]
                agg["same_seg"] += (
                    r["p_adjacent_same_unit_per_1000"] * r["pairs"] / 1000
                )
                lines.append(
                    f"| {lang} | {'78' if use78 else '52'} | {seed} | {r['tokens']} | {r['doubles']} | "
                    f"{r['per_1000']:.2f} ({r['wilson_95'][0]:.1f}–{r['wilson_95'][1]:.1f}) | "
                    f"{r['p_adjacent_same_unit_per_1000']:.1f} | {r['p_same_token_given_same_unit']:.3f} | "
                    f"{r['unigram_share_of_units']:.2f} | {r['doubles_from_different_units']} | "
                    + ", ".join(f"{t}×{n}" for t, n in r["top_doubled"][:3])
                    + " |"
                )
            lo, hi = wilson(agg["doubles"], agg["pairs"])
            lines.append(
                f"| **{lang}** | **{'78' if use78 else '52'}** | all | | {agg['doubles']} | "
                f"**{1000*agg['doubles']/agg['pairs']:.2f}** ({1000*lo:.1f}–{1000*hi:.1f}) | "
                f"{1000*agg['same_seg']/agg['pairs']:.1f} | | | | |"
            )
    lines.append("")
    lines.append(
        "Reference: VMS 9.29/1000 Boxer-style on his csv; 6.6–10.4/1000 across transliterations/policies "
        "(`doubling_report.md`). Arithmetic cipher is *tuned* to 0.0092 (task 0.7)."
    )
    (OUT / "naibbe_doubling.md").write_text("\n".join(lines))
    (OUT / "naibbe_doubling.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
