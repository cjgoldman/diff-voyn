"""How much geminate-collapsing would be needed for Boxer's reuse prob s = 1?

Under Boxer's model  VMS_doubling ≈ plaintext_letter_doubling × s.  Observed
VMS doubling is ~9.2/1000, plaintext letter doubling is 26–44/1000, hence
s ≈ 0.2–0.35.  If instead the scribes' source orthography wrote the commonest
doubled letters as a *single* symbol (German ss → ß, Latin nasal macrons …),
the plaintext the cipher actually saw would double less often, and s could be
larger — possibly 1 ("always reuse the same word", a rule not a preference).

This script collapses the n most frequent geminates (greedy, most frequent
first) and reports the residual doubling rate, so we can read off the n at
which the residual meets the VMS rate (s = 1).

Usage:  uv run python scripts/doubling_collapse.py [--chars N] [--target 9.2]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

DATA_ROOT = Path("/workspace/data")
LANGS = ("latin", "italian", "german")
# Per-hand VMS doubling rates (Takahashi, paragraph text, within-line, certain
# tokens) from scripts/doubling_rate.py — used as alternative targets.
VMS_RATE_OVERALL = 9.2


def load_stream(lang: str, max_chars: int) -> str:
    files = sorted(glob.glob(str(DATA_ROOT / f"corpora/v1/{lang}/docs/*")))
    parts, total = [], 0
    for f in files:
        t = re.sub(r"\s+", "", Path(f).read_text(encoding="utf-8"))
        parts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "".join(parts)[:max_chars]


def doubling_rate(s: str) -> float:
    n = len(s)
    return 1000 * sum(1 for i in range(n - 1) if s[i] == s[i + 1]) / (n - 1)


def geminate_counts(s: str) -> Counter:
    c: Counter = Counter()
    i = 0
    while i < len(s) - 1:  # count non-overlapping, as a collapse would consume
        if s[i] == s[i + 1]:
            c[s[i]] += 1
            i += 2
        else:
            i += 1
    return c


def collapse(s: str, letters: list[str]) -> str:
    """Replace each 'xx' (x in letters) by one private-use symbol."""
    for j, ch in enumerate(letters):
        s = s.replace(ch * 2, chr(0xE000 + j))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=3_000_000)
    ap.add_argument("--target", type=float, default=VMS_RATE_OVERALL)
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "analysis/doubling")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results: dict = {"target_per_1000": args.target, "languages": {}}
    report = [
        "# Geminate collapsing vs Boxer's reuse probability s",
        "",
        f"Target: residual plaintext doubling = VMS rate {args.target}/1000 (s = 1).",
        "",
    ]
    for lang in LANGS:
        text = load_stream(lang, args.chars)
        order = [ch for ch, _ in geminate_counts(text).most_common()]
        rows = []
        n_star = None
        for n in range(min(len(order), 16) + 1):
            cur = collapse(text, order[:n])
            r = doubling_rate(cur)
            rows.append(
                {
                    "n": n,
                    "collapsed": [c * 2 for c in order[:n]],
                    "residual_per_1000": r,
                    "implied_s": args.target / r if r else float("nan"),
                    "chars_after": len(cur),
                }
            )
            if n_star is None and r <= args.target:
                n_star = n
        results["languages"][lang] = {"n_star": n_star, "curve": rows}
        report += [
            f"\n## {lang}  (n* = {n_star})",
            "",
            "| n | newest pair collapsed | residual /1000 | implied s |",
            "|---:|---|---:|---:|",
        ]
        for r in rows:
            pair = r["collapsed"][-1] if r["collapsed"] else "—"
            report.append(
                f"| {r['n']} | {pair} | {r['residual_per_1000']:.1f} | "
                f"{min(r['implied_s'], 1.0):.2f} |"
            )
    (args.out / "collapse_results.json").write_text(json.dumps(results, indent=2))
    (args.out / "collapse_report.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
