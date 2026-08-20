"""Side-quest: VMS token "doubling rate" and its variation with scribal hand.

Boxer (voynich-attack, ``voynpy/pseudo_vms/encoder.py::tune_to_vms``) hard-codes
a VMS doubling rate of 0.0092 (~9.1-9.2 per 1000 tokens), defined as

    #{i : tokens[i] == tokens[i+1]} / (N - 1)

over whitespace-split tokens.  This script (a) re-measures that number on the
IVTFF transliterations already fetched for task 0.8 and on Boxer's own csv
transcription, under a few explicit tokenization choices, and (b) breaks it
down by Lisa Fagin Davis' scribal hand (IVTFF page variable ``$H``), by
Currier hand (``$C``) and by Currier language (``$L``), with page-level
bootstrap confidence intervals and a chi-square homogeneity test.

Usage:  uv run python scripts/doubling_rate.py [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA_ROOT = Path("/workspace/data")
IVTFF_FILES = {
    "takahashi_IT2a": DATA_ROOT / "raw/vms/IT2a-n.txt",
    "reference_RF1b": DATA_ROOT / "raw/vms/RF1b-e.txt",
}
BOXER_CSV = DATA_ROOT / "external/voynich-attack/transcription/vms.csv"

_PAGE_RE = re.compile(r"^<(f[^.>,]+)>\s*(<!([^>]*)>)?\s*$")
_LOCUS_RE = re.compile(r"^<(f[^.>,]+)\.(\d+),([^>]*)>\s*(.*)$")
_VAR_RE = re.compile(r"\$([A-Z])=([^\s>]+)")
_HAND_TAG_RE = re.compile(r"<@H=([^>]+)>")
_ALT_READING_RE = re.compile(r"\[([^:\]]*):[^\]]*\]")
_MARKUP_RE = re.compile(r"<[^>]*>")
_EXT_EVA_RE = re.compile(r"@\d+;")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_ivtff_lines(path: Path) -> list[dict]:
    """Yield one record per locus: page vars, locus type, hand, tokens.

    Tokenization policy: alternate readings take the first alternative;
    ``{...}`` brace groups are kept (content only); inline markup removed;
    ``.`` and ``,`` are word separators; tokens containing ``?`` or an
    extended-EVA ``@nnn;`` code are flagged ``uncertain``.
    """
    records: list[dict] = []
    page_vars: dict[str, str] = {}
    page = None
    hand_override = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("#") or not raw.strip():
            continue
        m = _PAGE_RE.match(raw)
        if m and not _LOCUS_RE.match(raw):
            page = m.group(1)
            page_vars = dict(_VAR_RE.findall(m.group(3) or ""))
            hand_override = None
            continue
        lm = _LOCUS_RE.match(raw)
        if not lm or page is None:
            continue
        locus_flags, text = lm.group(3), lm.group(4)
        ltype = next((c for c in locus_flags if c.isalpha()), "?")
        ht = _HAND_TAG_RE.search(text)
        if ht:  # f115r: hand switches mid-page via a text tag
            hand_override = ht.group(1)
        hand = page_vars.get("H")
        if hand == "@":
            hand = hand_override or "?"
        text = _ALT_READING_RE.sub(r"\1", text)
        text = _MARKUP_RE.sub("", text)
        text = text.replace("{", "").replace("}", "")
        toks = []
        for w in re.split(r"[.,\s]+", text):
            if not w:
                continue
            unc = ("?" in w) or bool(_EXT_EVA_RE.search(w))
            toks.append((w, unc))
        records.append(
            {
                "page": page,
                "line": int(lm.group(2)),
                "ltype": ltype,
                "hand": hand,
                "currier_hand": page_vars.get("C"),
                "lang": page_vars.get("L"),
                "illus": page_vars.get("I"),
                "quire": page_vars.get("Q"),
                "tokens": toks,
            }
        )
    return records


def _boxer_page(folio: str) -> str:
    """Boxer's ``89ra`` -> IVTFF ``f89r1`` (pharma sub-pages); others ``fNNr``."""
    m = re.fullmatch(r"(\d+[rv])([ab])", folio)
    if m:
        return "f" + m.group(1) + ("1" if m.group(2) == "a" else "2")
    return "f" + folio


def parse_boxer_csv(path: Path) -> list[dict]:
    """Boxer's csv: folio,par,line,t1..t26 — tokens are comma-joined glyph lists."""
    records = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            toks = []
            for k in (f"t{i}" for i in range(1, 27)):
                t = (row.get(k) or "").strip()
                if not t or t == "$":
                    continue
                toks.append((t, "?" in t))
            records.append(
                {
                    "page": _boxer_page(row["folio"]),
                    "par": int(row["par"]),
                    "line": int(row["line"]),
                    "ltype": "P",
                    "tokens": toks,
                }
            )
    return records


# --------------------------------------------------------------------------- #
# Doubling statistics
# --------------------------------------------------------------------------- #
def pair_stats(records, *, within_line: bool, drop_uncertain: bool):
    """Return per-record (n_pairs, n_doubles, doubled_types) under a policy.

    ``within_line=False`` also counts the pair straddling consecutive lines of
    the same page (Boxer's ``split()`` over the whole text does this).
    """
    out = []
    prev_tok, prev_page = None, None
    for r in records:
        toks = [t for t in r["tokens"] if not (drop_uncertain and t[1])]
        words = [t[0] for t in toks]
        if within_line or r["page"] != prev_page:
            prev_tok = None
        pairs = 0
        doubles = Counter()
        for w in words:
            if prev_tok is not None:
                pairs += 1
                if w == prev_tok:
                    doubles[w] += 1
            prev_tok = w
        prev_page = r["page"]
        if not within_line:
            prev_tok = words[-1] if words else prev_tok
        out.append((pairs, sum(doubles.values()), doubles))
    return out


def rate_ci_wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cluster_bootstrap(page_counts, n_boot=4000, seed=0):
    """Page-level bootstrap of doubles/pairs (pages are the resampling unit)."""
    rng = np.random.default_rng(seed)
    arr = np.array(page_counts, dtype=float)  # (pages, 2): pairs, doubles
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    s = arr[idx].sum(axis=1)
    rates = np.where(s[:, 0] > 0, s[:, 1] / np.maximum(s[:, 0], 1), np.nan)
    return tuple(np.nanpercentile(rates, [2.5, 97.5]))


def group_table(records, stats, key_fn, min_pairs=200):
    """Aggregate pair stats by key; CIs are binomial (Wilson) and page-bootstrap."""
    pairs = Counter()
    doubles = Counter()
    types = defaultdict(Counter)
    unigrams = defaultdict(Counter)
    pages = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r, (np_, nd, dtypes) in zip(records, stats):
        k = key_fn(r)
        if k is None:
            continue
        pairs[k] += np_
        doubles[k] += nd
        types[k].update(dtypes)
        unigrams[k].update(w for w, unc in r["tokens"] if not unc)
        pages[k][r["page"]][0] += np_
        pages[k][r["page"]][1] += nd
    rows = []
    for k in sorted(pairs, key=lambda x: str(x)):
        n, d = pairs[k], doubles[k]
        if n < min_pairs:
            continue
        lo, hi = rate_ci_wilson(d, n)
        blo, bhi = cluster_bootstrap(list(pages[k].values()))
        cnt = np.array(list(unigrams[k].values()), float)
        tot = cnt.sum()
        p_same = (cnt * (cnt - 1)).sum() / (
            tot * (tot - 1)
        )  # P(two random tokens equal)
        rows.append(
            {
                "group": str(k),
                "pages": len(pages[k]),
                "pairs": n,
                "doubles": d,
                "per_1000": 1000 * d / n,
                "chance_per_1000": 1000 * p_same,
                "obs_over_chance": (d / n) / p_same if p_same > 0 else float("nan"),
                "wilson_95": [1000 * lo, 1000 * hi],
                "page_boot_95": [1000 * blo, 1000 * bhi],
                "top_types": types[k].most_common(5),
            }
        )
    return rows


def chi2_homogeneity(rows):
    """Pearson chi-square that the doubling probability is equal across groups."""
    from scipy.stats import chi2

    n = np.array([r["pairs"] for r in rows], float)
    d = np.array([r["doubles"] for r in rows], float)
    p = d.sum() / n.sum()
    exp_d, exp_nd = n * p, n * (1 - p)
    stat = (((d - exp_d) ** 2) / exp_d + (((n - d) - exp_nd) ** 2) / exp_nd).sum()
    dof = len(rows) - 1
    return float(stat), dof, float(chi2.sf(stat, dof))


def fmt_rows(rows, title):
    lines = [
        f"\n### {title}",
        "",
        "| group | pages | pairs | doubles | per 1000 | Wilson 95% | page-bootstrap 95% | chance /1000 | obs/chance | top doubled types |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for r in rows:
        w = "{:.1f}–{:.1f}".format(*r["wilson_95"])
        b = "{:.1f}–{:.1f}".format(*r["page_boot_95"])
        tt = ", ".join(f"{t}×{c}" for t, c in r["top_types"][:4])
        lines.append(
            f"| {r['group']} | {r['pages']} | {r['pairs']} | {r['doubles']} | "
            f"{r['per_1000']:.2f} | {w} | {b} | {r['chance_per_1000']:.2f} | "
            f"{r['obs_over_chance']:.2f} | {tt} |"
        )
    if len(rows) > 1:
        stat, dof, p = chi2_homogeneity(rows)
        lines.append(f"\nχ² homogeneity: {stat:.1f} on {dof} dof, p = {p:.2g}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "analysis/doubling")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report: list[str] = ["# VMS token doubling rate", ""]
    results: dict = {}

    # ---- Part A: verify the headline number under explicit policies -------
    report.append("## A. Overall doubling rate (Boxer's figure: 0.0092 ≈ 9.2/1000)\n")
    report.append(
        "| source | text | across lines | drop uncertain | tokens | pairs | doubles | per 1000 (95% Wilson) |"
    )
    report.append("|---|---|---|---|---:|---:|---:|---|")
    sources = {k: parse_ivtff_lines(p) for k, p in IVTFF_FILES.items()}
    sources["boxer_csv"] = parse_boxer_csv(BOXER_CSV)
    for name, recs in sources.items():
        for text_sel in ("P", "all"):
            sel = [r for r in recs if text_sel == "all" or r["ltype"] == "P"]
            if name == "boxer_csv" and text_sel == "all":
                continue
            for across in (False, True):
                for drop in (False, True):
                    st = pair_stats(sel, within_line=not across, drop_uncertain=drop)
                    n = sum(s[0] for s in st)
                    d = sum(s[1] for s in st)
                    ntok = sum(
                        1 for r in sel for t in r["tokens"] if not (drop and t[1])
                    )
                    lo, hi = rate_ci_wilson(d, n)
                    results.setdefault("overall", []).append(
                        {
                            "source": name,
                            "text": text_sel,
                            "across_lines": across,
                            "drop_uncertain": drop,
                            "tokens": ntok,
                            "pairs": n,
                            "doubles": d,
                            "per_1000": 1000 * d / n,
                            "wilson_95": [1000 * lo, 1000 * hi],
                        }
                    )
                    report.append(
                        f"| {name} | {text_sel} | {across} | {drop} | {ntok} | {n} | {d} | "
                        f"{1000*d/n:.2f} ({1000*lo:.1f}–{1000*hi:.1f}) |"
                    )

    # ---- Part B: by scribal hand ----------------------------------------
    # Primary policy: paragraph text, within-line pairs, uncertain tokens dropped.
    for name in IVTFF_FILES:
        recs = [r for r in sources[name] if r["ltype"] == "P"]
        st = pair_stats(recs, within_line=True, drop_uncertain=True)
        report.append(
            f"\n## B. {name}: breakdown (paragraph text, within-line, certain tokens)"
        )
        blocks = {
            "davis_hand": lambda r: r["hand"],
            "currier_lang": lambda r: r["lang"],
            "davis_hand_x_currier_lang": lambda r: (
                f"H{r['hand']}/{r['lang']}" if r["hand"] and r["lang"] else None
            ),
            "currier_hand": lambda r: r["currier_hand"],
            "illustration_section": lambda r: r["illus"],
        }
        for bname, fn in blocks.items():
            rows = group_table(recs, st, fn)
            results.setdefault(name, {})[bname] = rows
            report.append(fmt_rows(rows, f"{name} by {bname}"))

        # Sensitivity: Boxer-style (across lines, uncertain kept) by Davis hand
        st2 = pair_stats(recs, within_line=False, drop_uncertain=False)
        rows = group_table(recs, st2, blocks["davis_hand"])
        results[name]["davis_hand_boxer_style"] = rows
        report.append(
            fmt_rows(
                rows,
                f"{name} by davis_hand — Boxer-style counting (across lines, uncertain kept)",
            )
        )

    # Boxer's own transcription by Davis hand (hand map taken from IT2a headers)
    hand_of_page = {}
    for r in sources["takahashi_IT2a"]:
        hand_of_page.setdefault(r["page"], r["hand"] if r["hand"] != "?" else None)
    lang_of_page = {r["page"]: r["lang"] for r in sources["takahashi_IT2a"]}
    brecs = sources["boxer_csv"]
    for r in brecs:
        r["hand"] = hand_of_page.get(r["page"])
        r["lang"] = lang_of_page.get(r["page"])
    # Rosettes foldout: Boxer's f85r/f85v/f86r = IVTFF f85r1/f85r2/f86v3-6, all $H=2 $L=B.
    for r in brecs:
        if r["hand"] is None and r["page"] in ("f85r", "f85v", "f86r"):
            r["hand"], r["lang"] = "2", "B"
    unmapped = sorted({r["page"] for r in brecs if r["hand"] is None})
    st = pair_stats(brecs, within_line=True, drop_uncertain=True)
    rows = group_table(brecs, st, lambda r: r["hand"])
    results["boxer_csv"] = {"davis_hand": rows, "unmapped_pages": unmapped}
    report.append(
        "\n## C. Boxer's csv transcription by Davis hand (hand map from IT2a page headers)"
    )
    report.append(f"\nPages without a hand mapping (excluded): {unmapped}")
    report.append(fmt_rows(rows, "boxer_csv by davis_hand"))
    rows = group_table(brecs, st, lambda r: r["lang"])
    results["boxer_csv"]["currier_lang"] = rows
    report.append(fmt_rows(rows, "boxer_csv by currier_lang"))

    (args.out / "doubling_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    (args.out / "doubling_report.md").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
