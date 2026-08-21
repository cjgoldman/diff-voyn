"""Side-quest: does the VMS token-length distribution vary with scribal hand?

Boxer's arithmetic cipher (voynich-attack ``voynpy/pseudo_vms/encoder.py``)
makes token length arbitrary: each plaintext letter has homophones of lengths
2..6 glyphs and the scribe picks one per letter (his encoder samples from a
global length mix 10/22/26/26/16 %).  If length really is a free scribal
choice, the length distribution could differ between Lisa Fagin Davis' hands.

This script measures token length three ways

* ``boxer_glyphs``: Boxer's own csv transcription, already segmented into his
  glyph alphabet (comma-separated) — the unit his claim is stated in;
* ``eva_glyphs``: EVA with the benched gallows ``cth/ckh/cph/cfh`` and the
  digraphs ``ch``/``sh`` counted as one glyph (Takahashi IT2a, Reference RF1b);
* ``eva_chars``: raw EVA characters;
* ``eva_collapsed``: EVA glyphs with ``i``-runs (+ closing stroke) and
  ``e``-runs each counted as one unit — the coarsest defensible segmentation,

and, per Davis hand (IVTFF ``$H``), reports the length histogram, mean, short/
long shares and their ratio, with page-bootstrap CIs; then tests homogeneity
across hands with a token-level chi-square (anticonservative), a page-label
permutation test (pages are the clustering unit), and pairwise permutation
tests on the mean difference and the KS distance.  Because hand is confounded
with Currier language and section, the same comparisons are repeated within
Currier B, within the Herbal section, and with line-initial/final tokens
removed (the known line-position length effect).  A variance decomposition on
page means asks how much of the page-to-page spread in mean length hand
explains beyond language, and the IT2a-vs-RF1b per-hand difference gives the
transliteration noise floor.

Usage:  uv run python scripts/token_length_by_hand.py [--out DIR] [--perms N]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doubling_rate import (
    BOXER_CSV,
    DATA_ROOT,
    IVTFF_FILES,
    parse_boxer_csv,
    parse_ivtff_lines,
)

_EVA_GLYPH_RE = re.compile(r"c[tkpf]h|ch|sh|[a-z]")
_I_RUN_RE = re.compile(r"i+[nrlm]")  # iin/in/iir... + closing stroke -> one unit
_E_RUN_RE = re.compile(r"e+")  # ee/eee -> one unit
MAX_LEN = 10  # histogram top bin is ">= MAX_LEN"
HANDS = ["1", "2", "3", "4", "5"]


# --------------------------------------------------------------------------- #
# Token table
# --------------------------------------------------------------------------- #
def eva_glyph_len(w: str) -> int:
    return len(_EVA_GLYPH_RE.findall(w))


def eva_collapsed_len(w: str) -> int:
    """EVA glyphs with i-runs (+ closing n/r/l/m) and e-runs each counted once.

    This is the coarsest defensible segmentation: it treats the ``iin``/``in``
    and ``ee`` ligature families as single glyphs, the way Boxer's ``m``/``n``
    units do for the i-family."""
    return eva_glyph_len(_E_RUN_RE.sub("e", _I_RUN_RE.sub("N", w)))


def build_tokens(records, measure: str) -> list[dict]:
    """Flatten records into per-token dicts with a length under ``measure``.

    Paragraph text only; uncertain tokens dropped; keeps position-in-line.
    """
    out = []
    for r in records:
        if r["ltype"] != "P":
            continue
        toks = [w for w, unc in r["tokens"] if not unc]
        n = len(toks)
        for i, w in enumerate(toks):
            if measure == "boxer_glyphs":
                L = len([g for g in w.split(",") if g])
            elif measure == "eva_glyphs":
                L = eva_glyph_len(w)
            elif measure == "eva_collapsed":
                L = eva_collapsed_len(w)
            else:
                L = len(w)
            if L == 0:
                continue
            out.append(
                {
                    "page": r["page"],
                    "hand": str(r.get("hand")),
                    "lang": str(r.get("lang")),
                    "illus": str(r.get("illus")),
                    "word": w,
                    "len": L,
                    "pos": "first" if i == 0 else ("last" if i == n - 1 else "mid"),
                    "line_id": (r["page"], r.get("par", 0), r["line"]),
                    "idx": i,
                }
            )
    return out


def attach_hands_to_boxer(brecs, it2a_recs):
    hand_of_page, lang_of_page, illus_of_page = {}, {}, {}
    for r in it2a_recs:
        hand_of_page.setdefault(r["page"], r["hand"] if r["hand"] != "?" else None)
        lang_of_page.setdefault(r["page"], r["lang"])
        illus_of_page.setdefault(r["page"], r["illus"])
    for r in brecs:
        r["hand"] = hand_of_page.get(r["page"])
        r["lang"] = lang_of_page.get(r["page"])
        r["illus"] = illus_of_page.get(r["page"])
        if r["hand"] is None and r["page"] in ("f85r", "f85v", "f86r"):
            r["hand"], r["lang"], r["illus"] = "2", "B", "C"  # rosettes foldout
    return brecs


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #
def hist(lengths) -> np.ndarray:
    h = np.zeros(MAX_LEN, dtype=float)  # bins 1..MAX_LEN-1, last = >=MAX_LEN
    for L in lengths:
        h[min(L, MAX_LEN) - 1] += 1
    return h


def page_hists(tokens):
    """{page: (hand, hist)}"""
    per = defaultdict(list)
    meta = {}
    for t in tokens:
        per[t["page"]].append(t["len"])
        meta[t["page"]] = t["hand"]
    return {p: (meta[p], hist(ls)) for p, ls in per.items()}


def summarise(h: np.ndarray) -> dict:
    n = h.sum()
    lens = np.arange(1, MAX_LEN + 1)
    mean = (h * lens).sum() / n
    sd = math.sqrt(((h * (lens - mean) ** 2).sum()) / n)
    cdf = np.cumsum(h) / n
    median = int(np.searchsorted(cdf, 0.5) + 1)
    short = h[:3].sum() / n  # len <= 3
    long_ = h[5:].sum() / n  # len >= 6
    return {
        "n": int(n),
        "mean": mean,
        "sd": sd,
        "median": median,
        "p_short_le3": short,
        "p_long_ge6": long_,
        "long_over_short": long_ / short if short > 0 else float("nan"),
        "hist_pct": list(100 * h / n),
    }


def page_bootstrap(ph, hand, n_boot=3000, seed=0):
    rng = np.random.default_rng(seed)
    hs = np.array([h for p, (hd, h) in ph.items() if hd == hand])
    if len(hs) < 2:
        return {}
    idx = rng.integers(0, len(hs), size=(n_boot, len(hs)))
    tot = hs[idx].sum(axis=1)  # (n_boot, MAX_LEN)
    n = tot.sum(axis=1)
    lens = np.arange(1, MAX_LEN + 1)
    mean = (tot * lens).sum(axis=1) / n
    short = tot[:, :3].sum(axis=1) / n
    long_ = tot[:, 5:].sum(axis=1) / n
    ratio = long_ / np.maximum(short, 1e-9)
    q = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]
    return {
        "mean_ci": q(mean),
        "p_short_ci": q(short),
        "p_long_ci": q(long_),
        "ratio_ci": q(ratio),
    }


def type_level(tokens, hand):
    """Length distribution over distinct word types used by a hand."""
    types = {t["word"]: t["len"] for t in tokens if t["hand"] == hand}
    if not types:
        return None
    h = hist(types.values())
    s = summarise(h)
    s["n_types"] = s.pop("n")
    return s


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def chi2_stat(table: np.ndarray):
    """Pearson chi-square on a groups x bins table (empty bins dropped)."""
    table = table[:, table.sum(axis=0) > 0]
    exp = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    stat = ((table - exp) ** 2 / exp).sum()
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    return float(stat), int(dof)


def cramers_v(table: np.ndarray):
    stat, _ = chi2_stat(table)
    k = min(table.shape) - 1
    return math.sqrt(stat / (table.sum() * k)) if k > 0 else float("nan")


def homogeneity(ph, hands, n_perm, seed=0):
    """Token-level chi-square + page-label permutation p-value for it."""
    from scipy.stats import chi2

    pages = [(hd, h) for p, (hd, h) in ph.items() if hd in hands]
    labels = np.array([hd for hd, _ in pages])
    H = np.array([h for _, h in pages])

    def table_for(lbl):
        return np.array([H[lbl == hd].sum(axis=0) for hd in hands])

    obs_tab = table_for(labels)
    obs, dof = chi2_stat(obs_tab)
    p_naive = float(chi2.sf(obs, dof))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        s, _ = chi2_stat(table_for(perm))
        hits += s >= obs
    return {
        "hands": hands,
        "n_pages": [int((labels == hd).sum()) for hd in hands],
        "n_tokens": [int(x) for x in obs_tab.sum(axis=1)],
        "chi2": obs,
        "dof": dof,
        "cramers_v": cramers_v(obs_tab),
        "p_token_level": p_naive,
        "p_page_perm": (hits + 1) / (n_perm + 1),
    }


def pairwise(ph, a, b, n_perm, seed=0):
    """Page-permutation tests for mean difference and KS distance, a vs b."""
    pages = [(hd, h) for p, (hd, h) in ph.items() if hd in (a, b)]
    labels = np.array([hd for hd, _ in pages])
    H = np.array([h for _, h in pages])
    lens = np.arange(1, MAX_LEN + 1)

    def stats(lbl):
        ha, hb = H[lbl == a].sum(axis=0), H[lbl == b].sum(axis=0)
        ma = (ha * lens).sum() / ha.sum()
        mb = (hb * lens).sum() / hb.sum()
        ks = np.abs(np.cumsum(ha) / ha.sum() - np.cumsum(hb) / hb.sum()).max()
        return ma - mb, ks

    d_obs, ks_obs = stats(labels)
    rng = np.random.default_rng(seed)
    hd = hk = 0
    for _ in range(n_perm):
        d, ks = stats(rng.permutation(labels))
        hd += abs(d) >= abs(d_obs)
        hk += ks >= ks_obs
    return {
        "pair": f"H{a} vs H{b}",
        "mean_diff": float(d_obs),
        "p_mean_perm": (hd + 1) / (n_perm + 1),
        "ks_D": float(ks_obs),
        "p_ks_perm": (hk + 1) / (n_perm + 1),
    }


def variance_decomposition(tokens):
    """Share of token-weighted between-page variance in mean length explained
    by Currier language, by Davis hand, and by hand within language."""
    per = defaultdict(list)
    meta = {}
    for t in tokens:
        per[t["page"]].append(t["len"])
        meta[t["page"]] = (t["hand"], t["lang"])
    pages = [
        (meta[p][0], meta[p][1], len(v), float(np.mean(v))) for p, v in per.items()
    ]
    n = np.array([p[2] for p in pages], float)
    m = np.array([p[3] for p in pages])
    grand = (n * m).sum() / n.sum()
    ss_tot = (n * (m - grand) ** 2).sum()

    def ss_between(keyf):
        groups = defaultdict(list)
        for i, p in enumerate(pages):
            groups[keyf(p)].append(i)
        ss = 0.0
        for idx in groups.values():
            idx = np.array(idx)
            gm = (n[idx] * m[idx]).sum() / n[idx].sum()
            ss += n[idx].sum() * (gm - grand) ** 2
        return ss

    ss_lang = ss_between(lambda p: p[1])
    ss_hand = ss_between(lambda p: p[0])
    ss_cell = ss_between(lambda p: (p[0], p[1]))
    return {
        "n_pages": len(pages),
        "between_page_sd_of_page_means": float(math.sqrt(ss_tot / n.sum())),
        "R2_language": ss_lang / ss_tot,
        "R2_hand": ss_hand / ss_tot,
        "R2_hand_x_language": ss_cell / ss_tot,
        "R2_hand_given_language": (ss_cell - ss_lang) / ss_tot,
    }


def lag1_autocorr(tokens, hand, n_perm=500, seed=0, detrend=False):
    """Lag-1 correlation of interior token lengths within lines, vs a
    within-line shuffle null.  Boxer's encoder draws lengths i.i.d. per letter
    (apart from doublings), so it predicts r equal to the null.

    ``detrend=True`` first subtracts the hand's mean length at the token's
    relative line position (10 bins), so a smooth line-position gradient
    cannot masquerade as serial dependence."""
    lines = defaultdict(list)
    for t in tokens:
        if t["hand"] == hand:
            lines[t["line_id"]].append((t["idx"], t["len"]))
    seqs = []
    for v in lines.values():
        v = sorted(v)
        n = len(v)
        if n < 5:
            continue
        seqs.append([(i / (n - 1), L) for i, L in v][1:-1])  # interior only
    if len(seqs) < 20:
        return None
    if detrend:
        bins = defaultdict(list)
        for s in seqs:
            for rp, L in s:
                bins[min(int(rp * 10), 9)].append(L)
        mu = {b: float(np.mean(v)) for b, v in bins.items()}
        seqs = [np.array([L - mu[min(int(rp * 10), 9)] for rp, L in s]) for s in seqs]
    else:
        seqs = [np.array([L for _, L in s], float) for s in seqs]

    def r_of(seqs_):
        x = np.concatenate([s[:-1] for s in seqs_])
        y = np.concatenate([s[1:] for s in seqs_])
        return float(np.corrcoef(x, y)[0, 1])

    obs = r_of(seqs)
    rng = np.random.default_rng(seed)
    null = np.array([r_of([rng.permutation(s) for s in seqs]) for _ in range(n_perm)])
    return {
        "lines": len(seqs),
        "pairs": int(sum(len(s) - 1 for s in seqs)),
        "r_lag1": obs,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "z": (
            float((obs - null.mean()) / null.std()) if null.std() > 0 else float("nan")
        ),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def hand_table(tokens, title, min_tokens=150):
    ph = page_hists(tokens)
    rows = []
    for hand in HANDS:
        h = sum((hh for p, (hd, hh) in ph.items() if hd == hand), np.zeros(MAX_LEN))
        if h.sum() < min_tokens:
            continue
        s = summarise(h)
        s["hand"] = hand
        s["pages"] = sum(1 for p, (hd, _) in ph.items() if hd == hand)
        s.update(page_bootstrap(ph, hand))
        s["types"] = type_level(tokens, hand)
        rows.append(s)
    lines = [
        f"\n### {title}",
        "",
        "| hand | pages | tokens | mean len (page-boot 95%) | sd | median | "
        "P(len≤3) | P(len≥6) | long/short (95%) | type-level mean | "
        + " | ".join(f"L{i}" for i in range(1, MAX_LEN))
        + f" | L{MAX_LEN}+ |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|---:|" + "---:|" * MAX_LEN,
    ]
    for s in rows:
        mci = "{:.2f}–{:.2f}".format(*s["mean_ci"]) if "mean_ci" in s else "–"
        rci = "{:.2f}–{:.2f}".format(*s["ratio_ci"]) if "ratio_ci" in s else "–"
        tm = f"{s['types']['mean']:.2f}" if s["types"] else "–"
        lines.append(
            f"| {s['hand']} | {s['pages']} | {s['n']} | {s['mean']:.3f} ({mci}) | "
            f"{s['sd']:.2f} | {s['median']} | {s['p_short_le3']:.3f} | "
            f"{s['p_long_ge6']:.3f} | {s['long_over_short']:.2f} ({rci}) | {tm} | "
            + " | ".join(f"{x:.1f}" for x in s["hist_pct"])
            + " |"
        )
    return rows, "\n".join(lines)


def test_block(tokens, hands, n_perm, title):
    ph = page_hists(tokens)
    hands = [h for h in hands if sum(1 for p, (hd, _) in ph.items() if hd == h) >= 2]
    out = {"hands": hands}
    lines = [f"\n**{title}** — hands {', '.join('H' + h for h in hands)}"]
    if len(hands) < 2:
        lines.append("(fewer than two hands with ≥2 pages — skipped)")
        return out, "\n".join(lines)
    hom = homogeneity(ph, hands, n_perm)
    out["homogeneity"] = hom
    lines.append(
        f"- χ² = {hom['chi2']:.1f} on {hom['dof']} dof, Cramér's V = {hom['cramers_v']:.3f}; "
        f"token-level p = {hom['p_token_level']:.2g}; **page-permutation p = {hom['p_page_perm']:.3f}** "
        f"(pages per hand {hom['n_pages']}, tokens {hom['n_tokens']})"
    )
    pw = []
    for i, a in enumerate(hands):
        for b in hands[i + 1 :]:
            r = pairwise(ph, a, b, n_perm)
            pw.append(r)
            lines.append(
                f"- {r['pair']}: Δmean = {r['mean_diff']:+.3f} (perm p = {r['p_mean_perm']:.3f}), "
                f"KS D = {r['ks_D']:.3f} (perm p = {r['p_ks_perm']:.3f})"
            )
    out["pairwise"] = pw
    return out, "\n".join(lines)


def fig_distributions(all_tokens: dict, out: Path):
    """One panel per measure: per-hand length pmf (pages ≥ 2, tokens ≥ 150)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "1": "#4C6EF5",
        "2": "#F08C00",
        "3": "#2F9E44",
        "4": "#AE3EC9",
        "5": "#E03131",
    }
    measures = list(all_tokens)
    fig, axes = plt.subplots(
        1, len(measures), figsize=(5.2 * len(measures), 3.8), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, mname in zip(axes, measures):
        ph = page_hists(all_tokens[mname])
        for hand in HANDS:
            h = sum((hh for p, (hd, hh) in ph.items() if hd == hand), np.zeros(MAX_LEN))
            if h.sum() < 150:
                continue
            ax.plot(
                np.arange(1, MAX_LEN + 1),
                100 * h / h.sum(),
                marker="o",
                ms=3.5,
                lw=1.4,
                color=colors[hand],
                label=f"H{hand} (n={int(h.sum())})",
            )
        ax.set_title(mname.replace("_", " "), fontsize=11)
        ax.set_xlabel("token length")
        ax.set_xticks(range(1, MAX_LEN + 1))
        ax.set_xticklabels([str(i) for i in range(1, MAX_LEN)] + [f"{MAX_LEN}+"])
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% of tokens")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "VMS token length by Davis hand (paragraph text, certain tokens)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "analysis/token_length")
    ap.add_argument("--perms", type=int, default=4000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    it2a = parse_ivtff_lines(IVTFF_FILES["takahashi_IT2a"])
    rf1b = parse_ivtff_lines(IVTFF_FILES["reference_RF1b"])
    boxer = attach_hands_to_boxer(parse_boxer_csv(BOXER_CSV), it2a)

    datasets = {
        "boxer_csv / boxer_glyphs": build_tokens(boxer, "boxer_glyphs"),
        "takahashi_IT2a / eva_glyphs": build_tokens(it2a, "eva_glyphs"),
        "takahashi_IT2a / eva_chars": build_tokens(it2a, "eva_chars"),
        "takahashi_IT2a / eva_collapsed": build_tokens(it2a, "eva_collapsed"),
        "reference_RF1b / eva_glyphs": build_tokens(rf1b, "eva_glyphs"),
    }
    # Boxer's csv rows have no $I; also drop tokens from unmapped pages.
    for name, toks in datasets.items():
        datasets[name] = [t for t in toks if t["hand"] not in ("None", "?")]

    report = ["# VMS token length by scribal hand", ""]
    report.append(
        "Paragraph text, uncertain tokens dropped. Hands = Davis `$H` (IT2a headers; "
        "Boxer's csv pages mapped via IT2a). CIs and permutation tests use **pages** as "
        "the resampling unit; token-level χ² p-values are shown only for reference "
        "(they ignore within-page correlation and are anticonservative)."
    )
    results: dict = {}

    for name, toks in datasets.items():
        report.append(f"\n## {name}")
        res: dict = {}
        rows, txt = hand_table(toks, f"{name}: all paragraph tokens")
        res["by_hand"] = rows
        report.append(txt)

        # Boxer's calibration check: share within 2..6 normalised
        if name.startswith("boxer"):
            report.append(
                "\nBoxer's encoder hard-codes the L2..L6 mix 10/22/26/26/16 %. Observed per hand, "
                "renormalised to lengths 2–6:"
            )
            report.append(
                "\n| hand | L2 | L3 | L4 | L5 | L6 |\n|---|---:|---:|---:|---:|---:|"
            )
            for s in rows:
                seg = np.array(s["hist_pct"][1:6])
                seg = 100 * seg / seg.sum()
                report.append(
                    f"| {s['hand']} | " + " | ".join(f"{x:.0f}" for x in seg) + " |"
                )
            res["boxer_L2_6_mix"] = {
                s["hand"]: list(
                    100 * np.array(s["hist_pct"][1:6]) / sum(s["hist_pct"][1:6])
                )
                for s in rows
            }

        tests = {}
        report.append("\n#### Homogeneity tests")
        t, txt = test_block(
            toks, HANDS, args.perms, "All hands (confounded with language/section)"
        )
        tests["all"] = t
        report.append(txt)
        B = [x for x in toks if x["lang"] == "B"]
        t, txt = test_block(B, ["2", "3", "5"], args.perms, "Within Currier B")
        tests["currier_B"] = t
        report.append(txt)
        if not name.startswith("boxer"):
            Hb = [x for x in toks if x["illus"] == "H"]
            t, txt = test_block(
                Hb, HANDS, args.perms, "Within Herbal section (H1=A, H2/H3/H5=B)"
            )
            tests["herbal"] = t
            report.append(txt)
            HB = [x for x in Hb if x["lang"] == "B"]
            t, txt = test_block(HB, ["2", "3", "5"], args.perms, "Within Herbal-B only")
            tests["herbal_B"] = t
            report.append(txt)
        mid = [x for x in toks if x["pos"] == "mid"]
        rows_mid, txt = hand_table(mid, f"{name}: line-interior tokens only")
        res["by_hand_interior"] = rows_mid
        report.append(txt)
        t, txt = test_block(mid, HANDS, args.perms, "All hands, line-interior tokens")
        tests["all_interior"] = t
        report.append(txt)
        t, txt = test_block(
            [x for x in mid if x["lang"] == "B"],
            ["2", "3", "5"],
            args.perms,
            "Within Currier B, line-interior tokens",
        )
        tests["currier_B_interior"] = t
        report.append(txt)
        res["tests"] = tests

        vd = variance_decomposition(toks)
        res["variance_decomposition"] = vd
        report.append(
            f"\n#### Variance decomposition of page mean length ({vd['n_pages']} pages, "
            f"between-page SD of page means = {vd['between_page_sd_of_page_means']:.3f})\n\n"
            f"| explained by | R² |\n|---|---:|\n"
            f"| Currier language | {vd['R2_language']:.3f} |\n"
            f"| Davis hand | {vd['R2_hand']:.3f} |\n"
            f"| hand × language | {vd['R2_hand_x_language']:.3f} |\n"
            f"| hand beyond language | {vd['R2_hand_given_language']:.3f} |"
        )

        ac = {h: lag1_autocorr(toks, h) for h in HANDS}
        acd = {h: lag1_autocorr(toks, h, detrend=True) for h in HANDS}
        ac = {h: v for h, v in ac.items() if v}
        res["lag1_autocorr"] = ac
        res["lag1_autocorr_detrended"] = {h: v for h, v in acd.items() if v}
        report.append(
            "\n#### Lag-1 autocorrelation of interior token lengths within lines "
            "(Boxer's i.i.d. homophone draw predicts r = shuffle null)\n\n"
            "| hand | lines | pairs | r | shuffle-null mean ± sd | z | "
            "r (position-detrended) | null | z |\n|---|---:|---:|---:|---|---:|---:|---|---:|"
        )
        for h, v in ac.items():
            d = acd[h]
            report.append(
                f"| {h} | {v['lines']} | {v['pairs']} | {v['r_lag1']:+.3f} | "
                f"{v['null_mean']:+.3f} ± {v['null_sd']:.3f} | {v['z']:+.1f} | "
                f"{d['r_lag1']:+.3f} | {d['null_mean']:+.3f} ± {d['null_sd']:.3f} | {d['z']:+.1f} |"
            )
        results[name] = res

    # Transliteration noise floor: same hand, IT2a vs RF1b (eva_glyphs)
    report.append(
        "\n## Transliteration noise floor (same pages, IT2a vs RF1b, eva_glyphs)"
    )
    report.append("\n| hand | mean IT2a | mean RF1b | Δ |\n|---|---:|---:|---:|")
    a = {
        s["hand"]: s["mean"] for s in results["takahashi_IT2a / eva_glyphs"]["by_hand"]
    }
    b = {
        s["hand"]: s["mean"] for s in results["reference_RF1b / eva_glyphs"]["by_hand"]
    }
    floor = {}
    for h in HANDS:
        if h in a and h in b:
            floor[h] = a[h] - b[h]
            report.append(f"| {h} | {a[h]:.3f} | {b[h]:.3f} | {a[h]-b[h]:+.3f} |")
    results["transliteration_floor"] = floor

    fig_path = args.out / "token_length_by_hand.png"
    fig_distributions(
        {
            k.split(" / ")[1] + (" (Boxer)" if k.startswith("boxer") else " (IT2a)"): v
            for k, v in datasets.items()
            if not k.startswith("reference")
        },
        fig_path,
    )
    report.append(f"\nFigure: `{fig_path}`")

    (args.out / "token_length_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    (args.out / "token_length_report.md").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
