"""Side-quest: first-order Markov ("finite-state") model of VMS token length.

Estimates P(length of next token | length of this token) from adjacent
within-line pairs and asks whether the *transition structure* varies with
Davis hand (``$H``) or Currier language (``$L``), separately from the already-
known difference in the marginal length distributions.

States are token lengths 1..6 and 7+ (7 states).  For each comparison set of
groups three nested models are fitted:

* M0  per-group marginal, no dependence            P_g(j)
* M1s per-group marginal x SHARED lift (IPF, no     P_g(j|i) ∝ P_g(j)·l(i,j)
      three-way interaction group×i×j)
* M1g per-group full transition matrix             P_g(j|i)
* M2g per-group second-order matrix                P_g(k|i,j)   (how deep?)

and compared by (a) page-grouped 5-fold held-out log-likelihood per token and
(b) a page-permutation likelihood-ratio test of M1g vs M1s (is the group×i×j
interaction real once page clustering is respected?).  Lift matrices and
mutual information I(L_t; L_t+1) are reported per group.

Usage: uv run python scripts/length_transitions.py [--measure eva_collapsed]
       [--out DIR] [--perms N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import pairwise
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
from token_length_by_hand import attach_hands_to_boxer, build_tokens

K = 7  # states: 1..6, 7+
STATE_NAMES = ["1", "2", "3", "4", "5", "6", "7+"]
ALPHA = 0.5  # additive smoothing on all count tables


def st(L: int) -> int:
    return min(L, K) - 1


# --------------------------------------------------------------------------- #
# Sequences -> pairs / triples
# --------------------------------------------------------------------------- #
def build_lines(tokens):
    lines = defaultdict(list)
    for t in tokens:
        lines[t["line_id"]].append(t)
    out = []
    for ts in lines.values():
        ts.sort(key=lambda t: t["idx"])
        out.append(
            {
                "page": ts[0]["page"],
                "hand": ts[0]["hand"],
                "lang": ts[0]["lang"],
                "s": [st(t["len"]) for t in ts],
            }
        )
    return out


def pair_counts(lines, keyf, interior=False):
    """{group: K x K count matrix} of adjacent within-line pairs."""
    C = defaultdict(lambda: np.zeros((K, K)))
    for ln in lines:
        g = keyf(ln)
        if g is None:
            continue
        s = ln["s"][1:-1] if interior else ln["s"]
        for a, b in pairwise(s):
            C[g][a, b] += 1
    return dict(C)


def triple_counts(lines, keyf, interior=False):
    """{group: K x K x K} counts of (i, j, k) consecutive triples."""
    C = defaultdict(lambda: np.zeros((K, K, K)))
    for ln in lines:
        g = keyf(ln)
        if g is None:
            continue
        s = ln["s"][1:-1] if interior else ln["s"]
        for a, b, c in zip(s, s[1:], s[2:]):
            C[g][a, b, c] += 1
    return dict(C)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def rownorm(M):
    return M / M.sum(axis=-1, keepdims=True)


def ipf_no_three_way(N, iters=200, tol=1e-9):
    """MLE of the log-linear model [GI][GJ][IJ] for a G x K x K table N."""
    m = np.ones_like(N)
    tgt_gi, tgt_gj, tgt_ij = N.sum(2), N.sum(1), N.sum(0)
    for _ in range(iters):
        m *= (tgt_gi / m.sum(2))[:, :, None]
        m *= (tgt_gj / m.sum(1))[:, None, :]
        m *= (tgt_ij / m.sum(0))[None, :, :]
        if np.abs(m.sum(2) - tgt_gi).max() < tol:
            break
    return m


def fit_models(train_pairs, train_triples, groups):
    """Return dict of model -> per-group conditional tables (smoothed)."""
    N = np.stack([train_pairs[g] + ALPHA for g in groups])  # G x K x K
    models = {}
    models["M0"] = {g: rownorm(N[gi].sum(0)) for gi, g in enumerate(groups)}
    pooled = rownorm(N.sum(0))
    models["M1pooled"] = {g: pooled for g in groups}
    m = ipf_no_three_way(N)
    models["M1s"] = {g: rownorm(m[gi]) for gi, g in enumerate(groups)}
    models["M1g"] = {g: rownorm(N[gi]) for gi, g in enumerate(groups)}
    models["M2g"] = {g: rownorm(train_triples[g] + ALPHA) for g in groups}
    return models


def loglik(models, test_triples, groups):
    """Held-out mean log-lik (nats) of the 3rd token of each triple under each
    model; all models scored on the same targets so they are comparable."""
    out = {}
    for name, tabs in models.items():
        ll, n = 0.0, 0
        for g in groups:
            T = test_triples[g]
            n += T.sum()
            if name == "M0":
                p = tabs[g]  # (K,)
                ll += (T.sum((0, 1)) * np.log(p)).sum()
            elif name == "M2g":
                ll += (T * np.log(tabs[g])).sum()
            else:
                P = tabs[g]  # P(k|j)
                ll += (T.sum(0) * np.log(P)).sum()
        out[name] = ll / n
    return out


def cv_compare(lines, keyf, groups, interior=False, folds=5, seed=0):
    pages = sorted({ln["page"] for ln in lines if keyf(ln) in groups})
    rng = np.random.default_rng(seed)
    rng.shuffle(pages)
    fold_of = {p: i % folds for i, p in enumerate(pages)}
    tot = defaultdict(float)
    n_tot = 0
    for f in range(folds):
        tr = [ln for ln in lines if keyf(ln) in groups and fold_of[ln["page"]] != f]
        te = [ln for ln in lines if keyf(ln) in groups and fold_of[ln["page"]] == f]
        tp = pair_counts(tr, keyf, interior)
        tt = triple_counts(tr, keyf, interior)
        et = triple_counts(te, keyf, interior)
        for g in groups:
            tp.setdefault(g, np.zeros((K, K)))
            tt.setdefault(g, np.zeros((K, K, K)))
            et.setdefault(g, np.zeros((K, K, K)))
        n = sum(et[g].sum() for g in groups)
        if n == 0:
            continue
        ll = loglik(fit_models(tp, tt, groups), et, groups)
        for k, v in ll.items():
            tot[k] += v * n
        n_tot += n
    return {k: v / n_tot for k, v in tot.items()}, int(n_tot)


def lrt_interaction(lines, keyf, groups, n_perm, interior=False, seed=0):
    """G² for group×i×j interaction (M1g vs M1s) with page-permuted labels."""

    def g2(ls):
        C = pair_counts(ls, keyf, interior)
        N = np.stack([C.get(g, np.zeros((K, K))) for g in groups])
        m = ipf_no_three_way(N + ALPHA)
        Ns = N + ALPHA
        return float(2 * (Ns * (np.log(Ns) - np.log(m))).sum())

    sel = [ln for ln in lines if keyf(ln) in groups]
    obs = g2(sel)
    pages = sorted({ln["page"] for ln in sel})
    lab = {p: keyf(next(ln for ln in sel if ln["page"] == p)) for p in pages}
    rng = np.random.default_rng(seed)
    hits = 0
    labels = [lab[p] for p in pages]
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        remap = dict(zip(pages, perm))
        ls = [dict(ln, grp=remap[ln["page"]]) for ln in sel]
        s = g2_with(ls, groups, interior)
        hits += s >= obs
    dof = (len(groups) - 1) * (K - 1) ** 2
    from scipy.stats import chi2

    return {
        "G2": obs,
        "dof": dof,
        "p_token_level": float(chi2.sf(obs, dof)),
        "p_page_perm": (hits + 1) / (n_perm + 1),
        "n_pages": len(pages),
    }


def g2_with(ls, groups, interior):
    C = pair_counts(ls, lambda ln: ln["grp"], interior)
    N = np.stack([C.get(g, np.zeros((K, K))) for g in groups]) + ALPHA
    m = ipf_no_three_way(N)
    return float(2 * (N * (np.log(N) - np.log(m))).sum())


# --------------------------------------------------------------------------- #
# Descriptives
# --------------------------------------------------------------------------- #
def mutual_info(C):
    P = C / C.sum()
    pi, pj = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log2(P / (pi * pj)), 0.0)
    return float(t.sum())


def fmt_matrix(C, title, lift=False):
    P = rownorm(C + 1e-12)
    pj = C.sum(0) / C.sum()
    lines = [
        f"\n**{title}** (rows: this token; cols: next token; n = {int(C.sum())} pairs)",
        "",
    ]
    lines.append("| L | n | " + " | ".join(STATE_NAMES) + " |")
    lines.append("|---|---:|" + "---:|" * K)
    for i in range(K):
        vals = P[i] / pj if lift else 100 * P[i]
        fmt = "{:.2f}" if lift else "{:.0f}"
        lines.append(
            f"| {STATE_NAMES[i]} | {int(C[i].sum())} | "
            + " | ".join(fmt.format(v) for v in vals)
            + " |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--measure",
        default="eva_collapsed",
        choices=["eva_collapsed", "eva_glyphs", "eva_chars", "boxer_glyphs"],
    )
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "analysis/token_length")
    ap.add_argument("--perms", type=int, default=1000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    it2a = parse_ivtff_lines(IVTFF_FILES["takahashi_IT2a"])
    if args.measure == "boxer_glyphs":
        recs = attach_hands_to_boxer(parse_boxer_csv(BOXER_CSV), it2a)
        src = "boxer_csv"
    else:
        recs, src = it2a, "takahashi_IT2a"
    toks = [t for t in build_tokens(recs, args.measure) if t["hand"] in "12345"]
    lines = build_lines(toks)

    by_hand = lambda ln: ln["hand"]
    by_lang = lambda ln: ln["lang"] if ln["lang"] in ("A", "B") else None
    by_hand_B = lambda ln: (
        ln["hand"] if ln["lang"] == "B" and ln["hand"] in "235" else None
    )

    rep = [f"# First-order length-transition model — {src} / {args.measure}", ""]
    rep.append(
        "States = token length 1..6, 7+. Adjacent within-line pairs, paragraph text, certain tokens. "
        "Held-out numbers are page-grouped 5-fold mean log-likelihood per token (nats), higher is better; "
        "M0 = per-group marginal only, M1pooled = one shared transition matrix, "
        "M1s = per-group marginal × shared lift (no group×i×j interaction), M1g = per-group full matrix, "
        "M2g = per-group second-order."
    )
    res = {}

    # ---- descriptive matrices -------------------------------------------
    CH = pair_counts(lines, by_hand)
    CL = pair_counts(lines, by_lang)
    rep.append("\n## Transition matrices P(next | this), % by row")
    for g in ("A", "B"):
        rep.append(fmt_matrix(CL[g], f"Currier {g}"))
    for h in ("1", "2", "3", "5"):
        rep.append(fmt_matrix(CH[h], f"Hand {h}"))
    rep.append(
        "\n## Lift P(next | this) / P(next) — the dependence structure with the marginal divided out"
    )
    for g in ("A", "B"):
        rep.append(fmt_matrix(CL[g], f"Currier {g} lift", lift=True))
    for h in ("2", "3"):
        rep.append(fmt_matrix(CH[h], f"Hand {h} lift", lift=True))

    rep.append(
        "\n## Mutual information I(L_t ; L_t+1) in bits (all pairs / interior-only pairs)\n"
    )
    CHi = pair_counts(lines, by_hand, interior=True)
    CLi = pair_counts(lines, by_lang, interior=True)
    rep.append("| group | pairs | MI | MI interior |\n|---|---:|---:|---:|")
    mi = {}
    for name, C, Ci in [(f"Currier {g}", CL[g], CLi[g]) for g in "AB"] + [
        (f"Hand {h}", CH[h], CHi[h]) for h in "12345" if h in CH
    ]:
        mi[name] = (mutual_info(C), mutual_info(Ci))
        rep.append(
            f"| {name} | {int(C.sum())} | {mi[name][0]:.4f} | {mi[name][1]:.4f} |"
        )
    res["mutual_info_bits"] = mi
    res["transition_counts"] = {f"hand_{h}": C.tolist() for h, C in CH.items()} | {
        f"lang_{g}": C.tolist() for g, C in CL.items()
    }

    # ---- model comparison -----------------------------------------------
    comparisons = [
        ("Currier A vs B", by_lang, ["A", "B"]),
        ("All hands", by_hand, ["1", "2", "3", "4", "5"]),
        ("Within Currier B: H2 / H3 / H5", by_hand_B, ["2", "3", "5"]),
        ("Within Currier B: H2 / H3", by_hand_B, ["2", "3"]),
    ]
    rep.append("\n## Model comparison (held-out log-lik per token, nats; Δ vs M0)\n")
    rep.append(
        "| comparison | pairs scored | M0 | M1pooled | M1s | M1g | M2g | "
        "LRT M1g vs M1s: G² (dof) | token p | page-perm p |"
    )
    rep.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    res["comparisons"] = {}
    for title, keyf, groups in comparisons:
        for interior in (False, True):
            cv, n = cv_compare(lines, keyf, groups, interior=interior)
            lrt = lrt_interaction(lines, keyf, groups, args.perms, interior=interior)
            lab = title + (" (interior)" if interior else "")
            res["comparisons"][lab] = {"cv": cv, "n": n, "lrt": lrt}
            dd = {k: f"{cv[k] - cv['M0']:+.4f}" for k in cv}
            rep.append(
                f"| {lab} | {n} | {cv['M0']:.4f} | {dd['M1pooled']} | {dd['M1s']} | {dd['M1g']} | {dd['M2g']} | "
                f"{lrt['G2']:.1f} ({lrt['dof']}) | {lrt['p_token_level']:.2g} | {lrt['p_page_perm']:.3f} |"
            )

    # ---- per-row homogeneity within B -----------------------------------
    rep.append(
        "\n## Within Currier B, per current-length row: is P(next | this) the same for H2 and H3?\n"
        "(page-permutation χ² on the 2 × 7 table of next-length counts for that row)\n"
    )
    rep.append(
        "| this length | H2 pairs | H3 pairs | χ² | page-perm p |\n|---|---:|---:|---:|---:|"
    )
    rows = per_row_tests(lines, by_hand_B, ["2", "3"], args.perms)
    res["per_row_H2_H3"] = rows
    for r in rows:
        rep.append(
            f"| {r['row']} | {r['n'][0]} | {r['n'][1]} | {r['chi2']:.1f} | {r['p_page_perm']:.3f} |"
        )

    tag = f"{src}_{args.measure}"
    (args.out / f"length_transitions_{tag}.md").write_text("\n".join(rep))
    (args.out / f"length_transitions_{tag}.json").write_text(
        json.dumps(res, indent=2, default=str)
    )
    print("\n".join(rep))
    print(f"\nwritten: {args.out}/length_transitions_{tag}.*")


def per_row_tests(lines, keyf, groups, n_perm, seed=0):
    sel = [ln for ln in lines if keyf(ln) in groups]
    pages = sorted({ln["page"] for ln in sel})
    lab = {p: keyf(next(ln for ln in sel if ln["page"] == p)) for p in pages}
    rng = np.random.default_rng(seed)

    def chi2_rows(ls, kf):
        C = pair_counts(ls, kf)
        N = np.stack([C.get(g, np.zeros((K, K))) for g in groups])  # G x K x K
        out = []
        for i in range(K):
            T = N[:, i, :]
            T = T[:, T.sum(0) > 0]
            E = np.outer(T.sum(1), T.sum(0)) / max(T.sum(), 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                out.append(float(np.nansum(np.where(E > 0, (T - E) ** 2 / E, 0))))
        return out, N

    obs, N = chi2_rows(sel, keyf)
    hits = np.zeros(K)
    labels = [lab[p] for p in pages]
    for _ in range(n_perm):
        remap = dict(zip(pages, rng.permutation(labels)))
        ls = [dict(ln, grp=remap[ln["page"]]) for ln in sel]
        s, _ = chi2_rows(ls, lambda ln: ln["grp"])
        hits += np.array(s) >= np.array(obs)
    return [
        {
            "row": STATE_NAMES[i],
            "n": [int(N[gi, i].sum()) for gi in range(len(groups))],
            "chi2": obs[i],
            "p_page_perm": float((hits[i] + 1) / (n_perm + 1)),
        }
        for i in range(K)
    ]


if __name__ == "__main__":
    main()
