"""Task 3.5 — bound-fairness audit (design §5b, §9.3; requirement R1).

Question: does the per-language bound looseness — as estimated by the
calibration offsets ``NELBO − NLL_AR`` — correlate with language family,
corpus size, or morphology? And, before that: is the offset measuring the
*bound gap* at all, or the quality of the reference model it is measured
against? With three languages no correlation coefficient is significant, so
the audit reports three things that *can* be tested:

1. **Reference dependence.** The same backbone calibrated against different
   reference tiers (per-language AR v1 → v2 → multilingual v3). If the
   offsets move by more than their standard errors when only the reference
   changes, the table carries reference quality, not bound looseness.
2. **Language dependence beyond document dispersion.** Per-document offsets
   (window-mean NELBO − NLL_AR per held-out document) give 16 units across
   three languages: a one-way ANOVA / Kruskal–Wallis asks whether language
   explains offset variance beyond the between-document spread within a
   language.
3. **Correlates** (descriptive, n = 3): Spearman rank agreement of the
   offsets with train-corpus size, intrinsic entropy (AR NLL), n-gram
   structure gain (order-1 → order-5 entropy drop, a crude morphology /
   predictability proxy) and family — reported with the explicit caveat.

Every finding above noise is listed as ESCALATED with its consequence; the
G3 check refuses a report with an un-escalated above-noise finding.

Usage:
    uv run python scripts/fairness_audit.py [--tables v1 v2 v3 v3-25m ...] [--adopt v3]
Writes DATA_ROOT/analysis/phase3/fairness_audit.json and
docs/phase3_fairness_audit.md; ClearML ``task3.5``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats

from diff_voyn.ciphers.external import data_root
from diff_voyn.data.loader import LANG_TO_INDEX
from diff_voyn.metrology import FAMILY, CalibrationTable

LANGS = tuple(LANG_TO_INDEX)
REPO = Path(__file__).resolve().parent.parent


def load_table(root: Path, version: str):
    t = CalibrationTable.load(version, root)
    full = json.loads(Path(t.path).read_text())
    npz_path = root / "calibration" / f"calibration_{version}_windows.npz"
    npz = np.load(npz_path) if npz_path.exists() else None
    return t, full, npz


def per_document_offsets(npz, lang: str) -> dict[str, float] | None:
    if npz is None or f"{lang}/doc_index" not in npz:
        return None
    nelbo = npz[f"{lang}/nelbo"][:, LANG_TO_INDEX[lang]]
    nll = npz[f"{lang}/nll_ar"]
    idx = npz[f"{lang}/doc_index"]
    ids = [str(d) for d in npz[f"{lang}/doc_ids"]]
    return {ids[int(d)]: float((nelbo - nll)[idx == d].mean()) for d in np.unique(idx)}


def language_dependence_test(doc_offsets: dict[str, dict[str, float]]) -> dict:
    groups = [np.array(list(v.values())) for v in doc_offsets.values() if v]
    if sum(len(g) > 1 for g in groups) < 2:
        return {"available": False}
    f, p_anova = stats.f_oneway(*groups)
    h, p_kw = stats.kruskal(*groups)
    within = float(np.sqrt(np.mean([g.var(ddof=1) for g in groups if len(g) > 1])))
    means = [g.mean() for g in groups]
    return {
        "available": True,
        "units": "held-out documents",
        "n_per_language": {l: len(v) for l, v in doc_offsets.items()},
        "doc_offset_mean": {
            l: float(np.mean(list(v.values()))) for l, v in doc_offsets.items()
        },
        "doc_offset_sd": {
            l: float(np.std(list(v.values()), ddof=1)) if len(v) > 1 else None
            for l, v in doc_offsets.items()
        },
        "between_language_range_bits": float(max(means) - min(means)),
        "within_language_doc_sd_bits": within,
        "anova_F": float(f),
        "anova_p": float(p_anova),
        "kruskal_H": float(h),
        "kruskal_p": float(p_kw),
    }


def spearman(x: dict[str, float], y: dict[str, float]) -> float:
    xs = [x[l] for l in LANGS]
    ys = [y[l] for l in LANGS]
    r = stats.spearmanr(xs, ys).statistic
    return float(r) if np.isfinite(r) else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    p.add_argument(
        "--tables",
        nargs="+",
        default=["v1-arv1", "v1", "v2", "v3-phase_a", "v3", "v2-25m", "v3-25m"],
    )
    p.add_argument("--adopt", default="v3", help="table the audit recommends applying")
    p.add_argument("--no-clearml", action="store_true")
    args = p.parse_args()

    manifest = json.loads((root / "corpora/v1/manifest.json").read_text())
    splits = json.loads((root / "corpora/v1/splits_v1.json").read_text())
    ngram = json.loads((root / "ngram_lms/v1/summary.json").read_text())
    doc_chars = {
        d["doc_id"]: d.get("norm_chars") or d.get("chars") or 0
        for docs in manifest["documents"].values()
        for d in docs
    }
    train_chars = {
        l: sum(doc_chars.get(d["doc_id"], 0) for d in splits["languages"][l]["train"])
        for l in LANGS
    }
    structure_gain = {
        l: ngram[l]["per_order_bits"]["1"] - ngram[l]["per_order_bits"]["5"]
        for l in LANGS
    }

    tables, doc_tests = {}, {}
    for v in args.tables:
        try:
            t, full, npz = load_table(root, v)
        except FileNotFoundError:
            print(f"(table {v} not present — skipped)")
            continue
        docs = {l: per_document_offsets(npz, l) for l in LANGS}
        tables[v] = {
            "summary": t.summary(),
            "offsets_bits": t.offsets_bits,
            "offsets_sem": t.offsets_sem,
            "nelbo_bits": t.nelbo_bits,
            "nll_ar_bits": t.nll_ar_bits,
            "backbone_run": Path(t.backbone_path).parent.name,
            "ar_dir": full.get("ar_dir"),
            "per_document_offsets": docs,
        }
        if all(docs.values()):
            doc_tests[v] = language_dependence_test(docs)

    findings = []
    # 1. reference dependence: same backbone, different reference tiers
    by_backbone: dict[str, list[str]] = {}
    for v, t in tables.items():
        by_backbone.setdefault(t["backbone_run"], []).append(v)
    ref_dep = {}
    for run, vs in by_backbone.items():
        if len(vs) < 2:
            continue
        moves = {}
        for l in LANGS:
            vals = {v: tables[v]["offsets_bits"][l] for v in vs}
            sems = {v: tables[v]["offsets_sem"][l] for v in vs}
            rng_ = max(vals.values()) - min(vals.values())
            moves[l] = {
                "offsets": vals,
                "range_bits": rng_,
                "range_over_sem": rng_ / max(max(sems.values()), 1e-9),
            }
        ref_dep[run] = moves
        worst = max(m["range_over_sem"] for m in moves.values())
        above = worst > 3
        findings.append(
            {
                "id": f"reference-dependence/{run}",
                "above_noise": above,
                "escalated": above,
                "statement": (
                    f"{run}: swapping only the AR reference ({', '.join(vs)}) moves the offsets by "
                    + ", ".join(f"{l} {m['range_bits']:.3f}" for l, m in moves.items())
                    + f" bits/char (up to {worst:.0f}× s.e.m.)"
                ),
                "consequence": (
                    (
                        "the offset is dominated by reference quality, not bound looseness — "
                        f"adopt the most data-fair reference tier ({args.adopt}: one multilingual "
                        "model on the backbone's own mix) and treat remaining offsets as "
                        "bound-gap-plus-architecture terms, never as proof of comparable tightness"
                    )
                    if above
                    else "offsets are reference-stable within noise"
                ),
            }
        )
    # 2. language dependence beyond document dispersion, for the adopted table
    adopt = args.adopt if args.adopt in tables else next(iter(tables))
    lt = doc_tests.get(adopt, {"available": False})
    if lt.get("available"):
        above = lt["anova_p"] < 0.05
        findings.append(
            {
                "id": f"language-dependence/{adopt}",
                "above_noise": above,
                "escalated": above,
                "statement": (
                    f"{adopt}: per-document offsets differ by language — range "
                    f"{lt['between_language_range_bits']:.3f} bits vs within-language document "
                    f"sd {lt['within_language_doc_sd_bits']:.3f}; ANOVA F={lt['anova_F']:.1f} "
                    f"p={lt['anova_p']:.3g}, Kruskal p={lt['kruskal_p']:.3g} "
                    f"(n = {lt['n_per_language']})"
                ),
                "consequence": (
                    (
                        "a language-level offset exists beyond document noise — it is exactly what "
                        "the calibration subtracts, and it must be re-measured after every phase; "
                        "the 3.6 synthetic suite decides whether the corrected ranking is fair"
                    )
                    if above
                    else "language-level offsets are within document-to-document dispersion"
                ),
            }
        )
    # 3. correlates (descriptive)
    corr = {}
    for v, t in tables.items():
        off = t["offsets_bits"]
        corr[v] = {
            "train_chars": spearman(off, train_chars),
            "intrinsic_entropy_nll_ar": spearman(off, t["nll_ar_bits"]),
            "ngram_structure_gain": spearman(off, structure_gain),
            "family_germanic_minus_romance_bits": float(
                off["german"] - np.mean([off["latin"], off["italian"]])
            ),
        }
    a = corr[adopt]
    findings.append(
        {
            "id": f"correlates/{adopt}",
            "above_noise": False,
            "escalated": False,
            "statement": (
                f"{adopt}: Spearman(offset, train chars) = {a['train_chars']:+.1f}, "
                f"(offset, AR entropy) = {a['intrinsic_entropy_nll_ar']:+.1f}, "
                f"(offset, 1→5-gram gain) = {a['ngram_structure_gain']:+.1f}; Germanic − Romance "
                f"= {a['family_germanic_minus_romance_bits']:+.3f} bits/char"
            ),
            "consequence": (
                "descriptive only — with three languages no rank correlation is testable; the "
                "document-level test above is the inferential statement"
            ),
        }
    )

    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "3.5",
        "adopted_table": adopt,
        "tables": {
            v: {k: val for k, val in t.items() if k != "per_document_offsets"}
            for v, t in tables.items()
        },
        "per_document_offsets": {
            v: t["per_document_offsets"] for v, t in tables.items()
        },
        "covariates": {
            "train_chars": train_chars,
            "family": FAMILY,
            "ngram_structure_gain_bits": structure_gain,
            "ngram5_heldout_bits": {
                l: ngram[l]["heldout_bits_per_char"] for l in LANGS
            },
        },
        "reference_dependence": ref_dep,
        "language_dependence": doc_tests,
        "correlates": corr,
        "findings": findings,
        "unescalated_above_noise": [
            f["id"] for f in findings if f["above_noise"] and not f["escalated"]
        ],
    }
    out_dir = root / "analysis" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fairness_audit.json").write_text(json.dumps(report, indent=1))
    md = render(report)
    (REPO / "docs" / "phase3_fairness_audit.md").write_text(md)
    print(md)
    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        task = init_task(
            RunConfig(run_name="fairness-audit", phase="phase3"), root, tags=["task3.5"]
        )
        task.connect_configuration(report, name="fairness_audit")
        logger = task.get_logger()
        for v, t in tables.items():
            for l in LANGS:
                logger.report_scalar(
                    "calibration_offset_bits", f"{v}/{l}", t["offsets_bits"][l], 0
                )
        logger.flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()


def render(rep: dict) -> str:
    intro = (
        f"Generated {rep['created_utc'][:19]}Z by `scripts/fairness_audit.py`; adopted table "
        f"**{rep['adopted_table']}** (`CALIBRATION_VERSION`). Offsets are `NELBO − NLL_AR` in "
        "bits/char on the full tiled held-out split (positive = diffusion bound looser than the "
        "reference's likelihood)."
    )
    L = [
        "# Phase 3 — bound-fairness audit (task 3.5)",
        "",
        intro,
        "",
        "## Offsets by calibration table (backbone × reference tier)",
        "",
        "| table | backbone | reference | latin | italian | german | spread |",
        "|---|---|---|---|---|---|---|",
    ]
    for v, t in rep["tables"].items():
        o, s = t["offsets_bits"], t["offsets_sem"]
        L.append(
            f"| {v} | {t['backbone_run']} | {Path(t['ar_dir'] or '?').name} | "
            + " | ".join(f"{o[l]:+.3f} ± {s[l]:.3f}" for l in LANGS)
            + f" | {t['summary']['spread_bits']:.3f} |"
        )
    L += [
        "",
        "## Covariates",
        "",
        "| language | family | train chars | AR NLL (adopted) | 5-gram held-out | 1→5-gram gain |",
        "|---|---|---|---|---|---|",
    ]
    cov = rep["covariates"]
    ad = rep["tables"][rep["adopted_table"]]
    for l in LANGS:
        L.append(
            f"| {l} | {cov['family'][l]} | {cov['train_chars'][l]:,} | {ad['nll_ar_bits'][l]:.3f} | "
            f"{cov['ngram5_heldout_bits'][l]:.3f} | {cov['ngram_structure_gain_bits'][l]:.3f} |"
        )
    if rep["language_dependence"]:
        L += [
            "",
            "## Language dependence beyond document dispersion (per-document offsets)",
            "",
            "| table | n docs | doc-mean offset latin / italian / german | between-lang range | within-lang doc sd | ANOVA p | Kruskal p |",
            "|---|---|---|---|---|---|---|",
        ]
        for v, t in rep["language_dependence"].items():
            if not t.get("available"):
                continue
            L.append(
                f"| {v} | {t['n_per_language']} | "
                + " / ".join(f"{t['doc_offset_mean'][l]:+.3f}" for l in LANGS)
                + f" | {t['between_language_range_bits']:.3f} | {t['within_language_doc_sd_bits']:.3f} | "
                f"{t['anova_p']:.3g} | {t['kruskal_p']:.3g} |"
            )
    L += [
        "",
        "## Correlates (descriptive, n = 3 languages — not testable)",
        "",
        "| table | ρ(offset, corpus size) | ρ(offset, AR entropy) | ρ(offset, n-gram gain) | Germanic − Romance |",
        "|---|---|---|---|---|",
    ]
    for v, c in rep["correlates"].items():
        L.append(
            f"| {v} | {c['train_chars']:+.1f} | {c['intrinsic_entropy_nll_ar']:+.1f} | "
            f"{c['ngram_structure_gain']:+.1f} | {c['family_germanic_minus_romance_bits']:+.3f} |"
        )
    L += ["", "## Findings", ""]
    for f in rep["findings"]:
        tag = (
            "**ESCALATED**"
            if f["escalated"]
            else ("above noise, NOT escalated" if f["above_noise"] else "within noise")
        )
        L.append(f"- `{f['id']}` — {tag}. {f['statement']}. → {f['consequence']}.")
    if rep["unescalated_above_noise"]:
        L += [
            "",
            f"**Un-escalated above-noise findings: {rep['unescalated_above_noise']}** (G3 blocker)",
        ]
    else:
        L += ["", "No above-noise finding is left un-escalated."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
