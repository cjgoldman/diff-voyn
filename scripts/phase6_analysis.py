"""Phase 6, task 6.5 — length-sensitivity + family-confusion analysis.

Assembles, from the frozen-evaluator artifacts of Phases 3–5 and the Phase-6
control battery, what the data support at each granularity:

- accuracy-vs-length curves (language / family) for the 1:1 rung: the
  Phase-4 recovery suite (n = 50 per cell, phase_c-85m evaluator, v3-phase_c-ro),
  the Phase-5 rung-1 two-tier suite (n = 20 per cell) and the LID head's
  clean-decipherment curve (task 4.3 eval), plus the replicate flip-rate,
  the calibration-unresolved rate and the structure margins by length;
- confusion matrices (true language × called language) per cipher rung:
  within-Romance (latin ↔ italian) from the in-inventory suites; the
  within-Germanic cell is only realizable through the out-of-inventory
  contamination set (Dutch / English → ?), which the 6.3 battery supplies
  (absent until its report exists);
- a restatement of the claims at the granularity the data support.

Writes DATA_ROOT/analysis/phase6/length_family.{json,md,png}.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.ladder import write_json_atomic
from diff_voyn.metrology import FAMILY

LANGS = ("latin", "italian", "german")
LENGTHS = (50, 100, 200, 400, 700)


def recovery_curves(path: Path, primary: str) -> dict:
    r = json.loads(path.read_text())
    out = {
        "source": str(path),
        "backbone": r["backbone"].get("run_name"),
        "n_per_cell": None,
        "by_length": {},
        "confusion": {},
    }
    for L in LENGTHS:
        bl = r["by_length"][str(L)][f"calibration_{primary}"]
        cells = {lang: r["cells"][f"{lang}/L{L}"] for lang in LANGS}
        out["n_per_cell"] = cells["latin"]["n"]
        out["by_length"][L] = {
            "language": bl["language"],
            "language_ci95": bl["language_ci95"],
            "family": bl["family"],
            "family_ci95": bl["family_ci95"],
            "flip_rate": float(np.mean([c["flip_rate"] for c in cells.values()])),
            "margin_unresolved_rate": float(
                np.mean([c["margin_unresolved_rate"] for c in cells.values()])
            ),
            "true_minus_shuffled": float(
                np.mean([c["true_minus_shuffled_bits_mean"] for c in cells.values()])
            ),
            "wrong_minus_shuffled": float(
                np.mean([c["wrong_minus_shuffled_bits_mean"] for c in cells.values()])
            ),
            "ser_mean": float(np.mean([c["ser_mean"] for c in cells.values()])),
            "per_language": {
                lang: cells[lang]["accuracy"][f"calibration_{primary}"]["language"]
                for lang in LANGS
            },
        }
        out["confusion"][L] = {
            lang: cells[lang]["confusion"][f"calibration_{primary}"] for lang in LANGS
        }
    return out


def rung1_curves(path: Path) -> dict:
    r = json.loads(path.read_text())
    out = {"source": str(path), "by_length": {}}
    for L in LENGTHS:
        cells = {lang: r["cells"][f"{lang}/L{L}"] for lang in LANGS}
        out["by_length"][L] = {
            "language": float(np.mean([c["lang_acc_final"] for c in cells.values()])),
            "family": float(np.mean([c["family_acc_final"] for c in cells.values()])),
            "solved_final": float(np.mean([c["solved_final"] for c in cells.values()])),
            "ser_final": float(np.mean([c["ser_final"] for c in cells.values()])),
            "per_language_solved": {
                lang: cells[lang]["solved_final"] for lang in LANGS
            },
            "per_language_acc": {lang: cells[lang]["lang_acc_final"] for lang in LANGS},
            "n_per_cell": cells["latin"]["n"],
        }
    return out


def lid_curves(path: Path) -> dict:
    e = json.loads(path.read_text())
    out = {
        "source": str(path),
        "by_length": {},
        "controls_abstain": e["summary"].get("abstain_rates_controls"),
    }
    for key, v in e["curves"].items():
        parts = key.split("/")
        if parts[0] != "substitution" or parts[3] != "0.0":
            continue
        L = int(parts[1][1:])
        d = out["by_length"].setdefault(L, {"acc": [], "abstain": [], "pred": {}})
        d["acc"].append(v["acc"])
        d["abstain"].append(v["abstain_rate"])
        d["pred"][parts[2]] = v["pred_hist"]
    for L, d in out["by_length"].items():
        d["acc"] = float(np.mean(d["acc"]))
        d["abstain"] = float(np.mean(d["abstain"]))
    return out


def rung_confusions(out_dir5: Path) -> dict:
    res = {}
    for rung, key in (
        ("rung2", "rank_final"),
        ("rung3", "rank_final"),
        ("rung4", "rank_final"),
    ):
        p = out_dir5 / f"{rung}_report.json"
        if not p.exists():
            continue
        inst = json.loads(p.read_text())["instances"]
        conf = {lang: {l: 0 for l in LANGS} for lang in LANGS}
        for x in inst:
            conf[x["language"]][x[key]] += 1
        n = len(inst)
        lang_acc = sum(conf[l][l] for l in LANGS) / n
        fam_acc = (
            sum(v for a in LANGS for b, v in conf[a].items() if FAMILY[a] == FAMILY[b])
            / n
        )
        romance_confusions = conf["latin"]["italian"] + conf["italian"]["latin"]
        cross = sum(
            v for a in LANGS for b, v in conf[a].items() if FAMILY[a] != FAMILY[b]
        )
        res[rung] = {
            "n": n,
            "confusion": conf,
            "language_acc": lang_acc,
            "family_acc": fam_acc,
            "within_romance_confusions": romance_confusions,
            "cross_family_confusions": cross,
        }
    return res


def contamination_confusion(path: Path) -> dict | None:
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    s = r["summary"].get("contamination")
    return (
        {
            "confusion_called": s["confusion_called"],
            "confusion_mdl_top": s["confusion_mdl_top"],
            "family_correct_rate_mdl_top": s["family_correct_rate_mdl_top"],
            "family_correct_rate_when_called": s["family_correct_rate_when_called"],
            "abstain_rate": s["abstain_rate"],
            "n": s["n"],
        }
        if s
        else None
    )


def plot(rec: dict, png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    c4 = rec["recovery_phase4"]["by_length"]
    Ls = sorted(int(k) for k in c4)
    lang = [c4[L]["language"] for L in Ls]
    lo = [c4[L]["language_ci95"][0] for L in Ls]
    hi = [c4[L]["language_ci95"][1] for L in Ls]
    ax.fill_between(Ls, lo, hi, alpha=0.15, color="C0")
    ax.plot(
        Ls,
        lang,
        "o-",
        color="C0",
        label=f"language (1:1, n={rec['recovery_phase4']['n_per_cell']}/cell)",
    )
    ax.plot(
        Ls, [c4[L]["family"] for L in Ls], "s--", color="C0", alpha=0.7, label="family"
    )
    r1 = rec["rung1_phase5"]["by_length"]
    ax.plot(
        Ls,
        [r1[L]["language"] for L in Ls],
        "^-",
        color="C1",
        label="language (Phase-5 two-tier, n=20)",
    )
    lid = rec["lid_head"]["by_length"]
    Ll = sorted(int(k) for k in lid)
    ax.plot(
        Ll,
        [lid[L]["acc"] for L in Ll],
        "d:",
        color="C2",
        label="LID head (clean decipherments)",
    )
    ax.axhline(0.971, color="k", lw=0.5, ls=":", label="Hauer & Kondrak 97.1%")
    ax.axvline(50, color="grey", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(L) for L in Ls])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("plaintext length (chars)")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Accuracy vs length (frozen evaluator, v3-phase_c-ro)")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    ax.plot(
        Ls,
        [c4[L]["flip_rate"] for L in Ls],
        "o-",
        label="replicate flip-rate (B=64 × 4)",
    )
    ax.plot(
        Ls,
        [c4[L]["margin_unresolved_rate"] for L in Ls],
        "s-",
        label="margin < calibration uncertainty",
    )
    ax.plot(
        Ls,
        [1 - r1[L]["solved_final"] for L in Ls],
        "^-",
        label="unsolved (SER ≥ 5%), two-tier",
    )
    ax.set_xscale("log")
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(L) for L in Ls])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("plaintext length (chars)")
    ax.set_ylabel("rate")
    ax.set_title("Uncertainty vs length")
    ax.legend(fontsize=7)
    ax = axes[2]
    ax.plot(
        Ls,
        [-c4[L]["true_minus_shuffled"] for L in Ls],
        "o-",
        label="true decipherment: shuffled − decode",
    )
    ax.plot(
        Ls,
        [-c4[L]["wrong_minus_shuffled"] for L in Ls],
        "s-",
        label="wrong hypothesis: shuffled − decode",
    )
    ax.axhline(
        1.5, color="r", lw=0.8, ls="--", label="abstention threshold (1.5 bits/char)"
    )
    ax.set_xscale("log")
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(L) for L in Ls])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("plaintext length (chars)")
    ax.set_ylabel("structure margin (bits/char)")
    ax.set_title("Shuffled-text margin vs length")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(png, dpi=130)


def conf_md(title: str, conf: dict, cols) -> list[str]:
    md = [
        f"**{title}**",
        "",
        "| true \\ called | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for a, row in conf.items():
        md.append(f"| {a} | " + " | ".join(str(row.get(c, 0)) for c in cols) + " |")
    md.append("")
    return md


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    p.add_argument("--out-dir", type=Path, default=root / "analysis" / "phase6")
    p.add_argument("--primary", default="v3-phase_c-ro")
    args = p.parse_args()
    rec = {
        "task": "6.5",
        "created_utc": datetime.now(UTC).isoformat(),
        "recovery_phase4": recovery_curves(
            root / "analysis/phase4/recovery_report.json", args.primary
        ),
        "recovery_phase3": recovery_curves(
            root / "analysis/phase3/recovery_report.json", "v3-ro"
        ),
        "rung1_phase5": rung1_curves(root / "analysis/phase5/rung1_report.json"),
        "lid_head": lid_curves(root / "analysis/phase4/lid_eval_phase_c_85m.json"),
        "rung_confusions": rung_confusions(root / "analysis/phase5"),
        "contamination": contamination_confusion(
            args.out_dir / "controls" / "report.json"
        ),
    }
    c4 = rec["recovery_phase4"]["by_length"]
    # claims at the granularity the data support
    ge200_lang = float(np.mean([c4[L]["language"] for L in (200, 400, 700)]))
    ge200_fam = float(np.mean([c4[L]["family"] for L in (200, 400, 700)]))
    romance_total = sum(
        v["within_romance_confusions"] for v in rec["rung_confusions"].values()
    )
    cross_total = sum(
        v["cross_family_confusions"] for v in rec["rung_confusions"].values()
    )
    conf_1to1 = rec["recovery_phase4"]["confusion"]
    romance_1to1 = {
        L: conf_1to1[L]["latin"]["italian"] + conf_1to1[L]["italian"]["latin"]
        for L in LENGTHS
    }
    cross_1to1 = {
        L: sum(
            conf_1to1[L][a][b] for a in LANGS for b in LANGS if FAMILY[a] != FAMILY[b]
        )
        for L in LENGTHS
    }
    rec["claims"] = {
        "length_floor_chars": 200,
        "language_acc_ge200_1to1": ge200_lang,
        "family_acc_ge200_1to1": ge200_fam,
        "language_acc_L50_1to1": c4[50]["language"],
        "language_acc_L100_1to1": c4[100]["language"],
        "within_romance_confusion_rate_1to1_by_length": romance_1to1,
        "cross_family_confusion_rate_1to1_by_length": cross_1to1,
        "rungs2to4_within_romance_confusions": romance_total,
        "rungs2to4_cross_family_confusions": cross_total,
        "dominant_error_mode": (
            "cross-family (document heterogeneity: high-entropy Latin documents tie with the German condition), not the Romance pair"
            if sum(cross_1to1[L] for L in (200, 400, 700))
            >= sum(romance_1to1[L] for L in (200, 400, 700))
            else "within-Romance (latin ↔ italian)"
        ),
        "within_germanic": "not realizable inside the inventory (German is the only Germanic language); measured only via the out-of-inventory contamination set (Dutch / English)",
    }
    write_json_atomic(args.out_dir / "length_family.json", rec)
    plot(rec, args.out_dir / "length_family.png")
    md = [
        "### Length sensitivity and family confusion (task 6.5)",
        "",
        f"Frozen evaluator `phase_c-85m-seed0`, calibration `{args.primary}` (report-only). 1:1 suite: n = {rec['recovery_phase4']['n_per_cell']} per (language × length) cell.",
        "",
        "| L | language acc (95% CI) | family acc | per-language (la / it / de) | flip-rate | margin unresolved | SER | structure margin true / wrong | two-tier (Phase 5) language / solved | LID head acc |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    r1 = rec["rung1_phase5"]["by_length"]
    lid = rec["lid_head"]["by_length"]
    for L in LENGTHS:
        c = c4[L]
        pl = c["per_language"]
        md.append(
            f"| {L} | {c['language']:.3f} ({c['language_ci95'][0]:.2f}–{c['language_ci95'][1]:.2f}) | {c['family']:.3f} | {pl['latin']:.2f} / {pl['italian']:.2f} / {pl['german']:.2f} | {c['flip_rate']:.3f} | {c['margin_unresolved_rate']:.2f} | {c['ser_mean']:.3f} | {-c['true_minus_shuffled']:.2f} / {-c['wrong_minus_shuffled']:.2f} | {r1[L]['language']:.3f} / {r1[L]['solved_final']:.2f} | {lid.get(L, {}).get('acc', float('nan')):.3f} |"
        )
    md += ["", "Confusion (1:1 suite, rows = true language, fractions):", ""]
    for L in (50, 100, 200, 700):
        md += conf_md(
            f"L = {L}",
            {a: {b: round(v, 2) for b, v in conf_1to1[L][a].items()} for a in LANGS},
            LANGS,
        )
    for rung, v in rec["rung_confusions"].items():
        md += conf_md(
            f"{rung} (n = {v['n']}; language acc {v['language_acc']:.2f}, family {v['family_acc']:.2f})",
            v["confusion"],
            LANGS,
        )
    if rec["contamination"]:
        ct = rec["contamination"]
        md += conf_md(
            "out-of-inventory contamination — MDL-top language (rows = true out-of-inventory language)",
            ct["confusion_mdl_top"],
            LANGS,
        )
        md += conf_md(
            "out-of-inventory contamination — called language under the abstention rule",
            ct["confusion_called"],
            list(LANGS) + ["abstain"],
        )
        md.append(
            f"family-correct rate (MDL top) {ct['family_correct_rate_mdl_top']:.2f}; when called {ct['family_correct_rate_when_called']}; abstain rate {ct['abstain_rate']:.2f}"
        )
    md += ["", "**Claims at the granularity the data support**", ""]
    cl = rec["claims"]
    md += [
        f"- Language-level ranking is supported at ≥ {cl['length_floor_chars']} plaintext characters on the 1:1 rung ({cl['language_acc_ge200_1to1']:.3f} language / {cl['family_acc_ge200_1to1']:.3f} family); at 50 chars it is {cl['language_acc_L50_1to1']:.2f} and at 100 chars {cl['language_acc_L100_1to1']:.2f} — the design's ~50-char LID anchor holds for the LID head on clean text, not for ELBO ranking of decipherments.",
        f"- Dominant error mode at ≥ 200 chars: {cl['dominant_error_mode']} (1:1 suite cross-family confusions by length {cross_1to1}, within-Romance {romance_1to1}; rungs 2–4: {cl['rungs2to4_cross_family_confusions']} cross-family vs {cl['rungs2to4_within_romance_confusions']} within-Romance).",
        f"- Within-Germanic resolution: {cl['within_germanic']}.",
        "- Rung 4 (arithmetic) language calls sit at margins of the order of the calibration uncertainty (Phase 5: 7/9), so a rung-4 row of any VMS table carries family-level resolution at best.",
    ]
    (args.out_dir / "length_family.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
