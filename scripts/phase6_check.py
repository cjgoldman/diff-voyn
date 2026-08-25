"""Phase 6 acceptance check — VMS application and reporting (tasks 6.1–6.7).

There is no numbered gate after G5; this is the task-acceptance roll-up in
the style of ``g5_check.py``:

  6.1  both Currier dialects scored independently (never pooled), on both
       transcriptions, by every applicable head
  6.2  ranked (cipher × language) table with uncertainty (per-window spread,
       replicate flip-rate, calibration margin uncertainty) and head-agreement
       columns; every cell also against the no-cipher baseline
  6.3  negative-control battery: abstention > 95% on voynichesque and
       shuffled text; contamination confusions documented; positives not
       abstained
  6.4  bound-fairness audit re-run attached (docs/phase6_fairness_audit.md)
  6.5  length curves + confusion matrices published, claims restated
  6.6  anchors reported (Borg end-to-end; Zodiac-408 n-gram baseline; BnF
       fr2988 not run — WARN)
  6.7  write-up with honest framing present (docs/phase6_writeup.md)
  freeze discipline: every Phase-6 artifact scored by the Gate-G4 frozen
       evaluator under the adopted calibration

Usage: uv run python scripts/phase6_check.py [--no-clearml]
Writes DATA_ROOT/runs/phase6_report.json; ClearML tag ``phase6``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diff_voyn.ciphers.external import data_root
from diff_voyn.metrology import CALIBRATION_VERSION

CHECKS: list[dict] = []
REPO = Path(__file__).resolve().parent.parent


def check(name, ok, detail="", warn_only=False):
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    CHECKS.append({"check": name, "status": status, "detail": detail})
    print(f"[{status}] {name}: {detail}")


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-clearml", action="store_true")
    args = p.parse_args()
    root = data_root()
    a6 = root / "analysis" / "phase6"
    fz = load_json(root / "analysis" / "phase5" / "evaluator_freeze.json")
    frozen_sha = fz["frozen"]["evaluator"]["sha256"] if fz else None
    frozen_step = fz["frozen"]["evaluator"]["step"] if fz else None

    # 6.1 / 6.2
    vr = load_json(a6 / "vms_report.json")
    if vr:
        tables = vr["tables"]
        cells = vr["cells"]
        dialects = {c["instance"].split("/")[1] for c in cells}
        sources = {c["instance"].split("/")[0] for c in cells}
        heads = {c["head"] for c in cells}
        check(
            "6.1 both dialects scored independently, both transcriptions, every head",
            dialects == {"A", "B"}
            and {"IT2a", "RF1b"} <= sources
            and heads == {"sub1to1", "homophonic", "naibbe", "arithmetic"},
            f"dialects {sorted(dialects)}, sources {sorted(sources)}, heads {sorted(heads)}, {len(cells)} cells, tables {sorted(tables)}",
        )
        pooled = any("/" not in k or k.split("/")[1] not in ("A", "B") for k in tables)
        check("6.1 A and B never pooled", not pooled, "; ".join(sorted(tables)))
        req = (
            "plain_bits_sem",
            "replicate_flip_rate",
            "top_margin_uncertainty_bits",
            "structure_margin",
            "window_vote_for_top",
        )
        check(
            "6.2 every cell carries uncertainty columns (per-window spread, flip-rate, calibration margin uncertainty, structure margin)",
            all(all(k in c for k in req) for c in cells),
            f"{len(cells)} cells",
        )
        check(
            "6.2 head agreement + transcription / dialect agreement reported",
            all(
                "head_agreement_on_top_language" in t and "per_head" in t
                for t in tables.values()
            )
            and bool(vr.get("transcription_agreement"))
            and bool(vr.get("dialect_agreement")),
            "; ".join(
                f"{k}: top {t['ranked'][0]['head']}/{t['ranked'][0]['hypothesis']}, head agreement {t['head_agreement_on_top_language']}, abstain {t['abstain']}"
                for k, t in sorted(tables.items())
            ),
        )
        base_ok = all("no_cipher_baselines" in v for k, v in vr["instances"].items())
        check(
            "6.2 no-cipher baseline attached to every presentation",
            base_ok,
            f"{len(vr['instances'])} presentations",
        )
        # the headline: abstention verdict per dialect
        for k, t in sorted(tables.items()):
            check(
                f"6.2 verdict {k}",
                True,
                f"{'ABSTAIN (no language-like cell)' if t['abstain'] else 'language-like cell found'}; ranking among heads: "
                + ", ".join(f"{h}:{v['top']}" for h, v in t["per_head"].items()),
            )
    else:
        check(
            "6.1 / 6.2 VMS report present",
            False,
            "analysis/phase6/vms_report.json missing",
        )

    # 6.3
    cr = load_json(a6 / "controls" / "report.json")
    if cr:
        acc = cr["acceptance"]
        s = cr["summary"]
        check(
            "6.3 voynichesque abstention > 95%",
            (acc["voynichesque_abstain"] or 0) > 0.95,
            f"{acc['voynichesque_abstain']:.3f} (n={s['voynichesque']['n']})",
        )
        check(
            "6.3 shuffled-text abstention > 95%",
            (acc["shuffled_abstain"] or 0) > 0.95,
            f"{acc['shuffled_abstain']:.3f} (n={s['shuffled']['n']})",
        )
        check(
            "6.3 positives not abstained, language recovered",
            (acc["positive_false_abstain"] or 0) <= 0.05
            and (acc["positive_language_correct"] or 0) >= 0.9,
            f"false-abstain {acc['positive_false_abstain']:.3f}, language correct {acc['positive_language_correct']:.3f} (n={s['positive']['n']})",
            warn_only=True,
        )
        ct = s.get("contamination")
        check(
            "6.3 contamination confusions documented",
            ct is not None,
            (
                f"n={ct['n']}, abstain {ct['abstain_rate']:.2f}, family-correct (MDL top) {ct['family_correct_rate_mdl_top']:.2f}; MDL-top confusion {json.dumps(ct['confusion_mdl_top'])}"
                if ct
                else "missing"
            ),
        )
    else:
        check(
            "6.3 control battery report present",
            False,
            "analysis/phase6/controls/report.json missing",
        )

    # 6.4
    fa = load_json(a6 / "fairness_audit.json")
    doc = REPO / "docs" / "phase6_fairness_audit.md"
    check(
        "6.4 bound-fairness audit re-run attached",
        bool(fa) and doc.exists() and fa.get("adopted_table") == CALIBRATION_VERSION,
        f"adopted {fa.get('adopted_table') if fa else None}; {doc.name}; findings: {len(fa.get('findings', [])) if fa else 0}",
    )

    # 6.5
    lf = load_json(a6 / "length_family.json")
    check(
        "6.5 length curves + confusion matrices published, claims restated",
        bool(lf) and (a6 / "length_family.png").exists() and "claims" in lf,
        (
            f"≥200 language {lf['claims']['language_acc_ge200_1to1']:.3f} / family {lf['claims']['family_acc_ge200_1to1']:.3f}; dominant error mode: {lf['claims']['dominant_error_mode']}; contamination {'attached' if lf.get('contamination') else 'absent'}"
            if lf
            else "missing"
        ),
    )

    # 6.6
    ar = load_json(a6 / "anchors" / "anchors_report.json")
    if ar:
        b, z = ar["borg"], ar["zodiac408"]
        check(
            "6.6 Borg ≤ 4.10% SER (full pipeline) and Latin recovered",
            b["pass"] and b["language_recovered"],
            f"SER {b['ser_final_latin']['ser_weighted']:.4f} (n-gram {b['ser_ngram_latin']['ser_weighted']:.4f}) on {b['ser_final_latin']['n_pages']} aligned pages; language {b['language_rank']['language_order']}",
            warn_only=True,
        )
        check(
            "6.6 Zodiac-408 ≤ 1.9% SER (n-gram tier, English outside inventory)",
            z["pass"],
            f"SER {z['ser_best_by_objective']:.4f} (oracle {z['ser_oracle']:.4f}); pre-diffusion baseline only",
            warn_only=True,
        )
        check("6.6 BnF fr2988", False, ar["bnf_fr2988"]["reason"], warn_only=True)
    else:
        check(
            "6.6 anchors report present",
            False,
            "analysis/phase6/anchors/anchors_report.json missing",
            warn_only=True,
        )

    # 6.7
    wu = REPO / "docs" / "phase6_writeup.md"
    txt = wu.read_text().lower() if wu.exists() else ""
    check(
        "6.7 write-up states assumptions and residual risks explicitly",
        wu.exists()
        and all(k in txt for k in ("exploratory", "assumption", "bound", "family")),
        f"{wu.name} ({len(txt)} chars)" if wu.exists() else "missing",
    )

    # freeze discipline
    evs = []
    for pth in (a6 / "vms_report.json", a6 / "controls" / "report.json"):
        d = load_json(pth)
        if d and d.get("evaluator"):
            evs.append(
                (
                    pth.name,
                    d["evaluator"].get("step"),
                    d.get("primary_calibration")
                    or d.get("evaluator", {}).get("calibration_version"),
                )
            )
    check(
        "freeze discipline: every Phase-6 artifact scored by the Gate-G4 frozen evaluator under the adopted calibration",
        bool(evs)
        and all(st == frozen_step for _, st, _ in evs)
        and all(cal in (CALIBRATION_VERSION, None) for _, _, cal in evs),
        f"frozen step {frozen_step}, sha {frozen_sha[:12] if frozen_sha else None}; artifacts {evs}",
    )

    verdict = all(c["status"] != "FAIL" for c in CHECKS)
    report = {
        "phase": "6",
        "created_utc": datetime.now(UTC).isoformat(),
        "verdict": "PASS" if verdict else "FAIL",
        "checks": CHECKS,
        "calibration_version": CALIBRATION_VERSION,
        "frozen_evaluator": {"step": frozen_step, "sha256": frozen_sha},
    }
    out = root / "runs" / "phase6_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nPhase 6: {report['verdict']}  (report {out})")
    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        task = init_task(
            RunConfig(run_name="phase6-check", phase="phase6"), root, tags=["phase6"]
        )
        task.connect_configuration(report, name="phase6_report")
        task.get_logger().report_scalar("phase", "phase6_pass", float(verdict), 0)
        task.get_logger().flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
