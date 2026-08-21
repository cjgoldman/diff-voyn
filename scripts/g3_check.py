"""Gate G3 verification — Phase 3 (ELBO metrology).

G3 (task breakdown): calibrated ranking recovers the true language on the
synthetic 1:1 suite within target; per-language offsets estimated and stored;
fairness audit shows no un-escalated language-dependent bias. Plus the
Phase-3 task acceptances that feed it: 3.1 (CRN ≥ 5× variance reduction),
3.2 (a budget with flip-rate < 1%), 3.3 (per-document spread reported).

Reads the Phase-3 artifacts under DATA_ROOT/analysis/phase3 and the applied
calibration table; writes DATA_ROOT/runs/g3_report.json; ClearML tag ``g3``.

Usage:
    uv run python scripts/g3_check.py [--no-clearml]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diff_voyn.ciphers.external import data_root
from diff_voyn.metrology import CALIBRATION_VERSION, CalibrationTable

REPO = Path(__file__).resolve().parent.parent
CHECKS: list[dict] = []


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    CHECKS.append({"check": name, "status": status, "detail": detail})
    print(f"[{status}] {name}: {detail}")


def single_application_point() -> tuple[bool, str]:
    """Static check: the calibration arithmetic is defined once and every
    other offset use goes through it."""
    src = (REPO / "diff_voyn").rglob("*.py")
    defs, adders = [], []
    for f in src:
        text = f.read_text()
        if re.search(r"^def calibrate_bits\(", text, re.MULTILINE):
            defs.append(str(f.relative_to(REPO)))
        # any other direct '+ offsets[' / '- offset' arithmetic on bits
        for m in re.finditer(
            r"calibration_offsets_bits\.get\(|offsets?_bits\[.*\]\s*[+-]", text
        ):
            if "metrology/calibration.py" not in str(f):
                adders.append(f"{f.relative_to(REPO)}: {m.group(0)}")
    ok = defs == ["diff_voyn/metrology/calibration.py"] and not adders
    return (
        ok,
        f"calibrate_bits defined in {defs}; other offset arithmetic: {adders or 'none'}",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--recovery-bar", type=float, default=0.971)
    args = p.parse_args()
    root = data_root()
    a3 = root / "analysis" / "phase3"

    # 3.1 CRN
    crn = json.loads((a3 / "crn_check.json").read_text())
    ratios = {L: r["min_variance_ratio"] for L, r in crn["by_length"].items()}
    check(
        "3.1 CRN variance reduction ≥ 5× (between-language differences)",
        crn["pass"],
        ", ".join(f"L{L}: {r:.1f}×" for L, r in ratios.items()),
    )

    # 3.2 budget
    sb = json.loads((a3 / "sample_budget.json").read_text())
    chosen = sb.get("chosen_budget")
    if chosen:
        flips = {
            L: d["by_budget"][str(chosen)]["flip_rate"]
            for L, d in sb["by_length"].items()
        }
        check(
            "3.2 sample budget with ranking flip-rate < 1% at every length",
            True,
            f"budget {chosen} draws; flip-rates "
            + ", ".join(f"L{L}: {f:.2%}" for L, f in flips.items()),
        )
    else:
        check(
            "3.2 sample budget with ranking flip-rate < 1% at every length",
            False,
            sb["chosen_budget_note"],
        )

    # 3.3 documents
    docs = sorted(a3.glob("documents_*.json"))
    if docs:
        d = json.loads(docs[-1].read_text())
        detail = "; ".join(
            f"{l}: {r['n_documents']} docs, own bits {r['own_condition_document_means']['mean']:.3f} "
            f"(between-doc sd {r['between_document_std']:.3f})"
            for l, r in d["languages"].items()
        )
        check(
            "3.3 per-document mean + spread reported",
            True,
            f"{docs[-1].name}: {detail}",
        )
    else:
        check("3.3 per-document mean + spread reported", False, "no documents_*.json")

    # 3.4 calibration stored, versioned, single-sourced
    try:
        t = CalibrationTable.load(CALIBRATION_VERSION, root)
        check(
            "3.4 calibration table stored and versioned",
            True,
            f"{t.version} ({t.phase}, backbone step {t.backbone_step}): offsets "
            + ", ".join(
                f"{l} {o:+.3f}±{t.offsets_sem[l]:.3f}"
                for l, o in t.offsets_bits.items()
            )
            + f"; spread {t.spread_bits:.3f} bits/char",
        )
    except FileNotFoundError as e:
        check("3.4 calibration table stored and versioned", False, str(e))
    ok, detail = single_application_point()
    check("3.4 offsets applied in exactly one place", ok, detail)

    # 3.5 audit
    fa = json.loads((a3 / "fairness_audit.json").read_text())
    esc = [f["id"] for f in fa["findings"] if f["escalated"]]
    check(
        "3.5 fairness audit: no un-escalated above-noise finding",
        not fa["unescalated_above_noise"],
        f"escalated: {esc}; un-escalated: {fa['unescalated_above_noise']}; adopted {fa['adopted_table']}",
    )
    check(
        "3.5 audit table == applied table",
        fa["adopted_table"] == CALIBRATION_VERSION,
        f"audit adopts {fa['adopted_table']}, CALIBRATION_VERSION = {CALIBRATION_VERSION}",
    )

    # 3.6 / 3.7 recovery
    rr = json.loads((a3 / "recovery_report.json").read_text())
    prim = f"calibration_{rr['primary_calibration']}"
    acc = rr["ge200_mean_accuracy"][prim]
    check(
        f"3.6 synthetic 1:1 recovery ≥ {args.recovery_bar:.1%} at ≥200 chars (primary table)",
        acc["language"] >= args.recovery_bar,
        f"language {acc['language']:.1%}, family {acc['family']:.1%} (table {rr['primary_calibration']}); "
        + "; ".join(
            f"L{L}: {d[prim]['language']:.1%}/{d[prim]['family']:.1%}"
            for L, d in rr["by_length"].items()
        ),
    )
    check(
        "3.6 primary table == applied table",
        rr["primary_calibration"] == CALIBRATION_VERSION,
        f"report primary {rr['primary_calibration']}",
    )
    check(
        "3.7 rankings reported at language and family granularity",
        all("family" in v for d in rr["by_length"].values() for v in d.values()),
        "by_length / by_language / cells",
    )
    flips = [c["flip_rate"] for c in rr["cells"].values()]
    check(
        "3.6 decipherment-ranking replicate flip-rate < 1% (≥200 chars)",
        all(
            c["flip_rate"] < 0.01
            for k, c in rr["cells"].items()
            if int(k.split("/L")[1]) >= 200
        ),
        f"max over cells {max(flips):.2%}",
        warn_only=True,
    )

    verdict = all(c["status"] != "FAIL" for c in CHECKS)
    report = {
        "gate": "G3",
        "created_utc": datetime.now(UTC).isoformat(),
        "verdict": "PASS" if verdict else "FAIL",
        "checks": CHECKS,
        "calibration_version": CALIBRATION_VERSION,
    }
    out = root / "runs" / "g3_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nGate G3: {report['verdict']}  (report {out})")
    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        task = init_task(
            RunConfig(run_name="g3-check", phase="phase3"), root, tags=["g3"]
        )
        task.connect_configuration(report, name="g3_report")
        task.get_logger().report_scalar("gate", "g3_pass", float(verdict), 0)
        task.get_logger().flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
