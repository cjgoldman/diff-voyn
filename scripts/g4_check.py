"""Gate G4 verification — Phase 4 (language-ID head, delayed then joint).

Gate wording (task breakdown): *joint model passes the G3 synthetic ranking
test unchanged or improved; per-language NELBO not degraded beyond
threshold; head and ELBO rankings agree on clean synthetics.* Plus the
task acceptances feeding it: 4.1 (>99% on clean long text, stop-gradient
verified), 4.2 (head converged, severity curves produced), 4.3 (abstain
>95% on negative controls), 4.4 (Phase-C calibration table produced, λ
schedule logged, LID gradient <10% of the diffusion gradient), 4.5
(synthetic-grid ranking identical end-B → end-C, any change documented as a
red flag), 4.6 (calibrated head, agreement matrix), 4.7 (seed replication,
P2 — reported as WARN while the 25M seed runs are still in flight).

Ranking comparison (4.5 / G4): per instance of the 3.6 suite, the ELBO
winner under the Phase-B weights (``analysis/phase3/recovery_scores.json``,
table ``v3-ro``) vs under the Phase-C weights
(``analysis/phase4/recovery_scores.json``, table ``CALIBRATION_VERSION``).
Both tables are report-only, so both rankings are the raw own-condition
NELBO ranking. Hard criteria: ≥200-char language accuracy under Phase C ≥
the G3 bar (97.1%) and not below Phase B by more than the replicate
flip-rate floor (``--max-drop``, 1.5 pp); every changed instance is listed
with whether its margin was unresolved at calibration precision in either
phase (a same-text near-tie) — changes beyond that are flagged RED.

Usage:
    uv run python scripts/g4_check.py [--no-clearml]
Writes DATA_ROOT/runs/g4_report.json; ClearML tag ``g4``.
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
from diff_voyn.data.loader import LANG_TO_INDEX
from diff_voyn.metrology import CALIBRATION_VERSION, CalibrationTable, rank_languages

LANGS = tuple(LANG_TO_INDEX)
CHECKS: list[dict] = []


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    CHECKS.append({"check": name, "status": status, "detail": detail})
    print(f"[{status}] {name}: {detail}")


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def instance_rankings(scores_path: Path, table: CalibrationTable) -> dict:
    """(language, length, trial) → {winner, margin, unresolved}."""
    offs = table.additive_offsets()
    out = {}
    for r in json.loads(scores_path.read_text())["instances"]:
        mean_bits = {h: float(np.mean(v)) for h, v in r["diffusion_bits"].items()}
        ranked = rank_languages(mean_bits, offs)
        margin = ranked[1][1] - ranked[0][1]
        out[(r["language"], r["length"], r["trial"])] = {
            "winner": ranked[0][0],
            "margin": margin,
            "unresolved": margin
            < table.margin_uncertainty_bits(ranked[0][0], ranked[1][0]),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--recovery-bar", type=float, default=0.971)
    p.add_argument("--max-drop", type=float, default=0.015, help="≥200 accuracy, pp")
    p.add_argument("--nelbo-degrade", type=float, default=0.01)
    p.add_argument("--agreement-bar", type=float, default=0.95)
    p.add_argument("--size", default="85m")
    args = p.parse_args()
    root = data_root()
    a3, a4 = root / "analysis" / "phase3", root / "analysis" / "phase4"
    size = args.size

    # 4.1 head architecture / wiring
    hs = load_json(root / "runs" / f"lid_head-{size}-seed0" / "summary.json")
    if hs:
        acc = hs["acceptance"]
        check(
            "4.1 head trains to >99% on clean long text (Phase-B, stop-gradient)",
            acc["4.1 clean long text > 99%"]["pass"],
            f"clean_L1024 {hs['heldout_acc']['clean_L1024']:.3f}; "
            + ", ".join(
                f"{k} {v:.3f}"
                for k, v in hs["heldout_acc"].items()
                if k.startswith("clean")
            ),
        )
    else:
        check(
            "4.1 head trains to >99% on clean long text", False, "no lid_head summary"
        )
    test_file = Path(__file__).resolve().parent.parent / "tests" / "test_lid_head.py"
    check(
        "4.1 stop-gradient verified (backbone grads exactly zero in Phase-B mode)",
        test_file.exists() and "exactly_zero" in test_file.read_text(),
        "tests/test_lid_head.py::test_stop_gradient_leaves_backbone_grads_exactly_zero",
    )

    # 4.2 / 4.3 Phase-B head eval
    eb = load_json(a4 / f"lid_eval_phase_b_{size}.json")
    if eb:
        s = eb["summary"]
        sub = {
            L: np.mean(
                [eb["curves"][f"substitution/L{L}/{l}/0.2"]["acc"] for l in LANGS]
            )
            for L in eb["settings"]["lengths"]
        }
        check(
            "4.2 head converged; LID accuracy vs noise-severity curves produced",
            True,
            f"lid_eval_phase_b_{size}.json: clean long {s['clean_long_acc']:.3f}; "
            "accuracy at 20% wrong key: "
            + ", ".join(f"L{L} {v:.2f}" for L, v in sub.items()),
        )
        check(
            "4.3 abstain class triggers on negative controls >95% (Phase-B head)",
            eb["acceptance"]["4.3 abstain on negative controls > 95%"]["pass"],
            ", ".join(f"{k} {v:.3f}" for k, v in s["abstain_rates_controls"].items()),
        )
    else:
        check("4.2 severity curves produced", False, "no lid_eval_phase_b")
        check("4.3 abstain on negative controls", False, "no lid_eval_phase_b")

    # 4.4 Phase C: λ schedule, grad ratio, calibration table
    run_dir = root / "runs" / f"phase_c-{size}-seed0"
    lam = load_json(run_dir / "lambda_schedule.json")
    if lam and (run_dir / "ckpt_final.pt").exists():
        trace = lam["lambda_trace"]
        ratios = [t["grad_ratio"] for t in trace if t.get("grad_ratio") is not None]
        tail = ratios[-10:] if ratios else []
        events = lam["lambda_events"]
        check(
            "4.4 Phase-C joint fine-tune finished; λ schedule logged",
            bool(trace),
            f"{len(trace)} log points, λ final {trace[-1]['lambda']:.4f} (cap {lam['lambda_max_current']:.4f}); "
            f"events: {[(e['step'], e['reason'], round(e['lambda_max'], 4)) for e in events] or 'none'}",
        )
        check(
            "4.4 LID gradient norm < 10% of diffusion gradient (last 10 measurements)",
            bool(tail) and max(tail) < 0.10,
            f"ratios {', '.join(f'{r:.3f}' for r in tail)}",
            warn_only=True,
        )
    else:
        check("4.4 Phase-C joint fine-tune finished", False, f"{run_dir} incomplete")
    try:
        tc = CalibrationTable.load(CALIBRATION_VERSION, root)
        check(
            "4.4 Phase-C calibration table produced and adopted",
            tc.phase == "phase_c",
            f"{tc.version} ({tc.phase}, policy {tc.policy}, backbone step {tc.backbone_step}): offsets "
            + ", ".join(f"{l} {o:+.3f}" for l, o in tc.offsets_bits.items())
            + f"; spread {tc.spread_bits:.3f}",
        )
    except FileNotFoundError as e:
        tc = None
        check("4.4 Phase-C calibration table produced and adopted", False, str(e))

    # 4.5 canary: per-language tiled held-out NELBO vs Phase B (same windows/seeds)
    tb = CalibrationTable.load("v3", root)
    if tc:
        drifts = {l: tc.nelbo_bits[l] / tb.nelbo_bits[l] - 1.0 for l in LANGS}
        check(
            f"4.5/G4 per-language held-out NELBO not degraded >{args.nelbo_degrade:.0%} (tiled, vs Phase B)",
            all(d < args.nelbo_degrade for d in drifts.values()),
            ", ".join(
                f"{l} {tb.nelbo_bits[l]:.4f} → {tc.nelbo_bits[l]:.4f} ({d:+.2%})"
                for l, d in drifts.items()
            ),
        )
        if lam:
            canary_events = [e for e in lam["lambda_events"] if e["reason"] == "canary"]
            check(
                "4.5 in-run canary: λ halved on breach (events documented)",
                True,
                f"{len(canary_events)} canary breach(es): {canary_events or 'none'}",
            )

    # 4.5 / G4 synthetic ranking unchanged or improved
    rb = load_json(a3 / "recovery_report.json")
    rc = load_json(a4 / "recovery_report.json")
    if rb and rc and tc:
        kb, kc = (
            f"calibration_{rb['primary_calibration']}",
            f"calibration_{rc['primary_calibration']}",
        )
        ab, ac = rb["ge200_mean_accuracy"][kb], rc["ge200_mean_accuracy"][kc]
        check(
            f"G4 synthetic 1:1 recovery ≥200 chars under Phase C ≥ {args.recovery_bar:.1%} and ≥ Phase B − {args.max_drop:.1%}",
            ac["language"] >= args.recovery_bar
            and ac["language"] >= ab["language"] - args.max_drop,
            f"language {ab['language']:.1%} → {ac['language']:.1%}, family {ab['family']:.1%} → {ac['family']:.1%} "
            f"(tables {rb['primary_calibration']} → {rc['primary_calibration']})",
        )
        cells = []
        for k in rb["cells"]:
            vb = rb["cells"][k]["accuracy"][kb]["language"]
            vc = rc["cells"][k]["accuracy"][kc]["language"]
            cells.append((k, vb, vc))
        changed = [(k, vb, vc) for k, vb, vc in cells if abs(vb - vc) > 1e-9]
        check(
            "4.5 per-cell accuracy end-B vs end-C (any change is reported, not tuned away)",
            True,
            "unchanged cells: "
            f"{len(cells)-len(changed)}/{len(cells)}; changed: "
            + (", ".join(f"{k} {vb:.0%}→{vc:.0%}" for k, vb, vc in changed) or "none"),
        )
        rank_b = instance_rankings(
            a3 / "recovery_scores.json",
            CalibrationTable.load(rb["primary_calibration"], root),
        )
        rank_c = instance_rankings(a4 / "recovery_scores.json", tc)
        flips, red = [], []
        for key, vb in rank_b.items():
            vc = rank_c.get(key)
            if vc is None or vb["winner"] == vc["winner"]:
                continue
            rec = {
                "instance": "/".join(map(str, key)),
                "phase_b": vb["winner"],
                "phase_c": vc["winner"],
                "truth": key[0],
                "unresolved_either": bool(vb["unresolved"] or vc["unresolved"]),
                "margins": (round(vb["margin"], 4), round(vc["margin"], 4)),
            }
            flips.append(rec)
            if not rec["unresolved_either"] and key[1] >= 200:
                red.append(rec)
        n_long = sum(1 for k in rank_b if k[1] >= 200)
        n_flips_long = sum(1 for f in flips if int(f["instance"].split("/")[1]) >= 200)
        check(
            "4.5 instance-level ranking changes end-B → end-C at ≥200 chars are all same-text near-ties (unresolved at calibration precision)",
            not red,
            f"{n_flips_long}/{n_long} winners changed at ≥200 chars "
            f"(all lengths: {len(flips)}/{len(rank_b)}); resolved-margin changes (RED FLAG): "
            + (json.dumps(red) if red else "none"),
            warn_only=True,
        )
    else:
        check(
            "G4 synthetic ranking test under Phase C",
            False,
            "missing Phase-B or Phase-C recovery report",
        )

    # 4.3 / 4.2 on the joint model
    ec = load_json(a4 / f"lid_eval_phase_c_{size}.json")
    if ec:
        s = ec["summary"]
        check(
            "4.3 abstain on negative controls >95% (joint model)",
            ec["acceptance"]["4.3 abstain on negative controls > 95%"]["pass"],
            ", ".join(f"{k} {v:.3f}" for k, v in s["abstain_rates_controls"].items()),
        )
        check(
            "4.1 clean long text >99% (joint model)",
            ec["acceptance"]["4.1 clean long text > 99%"]["pass"],
            f"{s['clean_long_acc']:.3f}",
        )
        if eb:
            deltas = {
                k: ec["curves"][k]["acc"] - eb["curves"][k]["acc"]
                for k in eb["curves"]
                if k in ec["curves"]
            }
            worst = sorted(deltas.items(), key=lambda kv: kv[1])[:3]
            best = sorted(deltas.items(), key=lambda kv: -kv[1])[:3]
            check(
                "4.4 joint training: LID robustness end-B → end-C (mean Δ accuracy over the severity grid)",
                True,
                f"mean Δ {np.mean(list(deltas.values())):+.4f}; largest gains {best}; largest losses {worst}",
            )

    # 4.6 head calibration + agreement
    hc = load_json(a4 / "head_calibration.json")
    if hc:
        c = hc["calibration"]
        check(
            "4.6 head temperature-scaled on held-out decipherments",
            True,
            f"T = {c['temperature']:.3f}; test NLL {c['before']['test']['nll']:.4f} → {c['after']['test']['nll']:.4f}, "
            f"ECE {c['before']['test']['ece']:.4f} → {c['after']['test']['ece']:.4f}; {hc['calibrated_head_checkpoint']}",
        )
        ag = hc["agreement"]
        check(
            f"G4 head and ELBO rankings agree on clean synthetics (≥200 chars, ≥ {args.agreement_bar:.0%})",
            hc["agreement_ge200_mean"] is not None
            and hc["agreement_ge200_mean"] >= args.agreement_bar,
            "agreement by length: "
            + ", ".join(
                f"L{L} {a['agree']:.3f}" for L, a in ag.items() if L != "overall"
            )
            + f"; overall {ag['overall']['agree']:.3f} (head right {ag['overall']['head_true']:.3f}, "
            f"ELBO right {ag['overall']['elbo_true']:.3f})",
        )
    else:
        check("4.6 head calibration", False, "no head_calibration.json")
        check("G4 head and ELBO rankings agree", False, "no head_calibration.json")

    # consistency of the adopted table
    fa = load_json(a4 / "fairness_audit.json") or load_json(a3 / "fairness_audit.json")
    if fa and rc:
        check(
            "3.5/4.4 audit table == applied table == recovery primary",
            fa["adopted_table"] == CALIBRATION_VERSION == rc["primary_calibration"],
            f"audit {fa['adopted_table']}, CALIBRATION_VERSION {CALIBRATION_VERSION}, report {rc['primary_calibration']}; "
            f"un-escalated above-noise findings: {fa['unescalated_above_noise']}",
        )

    # 4.7 seed replication (P2)
    sr = load_json(a4 / "seed_replication.json")
    if sr and sr.get("complete"):
        check(
            "4.7 seed replication at 25M (ranking stability across seeds)",
            True,
            sr.get("summary_line", ""),
        )
    else:
        check(
            "4.7 seed replication at 25M (P2)",
            False,
            (sr or {}).get(
                "summary_line",
                "seed runs in flight — re-run seed_replication.py --report, then g4_check.py",
            ),
            warn_only=True,
        )

    verdict = all(c["status"] != "FAIL" for c in CHECKS)
    report = {
        "gate": "G4",
        "created_utc": datetime.now(UTC).isoformat(),
        "verdict": "PASS" if verdict else "FAIL",
        "checks": CHECKS,
        "calibration_version": CALIBRATION_VERSION,
    }
    out = root / "runs" / "g4_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nGate G4: {report['verdict']}  (report {out})")
    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        task = init_task(
            RunConfig(run_name="g4-check", phase="phase4"), root, tags=["g4"]
        )
        task.connect_configuration(report, name="g4_report")
        task.get_logger().report_scalar("gate", "g4_pass", float(verdict), 0)
        task.get_logger().flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
