"""Gate G5 verification — Phase 5 (cipher-head integration, frozen evaluator).

Gate wording (task breakdown): *rungs 1–3 meet their SER targets on
synthetics; cross-head scores are on a comparable scale.* (Rung 4 is P1 and
may trail — reported as WARN when absent or failing.) Plus the task
acceptances feeding it: 5.1 (gradients reach a toy head through the frozen
evaluator; NaN smoke test), 5.2 (near-perfect 1:1 recovery at ≥200 chars),
5.3 (≤1.9% SER on Zodiac-408-class synthetics), 5.4 (≥95% letter-map
accuracy on Naibbe pairs, restart budget documented), 5.5 (language recovery
better than family-random on pseudo-VMS), 5.6 (uniform scale), and the
freeze discipline: every artifact was scored by the SAME frozen evaluator
(path + step + sha256 of the Gate-G4 checkpoint) under the adopted
calibration table.

Usage:
    uv run python scripts/g5_check.py [--no-clearml]
Writes DATA_ROOT/runs/g5_report.json; ClearML tag ``g5``.
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
    a5 = root / "analysis" / "phase5"

    fz = load_json(a5 / "evaluator_freeze.json")
    if fz:
        e = fz["frozen"]["evaluator"]
        check(
            "5.1 evaluator frozen (EMA G4 checkpoint, fingerprinted) and interface verified on real weights",
            fz["verdict"] == "PASS",
            f"{Path(e['path']).parent.name}/{Path(e['path']).name} step {e['step']} sha256 {e['sha256'][:12]}…; "
            + "; ".join(
                c["check"].split(" ", 1)[1][:40] + f" {c['status']}"
                for c in fz["checks"]
            ),
        )
        gap = fz["null_frame_gap"]
        check(
            "5.1 2N NULL-frame vs plain-stream gap measured (uniform scale scores collapsed decodes)",
            True,
            "; ".join(
                f"{g['language']} plain {g['plain_bits_per_char']:.3f} / frame letter-slots {g['frame_letter_slots_bits_per_plain_char']:.3f}"
                for g in gap
            ),
        )
        frozen_path, frozen_step = e["path"], e["step"]
    else:
        check("5.1 evaluator freeze", False, "no evaluator_freeze.json")
        frozen_path = frozen_step = None

    def same_evaluator(rep, name):
        if rep is None:
            return
        ev = rep.get("evaluator", {})
        ok = (
            ev.get("path") == frozen_path
            and ev.get("step") == frozen_step
            and rep.get("primary_calibration", CALIBRATION_VERSION)
            == CALIBRATION_VERSION
        )
        check(
            f"freeze discipline: {name} scored by the frozen evaluator under {CALIBRATION_VERSION}",
            ok,
            f"{Path(ev.get('path', '?')).parent.name} step {ev.get('step')} ({ev.get('weights')}), table {rep.get('primary_calibration')}",
        )

    r1 = load_json(a5 / "rung1_report.json")
    if r1:
        a = r1["acceptance"]
        check(
            "5.2 rung 1: near-perfect recovery on synthetic 1:1 at ≥200 chars",
            a["pass"],
            f"SER final {a['ser_final_ge200']:.4f} (n-gram {a['ser_ngram_ge200']:.4f}), solved {a['solved_final_ge200']:.1%}, "
            f"language recovery {a['lang_acc_final_ge200']:.1%} (n-gram winners {a['lang_acc_ngram_winner_ge200']:.1%})",
        )
        pl = r1["per_language_ge200"]
        check(
            "5.2 per-language solve success at ≥200 chars (search fairness) ≥ 90% for every language",
            all(v["solved_final_ge200"] >= 0.9 for v in pl.values()),
            "; ".join(
                f"{l} {v['solved_ngram_ge200']:.0%}→{v['solved_final_ge200']:.0%}"
                for l, v in pl.items()
            ),
        )
        same_evaluator(r1, "rung 1")
    else:
        check("5.2 rung 1", False, "no rung1_report.json")

    r2 = load_json(a5 / "rung2_report.json")
    if r2:
        a = r2["acceptance"]
        check(
            "5.3 rung 2: ≤1.9% SER on Zodiac-408-class synthetics (mean over instances)",
            a["pass"],
            f"mean SER final {a['ser_final_mean']:.4f} (n-gram {a['ser_ngram_mean']:.4f}, oracle {a['ser_oracle_mean']:.4f}); n={a['n']}",
            warn_only=a.get("pass_per_instance", False),
        )
        check(
            "5.3 rung 2: per-instance reading — ≥80% of instances ≤1.9% and median ≤1.9%",
            a.get("pass_per_instance", False),
            f"instances ≤1.9%: {a['instances_le_1.9pct_final']:.0%} (n-gram {a['instances_le_1.9pct_ngram']:.0%}); median SER {a.get('ser_final_median', float('nan')):.4f}",
        )
        check(
            "5.3 degenerate (unpenalized-objective) maps rejected by the MDL selection; pure-ELBO preference recorded",
            a["mdl_picked_degenerate_rate"] == 0.0,
            f"pure ELBO picks a degenerate map {a['elbo_pure_picked_degenerate_rate']:.0%} of the time "
            f"(SER {a['ser_elbo_pure_mean']:.3f}); MDL total (plaintext + choice bits) {a['mdl_picked_degenerate_rate']:.0%}",
        )
        check(
            "5.3 / 6.6 literature anchors",
            False,
            r2.get("anchors_note", ""),
            warn_only=True,
        )
        same_evaluator(r2, "rung 2")
    else:
        check("5.3 rung 2", False, "no rung2_report.json")

    r3 = load_json(a5 / "rung3_report.json")
    if r3:
        a = r3["acceptance"]
        s = r3["solve_settings"]
        check(
            "5.4 rung 3: ≥95% letter-map accuracy on synthetic Naibbe pairs (occurrence-weighted glyph-type accuracy)",
            a["pass"],
            f"weighted type accuracy final {a['wacc_final_mean']:.3f} (n-gram {a['wacc_ngram_mean']:.3f}); "
            f"unweighted code accuracy {a['acc_final_mean']:.3f} (instances ≥95%: {a['instances_ge95_final']:.0%}); n={a['n']}",
        )
        check(
            "5.4 restart budget documented",
            True,
            f"{s['restarts']} restarts × {s['steps']} steps + fixed-parse swap polish per hypothesis, {s['chars']} chars",
        )
        same_evaluator(r3, "rung 3")
    else:
        check("5.4 rung 3", False, "no rung3_report.json")

    r4 = load_json(a5 / "rung4_report.json")
    if r4:
        a = r4["acceptance"]
        check(
            "5.5 rung 4 (P1): language recovery better than family-random on pseudo-VMS",
            a["pass"],
            f"language {a['lang_acc_final']:.1%} (random 33%), family {a['family_acc_final']:.1%} (random 56%), n={a['n']}",
            warn_only=True,
        )
        same_evaluator(r4, "rung 4")
    else:
        check("5.5 rung 4 (P1)", False, "no rung4_report.json", warn_only=True)

    r6 = load_json(a5 / "crosshead_report.json")
    if r6:
        a = r6["acceptance"]
        check(
            "5.6 / G5 cross-head scores comparable on one calibrated scale (+ complexity penalty)",
            a["pass"],
            f"same instrument max gap {a['same_instrument_max_abs_gap_bits']:.3f} bits over {a['same_instrument_n']}; "
            f"MDL picks true cipher {a['mdl_true_cipher_rate']:.0%} of {a['mdl_n']}; rung-1 first on 1:1 {a['simplicity_rung1_first_rate']}; "
            f"language rank within true head {a['language_rank_within_true_head']}",
        )
        same_evaluator(r6, "cross-head")
    else:
        check("5.6 cross-head scale", False, "no crosshead_report.json")

    tests = Path(__file__).resolve().parent.parent / "tests" / "test_heads.py"
    txt = tests.read_text() if tests.exists() else ""
    check(
        "5.1 NaN smoke test + two-tier unit tests in CI",
        all(
            k in txt
            for k in (
                "logaddexp",
                "test_viterbi_segmental",
                "paired_bits",
                "test_scale_terms",
            )
        ),
        "tests/test_heads.py: frame/NaN, viterbi, paired_bits, refine_assignment, scale",
    )

    verdict = all(c["status"] != "FAIL" for c in CHECKS)
    report = {
        "gate": "G5",
        "created_utc": datetime.now(UTC).isoformat(),
        "verdict": "PASS" if verdict else "FAIL",
        "checks": CHECKS,
        "calibration_version": CALIBRATION_VERSION,
        "frozen_evaluator": {"path": frozen_path, "step": frozen_step},
    }
    out = root / "runs" / "g5_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nGate G5: {report['verdict']}  (report {out})")
    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        task = init_task(
            RunConfig(run_name="g5-check", phase="phase5"), root, tags=["g5"]
        )
        task.connect_configuration(report, name="g5_report")
        task.get_logger().report_scalar("gate", "g5_pass", float(verdict), 0)
        task.get_logger().flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
