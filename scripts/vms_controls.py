"""Phase 6, task 6.3 — the negative-control battery through the VMS pipeline.

voynichesque (must abstain) · shuffled text (must abstain) · out-of-inventory
contamination (Dutch / English / French / Spanish under in-inventory-fit
ciphers; confusions documented) · in-inventory positives (must not abstain).
Every instance runs through ``diff_voyn.vms.apply`` exactly as the
manuscript does (same heads, outer tier, MDL scale, abstention rule).

Stages: generate → solve (CPU pool) → score (GPU) → report
Artifacts: DATA_ROOT/analysis/phase6/controls/{manifest,solves,scores,report}.*
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if "solve" in sys.argv or "generate" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.ladder import wilson, write_json_atomic
from diff_voyn.metrology import CALIBRATION_VERSION, CalibrationTable
from diff_voyn.vms.apply import (
    ABSTAIN_RULE,
    KEY,
    LANGS,
    cell_from_score,
    instance_record,
    load_instance,
    make_jobs,
    order0_entropy_bits,
    rank_table,
    run_scores,
    run_solves,
)

MIN_NAIBBE_TOKENS = 200


def stage_generate(args):
    from diff_voyn.vms.controls import build_controls

    m = build_controls(
        args.out_dir,
        per_language=args.per_language,
        length=args.length,
        naibbe_length=args.naibbe_length,
        seed=args.seed,
    )
    print(len(m), "instances")


def _manifest(args):
    return json.loads((args.out_dir / "manifest.json").read_text())


def all_jobs(args):
    jobs = []
    for m in _manifest(args):
        rec = instance_record(args.out_dir / m["file"])
        heads = ("naibbe",) if m["kind"] == "words" else ("sub1to1", "homophonic")
        if m["kind"] == "words" and rec["n_stream"] < MIN_NAIBBE_TOKENS:
            continue
        jobs += make_jobs(
            rec,
            heads=tuple(h for h in heads if h in args.heads),
            n_windows=1,
            w1=args.w1,
            w2=args.w2,
            w3=args.w3,
            restarts={"sub1to1": args.r1, "homophonic": args.r2, "naibbe": args.r3},
        )
    return jobs


def stage_solve(args):
    settings = {k: getattr(args, k) for k in ("w1", "w2", "w3", "r1", "r2", "r3")}
    run_solves(
        all_jobs(args),
        args.out_dir / "solves.json",
        workers=args.workers,
        settings=settings,
        fresh=args.fresh,
    )


def stage_score(args):
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator

    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(4)
    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    solves = json.loads((args.out_dir / "solves.json").read_text())["instances"]
    instances = {}
    for m in _manifest(args):
        inst = load_instance(args.out_dir / m["file"])
        instances[(inst["name"], inst["kind"])] = inst
    i, n = (int(x) for x in args.shard.split("/"))
    path = args.out_dir / (f"scores_shard{i}of{n}.json" if n > 1 else "scores.json")
    run_scores(
        solves,
        instances,
        path,
        ev=ev,
        budget=args.budget,
        seeds=tuple(range(args.seeds)),
        score_windows=args.score_windows,
        shard=(i, n),
        fresh=args.fresh,
        meta={
            "evaluator": ev.meta,
            "budget": args.budget,
            "seeds": args.seeds,
            "abstain_rule": ABSTAIN_RULE,
        },
    )


def _load_scores(out_dir: Path):
    recs, meta = {}, {}
    for p in sorted(out_dir.glob("scores*.json")):
        d = json.loads(p.read_text())
        meta = {k: v for k, v in d.items() if k != "instances"}
        for r in d["instances"]:
            recs[tuple(r[k] for k in KEY)] = r
    return meta, list(recs.values())


def stage_report(args):
    table = CalibrationTable.load(args.primary, data_root())
    meta, recs = _load_scores(args.out_dir)
    manifest = {(m["name"], m["kind"]): m for m in _manifest(args)}
    inst_cache = {}

    def inst_of(name, kind):
        if (name, kind) not in inst_cache:
            inst_cache[(name, kind)] = load_instance(
                args.out_dir / manifest[(name, kind)]["file"]
            )
        return inst_cache[(name, kind)]

    cells = []
    for r in recs:
        inst = inst_of(r["instance"], r["presentation"])
        cov = inst["coverage"]
        sym = np.asarray(inst["symbols"]) if inst["kind"] != "words" else None
        if sym is None:
            alpha = sorted(set("".join(inst["tokens"])))
            sym = np.array([alpha.index(c) for c in "".join(inst["tokens"])])
        c = cell_from_score(
            r,
            table,
            {
                "n_cipher_all": int(cov["n_chars"]),
                "order0_entropy_bits": order0_entropy_bits(sym),
            },
        )
        c["control"] = manifest[(r["instance"], r["presentation"])]["control"]
        c["truth"] = {
            k: v
            for k, v in inst["truth"].items()
            if k not in ("plain_ids", "sym_to_letter")
        }
        # SER for 1:1-enciphered instances (truth map available)
        if r["head"] in ("sub1to1", "homophonic") and "sym_to_letter" in inst["truth"]:
            m = np.asarray(r["final"]["key"]["map"])
            plain = np.asarray(inst["truth"]["plain_ids"])
            c["ser"] = float(np.mean(m[sym] != plain))
        cells.append(c)
    # per-instance verdicts: the instance's table = all its cells (eva + words)
    by_inst: dict[str, list[dict]] = {}
    for c in cells:
        by_inst.setdefault(c["instance"], []).append(c)
    verdicts = []
    for name, cs in by_inst.items():
        tab = rank_table(cs)
        truth = cs[0]["truth"]
        control = cs[0]["control"]
        top = tab["ranked"][0]
        # language call: best language-like cell if any, else the MDL top
        like = [r for r in tab["ranked"] if r["language_like_any"]]
        call = like[0] if like else None
        verdicts.append(
            {
                "instance": name,
                "control": control,
                "truth_language": truth["language"],
                "truth_family": truth["family"],
                "abstain": tab["abstain"],
                "mdl_top_head": top["head"],
                "mdl_top_language": top["hypothesis"],
                "called_language": call["hypothesis"] if call else None,
                "called_head": call["head"] if call else None,
                "per_head": {h: v["top"] for h, v in tab["per_head"].items()},
                "best_plain_bits": min(c["plain_bits"] for c in cs),
                "best_structure_margin": max(c["structure_margin"] for c in cs),
                "ser_sub1to1_true_hyp": next(
                    (
                        c.get("ser")
                        for c in cs
                        if c["head"] == "sub1to1"
                        and c["hypothesis"] == truth["language"]
                    ),
                    None,
                ),
            }
        )
    # aggregates
    summary = {}
    for control in ("voynichesque", "shuffled", "contamination", "positive"):
        vs = [v for v in verdicts if v["control"] == control]
        if not vs:
            continue
        n = len(vs)
        k_abs = sum(v["abstain"] for v in vs)
        s = {"n": n, "abstain_rate": k_abs / n, "abstain_ci95": wilson(k_abs, n)}
        if control == "positive":
            ok = [v["called_language"] == v["truth_language"] for v in vs]
            s["language_correct_rate"] = float(np.mean(ok))
            s["family_correct_rate"] = float(
                np.mean(
                    [
                        (v["called_language"] is not None)
                        and _fam(v["called_language"]) == v["truth_family"]
                        for v in vs
                    ]
                )
            )
            s["false_abstain_rate"] = k_abs / n
        if control == "contamination":
            conf: dict[str, dict[str, int]] = {}
            for v in vs:
                row = conf.setdefault(
                    v["truth_language"], {l: 0 for l in LANGS} | {"abstain": 0}
                )
                row[v["called_language"] or "abstain"] += 1
            s["confusion_called"] = conf
            conf_mdl: dict[str, dict[str, int]] = {}
            for v in vs:
                row = conf_mdl.setdefault(v["truth_language"], {l: 0 for l in LANGS})
                row[v["mdl_top_language"]] += 1
            s["confusion_mdl_top"] = conf_mdl
            s["family_correct_rate_when_called"] = (
                float(
                    np.mean(
                        [
                            _fam(v["called_language"]) == v["truth_family"]
                            for v in vs
                            if v["called_language"]
                        ]
                    )
                )
                if any(v["called_language"] for v in vs)
                else None
            )
            s["family_correct_rate_mdl_top"] = float(
                np.mean([_fam(v["mdl_top_language"]) == v["truth_family"] for v in vs])
            )
        summary[control] = s
    acc = {
        "criterion": "abstention > 95% on voynichesque and shuffled; contamination confusions documented; positives not abstained",
        "voynichesque_abstain": summary.get("voynichesque", {}).get("abstain_rate"),
        "shuffled_abstain": summary.get("shuffled", {}).get("abstain_rate"),
        "positive_false_abstain": summary.get("positive", {}).get("false_abstain_rate"),
        "positive_language_correct": summary.get("positive", {}).get(
            "language_correct_rate"
        ),
    }
    acc["pass"] = bool(
        (acc["voynichesque_abstain"] or 0) > 0.95
        and (acc["shuffled_abstain"] or 0) > 0.95
        and (
            acc["positive_false_abstain"]
            if acc["positive_false_abstain"] is not None
            else 1
        )
        <= 0.05
    )
    report = {
        "task": "6.3",
        "created_utc": datetime.now(UTC).isoformat(),
        "evaluator": meta.get("evaluator"),
        "budget": meta.get("budget"),
        "abstain_rule": ABSTAIN_RULE,
        "summary": summary,
        "verdicts": verdicts,
        "cells": cells,
        "acceptance": acc,
    }
    write_json_atomic(args.out_dir / "report.json", report)
    md = [
        "### Negative-control battery (task 6.3)",
        "",
        "| control | n | abstain rate (95% CI) | notes |",
        "|---|---|---|---|",
    ]
    for k, s in summary.items():
        notes = ""
        if k == "positive":
            notes = f"language correct {s['language_correct_rate']:.2f}, family {s['family_correct_rate']:.2f}"
        if k == "contamination":
            notes = f"called: {json.dumps(s['confusion_called'])}; MDL-top: {json.dumps(s['confusion_mdl_top'])}; family-correct when called {s['family_correct_rate_when_called']}, MDL-top {s['family_correct_rate_mdl_top']:.2f}"
        md.append(
            f"| {k} | {s['n']} | {s['abstain_rate']:.2f} ({s['abstain_ci95'][0]:.2f}–{s['abstain_ci95'][1]:.2f}) | {notes} |"
        )
    md += [
        "",
        "| instance | control | truth | abstain | called | MDL top (head/lang) | best plain bits | best structure margin | SER 1:1 true hyp |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        md.append(
            f"| {v['instance']} | {v['control']} | {v['truth_language']} | {v['abstain']} | {v['called_language']} | {v['mdl_top_head']}/{v['mdl_top_language']} | {v['best_plain_bits']:.3f} | {v['best_structure_margin']:.2f} | {v['ser_sub1to1_true_hyp'] if v['ser_sub1to1_true_hyp'] is None else round(v['ser_sub1to1_true_hyp'], 3)} |"
        )
    md.append("")
    md.append(f"acceptance: {json.dumps(acc)}")
    md = "\n".join(md)
    (args.out_dir / "report.md").write_text(md)
    print(md)


def _fam(lang):
    return {"latin": "romance", "italian": "romance", "german": "germanic"}.get(lang)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = data_root()
    p.add_argument(
        "--stage", choices=["generate", "solve", "score", "report"], required=True
    )
    p.add_argument(
        "--out-dir", type=Path, default=root / "analysis" / "phase6" / "controls"
    )
    p.add_argument("--per-language", type=int, default=3)
    p.add_argument("--length", type=int, default=2000)
    p.add_argument("--naibbe-length", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heads", nargs="+", default=["sub1to1", "homophonic", "naibbe"])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--w1", type=int, default=4000)
    p.add_argument("--w2", type=int, default=2000)
    p.add_argument("--w3", type=int, default=4000)
    p.add_argument("--r1", type=int, default=4)
    p.add_argument("--r2", type=int, default=32)
    p.add_argument("--r3", type=int, default=3)
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--score-windows", type=int, default=16)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--primary", default=CALIBRATION_VERSION)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    {
        "generate": stage_generate,
        "solve": stage_solve,
        "score": stage_score,
        "report": stage_report,
    }[args.stage](args)


if __name__ == "__main__":
    main()
