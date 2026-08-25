"""Phase 6, tasks 6.1 / 6.2 — the manuscript through the validated heads.

Stages (resumable; artifacts under DATA_ROOT/analysis/phase6/):
  prepare  build the presentations (EVA chars, Naibbe-parseable words,
           Boxer glyph streams) per transcription × Currier dialect
  solve    CPU pool: every (instance × presentation × head × window ×
           hypothesis) inner search → shortlist  (vms_solves.json)
  score    GPU: outer tier (paired ELBO, MDL selection, elbo_polish),
           full-stream decode scored on ≤ N windows × 4 seeds with a
           shuffled copy per window                 (vms_scores.json)
  report   ranked (cipher × language) table per dialect with uncertainty,
           head agreement and abstention            (vms_report.{json,md})

Currier A and B are never pooled (design §9): every instance is one
dialect of one transcription. Run per-dialect shards of the score stage on
the two GPUs with --shard i/n.

Usage:
  uv run python scripts/vms_apply.py --stage prepare
  uv run python scripts/vms_apply.py --stage solve --workers 12
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/vms_apply.py --stage score --shard 0/2
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/vms_apply.py --stage score --shard 1/2
  uv run python scripts/vms_apply.py --stage report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if "solve" in sys.argv or "prepare" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.ladder import write_json_atomic
from diff_voyn.metrology import CALIBRATION_VERSION, CalibrationTable
from diff_voyn.vms.apply import (
    ABSTAIN_RULE,
    HEADS,
    KEY,
    cell_from_score,
    ciphertext_baselines,
    fmt_table_md,
    instance_record,
    load_instance,
    make_jobs,
    order0_entropy_bits,
    rank_table,
    run_scores,
    run_solves,
)

# instance files of the VMS run: (file stem, heads, n_windows)
INSTANCES = [
    ("IT2a_A_eva", ("sub1to1", "homophonic"), 2),
    ("IT2a_B_eva", ("sub1to1", "homophonic"), 2),
    ("IT2a_A_words", ("naibbe",), 1),
    ("IT2a_B_words", ("naibbe",), 2),
    ("boxer20_A_boxer", ("sub1to1", "homophonic"), 2),
    ("boxer20_B_boxer", ("sub1to1", "homophonic"), 2),
    ("boxer16_A_boxer", ("arithmetic",), 2),
    ("boxer16_B_boxer", ("arithmetic",), 2),
    ("RF1b_A_eva", ("sub1to1", "homophonic"), 1),
    ("RF1b_B_eva", ("sub1to1", "homophonic"), 1),
    ("RF1b_A_words", ("naibbe",), 1),
    ("RF1b_B_words", ("naibbe",), 1),
]


def stage_prepare(args):
    from diff_voyn.heads.naibbe_parse import NaibbeParser
    from diff_voyn.vms.presentations import write_presentations

    s = write_presentations(args.pres_dir, NaibbeParser())
    for k, v in s.items():
        print(
            k,
            v["n_symbols"],
            v["n_stream"],
            {
                kk: (round(vv, 3) if isinstance(vv, float) else vv)
                for kk, vv in v["coverage"].items()
            },
        )


def all_jobs(args):
    jobs = []
    for stem, heads, n_windows in INSTANCES:
        if args.only and not any(o in stem for o in args.only):
            continue
        rec = instance_record(args.pres_dir / f"{stem}.json")
        jobs += make_jobs(
            rec,
            heads=tuple(h for h in heads if h in args.heads),
            n_windows=n_windows,
            w1=args.w1,
            w2=args.w2,
            w3=args.w3,
            w4=args.w4,
            restarts={
                "sub1to1": args.r1,
                "homophonic": args.r2,
                "naibbe": args.r3,
                "arithmetic": args.r4,
            },
        )
    return jobs


def stage_solve(args):
    jobs = all_jobs(args)
    settings = {
        k: getattr(args, k) for k in ("w1", "w2", "w3", "w4", "r1", "r2", "r3", "r4")
    }
    run_solves(
        jobs,
        args.out_dir / "vms_solves.json",
        workers=args.workers,
        settings=settings,
        fresh=args.fresh,
    )


def stage_score(args):
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator

    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(4)
    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    solves = json.loads((args.out_dir / "vms_solves.json").read_text())["instances"]
    if args.only:
        solves = [
            r
            for r in solves
            if any(
                o in r["instance"].replace("/", "_") + "_" + r["presentation"]
                for o in args.only
            )
        ]
    solves = [r for r in solves if r["head"] in args.heads]
    instances = {}
    for stem, _, _ in INSTANCES:
        inst = load_instance(args.pres_dir / f"{stem}.json")
        instances[(inst["name"], inst["kind"])] = inst
    i, n = (int(x) for x in args.shard.split("/"))
    path = args.out_dir / (
        f"vms_scores_shard{i}of{n}.json" if n > 1 else "vms_scores.json"
    )
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
            "score_windows": args.score_windows,
            "abstain_rule": ABSTAIN_RULE,
        },
    )


def _load_scores(out_dir: Path) -> tuple[dict, list[dict]]:
    recs, meta = {}, {}
    for p in sorted(out_dir.glob("vms_scores*.json")):
        d = json.loads(p.read_text())
        meta = {k: v for k, v in d.items() if k != "instances"}
        for r in d["instances"]:
            recs[tuple(r[k] for k in KEY)] = r
    return meta, list(recs.values())


def stage_report(args):
    table = CalibrationTable.load(args.primary, data_root())
    meta, recs = _load_scores(args.out_dir)
    inst_meta = {}
    for stem, _, _ in INSTANCES:
        inst = load_instance(args.pres_dir / f"{stem}.json")
        cov = inst["coverage"]
        n_all = cov.get("n_chars", cov.get("n_glyphs", inst["n_stream"]))
        sym = np.asarray(inst["symbols"]) if inst["kind"] != "words" else None
        if sym is None:
            alpha = sorted(set("".join(inst["tokens"])))
            sym = np.array([alpha.index(c) for c in "".join(inst["tokens"])])
        inst_meta[(inst["name"], inst["kind"])] = {
            "n_cipher_all": int(n_all),
            "order0_entropy_bits": order0_entropy_bits(sym),
            "coverage": cov,
            "no_cipher_baselines": ciphertext_baselines(sym, int(sym.max()) + 1),
        }
    cells = [
        cell_from_score(r, table, inst_meta[(r["instance"], r["presentation"])])
        for r in recs
    ]
    # group by (transcription-source, dialect): the Boxer presentations join the
    # dialect's table; RF1b is the replicate transcription, reported separately
    groups: dict[str, list[dict]] = {}
    for c in cells:
        src, dialect = c["instance"].split("/")
        key = f"{'IT2a+boxer' if src in ('IT2a', 'boxer20', 'boxer16') else src}/{dialect}"
        groups.setdefault(key, []).append(c)
    tables = {k: rank_table(v) for k, v in groups.items()}
    # transcription agreement: IT2a vs RF1b top language per head per dialect
    transcription_agreement = {}
    for d in ("A", "B"):
        a, b = tables.get(f"IT2a+boxer/{d}"), tables.get(f"RF1b/{d}")
        if a and b:
            transcription_agreement[d] = {
                h: {
                    "IT2a": a["per_head"][h]["top"],
                    "RF1b": b["per_head"][h]["top"],
                    "agree": a["per_head"][h]["top"] == b["per_head"][h]["top"],
                }
                for h in a["per_head"]
                if h in b["per_head"]
            }
    # dialect agreement (A vs B) on the primary transcription
    dialect_agreement = {}
    a, b = tables.get("IT2a+boxer/A"), tables.get("IT2a+boxer/B")
    if a and b:
        dialect_agreement = {
            h: {
                "A": a["per_head"][h]["top"],
                "B": b["per_head"][h]["top"],
                "agree": a["per_head"][h]["top"] == b["per_head"][h]["top"],
            }
            for h in a["per_head"]
            if h in b["per_head"]
        }
    report = {
        "task": "6.1/6.2",
        "created_utc": datetime.now(UTC).isoformat(),
        "evaluator": meta.get("evaluator"),
        "budget": meta.get("budget"),
        "seeds": meta.get("seeds"),
        "score_windows": meta.get("score_windows"),
        "primary_calibration": args.primary,
        "calibration_offsets_measured_bits": table.offsets_bits,
        "abstain_rule": ABSTAIN_RULE,
        "instances": {f"{k[0]}/{k[1]}": v for k, v in inst_meta.items()},
        "cells": cells,
        "tables": tables,
        "transcription_agreement": transcription_agreement,
        "dialect_agreement": dialect_agreement,
    }
    write_json_atomic(args.out_dir / "vms_report.json", report)
    md = [
        "### VMS (cipher × language) tables — MDL total per covered ciphertext symbol (task 6.2)",
        "",
    ]
    for k in sorted(tables):
        md.append(fmt_table_md(k, tables[k], groups[k]))
        base = {
            f"{n[0]}/{n[1]}": {
                kk: round(vv, 3)
                for kk, vv in m["no_cipher_baselines"].items()
                if kk != "n_symbols"
            }
            for n, m in inst_meta.items()
            if n[1] != "words"
            and (
                (
                    k.startswith("IT2a")
                    and n[0].split("/")[0] in ("IT2a", "boxer20", "boxer16")
                )
                or n[0].split("/")[0] == k.split("/")[0]
            )
            and n[0].endswith(k.split("/")[1])
        }
        md.append(
            "no-cipher baselines (bits per ciphertext symbol, held-out n-gram of the stream itself): "
            + json.dumps(base)
        )
        md.append("")
    md.append(
        "transcription agreement (IT2a vs RF1b): " + json.dumps(transcription_agreement)
    )
    md.append(
        "dialect agreement (A vs B, IT2a+boxer): " + json.dumps(dialect_agreement)
    )
    md = "\n".join(md)
    (args.out_dir / "vms_report.md").write_text(md)
    print(md)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = data_root()
    p.add_argument(
        "--stage", choices=["prepare", "solve", "score", "report"], required=True
    )
    p.add_argument("--out-dir", type=Path, default=root / "analysis" / "phase6")
    p.add_argument(
        "--pres-dir", type=Path, default=root / "analysis" / "phase6" / "presentations"
    )
    p.add_argument("--heads", nargs="+", default=list(HEADS))
    p.add_argument(
        "--only", nargs="*", default=None, help="substring filter on instance stems"
    )
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--w1", type=int, default=4000)
    p.add_argument("--w2", type=int, default=2000)
    p.add_argument("--w3", type=int, default=4000)
    p.add_argument("--w4", type=int, default=500)
    p.add_argument("--r1", type=int, default=4)
    p.add_argument("--r2", type=int, default=64)
    p.add_argument("--r3", type=int, default=3)
    p.add_argument("--r4", type=int, default=3)
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
        "prepare": stage_prepare,
        "solve": stage_solve,
        "score": stage_score,
        "report": stage_report,
    }[args.stage](args)


if __name__ == "__main__":
    main()
