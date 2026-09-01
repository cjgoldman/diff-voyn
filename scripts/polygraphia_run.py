"""Solve / score / report the Polygraphia anchor instances through the
word-homophonic machinery (docs/polygraphia_digitization_scope.md §6).

Instances come from scripts/polygraphia_instances.py; solve and score reuse
the Phase-6 harness (same frozen evaluator, outer tier, MDL scale,
ABSTAIN_RULE). The report adds what only these instances have: SER against
the TRUE plaintext and against the oracle (majority) type key — the two
differ because Polygraphia's column cipher is not type-deterministic.

  uv run python scripts/polygraphia_run.py --stage solve --workers 12
  uv run python scripts/polygraphia_run.py --stage score
  uv run python scripts/polygraphia_run.py --stage report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if "solve" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.ladder import write_json_atomic
from diff_voyn.metrology import CALIBRATION_VERSION, CalibrationTable
from diff_voyn.vms.apply import (
    ABSTAIN_RULE,
    LANGS,
    WORDHOM,
    cell_from_score,
    ciphertext_baselines,
    instance_record,
    load_instance,
    make_jobs,
    order0_entropy_bits,
    rank_table,
    run_scores,
    run_solves,
)


def _paths(args) -> list[Path]:
    man = json.loads((args.inst_dir / "manifest.json").read_text())
    return [args.inst_dir / m["file"] for m in man]


def _inst_meta(inst: dict) -> dict:
    alpha = sorted(set("".join(inst["all_tokens"])))
    sym = np.array([alpha.index(c) for c in "".join(inst["all_tokens"])])
    return {
        "n_cipher_all": int(inst["coverage"]["n_chars"]),
        "order0_entropy_bits": order0_entropy_bits(sym),
        "coverage": inst["coverage"],
        "no_cipher_baselines": ciphertext_baselines(sym, int(sym.max()) + 1),
    }


def stage_solve(args):
    jobs = []
    for p in _paths(args):
        rec = instance_record(p)
        js = make_jobs(
            rec,
            heads=(WORDHOM,),
            hypotheses=tuple(LANGS),
            n_windows=1,
            w5=args.w5,
            restarts={WORDHOM: args.restarts},
        )
        for j in js:
            j["sa_steps"] = args.sa_steps
        jobs += js
    settings = {
        "w5": args.w5,
        "restarts": args.restarts,
        "sa_steps": args.sa_steps,
        "n_windows": 1,
        "units": None,
    }
    run_solves(
        jobs,
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
    for p in _paths(args):
        inst = load_instance(p)
        instances[(inst["name"], inst["kind"])] = inst
    run_scores(
        solves,
        instances,
        args.out_dir / "scores.json",
        ev=ev,
        budget=args.budget,
        seeds=tuple(range(args.seeds)),
        score_windows=args.score_windows,
        fresh=args.fresh,
        meta={
            "evaluator": ev.meta,
            "budget": args.budget,
            "seeds": args.seeds,
            "score_windows": args.score_windows,
            "abstain_rule": ABSTAIN_RULE,
        },
    )


def stage_report(args):
    from diff_voyn.heads.wordhom import UnitTargets, expand_units, unit_ser

    table = CalibrationTable.load(args.primary, data_root())
    scores = json.loads((args.out_dir / "scores.json").read_text())["instances"]
    insts = {}
    for p in _paths(args):
        inst = load_instance(p)
        insts[(inst["name"], inst["kind"])] = inst

    cells = {}
    for rec in scores:
        inst = insts[(rec["instance"], rec["presentation"])]
        c = cell_from_score(rec, table, _inst_meta(inst))
        truth = inst["truth"]
        m = np.asarray(rec["final"]["key"]["map"], dtype=np.int64)
        sym = np.asarray(inst["symbols"], dtype=np.int64)
        tm = np.asarray(truth["sym_to_unit"], dtype=np.int64)
        targets = UnitTargets.from_list(rec["final"]["key"]["bigrams"])
        dec = expand_units(m[sym], targets)
        plain = np.asarray(truth["plain_ids"], dtype=np.int64)
        c["ser_vs_plaintext"] = unit_ser(dec, plain)
        c["ser_vs_oracle_key"] = unit_ser(dec, expand_units(tm[sym], targets))
        c["oracle_type_acc"] = truth["oracle_type_acc"]
        c["truth_language"] = truth["language"]
        c["shape"] = truth["shape"]
        c["tokens_per_type"] = len(sym) / max(inst["n_symbols"], 1)
        cells.setdefault(rec["instance"], []).append(c)

    rows = []
    for name, cs in sorted(cells.items()):
        rt = rank_table(cs)
        best = rt["ranked"][0]
        own = next(c for c in cs if c["hypothesis"] == c["truth_language"])
        rows.append(
            {
                "instance": name,
                "truth": own["truth_language"],
                "shape": own["shape"],
                "tokens_per_type": round(own["tokens_per_type"], 2),
                "oracle_type_acc": round(own["oracle_type_acc"], 4),
                "mdl_top_language": best["hypothesis"],
                "abstain": not own["language_like"],
                "own_plain_bits": own["plain_bits"],
                "own_structure_margin": own["structure_margin"],
                "own_ser_vs_plaintext": own["ser_vs_plaintext"],
                "own_ser_vs_oracle_key": own["ser_vs_oracle_key"],
            }
        )
    out = {"abstain_rule": ABSTAIN_RULE, "calibration": args.primary, "rows": rows}
    write_json_atomic(args.out_dir / "report.json", out)
    hdr = (
        "instance",
        "truth",
        "shape",
        "tok/type",
        "oracle",
        "MDL-top",
        "abstain",
        "plain b/c",
        "margin",
        "SER(plain)",
        "SER(oracle)",
    )
    print(" | ".join(hdr))
    for r in rows:
        print(
            " | ".join(
                str(r[k]) if not isinstance(r[k], float) else f"{r[k]:.3f}"
                for k in (
                    "instance",
                    "truth",
                    "shape",
                    "tokens_per_type",
                    "oracle_type_acc",
                    "mdl_top_language",
                    "abstain",
                    "own_plain_bits",
                    "own_structure_margin",
                    "own_ser_vs_plaintext",
                    "own_ser_vs_oracle_key",
                )
            )
        )


def main():
    p = argparse.ArgumentParser()
    root = data_root()
    p.add_argument("--stage", choices=["solve", "score", "report"], required=True)
    p.add_argument(
        "--inst-dir",
        type=Path,
        default=root / "analysis" / "polygraphia_anchor" / "instances",
    )
    p.add_argument(
        "--out-dir", type=Path, default=root / "analysis" / "polygraphia_anchor"
    )
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--w5", type=int, default=12000)
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument("--sa-steps", type=int, default=400_000)
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--score-windows", type=int, default=16)
    p.add_argument("--primary", default=CALIBRATION_VERSION)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    {"solve": stage_solve, "score": stage_score, "report": stage_report}[args.stage](
        args
    )


if __name__ == "__main__":
    main()
