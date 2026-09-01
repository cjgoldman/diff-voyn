"""Post-Phase-6 study — Boxer's hypothesis without the arithmetic: a
word-level homophonic cipher over letters + the language's top-5 bigrams
with the repeat rule (``diff_voyn/heads/wordhom.py``), through the Phase-6
pipeline (same frozen evaluator, outer tier, MDL scale, abstention rule,
control battery).

Stages (resumable; artifacts under DATA_ROOT/analysis/wordhom/):
  prepare   word-type presentations of the manuscript (per transcription ×
            dialect × K) and the control battery (positive / shuffled /
            voynichesque / contamination) at the same K
  solve     CPU pool: inner search per (instance × K × hypothesis)
  score     GPU: outer tier + full-stream scoring (paired shuffled copies)
  report    ranked tables for the manuscript, control verdicts, the
            repeat-rule doubling control, SER on positives

Usage:
  uv run python scripts/wordhom_study.py --stage prepare
  uv run python scripts/wordhom_study.py --stage solve --workers 12
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/wordhom_study.py --stage score --shard 0/2
  uv run python scripts/wordhom_study.py --stage report
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
from diff_voyn.heads.ladder import wilson, write_json_atomic
from diff_voyn.metrology import CALIBRATION_VERSION, CalibrationTable
from diff_voyn.vms.apply import (
    ABSTAIN_RULE,
    KEY,
    LANGS,
    WORDHOM,
    build_ngram_evaluator,
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

FAM = {"latin": "romance", "italian": "romance", "german": "germanic"}


def _ks(args):
    return [None if k in ("all", "0") else int(k) for k in args.k]


def _kind(k):
    return f"wordtypes{k if k is not None else 'all'}"


# -- prepare -------------------------------------------------------------------


def stage_prepare(args):
    from diff_voyn.heads.synth import HeldoutSampler
    from diff_voyn.heads.wordhom import (
        doubling_rate,
        language_targets,
        segment_units,
    )
    from diff_voyn.vms.controls import build_wordhom_controls
    from diff_voyn.vms.presentations import write_wordtypes_presentations

    ev = build_ngram_evaluator()
    s = write_wordtypes_presentations(args.pres_dir, n_tops=_ks(args))
    for k, v in s.items():
        print(k, v["n_symbols"], v["n_stream"], json.dumps(v["coverage"]))
    for k in _ks(args):
        m = build_wordhom_controls(
            args.ctrl_dir / _kind(k),
            ev,
            per_language=args.per_language,
            length=args.length,
            n_types=args.n_types,
            n_top=k,
            seed=args.seed,
            shapes=(
                [(t, int(l), int(n)) for t, l, n in (x.split(":") for x in args.shapes)]
                if args.shapes
                else None
            ),
        )
        print(_kind(k), len(m), "control instances")
    # the free control: doubled-unit rate of each language vs the manuscript
    from diff_voyn.corpus.splits import load_splits

    corpus_dir = data_root() / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    from diff_voyn.vocab import LETTERS

    dbl = {}
    for lang in LANGS:
        smp = HeldoutSampler(corpus_dir, splits, lang)
        t = language_targets(ev, lang)
        rng = np.random.default_rng(args.seed)
        rates_u, rates_l = [], []
        for _ in range(20):
            plain = smp.sample(20000, rng)
            rates_u.append(doubling_rate(segment_units(plain, t)))
            rates_l.append(doubling_rate(plain))
        dbl[lang] = {
            "bigrams": ["".join(LETTERS[i] for i in bg) for bg in t.bigrams],
            "doubled_units_per_1000": 1000 * float(np.mean(rates_u)),
            "doubled_units_per_1000_sd": 1000 * float(np.std(rates_u)),
            "doubled_letters_per_1000": 1000 * float(np.mean(rates_l)),
        }
    vms = {k: v["coverage"]["adjacent_repeats_per_1000"] for k, v in s.items()}
    write_json_atomic(
        args.out_dir / "doubling_control.json",
        {"languages": dbl, "vms_adjacent_token_repeats_per_1000": vms},
    )
    print(json.dumps({"languages": dbl, "vms": vms}, indent=1))


# -- jobs ----------------------------------------------------------------------


def vms_instances(args):
    out = []
    for tr in ("IT2a", "RF1b"):
        for d in ("A", "B"):
            for k in _ks(args):
                p = args.pres_dir / f"{tr}_{d}_{_kind(k)}.json"
                if p.exists():
                    out.append(p)
    return out


def control_instances(args):
    out = []
    for k in _ks(args):
        d = args.ctrl_dir / _kind(k)
        if not (d / "manifest.json").exists():
            continue
        for m in json.loads((d / "manifest.json").read_text()):
            out.append(d / m["file"])
    return out


def all_jobs(args, paths):
    jobs = []
    for p in paths:
        if args.only and not any(o in str(p) for o in args.only):
            continue
        rec = instance_record(p)
        js = make_jobs(
            rec,
            heads=(WORDHOM,),
            hypotheses=tuple(args.hyps),
            n_windows=args.n_windows,
            w5=args.w5,
            restarts={WORDHOM: args.restarts},
            units=args.units,
        )
        for j in js:
            j["sa_steps"] = args.sa_steps
        jobs += js
    return jobs


def _paths(args):
    return {"vms": vms_instances(args), "controls": control_instances(args)}[args.set]


def stage_solve(args):
    from diff_voyn.heads.wordhom import units_suffix

    settings = {
        k: getattr(args, k) for k in ("w5", "restarts", "sa_steps", "n_windows", "units")
    }
    run_solves(
        all_jobs(args, _paths(args)),
        args.out_dir / f"{args.set}_solves{units_suffix(args.units)}.json",
        workers=args.workers,
        settings=settings,
        fresh=args.fresh,
    )


def stage_score(args):
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator

    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(4)
    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    solves = json.loads((args.out_dir / f"{args.set}_solves.json").read_text())[
        "instances"
    ]
    if args.only:
        solves = [
            r
            for r in solves
            if any(o in r["instance"] + "_" + r["presentation"] for o in args.only)
        ]
    instances = {}
    for p in _paths(args):
        inst = load_instance(p)
        instances[(inst["name"], inst["kind"])] = inst
    i, n = (int(x) for x in args.shard.split("/"))
    path = args.out_dir / (
        f"{args.set}_scores_shard{i}of{n}.json" if n > 1 else f"{args.set}_scores.json"
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


# -- report --------------------------------------------------------------------


def _load_scores(out_dir: Path, which: str):
    recs, meta = {}, {}
    for p in sorted(out_dir.glob(f"{which}_scores*.json")):
        d = json.loads(p.read_text())
        meta = {k: v for k, v in d.items() if k != "instances"}
        for r in d["instances"]:
            recs[tuple(r[k] for k in KEY)] = r
    return meta, list(recs.values())


def _inst_meta(inst: dict) -> dict:
    """Uncovered tokens are charged at the character stream's own held-out
    n-gram cost (the Phase-6 convention); the unit is the character."""
    alpha = sorted(set("".join(inst["all_tokens"])))
    sym = np.array([alpha.index(c) for c in "".join(inst["all_tokens"])])
    return {
        "n_cipher_all": int(inst["coverage"]["n_chars"]),
        "order0_entropy_bits": order0_entropy_bits(sym),
        "coverage": inst["coverage"],
        "no_cipher_baselines": ciphertext_baselines(sym, int(sym.max()) + 1),
    }


def _extra_cell_fields(c: dict, r: dict, inst: dict):
    from diff_voyn.heads.wordhom import adjacency, rule_violations

    m = np.asarray(r["final"]["key"]["map"], dtype=np.int64)
    sym = np.asarray(inst["symbols"], dtype=np.int64)
    adj = adjacency(sym, np.asarray(inst["token_pos"], dtype=np.int64))
    c["violations_inner"] = rule_violations(m[sym], sym, adj)
    c["n_symbols"] = inst["n_symbols"]
    c["tokens_per_type"] = len(sym) / max(inst["n_symbols"], 1)
    c["n_hapax_in_key"] = int(
        sum(1 for s in np.bincount(np.asarray(inst["symbols"])) if s == 1)
    )
    truth = inst.get("truth", {})
    if truth.get("kind") == "wordhom":
        from diff_voyn.heads.wordhom import UnitTargets, expand_units, unit_ser

        m = np.asarray(r["final"]["key"]["map"], dtype=np.int64)
        sym = np.asarray(inst["symbols"], dtype=np.int64)
        tm = np.asarray(truth["sym_to_unit"], dtype=np.int64)
        c["unit_error_rate"] = float(np.mean(m[sym] != tm[sym]))
        c["type_error_rate"] = float(np.mean(m != tm))
        if r["hypothesis"] == truth["language"]:
            targets = UnitTargets.from_list(r["final"]["key"]["bigrams"])
            dec = expand_units(m[sym], targets)
            ref = expand_units(tm[sym], targets)
            c["ser_letters_covered"] = unit_ser(dec, ref)


def stage_report(args):
    table = CalibrationTable.load(args.primary, data_root())
    report = {
        "task": "post-6 wordhom study",
        "created_utc": datetime.now(UTC).isoformat(),
        "abstain_rule": ABSTAIN_RULE,
        "primary_calibration": args.primary,
    }
    md = [
        "### Word-level homophonic head (Boxer's hypothesis without the arithmetic)",
        "",
    ]
    dc = args.out_dir / "doubling_control.json"
    if dc.exists():
        report["doubling_control"] = json.loads(dc.read_text())
        md.append(
            "repeat-rule control — doubled units per 1000 in held-out text vs adjacent identical tokens per 1000 in the manuscript: "
            + json.dumps(report["doubling_control"])
        )
        md.append("")
    # manuscript
    meta, recs = _load_scores(args.out_dir, "vms")
    cells = []
    inst_cache = {}
    for r in recs:
        key = (r["instance"], r["presentation"])
        if key not in inst_cache:
            tr, d = r["instance"].split("/")
            inst_cache[key] = load_instance(
                args.pres_dir / f"{tr}_{d}_{r['presentation']}.json"
            )
        inst = inst_cache[key]
        c = cell_from_score(r, table, _inst_meta(inst))
        _extra_cell_fields(c, r, inst)
        cells.append(c)
    groups: dict[str, list[dict]] = {}
    for c in cells:
        groups.setdefault(c["instance"], []).append(c)
    tables = {k: rank_table(v) for k, v in groups.items()}
    report["vms"] = {
        "evaluator": meta.get("evaluator"),
        "cells": cells,
        "tables": tables,
    }
    for k in sorted(tables):
        md.append(fmt_table_md(k, tables[k], groups[k]))
        md.append(
            "per cell (presentation/hyp; tokens/type "
            + f"{groups[k][0]['tokens_per_type']:.1f}): "
            + "; ".join(
                f"{c['presentation']}/{c['hypothesis']}: plain {c['plain_bits']:.2f}, margin {c['structure_margin']:.2f}, coverage {c['coverage']:.2f}, violations {c['violations_inner']}, K {c['n_symbols']}"
                for c in sorted(
                    groups[k], key=lambda c: (c["presentation"], c["hypothesis"])
                )
            )
        )
        md.append("")
    # controls
    meta_c, recs_c = _load_scores(args.out_dir, "controls")
    ccells = []
    for r in recs_c:
        key = (r["instance"], r["presentation"])
        if key not in inst_cache:
            d = args.ctrl_dir / r["presentation"]
            man = {
                (m["name"], m["kind"]): m
                for m in json.loads((d / "manifest.json").read_text())
            }
            inst_cache[key] = load_instance(d / man[key]["file"])
            inst_cache[key]["_control"] = man[key]["control"]
        inst = inst_cache[key]
        c = cell_from_score(r, table, _inst_meta(inst))
        _extra_cell_fields(c, r, inst)
        c["control"] = inst["_control"]
        c["truth"] = {
            k: v
            for k, v in inst["truth"].items()
            if k not in ("plain_ids", "unit_ids", "sym_to_unit")
        }
        ccells.append(c)
    by_inst: dict[tuple, list[dict]] = {}
    for c in ccells:
        by_inst.setdefault((c["instance"], c["presentation"]), []).append(c)
    verdicts = []
    for (name, pres), cs in by_inst.items():
        tab = rank_table(cs)
        truth = cs[0]["truth"]
        like = [r for r in tab["ranked"] if r["language_like_any"]]
        call = like[0] if like else None
        own = [c for c in cs if c["hypothesis"] == truth["language"]]
        verdicts.append(
            {
                "instance": name,
                "presentation": pres,
                "control": cs[0]["control"],
                "truth_language": truth["language"],
                "truth_family": truth["family"],
                "abstain": tab["abstain"],
                "mdl_top_language": tab["ranked"][0]["hypothesis"],
                "called_language": call["hypothesis"] if call else None,
                "best_plain_bits": min(c["plain_bits"] for c in cs),
                "best_structure_margin": max(c["structure_margin"] for c in cs),
                "own_plain_bits": own[0]["plain_bits"] if own else None,
                "own_structure_margin": own[0]["structure_margin"] if own else None,
                "own_ser_letters": own[0].get("ser_letters_covered") if own else None,
                "own_unit_error_rate": own[0].get("unit_error_rate") if own else None,
                "coverage": cs[0]["coverage"],
            }
        )
    summary = {}
    for control in ("voynichesque", "shuffled", "contamination", "positive"):
        vs = [v for v in verdicts if v["control"] == control]
        if not vs:
            continue
        n = len(vs)
        k_abs = sum(v["abstain"] for v in vs)
        s = {"n": n, "abstain_rate": k_abs / n, "abstain_ci95": wilson(k_abs, n)}
        if control == "positive":
            s["language_correct_rate"] = float(
                np.mean([v["called_language"] == v["truth_language"] for v in vs])
            )
            s["mdl_top_correct_rate"] = float(
                np.mean([v["mdl_top_language"] == v["truth_language"] for v in vs])
            )
            sers = [
                v["own_ser_letters"] for v in vs if v["own_ser_letters"] is not None
            ]
            s["ser_letters_median"] = float(np.median(sers)) if sers else None
            s["ser_letters_max"] = float(np.max(sers)) if sers else None
        if control == "contamination":
            s["family_correct_rate_mdl_top"] = float(
                np.mean(
                    [FAM.get(v["mdl_top_language"]) == v["truth_family"] for v in vs]
                )
            )
        s["structure_margins"] = sorted(
            round(v["best_structure_margin"], 3) for v in vs
        )
        summary[control] = s
    report["controls"] = {
        "evaluator": meta_c.get("evaluator"),
        "summary": summary,
        "verdicts": verdicts,
        "cells": ccells,
    }
    md += [
        "#### controls",
        "",
        "| control | n | abstain rate (95% CI) | structure margins | notes |",
        "|---|---|---|---|---|",
    ]
    for k, s in summary.items():
        notes = ""
        if k == "positive":
            notes = f"language called correctly {s['language_correct_rate']:.2f}; MDL-top correct {s['mdl_top_correct_rate']:.2f}; letter SER median {s['ser_letters_median']} max {s['ser_letters_max']}"
        if k == "contamination":
            notes = f"family correct (MDL top) {s['family_correct_rate_mdl_top']:.2f}"
        md.append(
            f"| {k} | {s['n']} | {s['abstain_rate']:.2f} ({s['abstain_ci95'][0]:.2f}–{s['abstain_ci95'][1]:.2f}) | {s['structure_margins']} | {notes} |"
        )
    md += [
        "",
        "| instance | K | control | truth | abstain | called | MDL top | own plain bits | own margin | own letter SER | own unit err | coverage |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for v in sorted(
        verdicts, key=lambda v: (v["control"], v["instance"], v["presentation"])
    ):
        f = lambda x, nd=3: "" if x is None else round(x, nd)
        md.append(
            f"| {v['instance']} | {v['presentation']} | {v['control']} | {v['truth_language']} | {v['abstain']} | {v['called_language']} | {v['mdl_top_language']} | {f(v['own_plain_bits'])} | {f(v['own_structure_margin'])} | {f(v['own_ser_letters'])} | {f(v['own_unit_error_rate'])} | {f(v['coverage'],2)} |"
        )
    write_json_atomic(args.out_dir / "report.json", report)
    (args.out_dir / "report.md").write_text("\n".join(md))
    print("\n".join(md))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = data_root()
    p.add_argument(
        "--stage", choices=["prepare", "solve", "score", "report"], required=True
    )
    p.add_argument("--set", choices=["vms", "controls"], default="vms")
    p.add_argument("--out-dir", type=Path, default=root / "analysis" / "wordhom")
    p.add_argument(
        "--pres-dir", type=Path, default=root / "analysis" / "wordhom" / "presentations"
    )
    p.add_argument(
        "--ctrl-dir", type=Path, default=root / "analysis" / "wordhom" / "controls"
    )
    p.add_argument(
        "--k",
        nargs="+",
        default=["all"],
        help="top-K word types per instance ('all' = every type)",
    )
    p.add_argument("--hyps", nargs="+", default=list(LANGS))
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--per-language", type=int, default=3)
    p.add_argument("--length", type=int, default=8000)
    p.add_argument("--n-types", type=int, default=2500)
    p.add_argument(
        "--shapes",
        nargs="*",
        default=None,
        help="extra positive shapes tag:length:n_types (e.g. Alike:14000:5200)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--w5", type=int, default=12000)
    p.add_argument("--n-windows", type=int, default=1)
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument("--sa-steps", type=int, default=400_000)
    p.add_argument(
        "--units",
        default=None,
        help="wordhom unit-set spec for the solve stage (d5 default; d5b20 = doubles + "
        "top-20 bigrams, written to <set>_solves_d5b20.json)",
    )
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
