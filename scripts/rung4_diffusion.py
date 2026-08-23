"""Task 5.5 — rung 4 (arithmetic sum-to-target, pseudo-VMS) on the frozen
diffusion evaluator.

The rung-4 head's segmentation is latent over CHAR positions (the lattice
DP is n-gram-only), so the outer tier here is shortlist re-ranking of hard
decodes: every restart's final key (per language hypothesis) is Viterbi-
decoded, every decode is scored by the frozen diffusion evaluator under
every language condition (paired masks for equal-length decodes), and the
calibrated own-condition bits of the best decode per hypothesis rank the
languages. Acceptance 5.5: language recovery better than family-random on
synthetic pseudo-VMS ciphers (0.7 pinned per-language tables).

Stages:  solve (CPU pool, one (instance × hypothesis) per worker) → score → report
Artifacts: DATA_ROOT/analysis/phase5/rung4_{solves,scores,report}.*
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if "solve" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX
from diff_voyn.heads.ladder import load_done, run_pool, wilson, write_json_atomic
from diff_voyn.metrology import (
    CALIBRATION_VERSION,
    CalibrationTable,
    family_of,
    rank_languages,
)

LANGS = tuple(LANG_TO_INDEX)
KEY = ("language", "trial", "hypothesis")
_EV = None


def _build_ngram_evaluator():
    from diff_voyn.heads.evaluator import NgramEvaluator
    from diff_voyn.heads.harness import ngram_calibration_offsets
    from diff_voyn.heads.ngram import lm_dir, load_lm

    lms = {lang: load_lm(lm_dir() / f"{lang}.npz") for lang in LANGS}
    return NgramEvaluator(lms, calibration_offsets_bits=ngram_calibration_offsets(lms))


def build_suite(root, trials, letters, seed):
    from diff_voyn.ciphers.arithmetic import ArithmeticCipher
    from diff_voyn.heads.synth import HeldoutSampler, gen_arithmetic

    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    jobs = []
    for lang in LANGS:
        sampler = HeldoutSampler(corpus_dir, splits, lang)
        cipher = ArithmeticCipher(
            table_path=root / "ciphers" / f"pseudo_vms_{lang}.csv"
        )
        for trial in range(trials):
            key = f"phase5/rung4/{seed}/{lang}/{letters}/{trial}".encode()
            rng = np.random.default_rng(zlib.crc32(key))
            plain = sampler.sample(letters, rng)
            inst = gen_arithmetic(plain, lang, cipher, rng)
            base = {
                "language": lang,
                "trial": trial,
                "letters": letters,
                "plain_ids": inst.plain_ids.tolist(),
                "char_ids": inst.char_ids.tolist(),
                "true_v": inst.true_v.tolist(),
                "true_u": inst.true_u.tolist(),
                "token_starts": inst.token_starts.tolist(),
            }
            for hyp in LANGS:
                jobs.append({**base, "hypothesis": hyp})
    return jobs


def _solve_one(job):
    from diff_voyn.heads.rung4_arithmetic import ArithmeticHead, levenshtein_ser

    t0 = time.time()
    char_ids = np.asarray(job["char_ids"], dtype=np.int64)
    plain = np.asarray(job["plain_ids"], dtype=np.int64)
    true_v, true_u = np.asarray(job["true_v"]), np.asarray(job["true_u"])
    head = ArithmeticHead(_EV, seed=job["trial"])
    res = head.solve(char_ids, language=job["hypothesis"], restarts=job["restarts"])
    occ = np.bincount(plain, minlength=len(true_u))
    short = []
    for v, u, dec, score, raw_ll, rank in res.shortlist:
        short.append(
            {
                "v": v.tolist(),
                "u": u.tolist(),
                "decode": dec.tolist(),
                "score": score,
                "raw_ll": raw_ll,
                "ser": levenshtein_ser(dec, plain),
                "v_acc": float(np.mean(v == true_v)),
                "u_acc": float(((u == true_u) * occ).sum() / max(occ.sum(), 1)),
                "ngram_calibrated_bits": _EV.calibrated_bits_per_char(
                    raw_ll, len(dec), job["hypothesis"]
                ),
            }
        )
    out = {k: job[k] for k in KEY}
    out.update(
        letters=job["letters"],
        plain_ids=plain.tolist(),
        char_ids=char_ids.tolist(),
        true_v=true_v.tolist(),
        true_u=true_u.tolist(),
        shortlist=short,
        n_evals=res.n_evals,
        solve_seconds=round(time.time() - t0, 1),
    )
    return out


def stage_solve(args, root):
    global _EV
    path = args.out_dir / "rung4_solves.json"
    jobs = build_suite(root, args.trials, args.letters, args.seed)
    done = load_done(path, KEY) if not args.fresh else {}
    todo = [j for j in jobs if tuple(j[k] for k in KEY) not in done]
    for j in todo:
        j["restarts"] = args.restarts
    print(f"{len(done)} done, {len(todo)} to solve", flush=True)
    _EV = _build_ngram_evaluator()
    results = list(done.values())
    settings = {
        "trials": args.trials,
        "letters": args.letters,
        "restarts": args.restarts,
        "seed": args.seed,
        "kind": "arithmetic",
        "plaintext_source": "held-out split v1",
        "generator": "voynpy.pseudo_vms @ e324bee, Phase-0 per-language tuned tables",
    }

    def on_result(i, r, el):
        results.append(r)
        s = r["shortlist"][0]
        print(
            f"  solved {i}/{len(todo)} ({el:.0f}s) {r['language']} t{r['trial']} hyp={r['hypothesis']}: SER {s['ser']:.3f} u_acc {s['u_acc']:.3f} ({r['solve_seconds']:.0f}s)",
            flush=True,
        )
        write_json_atomic(
            path,
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "settings": settings,
                "instances": results,
            },
        )

    run_pool(_solve_one, todo, workers=args.workers, on_result=on_result)
    write_json_atomic(
        path,
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "settings": settings,
            "instances": results,
        },
    )
    print(f"written {path}")


def stage_score(args, root):
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator
    from diff_voyn.heads.two_tier import Candidate, rescore, select

    torch.set_float32_matmul_precision("high")
    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    solves = json.loads((args.out_dir / "rung4_solves.json").read_text())
    by_inst = {}
    for r in solves["instances"]:
        by_inst.setdefault((r["language"], r["trial"]), {})[r["hypothesis"]] = r
    path = args.out_dir / "rung4_scores.json"
    done = load_done(path, ("language", "trial")) if not args.fresh else {}
    results = list(done.values())
    meta = {
        "evaluator": ev.meta,
        "scoring": {
            "budget": args.budget,
            "crn": "paired masks for equal-length decodes",
        },
        "solve_settings": solves["settings"],
    }
    t0 = time.time()
    todo = [
        (k, v)
        for k, v in sorted(by_inst.items())
        if k not in done and len(v) == len(LANGS)
    ]
    for i, ((lang, trial), hyps) in enumerate(todo, 1):
        seed = zlib.crc32(f"score4/{lang}/{trial}".encode()) % (2**31)
        rec = {"language": lang, "trial": trial, "hypotheses": {}}
        cands_all = []
        for hyp, r in hyps.items():
            for c in r["shortlist"]:
                cands_all.append(
                    Candidate(
                        decode=np.asarray(c["decode"], dtype=np.int64),
                        key=(c["v"], c["u"]),
                        inner_score=c["score"],
                        source="restart",
                        extra={
                            "ser": c["ser"],
                            "u_acc": c["u_acc"],
                            "hyp": hyp,
                            "raw_ll": c["raw_ll"],
                            "ngram_calibrated_bits": c["ngram_calibrated_bits"],
                        },
                    )
                )
        rescore(
            ev,
            cands_all,
            language=lang,
            conditions=list(LANGS),
            n_strata=args.budget,
            seed=seed,
            batch=96,
        )
        for hyp in LANGS:
            cands = [c for c in cands_all if c.extra["hyp"] == hyp]
            pick = select(cands, language=hyp)
            rec["hypotheses"][hyp] = {
                "n_candidates": len(cands),
                "ngram": pick["ngram"].as_dict(),
                "diffusion": pick["diffusion"].as_dict(),
                "oracle": pick["oracle"].as_dict(),
                "final": pick["diffusion"].as_dict(),
                "shortlist": [c.as_dict() for c in cands],
            }
        results.append(rec)
        print(
            f"  scored {i}/{len(todo)} ({time.time()-t0:.0f}s) {lang} t{trial}",
            flush=True,
        )
        write_json_atomic(
            path,
            {
                "created_utc": datetime.now(UTC).isoformat(),
                **meta,
                "instances": results,
            },
        )
    write_json_atomic(
        path,
        {"created_utc": datetime.now(UTC).isoformat(), **meta, "instances": results},
    )


def stage_report(args, root):
    data = json.loads((args.out_dir / "rung4_scores.json").read_text())
    table = CalibrationTable.load(args.primary, root)
    offs = table.additive_offsets()
    rows, per_lang = [], {}
    for r in data["instances"]:
        t = r["language"]
        h = r["hypotheses"]
        rec = {"language": t, "trial": r["trial"]}
        for v in ("ngram", "diffusion", "oracle"):
            rec[f"ser_{v}"] = h[t][v]["ser"]
            rec[f"u_acc_{v}"] = h[t][v]["u_acc"]
        fin = {hyp: h[hyp]["final"]["bits"][hyp] for hyp in LANGS}
        ranked = rank_languages(fin, offs)
        rec["rank_final"], rec["margin"] = ranked[0][0], ranked[1][1] - ranked[0][1]
        rec["rank_ngram_excess"] = min(
            LANGS, key=lambda hyp: h[hyp]["ngram"]["ngram_calibrated_bits"]
        )
        rec["calibrated_bits"] = {hyp: dict(ranked)[hyp] for hyp in LANGS}
        rows.append(rec)
        per_lang.setdefault(t, []).append(rec)
    cells = {}
    for lang, recs in per_lang.items():
        n = len(recs)
        k = sum(x["rank_final"] == lang for x in recs)
        kf = sum(family_of(x["rank_final"]) == family_of(lang) for x in recs)
        cells[lang] = {
            "n": n,
            "lang_acc_final": k / n,
            "lang_acc_ci95": wilson(k, n),
            "family_acc_final": kf / n,
            "lang_acc_ngram_excess": float(
                np.mean([x["rank_ngram_excess"] == lang for x in recs])
            ),
            **{
                f"ser_{v}": float(np.mean([x[f"ser_{v}"] for x in recs]))
                for v in ("ngram", "diffusion", "oracle")
            },
            **{
                f"u_acc_{v}": float(np.mean([x[f"u_acc_{v}"] for x in recs]))
                for v in ("ngram", "diffusion", "oracle")
            },
            "margin_median": float(np.median([x["margin"] for x in recs])),
        }
    n = len(rows)
    k = sum(x["rank_final"] == x["language"] for x in rows)
    kf = sum(family_of(x["rank_final"]) == family_of(x["language"]) for x in rows)
    acc = {
        "criterion": "language recovery better than family-random on synthetic pseudo-VMS ciphers",
        "n": n,
        "lang_acc_final": k / n,
        "lang_acc_ci95": wilson(k, n),
        "family_acc_final": kf / n,
        "language_random": 1 / 3,
        "family_random": 5 / 9,
        "lang_acc_ngram_excess": float(
            np.mean([x["rank_ngram_excess"] == x["language"] for x in rows])
        ),
    }
    acc["pass"] = bool(
        acc["family_acc_final"] > 5 / 9 and acc["lang_acc_final"] > 1 / 3
    )
    report = {
        "task": "5.5",
        "created_utc": datetime.now(UTC).isoformat(),
        "evaluator": data["evaluator"],
        "scoring": data["scoring"],
        "solve_settings": data["solve_settings"],
        "primary_calibration": args.primary,
        "cells": cells,
        "instances": rows,
        "acceptance": acc,
    }
    write_json_atomic(args.out_dir / "rung4_report.json", report)
    md = [
        f"### Rung 4 (arithmetic pseudo-VMS, {data['solve_settings']['letters']} letters, {data['solve_settings']['restarts']} restarts) — diffusion shortlist re-ranking",
        "",
        "| language | n | SER n-gram pick | SER diffusion pick | SER oracle | u-map acc n-gram / diffusion / oracle | lang acc final (CI) | family acc | lang acc n-gram excess bits | margin median |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lang, c in cells.items():
        md.append(
            f"| {lang} | {c['n']} | {c['ser_ngram']:.3f} | {c['ser_diffusion']:.3f} | {c['ser_oracle']:.3f} | {c['u_acc_ngram']:.3f} / {c['u_acc_diffusion']:.3f} / {c['u_acc_oracle']:.3f} | "
            f"**{c['lang_acc_final']:.0%}** ({c['lang_acc_ci95'][0]:.2f}–{c['lang_acc_ci95'][1]:.2f}) | {c['family_acc_final']:.0%} | {c['lang_acc_ngram_excess']:.0%} | {c['margin_median']:.3f} |"
        )
    md += [
        "",
        "| instance | calibrated bits/char (latin / italian / german) | winner | margin |",
        "|---|---|---|---|",
    ]
    for x in rows:
        b = x["calibrated_bits"]
        md.append(
            f"| {x['language']} t{x['trial']} | {b['latin']:.3f} / {b['italian']:.3f} / {b['german']:.3f} | {x['rank_final']} | {x['margin']:.3f} |"
        )
    md += [
        "",
        f"all: language top-1 {acc['lang_acc_final']:.1%} (random 33%), family top-1 {acc['family_acc_final']:.1%} (random 56%); n-gram excess-bits ranking {acc['lang_acc_ngram_excess']:.1%} → **{'PASS' if acc['pass'] else 'FAIL'}**",
    ]
    md = "\n".join(md)
    (args.out_dir / "rung4_report.md").write_text(md)
    print(md)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = data_root()
    p.add_argument("--stage", choices=["solve", "score", "report"], required=True)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--letters", type=int, default=300)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--primary", default=CALIBRATION_VERSION)
    p.add_argument("--out-dir", type=Path, default=root / "analysis" / "phase5")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    {"solve": stage_solve, "score": stage_score, "report": stage_report}[args.stage](
        args, root
    )


if __name__ == "__main__":
    main()
