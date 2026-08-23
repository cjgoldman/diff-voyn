"""Task 5.2 — rung 1 (1:1 substitution) on the frozen diffusion evaluator:
the two-tier protocol and its delta over the n-gram baseline.

Per (language × length × trial) instance built from HELD-OUT plaintext under
a random bijective key, and per language hypothesis:

  inner tier (CPU)  rung-1 Sinkhorn head on the frozen n-gram evaluator —
                    unchanged CH.3 search — now returning its SHORTLIST of
                    distinct local optima (restart argmaxes, ILS optima);
  outer tier (GPU)  every shortlist decode scored by the frozen diffusion
                    evaluator under every language condition with paired
                    masks (budget 64); the ELBO picks the map; the ELBO
                    winner is then refined through the backbone (soft
                    Sinkhorn map, expected-embedding inputs, R3) and the
                    refined map re-enters the shortlist.

Reported (prototyping doc §9 "measure the delta"): SER of the n-gram
winner vs the diffusion-shortlist winner vs the refined map vs the oracle
(best SER in the shortlist), per cell; language recovery of the calibrated
ranking over the final per-hypothesis maps (vs the Phase-3/4 protocol
that ranked n-gram winners); per-language solve success at matched
difficulty (the search-fairness question of the Phase-4 assessment).

Stages (resumable):  solve (CPU pool) → score (GPU) → report
    uv run python scripts/rung1_diffusion.py --stage solve --workers 12
    uv run python scripts/rung1_diffusion.py --stage score
    uv run python scripts/rung1_diffusion.py --stage report
Artifacts: DATA_ROOT/analysis/phase5/rung1_{solves,scores,report}.*
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
from diff_voyn.heads.ladder import (
    load_done,
    refine_assignment,
    run_pool,
    wilson,
    write_json_atomic,
)
from diff_voyn.metrology import (
    CALIBRATION_VERSION,
    CalibrationTable,
    family_of,
    rank_languages,
)

LANGS = tuple(LANG_TO_INDEX)
LENGTHS = (50, 100, 200, 400, 700)
KEY = ("language", "length", "trial")
_EV = None


def _build_ngram_evaluator():
    from diff_voyn.heads.evaluator import NgramEvaluator
    from diff_voyn.heads.harness import ngram_calibration_offsets
    from diff_voyn.heads.ngram import lm_dir, load_lm

    lms = {lang: load_lm(lm_dir() / f"{lang}.npz") for lang in LANGS}
    return NgramEvaluator(lms, calibration_offsets_bits=ngram_calibration_offsets(lms))


def build_suite(root: Path, trials: int, lengths, seed: int) -> list[dict]:
    from diff_voyn.heads.synth import HeldoutSampler, gen_substitution

    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    jobs = []
    for lang in LANGS:
        sampler = HeldoutSampler(corpus_dir, splits, lang)
        for L in lengths:
            for trial in range(trials):
                key = f"phase5/rung1/{seed}/{lang}/{L}/{trial}".encode()
                rng = np.random.default_rng(zlib.crc32(key))
                plain = sampler.sample(L, rng)
                c = gen_substitution(plain, lang, rng)
                jobs.append(
                    {
                        "language": lang,
                        "length": L,
                        "trial": trial,
                        "plain_ids": c.plain_ids.tolist(),
                        "cipher_ids": c.cipher_ids.tolist(),
                        "true_map": c.true_map.tolist(),
                    }
                )
    return jobs


def _solve_one(job: dict) -> dict:
    from diff_voyn.heads.rung1_sinkhorn import SinkhornSubstitutionHead

    cipher = np.asarray(job["cipher_ids"], dtype=np.int64)
    plain = np.asarray(job["plain_ids"], dtype=np.int64)
    out = {k: job[k] for k in KEY}
    out.update(
        plain_ids=plain.tolist(), cipher_ids=cipher.tolist(), true_map=job["true_map"]
    )
    out["hypotheses"] = {}
    t0 = time.time()
    for hyp in LANGS:
        head = SinkhornSubstitutionHead(_EV, seed=job["trial"])
        res = head.solve(
            cipher, language=hyp, restarts=job["restarts"], shortlist=job["shortlist"]
        )
        out["hypotheses"][hyp] = {
            "shortlist": [
                {
                    "perm": perm.tolist(),
                    "ngram_hard": hard,
                    "source": src,
                    "ser": float(np.mean(perm[cipher] != plain)),
                }
                for perm, hard, src in res.shortlist
            ],
            "n_evals": res.n_evals,
        }
    out["solve_seconds"] = round(time.time() - t0, 1)
    return out


def stage_solve(args, root):
    global _EV
    path = args.out_dir / "rung1_solves.json"
    jobs = build_suite(root, args.trials, args.lengths, args.seed)
    done = load_done(path, KEY) if not args.fresh else {}
    todo = [j for j in jobs if tuple(j[k] for k in KEY) not in done]
    for j in todo:
        j["restarts"], j["shortlist"] = args.restarts, args.shortlist
    todo.sort(key=lambda j: -j["length"])
    print(f"{len(done)} done, {len(todo)} to solve")
    _EV = _build_ngram_evaluator()
    results = list(done.values())
    settings = {
        "trials": args.trials,
        "lengths": list(args.lengths),
        "restarts": args.restarts,
        "shortlist": args.shortlist,
        "seed": args.seed,
        "kind": "sub1to1",
        "plaintext_source": "held-out split v1",
    }

    def on_result(i, r, el):
        results.append(r)
        if i % 10 == 0 or i == len(todo):
            ser = r["hypotheses"][r["language"]]["shortlist"][0]["ser"]
            print(
                f"  solved {i}/{len(todo)} ({el:.0f}s) last {r['language']} L={r['length']} SER={ser:.3f}",
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
    ng = _build_ngram_evaluator()  # exact n-gram score of refined maps (record only)
    solves = json.loads((args.out_dir / "rung1_solves.json").read_text())
    path = args.out_dir / "rung1_scores.json"
    done = load_done(path, KEY) if not args.fresh else {}
    inst = [r for r in solves["instances"] if tuple(r[k] for k in KEY) not in done]
    inst.sort(key=lambda r: (r["length"], r["language"], r["trial"]))
    print(f"{len(done)} scored, {len(inst)} to score")
    results = list(done.values())
    t0 = time.time()
    for i, r in enumerate(inst, 1):
        cipher = np.asarray(r["cipher_ids"], dtype=np.int64)
        plain = np.asarray(r["plain_ids"], dtype=np.int64)
        seed = zlib.crc32(
            f"score/{r['language']}/{r['length']}/{r['trial']}".encode()
        ) % (2**31)
        rec = {k: r[k] for k in KEY}
        rec["hypotheses"] = {}
        cands_all = []
        for hyp in LANGS:
            cands = [
                Candidate(
                    decode=np.asarray(c["perm"])[cipher],
                    key=np.asarray(c["perm"]),
                    inner_score=c["ngram_hard"],
                    source=c["source"],
                    extra={"ser": c["ser"], "hyp": hyp},
                )
                for c in r["hypotheses"][hyp]["shortlist"]
            ]
            cands_all.extend(cands)
        rescore(
            ev,
            cands_all,
            language=r["language"],
            conditions=list(LANGS),
            n_strata=args.budget,
            seed=seed,
            batch=96,
        )
        for hyp in LANGS:
            cands = [c for c in cands_all if c.extra["hyp"] == hyp]
            pick = select(cands, language=hyp)
            t1 = time.time()
            refined_perm, losses = refine_assignment(
                ev,
                cipher,
                pick["diffusion"].key,
                language=hyp,
                bijective=True,
                steps=args.refine_steps,
                lr=args.refine_lr,
                n_strata=args.refine_strata,
                seed=seed,
            )
            ref = Candidate(
                decode=refined_perm[cipher],
                key=refined_perm,
                inner_score=float(
                    ng.score_hard(refined_perm[cipher], language=hyp, order=5)
                ),
                source="refined",
                extra={
                    "ser": float(np.mean(refined_perm[cipher] != plain)),
                    "hyp": hyp,
                },
            )
            rescore(
                ev,
                [ref],
                language=hyp,
                conditions=list(LANGS),
                n_strata=args.budget,
                seed=seed,
            )
            final = min(cands + [ref], key=lambda c: c.bits[hyp])
            rec["hypotheses"][hyp] = {
                "n_candidates": len(cands),
                "ngram": pick["ngram"].as_dict(),
                "diffusion": pick["diffusion"].as_dict(),
                "oracle": pick["oracle"].as_dict(),
                "refined": {
                    **ref.as_dict(),
                    "changed": bool((refined_perm != pick["diffusion"].key).any()),
                    "loss_first_last": [losses[0], losses[-1]] if losses else None,
                    "seconds": round(time.time() - t1, 1),
                },
                "final": {**final.as_dict(), "ser": final.extra["ser"]},
                "shortlist": [c.as_dict() for c in cands],
            }
        results.append(rec)
        if i % 5 == 0 or i == len(inst):
            print(
                f"  scored {i}/{len(inst)} ({time.time()-t0:.0f}s) {r['language']} L={r['length']}",
                flush=True,
            )
            write_json_atomic(
                path,
                {
                    "created_utc": datetime.now(UTC).isoformat(),
                    "evaluator": ev.meta,
                    "scoring": {
                        "budget": args.budget,
                        "refine_steps": args.refine_steps,
                        "refine_lr": args.refine_lr,
                        "refine_strata": args.refine_strata,
                        "crn": "paired masks across all candidates and conditions of an instance",
                    },
                    "solve_settings": solves["settings"],
                    "instances": results,
                },
            )
    write_json_atomic(
        path,
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "evaluator": ev.meta,
            "scoring": {
                "budget": args.budget,
                "refine_steps": args.refine_steps,
                "refine_lr": args.refine_lr,
                "refine_strata": args.refine_strata,
                "crn": "paired masks across all candidates and conditions of an instance",
            },
            "solve_settings": solves["settings"],
            "instances": results,
        },
    )
    print(f"written {path}")


def stage_report(args, root):
    data = json.loads((args.out_dir / "rung1_scores.json").read_text())
    table = CalibrationTable.load(args.primary, root)
    offs = table.additive_offsets()
    inst = data["instances"]
    cells = {}
    variants = ("ngram", "diffusion", "refined", "final", "oracle")
    for r in inst:
        t = r["language"]
        h = r["hypotheses"]
        rec = {v: h[t][v]["ser"] for v in variants}
        rec["refine_changed"] = h[t]["refined"]["changed"]
        rec["diffusion_changed_choice"] = (
            h[t]["diffusion"]["ser"] != h[t]["ngram"]["ser"]
            or h[t]["diffusion"]["bits"][t] != h[t]["ngram"]["bits"][t]
        )
        # language ranking on final maps (calibrated own-condition bits)
        fin = {hyp: h[hyp]["final"]["bits"][hyp] for hyp in LANGS}
        ng = {hyp: h[hyp]["ngram"]["bits"][hyp] for hyp in LANGS}
        rec["rank_final"] = rank_languages(fin, offs)[0][0]
        rec["rank_ngram_winner"] = rank_languages(ng, offs)[0][0]
        ranked = rank_languages(fin, offs)
        rec["margin"] = ranked[1][1] - ranked[0][1]
        cells.setdefault((t, r["length"]), []).append(rec)
    out_cells, by_len = {}, {}
    for (lang, L), recs in sorted(cells.items()):
        n = len(recs)
        e = {"n": n}
        for v in variants:
            e[f"ser_{v}"] = float(np.mean([x[v] for x in recs]))
            e[f"solved_{v}"] = float(np.mean([x[v] < 0.05 for x in recs]))
        e["diffusion_better_than_ngram"] = float(
            np.mean([x["diffusion"] < x["ngram"] for x in recs])
        )
        e["diffusion_worse_than_ngram"] = float(
            np.mean([x["diffusion"] > x["ngram"] for x in recs])
        )
        e["refine_changed_rate"] = float(np.mean([x["refine_changed"] for x in recs]))
        e["refined_better_than_diffusion"] = float(
            np.mean([x["refined"] < x["diffusion"] for x in recs])
        )
        e["refined_worse_than_diffusion"] = float(
            np.mean([x["refined"] > x["diffusion"] for x in recs])
        )
        for name in ("final", "ngram_winner"):
            k = sum(x[f"rank_{name}"] == lang for x in recs)
            kf = sum(family_of(x[f"rank_{name}"]) == family_of(lang) for x in recs)
            e[f"lang_acc_{name}"] = k / n
            e[f"lang_acc_{name}_ci95"] = wilson(k, n)
            e[f"family_acc_{name}"] = kf / n
        out_cells[f"{lang}/L{L}"] = e
        a = by_len.setdefault(
            L, {"n": 0, "final": 0, "ngram_winner": 0, "fam_final": 0}
        )
        a["n"] += n
        a["final"] += sum(x["rank_final"] == lang for x in recs)
        a["ngram_winner"] += sum(x["rank_ngram_winner"] == lang for x in recs)
        a["fam_final"] += sum(
            family_of(x["rank_final"]) == family_of(lang) for x in recs
        )
    ge200 = [k for k in out_cells if int(k.split("/L")[1]) >= 200]
    acc = {
        "ser_final_ge200": float(np.mean([out_cells[k]["ser_final"] for k in ge200])),
        "ser_ngram_ge200": float(np.mean([out_cells[k]["ser_ngram"] for k in ge200])),
        "solved_final_ge200": float(
            np.mean([out_cells[k]["solved_final"] for k in ge200])
        ),
        "lang_acc_final_ge200": float(
            np.mean([out_cells[k]["lang_acc_final"] for k in ge200])
        ),
        "lang_acc_ngram_winner_ge200": float(
            np.mean([out_cells[k]["lang_acc_ngram_winner"] for k in ge200])
        ),
    }
    acc["pass"] = bool(
        acc["ser_final_ge200"] <= 0.02 and acc["lang_acc_final_ge200"] >= 0.971
    )
    per_lang = {}
    for lang in LANGS:
        ks = [k for k in ge200 if k.startswith(lang)]
        per_lang[lang] = {
            "solved_ngram_ge200": float(
                np.mean([out_cells[k]["solved_ngram"] for k in ks])
            ),
            "solved_final_ge200": float(
                np.mean([out_cells[k]["solved_final"] for k in ks])
            ),
            "ser_final_ge200": float(np.mean([out_cells[k]["ser_final"] for k in ks])),
        }
    report = {
        "task": "5.2",
        "created_utc": datetime.now(UTC).isoformat(),
        "evaluator": data["evaluator"],
        "scoring": data["scoring"],
        "solve_settings": data["solve_settings"],
        "primary_calibration": args.primary,
        "policy": table.policy,
        "cells": out_cells,
        "by_length": {
            str(L): {
                "n": a["n"],
                "lang_acc_final": a["final"] / a["n"],
                "family_acc_final": a["fam_final"] / a["n"],
                "lang_acc_ngram_winner": a["ngram_winner"] / a["n"],
            }
            for L, a in sorted(by_len.items())
        },
        "per_language_ge200": per_lang,
        "acceptance": {
            "criterion": "near-perfect recovery on synthetic 1:1 at ≥200 chars: mean SER ≤ 2% and calibrated language recovery ≥ 97.1%",
            **acc,
        },
    }
    write_json_atomic(args.out_dir / "rung1_report.json", report)
    md = render_md(report)
    (args.out_dir / "rung1_report.md").write_text(md)
    print(md)


def render_md(rep):
    lines = [
        "### Rung 1 (1:1 substitution) — two-tier on the frozen diffusion evaluator",
        "",
        "| language | L | n | SER n-gram | SER diffusion shortlist | SER refined | **SER final** | SER oracle | diff. better / worse | refine changed / better / worse | lang acc final | lang acc (n-gram winners) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, c in rep["cells"].items():
        lang, L = key.split("/L")
        lines.append(
            f"| {lang} | {L} | {c['n']} | {c['ser_ngram']:.3f} | {c['ser_diffusion']:.3f} | {c['ser_refined']:.3f} | **{c['ser_final']:.3f}** | {c['ser_oracle']:.3f} | "
            f"{c['diffusion_better_than_ngram']:.0%} / {c['diffusion_worse_than_ngram']:.0%} | {c['refine_changed_rate']:.0%} / {c['refined_better_than_diffusion']:.0%} / {c['refined_worse_than_diffusion']:.0%} | "
            f"**{c['lang_acc_final']:.1%}** | {c['lang_acc_ngram_winner']:.1%} |"
        )
    lines += [
        "",
        "| L | n | lang acc final / family | lang acc n-gram winners |",
        "|---|---|---|---|",
    ]
    for L, d in rep["by_length"].items():
        lines.append(
            f"| {L} | {d['n']} | {d['lang_acc_final']:.1%} / {d['family_acc_final']:.1%} | {d['lang_acc_ngram_winner']:.1%} |"
        )
    a = rep["acceptance"]
    lines += [
        "",
        (
            f"≥200: SER final {a['ser_final_ge200']:.4f} (n-gram {a['ser_ngram_ge200']:.4f}), solved {a['solved_final_ge200']:.1%}, "
            f"language recovery {a['lang_acc_final_ge200']:.1%} (n-gram winners {a['lang_acc_ngram_winner_ge200']:.1%}) → **{'PASS' if a['pass'] else 'FAIL'}**"
        ),
        "",
        "per-language ≥200 (search fairness): "
        + "; ".join(
            f"{l}: solved n-gram {v['solved_ngram_ge200']:.0%} → final {v['solved_final_ge200']:.0%}"
            for l, v in rep["per_language_ge200"].items()
        ),
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = data_root()
    p.add_argument("--stage", choices=["solve", "score", "report"], required=True)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--shortlist", type=int, default=8)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--refine-steps", type=int, default=20)
    p.add_argument("--refine-lr", type=float, default=0.1)
    p.add_argument("--refine-strata", type=int, default=4)
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
