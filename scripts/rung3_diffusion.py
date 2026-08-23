"""Task 5.4 — rung 3 (Naibbe mixed unigram-bigram) on the frozen diffusion
evaluator: block-Sinkhorn inner search + fixed-parse polish, then the outer
tier (paired diffusion scoring of every restart's decode, ELBO pick,
expected-embedding refinement on the 2N-slot frame, polish, final).

Instances: held-out plaintext (``--chars`` letters, default 10k as in CH.6)
enciphered by the pinned ``naibbe_v2`` generator (aligned segments = the
Greshko ground truth); the glyph→letter key is the published apparatus
(``NaibbeParser.block_truth``), so map accuracy is code-level accuracy over
the 18 (state × table) bijections. All three languages are run (Naibbe's
23-letter alphabet pre-maps k→c, w→uu) so per-language solve success is
measured at matched difficulty.

Stages:  solve (CPU pool, one instance per worker) → score (GPU) → report
Artifacts: DATA_ROOT/analysis/phase5/rung3_{solves,scores,report}.*
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
KEY = ("language", "trial")
BLOCK_KEYS = None
_EV = None
_PARSER = None


def _build_ngram_evaluator():
    from diff_voyn.heads.evaluator import NgramEvaluator
    from diff_voyn.heads.harness import ngram_calibration_offsets
    from diff_voyn.heads.ngram import lm_dir, load_lm

    lms = {lang: load_lm(lm_dir() / f"{lang}.npz") for lang in LANGS}
    return NgramEvaluator(lms, calibration_offsets_bits=ngram_calibration_offsets(lms))


def maps_to_json(maps):
    return {f"{s}/{t}": np.asarray(v).tolist() for (s, t), v in maps.items()}


def maps_from_json(d):
    return {tuple(k.split("/")): np.asarray(v, dtype=np.int64) for k, v in d.items()}


def build_suite(root, trials, chars, seed):
    from diff_voyn.heads.synth import HeldoutSampler, gen_naibbe

    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    jobs = []
    for lang in LANGS:
        sampler = HeldoutSampler(corpus_dir, splits, lang)
        for trial in range(trials):
            key = f"phase5/rung3/{seed}/{lang}/{chars}/{trial}".encode()
            rng = np.random.default_rng(zlib.crc32(key))
            plain = sampler.sample(chars, rng)
            inst = gen_naibbe(plain, lang, rng)
            jobs.append(
                {
                    "language": lang,
                    "trial": trial,
                    "chars": chars,
                    "tokens": inst.tokens,
                    "plain_ids": inst.plain_ids.tolist(),
                    "cipher_seed": inst.cipher_seed,
                }
            )
    return jobs


def _solve_one(job):
    from diff_voyn.heads.rung3_naibbe import BlockResult, NaibbeBlockHead
    from diff_voyn.heads.rung4_arithmetic import levenshtein_ser

    t0 = time.time()
    plain = np.asarray(job["plain_ids"], dtype=np.int64)
    out = {k: job[k] for k in KEY}
    out.update(
        chars=job["chars"],
        tokens=job["tokens"],
        plain_ids=plain.tolist(),
        cipher_seed=job["cipher_seed"],
    )
    out["hypotheses"] = {}
    parses = _PARSER.parse_stream(job["tokens"])
    for hyp in job["hypotheses"]:
        t1 = time.time()
        head = NaibbeBlockHead(_EV, _PARSER, steps=job["steps"], seed=job["trial"])
        res = head.solve(
            job["tokens"], language=hyp, restarts=job["restarts"], polish=True
        )
        short = []
        for maps, score, src in res.shortlist:
            letters, _, _ = head.decode(parses, maps, hyp)
            short.append(
                {
                    "maps": maps_to_json(maps),
                    "dp_score": score,
                    "source": src,
                    "code_acc": BlockResult(maps, score, 0, 0).code_accuracy(_PARSER),
                    "decode": letters.tolist(),
                    "ser": levenshtein_ser(letters, plain),
                }
            )
        out["hypotheses"][hyp] = {
            "shortlist": short,
            "n_evals": res.n_evals,
            "seconds": round(time.time() - t1, 1),
        }
    out["solve_seconds"] = round(time.time() - t0, 1)
    return out


def stage_solve(args, root):
    global _EV, _PARSER
    from diff_voyn.heads.naibbe_parse import NaibbeParser

    path = args.out_dir / "rung3_solves.json"
    jobs = build_suite(root, args.trials, args.chars, args.seed)
    done = load_done(path, KEY) if not args.fresh else {}
    todo = [j for j in jobs if tuple(j[k] for k in KEY) not in done]
    hyps = list(LANGS) if args.all_hypotheses else None
    for j in todo:
        j.update(
            restarts=args.restarts, steps=args.steps, hypotheses=hyps or [j["language"]]
        )
    print(f"{len(done)} done, {len(todo)} to solve", flush=True)
    _EV = _build_ngram_evaluator()
    _PARSER = NaibbeParser()
    _PARSER.build_blocks()
    results = list(done.values())
    settings = {
        "trials": args.trials,
        "chars": args.chars,
        "restarts": args.restarts,
        "steps": args.steps,
        "seed": args.seed,
        "kind": "naibbe",
        "all_hypotheses": bool(args.all_hypotheses),
        "plaintext_source": "held-out split v1",
        "generator": "naibbe_v2 @ df3d074 (pinned)",
    }

    def on_result(i, r, el):
        results.append(r)
        s = r["hypotheses"][r["language"]]["shortlist"][0]
        print(
            f"  solved {i}/{len(todo)} ({el:.0f}s) {r['language']} t{r['trial']}: acc {s['code_acc']['all']:.3f} SER {s['ser']:.3f} ({r['solve_seconds']:.0f}s)",
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
    from diff_voyn.heads.naibbe_parse import NaibbeParser
    from diff_voyn.heads.rung3_naibbe import BlockResult, NaibbeBlockHead
    from diff_voyn.heads.rung4_arithmetic import levenshtein_ser
    from diff_voyn.heads.two_tier import Candidate, rescore, select

    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(4)
    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    ng = _build_ngram_evaluator()
    parser = NaibbeParser()
    parser.build_blocks()
    solves = json.loads((args.out_dir / "rung3_solves.json").read_text())
    path = args.out_dir / "rung3_scores.json"
    done = load_done(path, KEY) if not args.fresh else {}
    inst = [r for r in solves["instances"] if tuple(r[k] for k in KEY) not in done]
    results = list(done.values())
    meta = {
        "evaluator": ev.meta,
        "scoring": {
            "budget": args.budget,
            "refine_steps": args.refine_steps,
            "refine_lr": args.refine_lr,
            "refine_strata": args.refine_strata,
            "crn": "paired masks across candidates/conditions of an instance (equal-length decodes)",
        },
        "solve_settings": solves["settings"],
    }
    t0 = time.time()
    for i, r in enumerate(inst, 1):
        plain = np.asarray(r["plain_ids"], dtype=np.int64)
        parses = parser.parse_stream(r["tokens"])
        seed = zlib.crc32(f"score3/{r['language']}/{r['trial']}".encode()) % (2**31)
        rec = {k: r[k] for k in KEY}
        rec["hypotheses"] = {}
        for hyp, hdata in r["hypotheses"].items():
            head = NaibbeBlockHead(ng, parser, seed=r["trial"])
            cands = [
                Candidate(
                    decode=np.asarray(c["decode"]),
                    key=maps_from_json(c["maps"]),
                    inner_score=c["dp_score"],
                    source=c["source"],
                    extra={
                        "ser": c["ser"],
                        "code_acc": c["code_acc"]["all"],
                        "hyp": hyp,
                    },
                )
                for c in hdata["shortlist"]
            ]
            rescore(
                ev,
                cands,
                language=hyp,
                conditions=list(LANGS),
                n_strata=args.budget,
                seed=seed,
                batch=32,
            )
            pick = select(cands, language=hyp)
            t1 = time.time()
            ref_maps, losses = head.refine_frame(
                ev,
                parses,
                pick["diffusion"].key,
                hyp,
                steps=args.refine_steps,
                lr=args.refine_lr,
                n_strata=args.refine_strata,
                seed=seed,
            )
            changed = any(
                (ref_maps[k] != pick["diffusion"].key[k]).any() for k in ref_maps
            )
            ref_maps_p, dp, _ = head.polish(parses, ref_maps, hyp, rounds=1)
            letters, _, _ = head.decode(parses, ref_maps_p, hyp)
            ref = Candidate(
                decode=letters,
                key=ref_maps_p,
                inner_score=float(dp),
                source="refined+polish",
                extra={
                    "ser": levenshtein_ser(letters, plain),
                    "code_acc": BlockResult(ref_maps_p, dp, 0, 0).code_accuracy(parser)[
                        "all"
                    ],
                    "hyp": hyp,
                    "refine_changed": bool(changed),
                    "code_acc_refined_unpolished": BlockResult(
                        ref_maps, 0, 0, 0
                    ).code_accuracy(parser)["all"],
                },
            )
            rescore(
                ev,
                [ref],
                language=hyp,
                conditions=list(LANGS),
                n_strata=args.budget,
                seed=seed,
                batch=32,
            )
            final = min(cands + [ref], key=lambda c: c.bits[hyp])

            def d(c):
                return {**c.as_dict(), "maps": maps_to_json(c.key)}

            rec["hypotheses"][hyp] = {
                "n_candidates": len(cands),
                "ngram": d(pick["ngram"]),
                "diffusion": d(pick["diffusion"]),
                "oracle": d(pick["oracle"]),
                "refined": {
                    **d(ref),
                    "loss_first_last": [losses[0], losses[-1]] if losses else None,
                    "seconds": round(time.time() - t1, 1),
                },
                "final": d(final),
                "shortlist": [c.as_dict() for c in cands],
            }
            print(
                f"  [{i}/{len(inst)}] {r['language']} t{r['trial']} hyp={hyp}: acc ngram {pick['ngram'].extra['code_acc']:.3f} "
                f"diff {pick['diffusion'].extra['code_acc']:.3f} refined {ref.extra['code_acc']:.3f} final {final.extra['code_acc']:.3f} "
                f"(oracle {pick['oracle'].extra['code_acc']:.3f}) {time.time()-t0:.0f}s",
                flush=True,
            )
        results.append(rec)
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


def weighted_type_accuracy(head, parser, maps, parses) -> float:
    """Occurrence-weighted accuracy of the glyph-TYPE → letter map implied
    by the block maps (weights = type occurrences in the ciphertext, every
    parse branch of a token counted once)."""
    got, want = head._type_letter(maps), head._type_letter(parser.block_truth)
    num = den = 0.0
    for p in parses:
        if p.uni is not None:
            num += int(got["unigram"][p.uni] == want["unigram"][p.uni])
            den += 1
        for pre, suf in p.bi:
            num += int(got["prefix"][pre] == want["prefix"][pre])
            num += int(got["suffix"][suf] == want["suffix"][suf])
            den += 2
    return float(num / max(den, 1))


def stage_report(args, root):
    from diff_voyn.heads.naibbe_parse import NaibbeParser
    from diff_voyn.heads.rung3_naibbe import NaibbeBlockHead

    data = json.loads((args.out_dir / "rung3_scores.json").read_text())
    solves = {
        (r["language"], r["trial"]): r
        for r in json.loads((args.out_dir / "rung3_solves.json").read_text())[
            "instances"
        ]
    }
    parser = NaibbeParser()
    parser.build_blocks()
    head = NaibbeBlockHead(None, parser)
    table = CalibrationTable.load(args.primary, root)
    offs = table.additive_offsets()
    variants = ("ngram", "diffusion", "refined", "final", "oracle")
    rows, per_lang = [], {}
    for r in data["instances"]:
        t = r["language"]
        h = r["hypotheses"]
        parses = parser.parse_stream(solves[(t, r["trial"])]["tokens"])
        rec = {"language": t, "trial": r["trial"]}
        for v in variants:
            rec[f"acc_{v}"] = h[t][v]["code_acc"]
            rec[f"ser_{v}"] = h[t][v]["ser"]
            rec[f"wacc_{v}"] = (
                weighted_type_accuracy(
                    head, parser, maps_from_json(h[t][v]["maps"]), parses
                )
                if "maps" in h[t][v]
                else float("nan")
            )
        rec["refine_changed"] = h[t]["refined"].get("refine_changed")
        rec["acc_refined_unpolished"] = h[t]["refined"].get(
            "code_acc_refined_unpolished"
        )
        if len(h) == len(LANGS):
            fin = {hyp: h[hyp]["final"]["bits"][hyp] for hyp in LANGS}
            ranked = rank_languages(fin, offs)
            rec["rank_final"], rec["margin"] = ranked[0][0], ranked[1][1] - ranked[0][1]
        rows.append(rec)
        per_lang.setdefault(t, []).append(rec)
    cells = {}
    for lang, recs in per_lang.items():
        n = len(recs)
        e = {"n": n}
        for v in variants:
            e[f"acc_{v}"] = float(np.mean([x[f"acc_{v}"] for x in recs]))
            e[f"ge95_{v}"] = float(np.mean([x[f"acc_{v}"] >= 0.95 for x in recs]))
            e[f"ser_{v}"] = float(np.mean([x[f"ser_{v}"] for x in recs]))
            e[f"wacc_{v}"] = float(np.mean([x[f"wacc_{v}"] for x in recs]))
        e["refine_changed_rate"] = float(
            np.mean([bool(x["refine_changed"]) for x in recs])
        )
        e["refined_better_than_diffusion"] = float(
            np.mean([x["acc_refined"] > x["acc_diffusion"] for x in recs])
        )
        e["refined_worse_than_diffusion"] = float(
            np.mean([x["acc_refined"] < x["acc_diffusion"] for x in recs])
        )
        if all("rank_final" in x for x in recs):
            k = sum(x["rank_final"] == lang for x in recs)
            e["lang_acc_final"], e["lang_acc_final_ci95"] = k / n, wilson(k, n)
            e["family_acc_final"] = float(
                np.mean([family_of(x["rank_final"]) == family_of(lang) for x in recs])
            )
        cells[lang] = e
    acc = {
        "criterion": "≥95% letter-map accuracy on synthetic Naibbe pairs (occurrence-weighted glyph-type accuracy; unweighted code accuracy reported alongside); restart budget documented",
        "wacc_final_mean": float(np.mean([x["wacc_final"] for x in rows])),
        "wacc_ngram_mean": float(np.mean([x["wacc_ngram"] for x in rows])),
        "acc_final_mean": float(np.mean([x["acc_final"] for x in rows])),
        "acc_ngram_mean": float(np.mean([x["acc_ngram"] for x in rows])),
        "instances_ge95_final": float(np.mean([x["acc_final"] >= 0.95 for x in rows])),
        "instances_ge95_ngram": float(np.mean([x["acc_ngram"] >= 0.95 for x in rows])),
        "n": len(rows),
    }
    acc["pass"] = bool(acc["wacc_final_mean"] >= 0.95)
    report = {
        "task": "5.4",
        "created_utc": datetime.now(UTC).isoformat(),
        "evaluator": data["evaluator"],
        "scoring": data["scoring"],
        "solve_settings": data["solve_settings"],
        "primary_calibration": args.primary,
        "cells": cells,
        "instances": rows,
        "acceptance": acc,
    }
    write_json_atomic(args.out_dir / "rung3_report.json", report)
    md = [
        f"### Rung 3 (Naibbe, {data['solve_settings']['chars']} chars, {data['solve_settings']['restarts']} restarts × {data['solve_settings']['steps']} steps + polish) — two-tier",
        "",
        "| language | n | weighted acc n-gram → final | acc n-gram | acc diffusion pick | acc refined(+polish) | **acc final** | acc oracle | ≥95% final (n-gram) | SER final | refine changed/better/worse | lang acc final |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lang, c in cells.items():
        md.append(
            f"| {lang} | {c['n']} | {c['wacc_ngram']:.3f} → **{c['wacc_final']:.3f}** | {c['acc_ngram']:.3f} | {c['acc_diffusion']:.3f} | {c['acc_refined']:.3f} | **{c['acc_final']:.3f}** | {c['acc_oracle']:.3f} | "
            f"{c['ge95_final']:.0%} ({c['ge95_ngram']:.0%}) | {c['ser_final']:.3f} | {c['refine_changed_rate']:.0%}/{c['refined_better_than_diffusion']:.0%}/{c['refined_worse_than_diffusion']:.0%} | "
            f"{c.get('lang_acc_final', float('nan')):.0%} |"
        )
    md += [
        "",
        f"all: occurrence-weighted type accuracy final {acc['wacc_final_mean']:.3f} (n-gram {acc['wacc_ngram_mean']:.3f}); unweighted code accuracy final {acc['acc_final_mean']:.3f} (n-gram {acc['acc_ngram_mean']:.3f}); instances ≥95% unweighted: {acc['instances_ge95_final']:.0%} (n-gram {acc['instances_ge95_ngram']:.0%}) → **{'PASS' if acc['pass'] else 'FAIL'}**",
    ]
    md = "\n".join(md)
    (args.out_dir / "rung3_report.md").write_text(md)
    print(md)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = data_root()
    p.add_argument("--stage", choices=["solve", "score", "report"], required=True)
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--chars", type=int, default=10_000)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--steps", type=int, default=350)
    p.add_argument(
        "--all-hypotheses",
        action="store_true",
        help="solve under every language hypothesis (language probe)",
    )
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--refine-steps", type=int, default=40)
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
