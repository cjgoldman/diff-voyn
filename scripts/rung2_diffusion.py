"""Task 5.3 — rung 2 (unigram homophonic) on the frozen diffusion evaluator.

Zodiac-408-class synthetics (408 held-out chars, 54 symbols, homophones
allocated by letter frequency) — the literature's anchor class; the real
Zodiac-408 plaintext is English, outside the frozen language inventory, so
the anchor itself cannot be scored by this instrument (recorded, task 6.6).

Per instance and language hypothesis:
  inner tier (CPU)  rung-2 SA on the penalized pentagram objective
                    (CH.5; restarts fanned over forked workers) → shortlist
                    of distinct restart optima; PLUS a few restarts on the
                    UNPENALIZED LM objective — the hyper-likely degenerate
                    maps the n-gram objective prefers (CH.5 finding) — so
                    the outer tier is tested against them: does the
                    diffusion ELBO also prefer repetitive junk to language?
  outer tier (GPU)  paired diffusion scoring of every candidate under every
                    condition; ELBO pick; soft refinement of the pick
                    (row-stochastic map, expected embeddings, R3); final.

Stages:  solve → score → report (resumable); artifacts
DATA_ROOT/analysis/phase5/rung2_{solves,scores,report}.*
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
    elbo_polish,
    load_done,
    refine_assignment,
    wilson,
    write_json_atomic,
)
from diff_voyn.heads.scale import choice_bits
from diff_voyn.metrology import (
    CALIBRATION_VERSION,
    CalibrationTable,
    family_of,
    rank_languages,
)

LANGS = tuple(LANG_TO_INDEX)
KEY = ("language", "length", "trial")


def _build_ngram_evaluator():
    from diff_voyn.heads.evaluator import NgramEvaluator
    from diff_voyn.heads.harness import ngram_calibration_offsets
    from diff_voyn.heads.ngram import lm_dir, load_lm

    lms = {lang: load_lm(lm_dir() / f"{lang}.npz") for lang in LANGS}
    return NgramEvaluator(lms, calibration_offsets_bits=ngram_calibration_offsets(lms))


def build_suite(root, trials, length, n_symbols, seed):
    from diff_voyn.heads.synth import HeldoutSampler, gen_homophonic

    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    jobs = []
    for lang in LANGS:
        sampler = HeldoutSampler(corpus_dir, splits, lang)
        for trial in range(trials):
            key = f"phase5/rung2/{seed}/{lang}/{length}/{trial}".encode()
            rng = np.random.default_rng(zlib.crc32(key))
            plain = sampler.sample(length, rng)
            c = gen_homophonic(plain, lang, rng, n_symbols=n_symbols)
            jobs.append(
                {
                    "language": lang,
                    "length": length,
                    "trial": trial,
                    "n_symbols": n_symbols,
                    "plain_ids": c.plain_ids.tolist(),
                    "cipher_ids": c.cipher_ids.tolist(),
                    "true_map": c.true_map.tolist(),
                }
            )
    return jobs


def stage_solve(args, root):
    from diff_voyn.heads.rung2_homophonic import HomophonicHead

    torch.set_num_threads(1)
    ev = _build_ngram_evaluator()
    path = args.out_dir / "rung2_solves.json"
    jobs = build_suite(root, args.trials, args.length, args.n_symbols, args.seed)
    done = load_done(path, KEY) if not args.fresh else {}
    todo = [j for j in jobs if tuple(j[k] for k in KEY) not in done]
    if args.only:  # targeted re-solve (e.g. a larger restart budget for basin misses)
        keys = {(o.split("/")[0], int(o.split("/")[1])) for o in args.only}
        todo = [j for j in jobs if (j["language"], j["trial"]) in keys]
        done = {k: v for k, v in done.items() if (k[0], k[2]) not in keys}
    print(f"{len(done)} done, {len(todo)} to solve", flush=True)
    results = list(done.values())
    settings = {
        "trials": args.trials,
        "length": args.length,
        "n_symbols": args.n_symbols,
        "restarts": args.restarts,
        "restarts_nopenalty": args.restarts_nopenalty,
        "sa_steps": args.sa_steps,
        "shortlist": args.shortlist,
        "seed": args.seed,
        "kind": "homophonic",
        "plaintext_source": "held-out split v1",
    }
    t0 = time.time()
    for i, job in enumerate(todo, 1):
        cipher = np.asarray(job["cipher_ids"], dtype=np.int64)
        plain = np.asarray(job["plain_ids"], dtype=np.int64)
        rec = {k: job[k] for k in KEY}
        rec.update(
            n_symbols=job["n_symbols"],
            plain_ids=plain.tolist(),
            cipher_ids=cipher.tolist(),
            true_map=job["true_map"],
        )
        rec["hypotheses"] = {}
        for hyp in LANGS:
            t1 = time.time()
            head = HomophonicHead(ev, seed=job["trial"])
            res = head.solve_parallel(
                cipher,
                job["n_symbols"],
                language=hyp,
                restarts=args.restarts,
                workers=args.workers,
                sa_steps=args.sa_steps,
                shortlist=args.shortlist,
            )
            short = [
                {
                    "map": m.tolist(),
                    "penalized": s,
                    "raw_ll": r,
                    "source": "penalized",
                    "ser": float(np.mean(m[cipher] != plain)),
                }
                for m, s, r in res.shortlist
            ]
            head0 = HomophonicHead(ev, seed=job["trial"] + 777, freq_penalty_weight=0.0)
            res0 = head0.solve_parallel(
                cipher,
                job["n_symbols"],
                language=hyp,
                restarts=args.restarts_nopenalty,
                workers=args.workers,
                sa_steps=args.sa_steps,
                shortlist=args.shortlist // 2,
            )
            short += [
                {
                    "map": m.tolist(),
                    "penalized": float(head._objective(m[cipher], hyp)),
                    "raw_ll": r,
                    "source": "nopenalty",
                    "ser": float(np.mean(m[cipher] != plain)),
                }
                for m, s, r in res0.shortlist
            ]
            rec["hypotheses"][hyp] = {
                "shortlist": short,
                "n_evals": res.n_evals + res0.n_evals,
                "seconds": round(time.time() - t1, 1),
            }
            print(
                f"  [{i}/{len(todo)}] {job['language']} t{job['trial']} hyp={hyp}: SER best {short[0]['ser']:.3f} "
                f"(oracle {min(x['ser'] for x in short):.3f}; nopenalty best SER {min(x['ser'] for x in short if x['source']=='nopenalty'):.3f}) "
                f"{time.time()-t1:.0f}s  total {time.time()-t0:.0f}s",
                flush=True,
            )
        results.append(rec)
        write_json_atomic(
            path,
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "settings": settings,
                "instances": results,
            },
        )
    print(f"written {path}")


def _mdl_annotate(c, hyp):
    """Selection key of the outer tier for a homophonic head: own-condition
    plaintext bits + the cipher's choice bits per plaintext char (the uniform
    scale's MDL total; key bits are identical across candidates). The pure
    ELBO prefers hyper-repetitive degenerate decodes (they are *very*
    predictable text) — the choice term is what makes them pay for the
    freedom that produced them."""
    from diff_voyn.heads.scale import choice_bits

    cb = choice_bits("homophonic", c.decode, sym_to_letter=np.asarray(c.key))
    c.extra["choice_bits_per_char"] = float(cb / max(len(c.decode), 1))
    c.extra["mdl_bits"] = float(c.bits[hyp] + c.extra["choice_bits_per_char"])
    c.bits["mdl"] = c.extra["mdl_bits"]


def _mdl_annotate(c, hyp):
    """Selection key of the outer tier for a homophonic head: own-condition
    plaintext bits + the cipher's choice bits per plaintext char (the uniform
    scale's MDL total; key bits are identical across candidates). The pure
    ELBO prefers hyper-repetitive degenerate decodes (they are *very*
    predictable text) — the choice term makes them pay for the freedom that
    produced them."""
    from diff_voyn.heads.scale import choice_bits

    cb = choice_bits("homophonic", c.decode, sym_to_letter=np.asarray(c.key))
    c.extra["choice_bits_per_char"] = float(cb / max(len(c.decode), 1))
    c.extra["mdl_bits"] = float(c.bits[hyp] + c.extra["choice_bits_per_char"])
    c.bits["mdl"] = c.extra["mdl_bits"]


def stage_score(args, root):
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator
    from diff_voyn.heads.rung2_homophonic import HomophonicHead
    from diff_voyn.heads.two_tier import Candidate, rescore, select

    torch.set_float32_matmul_precision("high")
    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    ng = _build_ngram_evaluator()
    solves = json.loads((args.out_dir / "rung2_solves.json").read_text())
    path = args.out_dir / "rung2_scores.json"
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
            "crn": "paired masks across all candidates and conditions of an instance",
        },
        "solve_settings": solves["settings"],
    }
    t0 = time.time()
    for i, r in enumerate(inst, 1):
        cipher = np.asarray(r["cipher_ids"], dtype=np.int64)
        plain = np.asarray(r["plain_ids"], dtype=np.int64)
        seed = zlib.crc32(f"score2/{r['language']}/{r['trial']}".encode()) % (2**31)
        rec = {k: r[k] for k in KEY}
        rec["hypotheses"] = {}
        cands_all = []
        for hyp in LANGS:
            for c in r["hypotheses"][hyp]["shortlist"]:
                m = np.asarray(c["map"], dtype=np.int64)
                cands_all.append(
                    Candidate(
                        decode=m[cipher],
                        key=m,
                        inner_score=c["penalized"],
                        source=c["source"],
                        extra={"ser": c["ser"], "hyp": hyp, "raw_ll": c["raw_ll"]},
                    )
                )
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
            pen = [c for c in cands if c.source == "penalized"]
            for c in cands:
                _mdl_annotate(c, hyp)
            pick = select(cands, language=hyp)  # pure-ELBO pick / oracle over ALL
            pick_mdl = min(cands, key=lambda c: c.extra["mdl_bits"])
            ngram_pick = max(pen, key=lambda c: c.inner_score)
            rawll_pick = max(
                cands, key=lambda c: c.extra["raw_ll"]
            )  # unpenalized n-gram
            t1 = time.time()
            refined_map, losses = refine_assignment(
                ev,
                cipher,
                pick_mdl.key,
                language=hyp,
                bijective=False,
                steps=args.refine_steps,
                lr=args.refine_lr,
                n_strata=args.refine_strata,
                seed=seed,
            )
            ref = Candidate(
                decode=refined_map[cipher],
                key=refined_map,
                inner_score=float(
                    ng.score_hard(refined_map[cipher], language=hyp, order=5)
                ),
                source="refined",
                extra={"ser": float(np.mean(refined_map[cipher] != plain)), "hyp": hyp},
            )
            rescore(
                ev,
                [ref],
                language=hyp,
                conditions=list(LANGS),
                n_strata=args.budget,
                seed=seed,
            )
            _mdl_annotate(ref, hyp)
            # pair-swap polish of the MDL pick under the penalized n-gram
            # objective (CH.5 lever; cheap, CPU) — a further candidate
            head2 = HomophonicHead(ng, seed=r["trial"])
            pol_map, pol_score, _ = head2.polish_pairs(cipher, pick_mdl.key, hyp)
            pol = Candidate(
                decode=pol_map[cipher],
                key=pol_map,
                inner_score=float(pol_score),
                source="pairpolish",
                extra={
                    "ser": float(np.mean(pol_map[cipher] != plain)),
                    "hyp": hyp,
                    "changed": bool((pol_map != pick_mdl.key).any()),
                },
            )
            rescore(
                ev,
                [pol],
                language=hyp,
                conditions=list(LANGS),
                n_strata=args.budget,
                seed=seed,
            )
            _mdl_annotate(pol, hyp)
            # ELBO-scored discrete polish (outer tier refining the inner
            # objective's optimum): from the MDL-best so far, single moves +
            # pair swaps scored by paired NELBO + choice bits
            best_so_far = min(cands + [ref, pol], key=lambda c: c.extra["mdl_bits"])
            t2 = time.time()
            ep_map, ep_info = elbo_polish(
                ev,
                cipher,
                best_so_far.key,
                language=hyp,
                seed=seed,
                # objective: ELBO alone unless --polish-choice-term (the recorded
                # Phase-5 runs used the MDL total; see docs/race_polish_plan.md §7)
                choice_fn=(
                    (
                        lambda m, dec: choice_bits("homophonic", dec, sym_to_letter=m)
                        / max(len(dec), 1)
                    )
                    if args.polish_choice_term
                    else None
                ),
                choice_term_in_polish=args.polish_choice_term,
                sweeps=args.elbo_sweeps,
                budget=args.elbo_budget,
                pair_swaps=False,
            )
            ep = Candidate(
                decode=ep_map[cipher],
                key=ep_map,
                inner_score=float(head2._objective(ep_map[cipher], hyp)),
                source="elbopolish",
                extra={
                    "ser": float(np.mean(ep_map[cipher] != plain)),
                    "hyp": hyp,
                    "changed": bool((ep_map != best_so_far.key).any()),
                    "accepted": ep_info["accepted"],
                    "n_changed": ep_info["n_changed"],
                    "sweeps": len(ep_info["trace"]),
                    "seconds": round(time.time() - t2, 1),
                },
            )
            rescore(
                ev,
                [ep],
                language=hyp,
                conditions=list(LANGS),
                n_strata=args.budget,
                seed=seed,
            )
            _mdl_annotate(ep, hyp)
            final = min(cands + [ref, pol, ep], key=lambda c: c.extra["mdl_bits"])
            rec["hypotheses"][hyp] = {
                "n_candidates": len(cands),
                "ngram": ngram_pick.as_dict(),
                "rawll": rawll_pick.as_dict(),
                "elbo_pure": pick["diffusion"].as_dict(),
                "diffusion": pick_mdl.as_dict(),
                "oracle": pick["oracle"].as_dict(),
                "elbo_pure_picked_nopenalty": pick["diffusion"].source == "nopenalty",
                "diffusion_picked_nopenalty": pick_mdl.source == "nopenalty",
                "refined": {
                    **ref.as_dict(),
                    "changed": bool((refined_map != pick_mdl.key).any()),
                    "loss_first_last": [losses[0], losses[-1]] if losses else None,
                    "seconds": round(time.time() - t1, 1),
                },
                "pairpolish": pol.as_dict(),
                "elbopolish": ep.as_dict(),
                "final": final.as_dict(),
                "shortlist": [c.as_dict() for c in cands],
            }
        results.append(rec)
        print(
            f"  scored {i}/{len(inst)} ({time.time()-t0:.0f}s) {r['language']} t{r['trial']}: "
            f"SER ngram {rec['hypotheses'][r['language']]['ngram']['ser']:.3f} → final {rec['hypotheses'][r['language']]['final']['ser']:.3f}",
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
    data = json.loads((args.out_dir / "rung2_scores.json").read_text())
    table = CalibrationTable.load(args.primary, root)
    offs = table.additive_offsets()
    variants = (
        "ngram",
        "rawll",
        "elbo_pure",
        "diffusion",
        "refined",
        "pairpolish",
        "elbopolish",
        "final",
        "oracle",
    )
    per_lang = {}
    rows = []
    for r in data["instances"]:
        t = r["language"]
        h = r["hypotheses"]
        rec = {"language": t, "trial": r["trial"]}
        for v in variants:
            rec[v] = h[t][v]["ser"]
        rec["picked_nopenalty"] = h[t]["diffusion_picked_nopenalty"]
        rec["elbo_pure_picked_nopenalty"] = h[t]["elbo_pure_picked_nopenalty"]
        rec["refine_changed"] = h[t]["refined"]["changed"]
        rec["pairpolish_changed"] = h[t].get("pairpolish", {}).get("changed", False)
        rec["elbopolish_accepted"] = h[t].get("elbopolish", {}).get("accepted", False)
        # language ranking: MDL total (plaintext + choice bits) per hypothesis
        # (primary for a verbose cipher); calibrated plaintext bits alone and
        # the n-gram excess bits as variants
        mdl = {hyp: h[hyp]["final"]["mdl_bits"] for hyp in LANGS}
        ranked = sorted(mdl.items(), key=lambda kv: kv[1])
        rec["rank_final"], rec["margin"] = ranked[0][0], ranked[1][1] - ranked[0][1]
        fin = {hyp: h[hyp]["final"]["bits"][hyp] for hyp in LANGS}
        rec["rank_plain_bits"] = rank_languages(fin, offs)[0][0]
        rec["rank_ngram"] = rank_languages(
            {hyp: h[hyp]["ngram"]["bits"][hyp] for hyp in LANGS}, offs
        )[0][0]
        rec["mdl_bits"] = mdl
        rows.append(rec)
        per_lang.setdefault(t, []).append(rec)
    cells = {}
    for lang, recs in per_lang.items():
        n = len(recs)
        e = {"n": n}
        for v in variants:
            e[f"ser_{v}"] = float(np.mean([x[v] for x in recs]))
            e[f"le_1.9pct_{v}"] = float(np.mean([x[v] <= 0.019 for x in recs]))
        e["diffusion_better_than_ngram"] = float(
            np.mean([x["diffusion"] < x["ngram"] for x in recs])
        )
        e["diffusion_worse_than_ngram"] = float(
            np.mean([x["diffusion"] > x["ngram"] for x in recs])
        )
        e["diffusion_picked_nopenalty_rate"] = float(
            np.mean([x["picked_nopenalty"] for x in recs])
        )
        e["elbo_pure_picked_nopenalty_rate"] = float(
            np.mean([x["elbo_pure_picked_nopenalty"] for x in recs])
        )
        e["refine_changed_rate"] = float(np.mean([x["refine_changed"] for x in recs]))
        e["refined_better_than_diffusion"] = float(
            np.mean([x["refined"] < x["diffusion"] for x in recs])
        )
        e["refined_worse_than_diffusion"] = float(
            np.mean([x["refined"] > x["diffusion"] for x in recs])
        )
        k = sum(x["rank_final"] == lang for x in recs)
        e["lang_acc_final"], e["lang_acc_final_ci95"] = k / n, wilson(k, n)
        e["family_acc_final"] = float(
            np.mean([family_of(x["rank_final"]) == family_of(lang) for x in recs])
        )
        e["lang_acc_plain_bits"] = float(
            np.mean([x["rank_plain_bits"] == lang for x in recs])
        )
        e["lang_acc_ngram"] = float(np.mean([x["rank_ngram"] == lang for x in recs]))
        e["margin_median"] = float(np.median([x["margin"] for x in recs]))
        cells[lang] = e
    allr = rows
    acc = {
        "criterion": "≤1.9% SER on Zodiac-408-class synthetics (mean final SER over instances; per-instance rate reported)",
        "ser_final_mean": float(np.mean([x["final"] for x in allr])),
        "ser_ngram_mean": float(np.mean([x["ngram"] for x in allr])),
        "ser_oracle_mean": float(np.mean([x["oracle"] for x in allr])),
        "ser_elbo_pure_mean": float(np.mean([x["elbo_pure"] for x in allr])),
        "instances_le_1.9pct_final": float(
            np.mean([x["final"] <= 0.019 for x in allr])
        ),
        "instances_le_1.9pct_ngram": float(
            np.mean([x["ngram"] <= 0.019 for x in allr])
        ),
        "ser_final_median": float(np.median([x["final"] for x in allr])),
        "pairpolish_changed_rate": float(
            np.mean([x.get("pairpolish_changed", False) for x in allr])
        ),
        "elbopolish_accepted_rate": float(
            np.mean([x.get("elbopolish_accepted", False) for x in allr])
        ),
        "ser_elbopolish_mean": float(np.mean([x["elbopolish"] for x in allr])),
        "lang_acc_final": float(
            np.mean([x["rank_final"] == x["language"] for x in allr])
        ),
        "lang_acc_plain_bits": float(
            np.mean([x["rank_plain_bits"] == x["language"] for x in allr])
        ),
        "lang_acc_ngram": float(
            np.mean([x["rank_ngram"] == x["language"] for x in allr])
        ),
        "elbo_pure_picked_degenerate_rate": float(
            np.mean([x["elbo_pure_picked_nopenalty"] for x in allr])
        ),
        "mdl_picked_degenerate_rate": float(
            np.mean([x["picked_nopenalty"] for x in allr])
        ),
        "n": len(allr),
    }
    acc["pass"] = bool(acc["ser_final_mean"] <= 0.019)
    acc["pass_per_instance"] = bool(
        acc["instances_le_1.9pct_final"] >= 0.8 and acc["ser_final_median"] <= 0.019
    )
    report = {
        "task": "5.3",
        "created_utc": datetime.now(UTC).isoformat(),
        "evaluator": data["evaluator"],
        "scoring": data["scoring"],
        "solve_settings": data["solve_settings"],
        "primary_calibration": args.primary,
        "language_ranking_rule": "MDL total per hypothesis: own-condition plaintext bits + choice bits per plaintext char (calibration hook applied to the plaintext bits)",
        "cells": cells,
        "instances": rows,
        "acceptance": acc,
        "anchors_note": "Zodiac-408 (English) is outside the frozen inventory; Borg (Latin) / BnF fr2988 transcriptions not fetched — task 6.6",
    }
    write_json_atomic(args.out_dir / "rung2_report.json", report)
    md = [
        "### Rung 2 (unigram homophonic, Zodiac-408-class: 408 chars / 54 symbols) — two-tier, MDL selection",
        "",
        "| language | n | SER n-gram (penalized) | SER raw-LL pick | SER pure-ELBO pick | SER MDL pick | SER refined | SER pair-polish | SER ELBO-polish | **SER final** | SER oracle | ≤1.9% final | pure-ELBO / MDL picked degenerate map | MDL better/worse vs n-gram | refine changed/better/worse | lang acc MDL (plain bits / n-gram) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lang, c in cells.items():
        md.append(
            f"| {lang} | {c['n']} | {c['ser_ngram']:.3f} | {c['ser_rawll']:.3f} | {c['ser_elbo_pure']:.3f} | {c['ser_diffusion']:.3f} | {c['ser_refined']:.3f} | {c['ser_pairpolish']:.3f} | {c['ser_elbopolish']:.3f} | **{c['ser_final']:.3f}** | {c['ser_oracle']:.3f} | "
            f"{c['le_1.9pct_final']:.0%} | {c['elbo_pure_picked_nopenalty_rate']:.0%} / {c['diffusion_picked_nopenalty_rate']:.0%} | {c['diffusion_better_than_ngram']:.0%}/{c['diffusion_worse_than_ngram']:.0%} | "
            f"{c['refine_changed_rate']:.0%}/{c['refined_better_than_diffusion']:.0%}/{c['refined_worse_than_diffusion']:.0%} | **{c['lang_acc_final']:.0%}** ({c['lang_acc_plain_bits']:.0%} / {c['lang_acc_ngram']:.0%}) |"
        )
    md += [
        "",
        (
            f"all: SER final {acc['ser_final_mean']:.4f} (n-gram {acc['ser_ngram_mean']:.4f}, oracle {acc['ser_oracle_mean']:.4f}, pure-ELBO pick {acc['ser_elbo_pure_mean']:.4f}); "
            f"instances ≤1.9%: final {acc['instances_le_1.9pct_final']:.0%} (n-gram {acc['instances_le_1.9pct_ngram']:.0%}); "
            f"pure ELBO picks a degenerate map {acc['elbo_pure_picked_degenerate_rate']:.0%} of the time, the MDL selection {acc['mdl_picked_degenerate_rate']:.0%}; "
            f"language recovery by MDL total {acc['lang_acc_final']:.1%} (plaintext bits alone {acc['lang_acc_plain_bits']:.1%}, n-gram excess bits {acc['lang_acc_ngram']:.1%}); median SER {acc['ser_final_median']:.4f} → mean-SER criterion **{'PASS' if acc['pass'] else 'FAIL'}**, per-instance (≥80% of instances ≤1.9% and median ≤1.9%) **{'PASS' if acc['pass_per_instance'] else 'FAIL'}**"
        ),
    ]
    md = "\n".join(md)
    (args.out_dir / "rung2_report.md").write_text(md)
    print(md)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = data_root()
    p.add_argument("--stage", choices=["solve", "score", "report"], required=True)
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--length", type=int, default=408)
    p.add_argument("--n-symbols", type=int, default=54)
    p.add_argument("--restarts", type=int, default=120)
    p.add_argument("--restarts-nopenalty", type=int, default=24)
    p.add_argument("--sa-steps", type=int, default=100_000)
    p.add_argument("--shortlist", type=int, default=12)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--fresh", action="store_true")
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="re-solve these language/trial instances",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--elbo-sweeps", type=int, default=6)
    p.add_argument(
        "--polish-choice-term",
        action="store_true",
        help="put the MDL choice term in the elbo_polish objective (the recorded "
        "Phase-5 behaviour; harmful at Borg scale — docs/race_polish_plan.md §7)",
    )
    p.add_argument("--elbo-budget", type=int, default=8)
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
