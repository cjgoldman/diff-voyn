"""Tasks 3.6 + 3.7 — synthetic language-recovery validation on 1:1 ciphers.

The end-to-end test of the calibrated instrument (design §5b.4, §9.1): for
every (language × length ∈ {50,100,200,400,700}) cell, ≥50 ciphers built from
HELD-OUT plaintext under a random bijective key. Each cipher is deciphered
under *every* language hypothesis with the rung-1 head on the frozen n-gram
evaluator (the cheap inner search of design §7.4), and each hypothesis's
candidate plaintext is scored by the diffusion evaluator under its own
language condition with common random numbers (same masks for every
hypothesis of a cipher). The calibrated ranking (``CalibrationTable``, the
single application point) must recover the true language; reported at
language and family granularity (3.7), per cell, with Wilson intervals.

Also reported, for the honest reading of the result:

- the n-gram-only ranking (the CH.9 baseline — what the inner search alone
  would conclude), and the uncalibrated / alternative-table rankings, so the
  effect of the calibration is visible rather than assumed;
- replicate flip-rate of the diffusion ranking at the chosen budget (3.2);
- the margin between the winning decipherment and a shuffled-letter version
  of the plaintext under the same condition — the "no structure" ceiling
  that any Phase-4/6 abstention threshold must be set against (on the
  Phase-B weights, as the G2 carry-over requires).

Stages (each resumable from the previous one's artifact):
    solve   CPU, forked single-thread workers: rung-1 decipherments
    score   GPU: diffusion scoring of every hypothesis's decipherment
    report  rankings, accuracies, markdown table, ClearML

Usage:
    uv run python scripts/language_recovery.py --stage solve --workers 8
    uv run python scripts/language_recovery.py --stage score --ckpt ...
    uv run python scripts/language_recovery.py --stage report
Artifacts under DATA_ROOT/analysis/phase3/recovery_{solves,scores,report}.*
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "solve" in sys.argv:  # CPU-only stage: keep CUDA out of the forked workers
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX
from diff_voyn.data.noise import LETTER_BASE
from diff_voyn.metrology import (
    CALIBRATION_VERSION,
    CalibrationTable,
    ScoreSettings,
    calibrate_bits,
    family_of,
    rank_languages,
    score_conditions,
)

LANGS = tuple(LANG_TO_INDEX)
LENGTHS = (50, 100, 200, 400, 700)
OUT = "recovery"

# ---------------------------------------------------------------- solve stage

_EV = None  # fork-inherited n-gram evaluator


def _init_worker():
    os.nice(10)
    torch.set_num_threads(1)


def _build_evaluator():
    from diff_voyn.heads.evaluator import NgramEvaluator
    from diff_voyn.heads.harness import ngram_calibration_offsets
    from diff_voyn.heads.ngram import lm_dir, load_lm

    lms = {lang: load_lm(lm_dir() / f"{lang}.npz") for lang in LANGS}
    return NgramEvaluator(lms, calibration_offsets_bits=ngram_calibration_offsets(lms))


def _solve_one(job: dict) -> dict:
    """Decipher one instance under every language hypothesis."""
    from diff_voyn.heads.rung1_sinkhorn import SinkhornSubstitutionHead

    cipher_ids = np.asarray(job["cipher_ids"], dtype=np.int64)
    plain = np.asarray(job["plain_ids"], dtype=np.int64)
    out = {k: job[k] for k in ("language", "length", "trial")}
    out["decipherments"], out["ngram_hard_nats"], out["ngram_calibrated_bits"] = (
        {},
        {},
        {},
    )
    t0 = time.time()
    for hyp in LANGS:
        head = SinkhornSubstitutionHead(_EV, seed=job["trial"])
        res = head.solve(cipher_ids, language=hyp, restarts=job["restarts"])
        dec = res.sym_to_letter[cipher_ids]
        out["decipherments"][hyp] = dec.astype(np.int64).tolist()
        out["ngram_hard_nats"][hyp] = float(res.hard_score)
        out["ngram_calibrated_bits"][hyp] = _EV.calibrated_bits_per_char(
            res.hard_score, len(cipher_ids), hyp
        )
    true_dec = np.asarray(out["decipherments"][job["language"]])
    out["ser_true_hypothesis"] = float(np.mean(true_dec != plain))
    out["plain_ids"] = plain.tolist()
    out["cipher_ids"] = cipher_ids.tolist()
    out["solve_seconds"] = round(time.time() - t0, 1)
    return out


def build_suite(
    corpus_dir: Path, splits: dict, trials: int, lengths, seed: int
) -> list[dict]:
    from diff_voyn.heads.synth import HeldoutSampler, gen_substitution

    jobs = []
    for lang in LANGS:
        sampler = HeldoutSampler(corpus_dir, splits, lang)
        for L in lengths:
            for trial in range(trials):
                key = f"recovery/{seed}/{lang}/sub1to1/{L}/{trial}".encode()
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
                    }
                )
    return jobs


def stage_solve(args, root: Path) -> None:
    global _EV
    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    jobs = build_suite(corpus_dir, splits, args.trials, args.lengths, args.seed)
    for j in jobs:
        j["restarts"] = args.restarts
    out_path = args.solves
    done = {}
    if out_path.exists() and not args.fresh:
        for r in json.loads(out_path.read_text())["instances"]:
            done[(r["language"], r["length"], r["trial"])] = r
        print(f"resuming: {len(done)} of {len(jobs)} instances already solved")
    todo = [j for j in jobs if (j["language"], j["length"], j["trial"]) not in done]
    # long instances first for load balance
    todo.sort(key=lambda j: -j["length"])
    torch.set_num_threads(1)
    _EV = _build_evaluator()
    t0 = time.time()
    ctx = mp.get_context("fork")
    results = list(done.values())
    with ctx.Pool(args.workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(_solve_one, todo, chunksize=1), 1):
            results.append(r)
            if i % 10 == 0 or i == len(todo):
                print(
                    f"  solved {i}/{len(todo)}  ({time.time()-t0:.0f}s)  last: "
                    f"{r['language']} L={r['length']} SER={r['ser_true_hypothesis']:.3f}",
                    flush=True,
                )
                _write_solves(out_path, results, args)
    _write_solves(out_path, results, args)
    print(f"written {out_path}")


def _write_solves(path: Path, results: list[dict], args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "settings": {
                    "trials": args.trials,
                    "lengths": list(args.lengths),
                    "restarts": args.restarts,
                    "seed": args.seed,
                    "kind": "sub1to1",
                    "plaintext_source": "held-out split v1",
                },
                "instances": sorted(
                    results, key=lambda r: (r["language"], r["length"], r["trial"])
                ),
            }
        )
    )
    os.replace(tmp, path)


# ---------------------------------------------------------------- score stage


def stage_score(args, root: Path) -> None:
    """Diffusion scoring of every hypothesis's decipherment, at the primary
    budget (``--budget``, ``reps`` replicate seeds) and at a budget sweep
    (``--budgets``) — the task-3.2 study in the *decipherment* regime, where
    the hypotheses differ in text (large margins) rather than only in the
    conditioning of one fixed text (the clean-text regime of
    ``sample_budget.py``)."""
    from diff_voyn.infra.checkpoint import load_backbone

    solves = json.loads(args.solves.read_text())
    inst = solves["instances"]
    budget = args.budget
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
    model, meta = load_backbone(args.ckpt, args.device)
    rng = np.random.default_rng(args.seed)
    cells = {}
    for r in inst:
        cells.setdefault((r["language"], r["length"]), []).append(r)
    scored = []
    t0 = time.time()
    for (lang, L), rs in sorted(cells.items()):
        rs.sort(key=lambda r: r["trial"])
        texts = {
            h: np.array([r["decipherments"][h] for r in rs]) + LETTER_BASE
            for h in LANGS
        }
        plain = np.array([r["plain_ids"] for r in rs]) + LETTER_BASE
        shuffled = np.stack([rng.permutation(row) for row in plain])
        budgets = sorted(set(args.budgets) | {budget})
        by_budget = {}
        for B in budgets:
            reps = args.reps if B == budget else args.sweep_reps
            runs = []
            for k in range(reps):
                st = ScoreSettings(
                    n_strata=B, seed=args.seed + 1000 * k, batch=args.batch
                )
                runs.append(
                    score_conditions(
                        model, texts, LANGS, settings=st, device=args.device
                    )
                )
            by_budget[B] = np.stack(runs)  # [R, N, C]
        dec_bits = by_budget[budget]
        st = ScoreSettings(n_strata=budget, seed=args.seed, batch=args.batch)
        plain_bits = score_conditions(
            model, plain, LANGS, settings=st, device=args.device
        )
        shuf_bits = score_conditions(
            model, shuffled, LANGS, settings=st, device=args.device
        )
        for i, r in enumerate(rs):
            plain_i = np.asarray(r["plain_ids"])
            scored.append(
                {
                    "language": lang,
                    "length": L,
                    "trial": r["trial"],
                    "ser_true_hypothesis": r["ser_true_hypothesis"],
                    "ser_by_hypothesis": {
                        h: float(np.mean(np.asarray(r["decipherments"][h]) != plain_i))
                        for h in LANGS
                    },
                    "ngram_calibrated_bits": r["ngram_calibrated_bits"],
                    "diffusion_bits": {
                        h: [float(dec_bits[k, i, j]) for k in range(dec_bits.shape[0])]
                        for j, h in enumerate(LANGS)
                    },
                    "diffusion_bits_by_budget": {
                        str(B): {
                            h: [float(arr[k, i, j]) for k in range(arr.shape[0])]
                            for j, h in enumerate(LANGS)
                        }
                        for B, arr in by_budget.items()
                    },
                    "plain_bits_by_condition": {
                        h: float(plain_bits[i, j]) for j, h in enumerate(LANGS)
                    },
                    "shuffled_bits_by_condition": {
                        h: float(shuf_bits[i, j]) for j, h in enumerate(LANGS)
                    },
                }
            )
        print(f"  scored {lang} L={L} n={len(rs)}  ({time.time()-t0:.0f}s)", flush=True)
    out = args.out_dir / f"{OUT}_scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "backbone": meta,
                "scoring": {
                    "budget": budget,
                    "reps": args.reps,
                    "budgets": budgets,
                    "sweep_reps": args.sweep_reps,
                    "seed": args.seed,
                    "batch": args.batch,
                    "crn": "same masks for every hypothesis of a cipher (per chunk)",
                },
                "solve_settings": solves["settings"],
                "instances": scored,
            },
            indent=1,
        )
    )
    print(f"written {out}")


# --------------------------------------------------------------- report stage


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


def _rank_variants(root: Path, versions: list[str]) -> dict[str, dict[str, float]]:
    """Name -> additive offsets (the hook's convention)."""
    variants = {"uncalibrated": {l: 0.0 for l in LANGS}}
    for v in versions:
        try:
            variants[f"calibration_{v}"] = CalibrationTable.load(
                v, root
            ).additive_offsets()
        except FileNotFoundError:
            pass
    return variants


def stage_report(args, root: Path) -> dict:
    path = args.out_dir / f"{OUT}_scores.json"
    data = json.loads(path.read_text())
    inst = data["instances"]
    variants = _rank_variants(root, args.calibrations)
    primary = f"calibration_{args.primary}"
    if primary not in variants:
        raise SystemExit(f"primary calibration table {args.primary} not found")
    primary_table = CalibrationTable.load(args.primary, root)
    # per-instance rankings
    per_cell: dict[tuple, list[dict]] = {}
    for r in inst:
        rec = {
            "trial": r["trial"],
            "ser": r["ser_true_hypothesis"],
            "rankings": {},
            "ser_by_hypothesis": r.get("ser_by_hypothesis", {}),
        }
        mean_bits = {h: float(np.mean(v)) for h, v in r["diffusion_bits"].items()}
        for name, offs in variants.items():
            rec["rankings"][name] = rank_languages(mean_bits, offs)[0][0]
        # replicate flip-rate under the primary table
        reps = len(next(iter(r["diffusion_bits"].values())))
        tops = [
            rank_languages(
                {h: r["diffusion_bits"][h][k] for h in LANGS}, variants[primary]
            )[0][0]
            for k in range(reps)
        ]
        rec["flip"] = (
            float(np.mean([a != b for a, b in itertools.combinations(tops, 2)]))
            if reps > 1
            else 0.0
        )
        rec["rankings"]["ngram_excess_bits"] = min(
            r["ngram_calibrated_bits"], key=r["ngram_calibrated_bits"].get
        )
        # margins under the primary table
        cal = {h: calibrate_bits(mean_bits[h], h, variants[primary]) for h in LANGS}
        order = sorted(cal, key=cal.get)
        rec["margin_bits"] = cal[order[1]] - cal[order[0]]
        # resolvable at the calibration's precision? (margin vs the systematic
        # uncertainty the measured offsets imply for this pair)
        rec["margin_unresolved"] = rec[
            "margin_bits"
        ] < primary_table.margin_uncertainty_bits(order[0], order[1])
        t = r["language"]
        rec["true_minus_shuffled_bits"] = (
            mean_bits[t] - r["shuffled_bits_by_condition"][t]
        )
        rec["wrong_minus_shuffled_bits"] = min(
            mean_bits[h] - r["shuffled_bits_by_condition"][h] for h in LANGS if h != t
        )
        rec["true_hyp_minus_plain_bits"] = (
            mean_bits[t] - r["plain_bits_by_condition"][t]
        )
        per_cell.setdefault((r["language"], r["length"]), []).append(rec)

    rank_names = list(variants) + ["ngram_excess_bits"]
    cells, by_length, by_language = {}, {}, {}
    for (lang, L), recs in sorted(per_cell.items()):
        n = len(recs)
        entry = {
            "n": n,
            "ser_mean": float(np.mean([x["ser"] for x in recs])),
            "ser_by_hypothesis_mean": {
                h: float(np.mean([x["ser_by_hypothesis"].get(h, np.nan) for x in recs]))
                for h in LANGS
            },
            "wrong_hypothesis_decodes_truth_rate": float(
                np.mean(
                    [
                        any(
                            x["ser_by_hypothesis"].get(h, 1.0) < 0.05
                            for h in LANGS
                            if h != lang
                        )
                        for x in recs
                    ]
                )
            ),
            "flip_rate": float(np.mean([x["flip"] for x in recs])),
            "margin_bits_median": float(np.median([x["margin_bits"] for x in recs])),
            "margin_unresolved_rate": float(
                np.mean([x["margin_unresolved"] for x in recs])
            ),
            "true_minus_shuffled_bits_mean": float(
                np.mean([x["true_minus_shuffled_bits"] for x in recs])
            ),
            "wrong_minus_shuffled_bits_mean": float(
                np.mean([x["wrong_minus_shuffled_bits"] for x in recs])
            ),
            "true_hyp_minus_plain_bits_mean": float(
                np.mean([x["true_hyp_minus_plain_bits"] for x in recs])
            ),
            "accuracy": {},
        }
        for name in rank_names:
            k_lang = sum(x["rankings"][name] == lang for x in recs)
            k_fam = sum(family_of(x["rankings"][name]) == family_of(lang) for x in recs)
            entry["accuracy"][name] = {
                "language": k_lang / n,
                "language_ci95": wilson(k_lang, n),
                "family": k_fam / n,
                "family_ci95": wilson(k_fam, n),
            }
        entry["confusion"] = {
            name: {h: sum(x["rankings"][name] == h for x in recs) / n for h in LANGS}
            for name in rank_names
        }
        cells[f"{lang}/L{L}"] = entry
        for agg, key in ((by_length, L), (by_language, lang)):
            a = agg.setdefault(str(key), {name: [0, 0, 0] for name in rank_names})
            for name in rank_names:
                a[name][0] += sum(x["rankings"][name] == lang for x in recs)
                a[name][1] += sum(
                    family_of(x["rankings"][name]) == family_of(lang) for x in recs
                )
                a[name][2] += n

    def fold(agg):
        return {
            k: {
                name: {
                    "language": v[0] / v[2],
                    "language_ci95": wilson(v[0], v[2]),
                    "family": v[1] / v[2],
                    "family_ci95": wilson(v[1], v[2]),
                    "n": v[2],
                }
                for name, v in d.items()
            }
            for k, d in agg.items()
        }

    by_length, by_language = fold(by_length), fold(by_language)
    overall = {name: [0, 0, 0] for name in rank_names}
    for d in by_length.values():
        for name, v in d.items():
            overall[name][0] += round(v["language"] * v["n"])
            overall[name][1] += round(v["family"] * v["n"])
            overall[name][2] += v["n"]
    overall = {
        name: {"language": v[0] / v[2], "family": v[1] / v[2], "n": v[2]}
        for name, v in overall.items()
    }

    # budget sweep (3.2, decipherment regime): replicate flip-rate per (length, budget)
    budget_sweep = {}
    for r in inst:
        for B, bits in r.get("diffusion_bits_by_budget", {}).items():
            reps = len(next(iter(bits.values())))
            tops = [
                rank_languages({h: bits[h][k] for h in LANGS}, variants[primary])[0][0]
                for k in range(reps)
            ]
            flips = [a != b for a, b in itertools.combinations(tops, 2)]
            d = budget_sweep.setdefault(str(r["length"]), {}).setdefault(
                B, {"flip": [], "true": []}
            )
            d["flip"].extend(flips)
            d["true"].extend(t == r["language"] for t in tops)
    budget_sweep = {
        L: {
            B: {
                "flip_rate": float(np.mean(v["flip"])),
                "language_acc": float(np.mean(v["true"])),
            }
            for B, v in sorted(d.items(), key=lambda kv: int(kv[0]))
        }
        for L, d in sorted(budget_sweep.items(), key=lambda kv: int(kv[0]))
    }
    chosen = None
    if budget_sweep:
        for B in sorted({int(B) for d in budget_sweep.values() for B in d}):
            if all(
                str(B) in d and d[str(B)]["flip_rate"] < 0.01
                for d in budget_sweep.values()
            ):
                chosen = B
                break

    long_lengths = [L for L in sorted({r["length"] for r in inst}) if L >= 200]
    long_acc = {
        name: {
            "language": np.mean(
                [
                    cells[f"{l}/L{L}"]["accuracy"][name]["language"]
                    for l in LANGS
                    for L in long_lengths
                    if f"{l}/L{L}" in cells
                ]
            ),
            "family": np.mean(
                [
                    cells[f"{l}/L{L}"]["accuracy"][name]["family"]
                    for l in LANGS
                    for L in long_lengths
                    if f"{l}/L{L}" in cells
                ]
            ),
        }
        for name in rank_names
    }
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "3.6/3.7",
        "backbone": data["backbone"],
        "scoring": data["scoring"],
        "solve_settings": data["solve_settings"],
        "primary_calibration": args.primary,
        "primary_calibration_policy": primary_table.policy,
        "primary_calibration_summary": primary_table.summary(),
        "ranking_variants": rank_names,
        "cells": cells,
        "by_length": by_length,
        "by_language": by_language,
        "overall": overall,
        "ge200_mean_accuracy": {
            k: {kk: float(vv) for kk, vv in v.items()} for k, v in long_acc.items()
        },
        "budget_sweep_decipherment_regime": budget_sweep,
        "budget_sweep_chosen": chosen,
        "budget_sweep_note": "smallest budget with replicate flip-rate < 1% of the calibrated "
        "decipherment ranking at every length (task 3.2, decipherment regime)",
        "hauer_kondrak_bar": 0.971,
        "acceptance": {
            "criterion": "near-ceiling language recovery on 1:1 at ≥200 chars (bar 97.1%) "
            "under the primary calibrated ranking",
            "primary_language_acc_ge200": float(long_acc[primary]["language"]),
            "primary_family_acc_ge200": float(long_acc[primary]["family"]),
            "pass": bool(long_acc[primary]["language"] >= 0.971),
        },
    }
    out = args.out_dir / f"{OUT}_report.json"
    out.write_text(json.dumps(report, indent=1))
    md = render_markdown(report)
    (args.out_dir / f"{OUT}_report.md").write_text(md)
    print(md)
    print(f"written {out}")
    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        task = init_task(
            RunConfig(run_name="language-recovery", phase=args.phase_tag),
            root,
            tags=["task3.6", "task3.7"],
        )
        task.connect_configuration(report, name="recovery_report")
        logger = task.get_logger()
        for L, d in by_length.items():
            for name, v in d.items():
                logger.report_scalar(
                    "recovery_language_acc", name, v["language"], int(L)
                )
                logger.report_scalar("recovery_family_acc", name, v["family"], int(L))
        logger.flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()
    return report


def render_markdown(rep: dict) -> str:
    prim = f"calibration_{rep['primary_calibration']}"
    lines = [
        (
            f"### Language recovery on 1:1 ciphers — primary ranking `{prim}` "
            f"(policy {rep.get('primary_calibration_policy', 'apply')})"
        ),
        "",
        "| language | L | n | SER (true hyp.) | wrong hyp. decodes truth | lang acc | 95% CI | family acc | uncalibrated | n-gram excess | flip-rate | margin (bits, median) | unresolved at calib. precision | true−shuffled | wrong−shuffled |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, c in rep["cells"].items():
        lang, L = key.split("/L")
        a = c["accuracy"][prim]
        lines.append(
            f"| {lang} | {L} | {c['n']} | {c['ser_mean']:.3f} | {c['wrong_hypothesis_decodes_truth_rate']:.0%} | **{a['language']:.1%}** | "
            f"{a['language_ci95'][0]:.2f}–{a['language_ci95'][1]:.2f} | {a['family']:.1%} | "
            f"{c['accuracy']['uncalibrated']['language']:.1%} | "
            f"{c['accuracy']['ngram_excess_bits']['language']:.1%} | {c['flip_rate']:.2%} | "
            f"{c['margin_bits_median']:.3f} | {c['margin_unresolved_rate']:.0%} | "
            f"{c['true_minus_shuffled_bits_mean']:+.2f} | "
            f"{c['wrong_minus_shuffled_bits_mean']:+.2f} |"
        )
    lines += [
        "",
        "| length | " + " | ".join(rep["ranking_variants"]) + " |",
        "|---|" + "---|" * len(rep["ranking_variants"]),
    ]
    for L, d in rep["by_length"].items():
        lines.append(
            f"| {L} | "
            + " | ".join(
                f"{d[n]['language']:.1%} / {d[n]['family']:.1%}"
                for n in rep["ranking_variants"]
            )
            + " |"
        )
    lines.append(
        "| **≥200 mean** | "
        + " | ".join(
            f"{rep['ge200_mean_accuracy'][n]['language']:.1%} / {rep['ge200_mean_accuracy'][n]['family']:.1%}"
            for n in rep["ranking_variants"]
        )
        + " |"
    )
    if rep.get("budget_sweep_decipherment_regime"):
        sw = rep["budget_sweep_decipherment_regime"]
        budgets = sorted({int(B) for d in sw.values() for B in d})
        lines += [
            "",
            "Budget sweep, decipherment regime (replicate flip-rate of the calibrated ranking / language accuracy):",
            "",
            "| length | " + " | ".join(f"B={B}" for B in budgets) + " |",
            "|---|" + "---|" * len(budgets),
        ]
        for L, d in sw.items():
            lines.append(
                f"| {L} | "
                + " | ".join(
                    (
                        f"{d[str(B)]['flip_rate']:.2%} / {d[str(B)]['language_acc']:.1%}"
                        if str(B) in d
                        else "—"
                    )
                    for B in budgets
                )
                + " |"
            )
        lines.append(
            f"\nchosen budget (flip < 1% at every length): **{rep['budget_sweep_chosen']}**"
        )
    verdict = "PASS" if rep["acceptance"]["pass"] else "FAIL"
    lines += [
        "",
        (
            f"(language / family accuracy; bar {rep['hauer_kondrak_bar']:.1%} at "
            f"≥200 chars; primary {verdict} at "
            f"{rep['acceptance']['primary_language_acc_ge200']:.1%})"
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = data_root()
    p.add_argument("--stage", choices=["solve", "score", "report"], required=True)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fresh", action="store_true", help="ignore existing solves")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_b-85m-seed0/ckpt_final.pt"
    )
    p.add_argument(
        "--budget",
        type=int,
        default=64,
        help="primary timestep-draw budget (design §5a)",
    )
    p.add_argument(
        "--reps", type=int, default=4, help="replicate seeds at the primary budget"
    )
    p.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128],
        help="budget sweep (3.2, decipherment regime)",
    )
    p.add_argument("--sweep-reps", type=int, default=4)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--calibrations", nargs="+", default=["v1", "v2", "v3"])
    p.add_argument("--primary", default=CALIBRATION_VERSION)
    p.add_argument(
        "--solves",
        type=Path,
        default=root / "analysis" / "phase3" / f"{OUT}_solves.json",
        help="the solve-stage artifact (rung-1 decipherments are backbone-"
        "independent, so Phase 4 re-scores the Phase-3 solves)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "analysis" / "phase3",
        help="where the score/report stages write (Phase 4: analysis/phase4)",
    )
    p.add_argument(
        "--phase-tag", default="phase3", help="ClearML phase tag of the report task"
    )
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.stage == "solve":
        stage_solve(args, root)
    elif args.stage == "score":
        stage_score(args, root)
    else:
        stage_report(args, root)


if __name__ == "__main__":
    main()
