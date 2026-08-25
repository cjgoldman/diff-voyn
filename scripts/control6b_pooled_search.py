"""Control experiment 6b — does the per-hypothesis inner search bias the
language ranking?  (docs/project_goals_and_progress.md §9, item 6b)

The Phase-5 two-tier pipeline solves every ciphertext three times, once
under each language's n-gram LM, and the three searches are not equally
successful (Latin 93–97 %, Italian / German 100 % at rung 1).  A language
that is harder to *search* presents the judge with a worse decipherment for
reasons unrelated to whether the text is in that language.  This script
removes the hypothesis from the search stage and asks whether the ranking
changes, on the synthetic suites with known answers (the frozen evaluator
is untouched):

  arm A  the current hybrid — per-hypothesis n-gram search (the frozen
         Phase-5 shortlists in ``rung{1,2}_solves.json``), diffusion pick /
         MDL pick + pair polish + ELBO polish under the hypothesis, each
         hypothesis' final decode scored under its own dial;
  arm B  a *language-pooled* n-gram (orders 1–5, trained on the three
         train corpora mixed with equal character weight) drives ONE search
         per ciphertext with the same per-search budget; the single decode
         is scored under the three language dials;
  arm B' as B, but the shortlist entry is selected by the language-free
         (NULL-language, "uncond") ELBO instead of the pooled n-gram score;
  arm C  arm B followed by the discrete ELBO polish (``ladder.elbo_polish``)
         run on the language-free dial (pair swaps only at rung 1 so the
         key stays bijective).

Per arm the report gives per-language solve success, language recovery,
and the replicate flip-rate (budget 64 × 4 masking seeds) on the
Latin–Italian and Latin–German pairs with the calibration-uncertainty
flag (``CalibrationTable.margin_uncertainty_bits``).

Stages (``--rung 1|2``):
    uv run python scripts/control6b_pooled_search.py --stage train-lm
    uv run python scripts/control6b_pooled_search.py --rung 1 --stage solve --workers 12
    uv run python scripts/control6b_pooled_search.py --rung 1 --stage score   # GPU
    uv run python scripts/control6b_pooled_search.py --rung 1 --stage report

Artifacts: ``DATA_ROOT/ngram_lms/v1/pooled.npz`` and
``DATA_ROOT/analysis/control6/control6b_rung{1,2}_{solves,scores,report}.{json,md}``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if "solve" in sys.argv or "train-lm" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU stages

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX
from diff_voyn.heads.ladder import (
    elbo_polish,
    load_done,
    run_pool,
    wilson,
    write_json_atomic,
)
from diff_voyn.heads.ngram import (
    NGRAM_LM_VERSION,
    NgramLM,
    _ngram_counts,
    _read_docs,
    _witten_bell,
    lm_dir,
    load_lm,
    save_lm,
)
from diff_voyn.heads.scale import choice_bits
from diff_voyn.metrology.calibration import (
    CALIBRATION_VERSION,
    CalibrationTable,
    family_of,
    rank_languages,
)
from diff_voyn.vocab import VOCAB_VERSION

LANGS = tuple(LANG_TO_INDEX)
POOLED = "pooled"
UNCOND = "uncond"
CONDS = list(LANGS) + [UNCOND]
KEY = ("language", "length", "trial")
PAIRS = (("latin", "italian"), ("latin", "german"))
SOLVED_SER = {1: 0.05, 2: 0.019}  # rung-1 "solved" / rung-2 acceptance thresholds
_EV = None


# ---------------------------------------------------------------------------
# pooled LM


def train_pooled_lm(corpus_dir: Path, splits: dict, k_max: int = 5) -> NgramLM:
    """Witten-Bell LM on the three train splits mixed with EQUAL character
    weight per language (raw pooling would be 74 % German: 88.7M vs 26.8M
    vs 3.6M chars).  Counts are re-weighted per language so each language
    contributes one third of every n-gram table; the Witten-Bell type counts
    (``counts > 0``) are unaffected by the scaling."""
    streams = {
        lang: _read_docs(
            corpus_dir, lang, [d["doc_id"] for d in splits["languages"][lang]["train"]]
        )
        for lang in LANGS
    }
    n_chars = {lang: sum(len(s) for s in streams[lang]) for lang in LANGS}
    target = float(min(n_chars.values()))
    weight = {lang: target / n_chars[lang] for lang in LANGS}
    logp: dict[int, np.ndarray] = {}
    uni = sum(weight[l] * _ngram_counts(streams[l], 1) for l in LANGS)
    p1 = (uni + 1.0) / (uni + 1.0).sum()
    logp[1] = np.log(p1).astype(np.float32)
    for k in range(2, k_max + 1):
        counts = sum(weight[l] * _ngram_counts(streams[l], k) for l in LANGS)
        logp[k] = _witten_bell(counts, logp[k - 1])
    lm = NgramLM(
        POOLED,
        k_max,
        logp,
        {
            "ngram_lm_version": NGRAM_LM_VERSION,
            "corpus_version": splits["corpus_version"],
            "splits_version": splits["splits_version"],
            "vocab_version": VOCAB_VERSION,
            "k_max": k_max,
            "train_docs": {l: len(streams[l]) for l in LANGS},
            "train_chars": n_chars,
            "language_weights": weight,
            "mixing": "equal character weight per language",
            "smoothing": "interpolated witten-bell",
        },
    )
    per_lang = {}
    for lang in LANGS:
        held = _read_docs(
            corpus_dir,
            lang,
            [d["doc_id"] for d in splits["languages"][lang]["heldout"]],
        )
        n = sum(len(h) for h in held)
        per_lang[lang] = float(sum(-lm.score_ids(h) for h in held) / (n * np.log(2.0)))
    lm.meta["heldout_bits_per_char_by_language"] = per_lang
    lm.meta["heldout_bits_per_char"] = float(np.mean(list(per_lang.values())))
    return lm


def stage_train_lm(args, root):
    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    t0 = time.time()
    lm = train_pooled_lm(corpus_dir, splits, k_max=args.k_max)
    path = save_lm(lm, lm_dir())
    own = {
        lang: load_lm(lm_dir() / f"{lang}.npz").meta["heldout_bits_per_char"]
        for lang in LANGS
    }
    summary = {
        "path": str(path),
        "train_seconds": round(time.time() - t0, 1),
        "meta": lm.meta,
        "own_language_lm_heldout_bits": own,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pooled_lm_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _build_evaluator(pooled: bool):
    from diff_voyn.heads.evaluator import NgramEvaluator
    from diff_voyn.heads.harness import ngram_calibration_offsets

    lms = {lang: load_lm(lm_dir() / f"{lang}.npz") for lang in LANGS}
    if pooled:
        lms[POOLED] = load_lm(lm_dir() / f"{POOLED}.npz")
    return NgramEvaluator(lms, calibration_offsets_bits=ngram_calibration_offsets(lms))


# ---------------------------------------------------------------------------
# solve (CPU): one pooled search per ciphertext, same per-search budget as
# one hypothesis of arm A


def _suite(args, root):
    if args.rung == 1:
        import rung1_diffusion as r1

        return r1.build_suite(root, args.trials, args.lengths, args.seed)
    import rung2_diffusion as r2

    return r2.build_suite(root, args.trials, args.length, args.n_symbols, args.seed)


def _solve_one_rung1(job: dict) -> dict:
    from diff_voyn.heads.rung1_sinkhorn import SinkhornSubstitutionHead

    cipher = np.asarray(job["cipher_ids"], dtype=np.int64)
    plain = np.asarray(job["plain_ids"], dtype=np.int64)
    out = {k: job[k] for k in KEY}
    out.update(
        plain_ids=plain.tolist(), cipher_ids=cipher.tolist(), true_map=job["true_map"]
    )
    t0 = time.time()
    head = SinkhornSubstitutionHead(_EV, seed=job["trial"])
    res = head.solve(
        cipher, language=POOLED, restarts=job["restarts"], shortlist=job["shortlist"]
    )
    out["pooled"] = {
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
    torch.set_num_threads(1)
    _EV = _build_evaluator(pooled=True)
    path = args.out_dir / f"control6b_rung{args.rung}_solves.json"
    jobs = _suite(args, root)
    done = load_done(path, KEY) if not args.fresh else {}
    todo = [j for j in jobs if tuple(j[k] for k in KEY) not in done]
    print(f"{len(done)} done, {len(todo)} to solve", flush=True)
    results = list(done.values())
    settings = {
        "rung": args.rung,
        "trials": args.trials,
        "seed": args.seed,
        "search_lm": POOLED,
        "plaintext_source": "held-out split v1",
    }
    if args.rung == 1:
        settings.update(
            lengths=list(args.lengths), restarts=args.restarts, shortlist=args.shortlist
        )
        for j in todo:
            j["restarts"], j["shortlist"] = args.restarts, args.shortlist
        todo.sort(key=lambda j: -j["length"])
        t0 = time.time()

        def on_result(_i, r, _elapsed):
            results.append(r)
            if len(results) % 10 == 0:
                write_json_atomic(
                    path,
                    {
                        "created_utc": datetime.now(UTC).isoformat(),
                        "settings": settings,
                        "instances": results,
                    },
                )
                print(f"  solved {len(results)} ({time.time()-t0:.0f}s)", flush=True)

        run_pool(_solve_one_rung1, todo, workers=args.workers, on_result=on_result)
    else:
        from diff_voyn.heads.rung2_homophonic import HomophonicHead

        settings.update(
            length=args.length,
            n_symbols=args.n_symbols,
            restarts=args.restarts2,
            restarts_nopenalty=args.restarts_nopenalty,
            sa_steps=args.sa_steps,
            shortlist=args.shortlist2,
        )
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
            t1 = time.time()
            head = HomophonicHead(_EV, seed=job["trial"])
            res = head.solve_parallel(
                cipher,
                job["n_symbols"],
                language=POOLED,
                restarts=args.restarts2,
                workers=args.workers,
                sa_steps=args.sa_steps,
                shortlist=args.shortlist2,
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
            head0 = HomophonicHead(
                _EV, seed=job["trial"] + 777, freq_penalty_weight=0.0
            )
            res0 = head0.solve_parallel(
                cipher,
                job["n_symbols"],
                language=POOLED,
                restarts=args.restarts_nopenalty,
                workers=args.workers,
                sa_steps=args.sa_steps,
                shortlist=args.shortlist2 // 2,
            )
            short += [
                {
                    "map": m.tolist(),
                    "penalized": float(head._objective(m[cipher], POOLED)),
                    "raw_ll": r,
                    "source": "nopenalty",
                    "ser": float(np.mean(m[cipher] != plain)),
                }
                for m, s, r in res0.shortlist
            ]
            rec["pooled"] = {
                "shortlist": short,
                "n_evals": res.n_evals + res0.n_evals,
                "seconds": round(time.time() - t1, 1),
            }
            results.append(rec)
            print(
                f"  [{i}/{len(todo)}] {job['language']} t{job['trial']}: SER best "
                f"{short[0]['ser']:.3f} (oracle {min(x['ser'] for x in short):.3f}) "
                f"{time.time()-t1:.0f}s total {time.time()-t0:.0f}s",
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
    write_json_atomic(
        path,
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "settings": settings,
            "instances": results,
        },
    )
    print(f"written {path}")


# ---------------------------------------------------------------------------
# score (GPU)


def _bits_matrix(ev, rows: np.ndarray, conds, seeds, budget, batch=96) -> np.ndarray:
    """[n_rows, len(conds), len(seeds)] bits/char, paired masks within a
    (seed) across rows and conditions."""
    from diff_voyn.heads.two_tier import paired_bits

    out = np.zeros((rows.shape[0], len(conds), len(seeds)))
    for k, sd in enumerate(seeds):
        out[:, :, k] = paired_bits(
            ev, rows, list(conds), n_strata=budget, seed=sd, batch=batch
        )
    return out


def _arm_record(
    decode_key, cipher, plain, bits, conds, choice_fn=None, **extra
) -> dict:
    key = np.asarray(decode_key, dtype=np.int64)
    dec = key[cipher]
    cb = float(choice_fn(key, dec)) if choice_fn else 0.0
    return {
        "key": key.tolist(),
        "ser": float(np.mean(dec != plain)),
        "choice_bits_per_char": cb,
        # bits[cond][seed]
        "bits": {c: [float(x) for x in bits[j]] for j, c in enumerate(conds)},
        **extra,
    }


def stage_score(args, root):
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator

    torch.set_float32_matmul_precision("high")
    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    ng = _build_evaluator(pooled=True)
    a_solves = json.loads(
        (root / "analysis" / "phase5" / f"rung{args.rung}_solves.json").read_text()
    )
    b_solves = json.loads(
        (args.out_dir / f"control6b_rung{args.rung}_solves.json").read_text()
    )
    a_by = {tuple(r[k] for k in KEY): r for r in a_solves["instances"]}
    path = args.out_dir / f"control6b_rung{args.rung}_scores.json"
    done = load_done(path, KEY) if not args.fresh else {}
    inst = [r for r in b_solves["instances"] if tuple(r[k] for k in KEY) not in done]
    inst.sort(key=lambda r: (r["length"], r["language"], r["trial"]))
    print(f"{len(done)} scored, {len(inst)} to score", flush=True)
    results = list(done.values())
    kind = "sub1to1" if args.rung == 1 else "homophonic"
    seeds_of = lambda base: [base + 1000 * k for k in range(args.reps)]
    t0 = time.time()

    def header():
        return {
            "created_utc": datetime.now(UTC).isoformat(),
            "evaluator": ev.meta,
            "scoring": {
                "budget": args.budget,
                "reps": args.reps,
                "conditions": CONDS,
                "elbo_sweeps": args.elbo_sweeps,
                "elbo_budget": args.elbo_budget,
                "crn": "paired masks across all rows and conditions within a seed; "
                "replicate seeds base+1000k",
            },
            "arm_a_solve_settings": a_solves["settings"],
            "arm_b_solve_settings": b_solves["settings"],
            "instances": results,
        }

    for i, r in enumerate(inst, 1):
        key = tuple(r[k] for k in KEY)
        a = a_by[key]
        cipher = np.asarray(r["cipher_ids"], dtype=np.int64)
        plain = np.asarray(r["plain_ids"], dtype=np.int64)
        assert a["cipher_ids"] == r["cipher_ids"], "suite mismatch between arms"
        base = zlib.crc32(
            f"score/{r['language']}/{r['length']}/{r['trial']}".encode()
        ) % (2**31)
        seeds = seeds_of(base)
        rec = {k: r[k] for k in KEY}
        t1 = time.time()
        if args.rung == 1:
            choice_fn = None
            a_keys = {
                h: [np.asarray(c["perm"]) for c in a["hypotheses"][h]["shortlist"]]
                for h in LANGS
            }
            b_keys = [np.asarray(c["perm"]) for c in r["pooled"]["shortlist"]]
        else:
            choice_fn = lambda m, dec: choice_bits(kind, dec, sym_to_letter=m) / max(
                len(dec), 1
            )
            a_keys = {
                h: [np.asarray(c["map"]) for c in a["hypotheses"][h]["shortlist"]]
                for h in LANGS
            }
            b_keys = [np.asarray(c["map"]) for c in r["pooled"]["shortlist"]]

        # --- shortlist pass: every candidate of every arm, all conditions, seed 0
        all_keys = [k_ for h in LANGS for k_ in a_keys[h]] + b_keys
        rows = np.stack([k_[cipher] for k_ in all_keys])
        sb = _bits_matrix(ev, rows, CONDS, seeds[:1], args.budget)[:, :, 0]  # [n, C]
        off = 0
        a_short = {}
        for h in LANGS:
            n = len(a_keys[h])
            a_short[h] = sb[off : off + n]
            off += n
        b_short = sb[off:]
        j = {c: CONDS.index(c) for c in CONDS}

        # --- arm A: per-hypothesis selection (+ rung-2 outer tier under the hypothesis)
        arm_a = {}
        for h in LANGS:
            cb = np.array(
                [choice_fn(k_, k_[cipher]) if choice_fn else 0.0 for k_ in a_keys[h]]
            )
            mdl = a_short[h][:, j[h]] + cb
            pick = int(np.argmin(mdl))
            cur = a_keys[h][pick].copy()
            info = {
                "pick_index": pick,
                "pick_source": a["hypotheses"][h]["shortlist"][pick]["source"],
            }
            if args.rung == 2:
                from diff_voyn.heads.rung2_homophonic import HomophonicHead

                pol, _, _ = HomophonicHead(ng, seed=r["trial"]).polish_pairs(
                    cipher, cur, h
                )
                ep, ep_info = elbo_polish(
                    ev,
                    cipher,
                    pol,
                    language=h,
                    seed=seeds[0],
                    choice_fn=choice_fn if args.polish_choice_term else None,
                    choice_term_in_polish=args.polish_choice_term,
                    sweeps=args.elbo_sweeps,
                    budget=args.elbo_budget,
                    pair_swaps=False,
                )
                trio = np.stack([cur[cipher], pol[cipher], ep[cipher]])
                tb = _bits_matrix(ev, trio, [h], seeds[:1], args.budget)[:, 0, 0]
                tm = tb + np.array([choice_fn(m, m[cipher]) for m in (cur, pol, ep)])
                best = int(np.argmin(tm))
                info.update(
                    pairpolish_changed=bool((pol != cur).any()),
                    elbopolish_accepted=ep_info["accepted"],
                    elbopolish_n_changed=ep_info["n_changed"],
                    final_source=["pick", "pairpolish", "elbopolish"][best],
                )
                cur = (cur, pol, ep)[best]
            arm_a[h] = (cur, info)

        # --- arm B / B' / C
        b_pick = 0  # pooled search's own best (shortlist is best-first)
        b_key = b_keys[b_pick].copy()
        bu_pick = int(
            np.argmin(
                b_short[:, j[UNCOND]]
                + np.array(
                    [choice_fn(k_, k_[cipher]) if choice_fn else 0.0 for k_ in b_keys]
                )
            )
        )
        bu_key = b_keys[bu_pick].copy()
        c_key, c_info = elbo_polish(
            ev,
            cipher,
            b_key,
            language=UNCOND,
            seed=seeds[0],
            choice_fn=choice_fn if args.polish_choice_term else None,
            choice_term_in_polish=args.polish_choice_term,
            sweeps=args.elbo_sweeps,
            budget=args.elbo_budget,
            pair_swaps=True,
            set_moves=(args.rung == 2),
        )

        # --- replicate pass: every arm's final decode, 4 seeds, all conditions
        finals = [arm_a[h][0] for h in LANGS] + [b_key, bu_key, c_key]
        rows = np.stack([k_[cipher] for k_ in finals])
        fb = _bits_matrix(ev, rows, CONDS, seeds, args.budget)  # [6, C, R]
        rec["seeds"] = seeds
        rec["arms"] = {
            "A": {
                h: _arm_record(
                    arm_a[h][0], cipher, plain, fb[hi], CONDS, choice_fn, **arm_a[h][1]
                )
                for hi, h in enumerate(LANGS)
            },
            "B": _arm_record(
                b_key,
                cipher,
                plain,
                fb[3],
                CONDS,
                choice_fn,
                pick_index=b_pick,
                pick_source=r["pooled"]["shortlist"][b_pick]["source"],
            ),
            "B_uncond": _arm_record(
                bu_key,
                cipher,
                plain,
                fb[4],
                CONDS,
                choice_fn,
                pick_index=bu_pick,
                pick_source=r["pooled"]["shortlist"][bu_pick]["source"],
            ),
            "C": _arm_record(
                c_key,
                cipher,
                plain,
                fb[5],
                CONDS,
                choice_fn,
                accepted=c_info["accepted"],
                n_changed=c_info["n_changed"],
                sweeps=len(c_info["trace"]),
                confirm_bits=c_info["confirm_bits"],
            ),
        }
        rec["shortlist_bits_seed0"] = {
            "A": {h: a_short[h].tolist() for h in LANGS},
            "B": b_short.tolist(),
            "B_ser": [c["ser"] for c in r["pooled"]["shortlist"]],
            "B_oracle_ser": float(min(c["ser"] for c in r["pooled"]["shortlist"])),
        }
        rec["seconds"] = round(time.time() - t1, 1)
        results.append(rec)
        print(
            f"  [{i}/{len(inst)}] {r['language']} L={r['length']} t{r['trial']}: "
            f"SER A/true {rec['arms']['A'][r['language']]['ser']:.3f} "
            f"B {rec['arms']['B']['ser']:.3f} B' {rec['arms']['B_uncond']['ser']:.3f} "
            f"C {rec['arms']['C']['ser']:.3f}  {rec['seconds']}s  total {time.time()-t0:.0f}s",
            flush=True,
        )
        if i % 5 == 0 or i == len(inst):
            write_json_atomic(path, header())
    write_json_atomic(path, header())
    print(f"written {path}")


# ---------------------------------------------------------------------------
# report


def _top(bits_by_lang: dict[str, float], offs) -> list[tuple[str, float]]:
    return rank_languages(bits_by_lang, offs)


def _flip(tops: list[str]) -> float:
    return (
        float(np.mean([a != b for a, b in itertools.combinations(tops, 2)]))
        if len(tops) > 1
        else 0.0
    )


def _instance_stats(rec, arm, rung, offs, table):
    """Per-instance ranking statistics of one arm.  Arm A: each hypothesis'
    final decode under its own dial; B/B'/C: one decode under three dials.
    Rung 2 ranks on the MDL total (bits + choice bits)."""
    truth = rec["language"]
    reps = len(rec["seeds"])
    if arm == "A":
        d = rec["arms"]["A"]
        tot = {
            h: [d[h]["bits"][h][k] + d[h]["choice_bits_per_char"] for k in range(reps)]
            for h in LANGS
        }
        ser = d[truth]["ser"]
    else:
        d = rec["arms"][arm]
        tot = {
            h: [d["bits"][h][k] + d["choice_bits_per_char"] for k in range(reps)]
            for h in LANGS
        }
        ser = d["ser"]
    ranked = [_top({h: tot[h][k] for h in LANGS}, offs) for k in range(reps)]
    tops = [rk[0][0] for rk in ranked]
    top0 = tops[0]
    margin = ranked[0][1][1] - ranked[0][0][1]
    unc = table.margin_uncertainty_bits(ranked[0][0][0], ranked[0][1][0])
    out = {
        "top": top0,
        "correct": top0 == truth,
        "family_correct": family_of(top0) == family_of(truth),
        "ser": ser,
        "solved": ser <= SOLVED_SER[rung] if rung == 2 else ser < SOLVED_SER[rung],
        "margin_bits": margin,
        "margin_uncertainty_bits": unc,
        "margin_unresolved": margin < unc,
        "flip": _flip(tops),
    }
    for a, b in PAIRS:
        pt = [a if tot[a][k] < tot[b][k] else b for k in range(reps)]
        out[f"flip_{a}_{b}"] = _flip(pt)
        out[f"pair_correct_{a}_{b}"] = (pt[0] == truth) if truth in (a, b) else None
    return out


def _agg(items: list[dict]) -> dict:
    n = len(items)
    if n == 0:
        return {"n": 0}, {}
    out = {"n": n}
    for k in ("correct", "family_correct", "solved", "margin_unresolved"):
        c = sum(bool(x[k]) for x in items)
        lo, hi = wilson(c, n)
        out[k] = c / n
        out[k + "_ci95"] = [lo, hi]
    out["ser_mean"] = float(np.mean([x["ser"] for x in items]))
    out["ser_median"] = float(np.median([x["ser"] for x in items]))
    out["flip_rate"] = float(np.mean([x["flip"] for x in items]))
    out["margin_median_bits"] = float(np.median([x["margin_bits"] for x in items]))
    for a, b in PAIRS:
        out[f"flip_rate_{a}_{b}"] = float(np.mean([x[f"flip_{a}_{b}"] for x in items]))
        pc = [
            x[f"pair_correct_{a}_{b}"]
            for x in items
            if x[f"pair_correct_{a}_{b}"] is not None
        ]
        out[f"pair_acc_{a}_{b}"] = float(np.mean(pc)) if pc else None
    conf = {t: {p: 0 for p in LANGS} for t in LANGS}
    return out, conf


def stage_report(args, root):
    data = json.loads(
        (args.out_dir / f"control6b_rung{args.rung}_scores.json").read_text()
    )
    table = CalibrationTable.load(args.primary, root)
    offs = table.additive_offsets()
    arms = ("A", "B", "B_uncond", "C")
    per = {arm: [] for arm in arms}
    for rec in data["instances"]:
        for arm in arms:
            s = _instance_stats(rec, arm, args.rung, offs, table)
            s.update(language=rec["language"], length=rec["length"], trial=rec["trial"])
            per[arm].append(s)
    ge = 200 if args.rung == 1 else 0
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "rung": args.rung,
        "n_instances": len(data["instances"]),
        "evaluator": data["evaluator"],
        "scoring": data["scoring"],
        "primary_calibration": args.primary,
        "arm_a_solve_settings": data["arm_a_solve_settings"],
        "arm_b_solve_settings": data["arm_b_solve_settings"],
        "arms": {},
    }
    for arm in arms:
        items = per[arm]
        sel = [x for x in items if x["length"] >= ge]
        a = {
            "all": _agg(items)[0],
            f"ge{ge}": _agg(sel)[0],
            "by_language": {
                l: _agg([x for x in sel if x["language"] == l])[0] for l in LANGS
            },
            "confusion": {
                t: {
                    p: sum(1 for x in sel if x["language"] == t and x["top"] == p)
                    for p in LANGS
                }
                for t in LANGS
            },
        }
        if args.rung == 1:
            a["by_length"] = {
                str(L): _agg([x for x in items if x["length"] == L])[0]
                for L in sorted({x["length"] for x in items})
            }
        report["arms"][arm] = a
    # paired comparisons: does the ranking change, instance by instance?
    idx = {
        arm: {(x["language"], x["length"], x["trial"]): x for x in per[arm]}
        for arm in arms
    }
    keys = [k for k in idx["A"] if k[1] >= ge]
    report["paired_vs_A"] = {}
    for arm in arms[1:]:
        diff = [(idx["A"][k]["top"], idx[arm][k]["top"]) for k in keys]
        changed = sum(1 for a_, b_ in diff if a_ != b_)
        a_right_b_wrong = sum(
            1 for k in keys if idx["A"][k]["correct"] and not idx[arm][k]["correct"]
        )
        b_right_a_wrong = sum(
            1 for k in keys if idx[arm][k]["correct"] and not idx["A"][k]["correct"]
        )
        report["paired_vs_A"][arm] = {
            "n": len(keys),
            "top_changed": changed,
            "A_correct_arm_wrong": a_right_b_wrong,
            "arm_correct_A_wrong": b_right_a_wrong,
            "changed_instances": [
                {
                    "key": list(k),
                    "A": idx["A"][k]["top"],
                    arm: idx[arm][k]["top"],
                    "A_ser": idx["A"][k]["ser"],
                    f"{arm}_ser": idx[arm][k]["ser"],
                }
                for k in keys
                if idx["A"][k]["top"] != idx[arm][k]["top"]
            ],
        }
    # arm C polish bookkeeping
    cs = [rec["arms"]["C"] for rec in data["instances"]]
    report["arm_c_polish"] = {
        "accepted_rate": float(np.mean([c["accepted"] for c in cs])),
        "mean_n_changed": float(np.mean([c["n_changed"] for c in cs])),
        "ser_B_mean": float(
            np.mean([rec["arms"]["B"]["ser"] for rec in data["instances"]])
        ),
        "ser_C_mean": float(np.mean([c["ser"] for c in cs])),
        "improved": sum(
            1
            for rec in data["instances"]
            if rec["arms"]["C"]["ser"] < rec["arms"]["B"]["ser"]
        ),
        "degraded": sum(
            1
            for rec in data["instances"]
            if rec["arms"]["C"]["ser"] > rec["arms"]["B"]["ser"]
        ),
    }
    report["instances"] = {arm: per[arm] for arm in arms}
    out = args.out_dir / f"control6b_rung{args.rung}_report.json"
    write_json_atomic(out, report)
    md = render_md(report)
    (args.out_dir / f"control6b_rung{args.rung}_report.md").write_text(md)
    print(md)
    print(f"written {out}")


def render_md(rep: dict) -> str:
    rung = rep["rung"]
    ge = "≥200 chars" if rung == 1 else "408 chars"
    names = {
        "A": "A hybrid (per-hypothesis search)",
        "B": "B pooled search",
        "B_uncond": "B' pooled search, uncond-ELBO pick",
        "C": "C = B + ELBO polish (uncond dial)",
    }
    L = [
        f"# Control 6b — rung {rung}: hypothesis-free inner search vs the hybrid\n",
        f"{rep['n_instances']} instances; budget {rep['scoring']['budget']} × {rep['scoring']['reps']} seeds; calibration `{rep['primary_calibration']}`.",
        (
            f"Arm-A search: {json.dumps({k: v for k, v in rep['arm_a_solve_settings'].items() if k in ('restarts','shortlist','restarts_nopenalty','sa_steps')})}; "
            f"arm-B search: {json.dumps({k: v for k, v in rep['arm_b_solve_settings'].items() if k in ('restarts','shortlist','restarts_nopenalty','sa_steps')})} (one search per ciphertext instead of three).\n"
        ),
        f"## Summary ({ge})\n",
        "| arm | solved (true-language decode) | language recovery | family | flip-rate (3-way) | flip L–I | flip L–G | pair acc L–I | pair acc L–G | margin unresolved | median margin bits | mean SER |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, a in rep["arms"].items():
        g = a[next(k for k in a if k.startswith("ge"))]
        L.append(
            f"| {names[arm]} | {g['solved']:.3f} | {g['correct']:.3f} [{g['correct_ci95'][0]:.2f},{g['correct_ci95'][1]:.2f}] | {g['family_correct']:.3f} | {g['flip_rate']:.3f} | {g['flip_rate_latin_italian']:.3f} | {g['flip_rate_latin_german']:.3f} | {g['pair_acc_latin_italian']:.3f} | {g['pair_acc_latin_german']:.3f} | {g['margin_unresolved']:.3f} | {g['margin_median_bits']:.3f} | {g['ser_mean']:.4f} |"
        )
    L.append(f"\n## Per language ({ge})\n")
    L.append(
        "| arm | lang | n | solved | mean SER | recovery | flip 3-way | flip L–I | flip L–G | unresolved |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for arm, a in rep["arms"].items():
        for l, g in a["by_language"].items():
            L.append(
                f"| {arm} | {l} | {g['n']} | {g['solved']:.3f} | {g['ser_mean']:.4f} | {g['correct']:.3f} | {g['flip_rate']:.3f} | {g['flip_rate_latin_italian']:.3f} | {g['flip_rate_latin_german']:.3f} | {g['margin_unresolved']:.3f} |"
            )
    L.append(f"\n## Confusion (rows = truth, cols = top-1, {ge})\n")
    for arm, a in rep["arms"].items():
        L.append(
            f"**{arm}**: "
            + "; ".join(
                f"{t}→" + ",".join(f"{p}:{a['confusion'][t][p]}" for p in LANGS)
                for t in LANGS
            )
        )
    if rung == 1:
        L.append("\n## By length (all languages)\n")
        L.append(
            "| length | "
            + " | ".join(f"{arm} solved / recovery / flip" for arm in rep["arms"])
            + " |"
        )
        L.append("|---|" + "---|" * len(rep["arms"]))
        for Ln in rep["arms"]["A"]["by_length"]:
            L.append(
                f"| {Ln} | "
                + " | ".join(
                    f"{rep['arms'][arm]['by_length'][Ln]['solved']:.2f} / {rep['arms'][arm]['by_length'][Ln]['correct']:.2f} / {rep['arms'][arm]['by_length'][Ln]['flip_rate']:.3f}"
                    for arm in rep["arms"]
                )
                + " |"
            )
    L.append(f"\n## Paired ranking changes vs arm A ({ge})\n")
    for arm, p in rep["paired_vs_A"].items():
        L.append(
            f"- **{arm}**: top-1 changed on {p['top_changed']}/{p['n']} instances; A right & {arm} wrong {p['A_correct_arm_wrong']}, {arm} right & A wrong {p['arm_correct_A_wrong']}."
        )
        for c in p["changed_instances"][:12]:
            L.append(
                f"  - {c['key']}: A→{c['A']} (SER {c['A_ser']:.3f}), {arm}→{c[arm]} (SER {c[arm+'_ser']:.3f})"
            )
    c = rep["arm_c_polish"]
    L.append(
        f"\n## Arm C polish (uncond dial)\n\naccepted {c['accepted_rate']:.2f}, mean symbols changed {c['mean_n_changed']:.2f}; SER B {c['ser_B_mean']:.4f} → C {c['ser_C_mean']:.4f} (improved {c['improved']}, degraded {c['degraded']})."
    )
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------


def main():
    root = data_root()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stage", required=True, choices=["train-lm", "solve", "score", "report"]
    )
    ap.add_argument("--rung", type=int, default=1, choices=[1, 2])
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    # rung 1 (frozen Phase-5 suite settings)
    ap.add_argument("--trials", type=int, default=None, help="rung 1: 20, rung 2: 6")
    ap.add_argument("--lengths", type=int, nargs="+", default=[50, 100, 200, 400, 700])
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--shortlist", type=int, default=8)
    # rung 2
    ap.add_argument("--length", type=int, default=408)
    ap.add_argument("--n-symbols", type=int, default=54)
    ap.add_argument("--restarts2", type=int, default=120)
    ap.add_argument("--restarts-nopenalty", type=int, default=24)
    ap.add_argument("--sa-steps", type=int, default=100_000)
    ap.add_argument("--shortlist2", type=int, default=12)
    # scoring
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=root / "runs" / "phase_c-85m-seed0" / "ckpt_final.pt",
    )
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--elbo-sweeps", type=int, default=6)
    ap.add_argument(
        "--polish-choice-term",
        action="store_true",
        help="put the MDL choice term in the elbo_polish objective (the recorded "
        "arm-C behaviour; harmful at Borg scale — docs/race_polish_plan.md §7)",
    )
    ap.add_argument("--elbo-budget", type=int, default=8)
    ap.add_argument("--primary", default=CALIBRATION_VERSION)
    ap.add_argument("--out-dir", type=Path, default=root / "analysis" / "control6")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.trials is None:
        args.trials = 20 if args.rung == 1 else 6
    args.out_dir.mkdir(parents=True, exist_ok=True)
    {
        "train-lm": stage_train_lm,
        "solve": stage_solve,
        "score": stage_score,
        "report": stage_report,
    }[args.stage](args, root)


if __name__ == "__main__":
    main()
