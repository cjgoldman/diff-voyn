"""Task 5.6 — uniform-scale cross-head comparison (design §8 R5).

Assembles every Phase-5 (cipher × language) cell on ONE scale from the rung
artifacts (``rung{1..4}_scores.json``: the true-hypothesis final decodes
under every language condition) and adds the CROSS-APPLICATION cells — the
rung-1 and rung-2 heads attacking the OTHER ciphers' ciphertexts (a 1:1
head on a Naibbe glyph stream, a homophonic head on an arithmetic stream,
…), which is what a (cipher × language) table of the VMS will contain:
every head applied to the same ciphertext.

Scale (``diff_voyn/heads/scale.py``): per cell
    plaintext bits   = calibrated bits/plaintext-char × n_plain   (frozen evaluator)
    key bits         = description length of the key class
    choice bits      = the cipher's encoding freedom given plaintext + key
    total            = plaintext + key + choice, reported PER CIPHERTEXT SYMBOL
Within a cipher hypothesis the language ranking uses the calibrated
plaintext bits/char (identical penalties cancel); across cipher hypotheses
the total per ciphertext symbol is the MDL comparator — heads emit
different plaintext lengths for the same ciphertext, so "bits per plaintext
char" alone is not comparable across heads.

Acceptance (5.6, "cross-head scores demonstrably comparable on a shared
scale"), checked on the synthetic instances:
  (a) same instrument: every cell's plaintext bits come from the frozen
      evaluator's letter-stream estimator (the Phase-3 scale) — verified by
      scoring each true plaintext alongside its decode;
  (b) MDL picks the true cipher class: on each ciphertext the true head's
      total per ciphertext symbol beats every cross-applied head's;
  (c) the penalty orders equally-good decodes by simplicity: on 1:1
      ciphertexts the rung-2 head recovers the same text as rung 1 and the
      description-length term ranks rung 1 first.

Stages:  cross (CPU pool: cross-application solves) → score (GPU) → report
Artifacts: DATA_ROOT/analysis/phase5/crosshead_{solves,scores,report}.*
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
if "cross" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.data.loader import LANG_TO_INDEX
from diff_voyn.heads.ladder import load_done, run_pool, write_json_atomic
from diff_voyn.heads.naibbe_parse import NaibbeParser
from diff_voyn.heads.ngram import A
from diff_voyn.heads.scale import choice_bits, key_bits
from diff_voyn.metrology import CALIBRATION_VERSION, CalibrationTable, calibrate_bits

LANGS = tuple(LANG_TO_INDEX)
KEY = ("cipher", "language", "trial", "head", "hypothesis")
_EV = None


def _build_ngram_evaluator():
    from diff_voyn.heads.evaluator import NgramEvaluator
    from diff_voyn.heads.harness import ngram_calibration_offsets
    from diff_voyn.heads.ngram import lm_dir, load_lm

    lms = {lang: load_lm(lm_dir() / f"{lang}.npz") for lang in LANGS}
    return NgramEvaluator(lms, calibration_offsets_bits=ngram_calibration_offsets(lms))


# -- the ciphertexts, as symbol streams, from the rung artifacts -------------


def load_ciphertexts(out_dir: Path, per_language: int, max_symbols: int) -> list[dict]:
    """One record per (cipher, language, trial): symbol stream (ids), its
    plaintext, n_cipher_symbols, and the rung's own true-hypothesis final
    decode bits from the scores file."""
    recs = []
    # rung 1 / 2: cipher_ids are the symbol stream
    for rung, kind, length_sel in (
        ("rung1", "sub1to1", 700),
        ("rung2", "homophonic", None),
    ):
        sol = json.loads((out_dir / f"{rung}_solves.json").read_text())["instances"]
        for lang in LANGS:
            rows = [
                r
                for r in sol
                if r["language"] == lang
                and (length_sel is None or r["length"] == length_sel)
            ]
            for r in sorted(rows, key=lambda r: r["trial"])[:per_language]:
                recs.append(
                    {
                        "cipher": kind,
                        "language": lang,
                        "trial": r["trial"],
                        "symbols": r["cipher_ids"],
                        "n_symbols": int(max(r["cipher_ids"]) + 1),
                        "plain_ids": r["plain_ids"],
                        "true_map": r.get("true_map"),
                    }
                )
    # rung 3: glyph characters of the token stream are the symbols
    p3 = out_dir / "rung3_solves.json"
    if p3.exists():
        sol = json.loads(p3.read_text())["instances"]
        for lang in LANGS:
            rows = [r for r in sol if r["language"] == lang]
            for r in sorted(rows, key=lambda r: r["trial"])[:per_language]:
                stream = "".join(r["tokens"])
                alphabet = sorted(set(stream))
                sym = [alphabet.index(c) for c in stream]
                recs.append(
                    {
                        "cipher": "naibbe",
                        "language": lang,
                        "trial": r["trial"],
                        "symbols": sym[:max_symbols],
                        "n_symbols": len(alphabet),
                        "plain_ids": r["plain_ids"],
                        "n_tokens": len(r["tokens"]),
                        "n_cipher_chars_full": len(stream),
                        "window_note": f"first {min(max_symbols, len(stream))} of {len(stream)} glyph chars for cross-application",
                    }
                )
    p4 = out_dir / "rung4_solves.json"
    if p4.exists():
        sol = json.loads(p4.read_text())["instances"]
        for lang in LANGS:
            rows = [r for r in sol if r["language"] == lang and r["hypothesis"] == lang]
            for r in sorted(rows, key=lambda r: r["trial"])[:per_language]:
                recs.append(
                    {
                        "cipher": "arithmetic",
                        "language": lang,
                        "trial": r["trial"],
                        "symbols": r["char_ids"],
                        "n_symbols": 16,
                        "plain_ids": r["plain_ids"],
                    }
                )
    return recs


def _cross_one(job):
    from diff_voyn.heads.rung1_sinkhorn import SinkhornSubstitutionHead
    from diff_voyn.heads.rung2_homophonic import HomophonicHead

    t0 = time.time()
    sym = np.asarray(job["symbols"], dtype=np.int64)
    out = {k: job[k] for k in KEY}
    if job["head"] == "sub1to1":
        head = SinkhornSubstitutionHead(_EV, seed=job["trial"])
        res = head.solve(sym, language=job["hypothesis"], restarts=2, shortlist=4)
        cands = [(perm[sym], hard, perm) for perm, hard, _ in res.shortlist]
    else:
        head = HomophonicHead(_EV, seed=job["trial"])
        res = head.solve(
            sym,
            job["n_symbols"],
            language=job["hypothesis"],
            restarts=job["restarts"],
            sa_steps=job["sa_steps"],
        )
        cands = [(res.sym_to_letter[sym], res.hard_score, res.sym_to_letter)]
    out["candidates"] = [
        {"decode": d.tolist(), "inner": float(s), "map": m.tolist()}
        for d, s, m in cands
    ]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def stage_cross(args, root):
    global _EV
    path = args.out_dir / "crosshead_solves.json"
    texts = load_ciphertexts(args.out_dir, args.per_language, args.max_symbols)
    jobs = []
    for t in texts:
        for head in ("sub1to1", "homophonic"):
            if head == "sub1to1" and t["n_symbols"] > A:
                continue  # a bijective head cannot absorb more symbols than letters
            for hyp in LANGS:
                jobs.append(
                    {
                        **t,
                        "head": head,
                        "hypothesis": hyp,
                        "restarts": args.restarts,
                        "sa_steps": args.sa_steps,
                    }
                )
    done = load_done(path, KEY) if not args.fresh else {}
    todo = [j for j in jobs if tuple(j[k] for k in KEY) not in done]
    print(
        f"{len(texts)} ciphertexts, {len(done)} cells done, {len(todo)} to solve",
        flush=True,
    )
    _EV = _build_ngram_evaluator()
    results = list(done.values())

    def on_result(i, r, el):
        results.append(r)
        print(
            f"  [{i}/{len(todo)}] {r['cipher']} {r['language']} t{r['trial']} head={r['head']} hyp={r['hypothesis']} ({r['seconds']}s, {el:.0f}s)",
            flush=True,
        )
        write_json_atomic(
            path,
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "settings": {
                    "per_language": args.per_language,
                    "max_symbols": args.max_symbols,
                    "restarts": args.restarts,
                    "sa_steps": args.sa_steps,
                },
                "instances": results,
            },
        )

    run_pool(_cross_one, todo, workers=args.workers, on_result=on_result)
    write_json_atomic(
        path,
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "settings": {
                "per_language": args.per_language,
                "max_symbols": args.max_symbols,
                "restarts": args.restarts,
                "sa_steps": args.sa_steps,
            },
            "instances": results,
        },
    )


def stage_score(args, root):
    """Diffusion bits of every cross-application decode (own condition) and
    of every ciphertext's true plaintext under every condition."""
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator
    from diff_voyn.heads.two_tier import paired_bits

    torch.set_float32_matmul_precision("high")
    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    sol = json.loads((args.out_dir / "crosshead_solves.json").read_text())
    texts = load_ciphertexts(args.out_dir, args.per_language, args.max_symbols)
    out = {"evaluator": ev.meta, "budget": args.budget, "cross": [], "plain": []}
    for t in texts:
        seed = zlib.crc32(
            f"cross/{t['cipher']}/{t['language']}/{t['trial']}".encode()
        ) % (2**31)
        plain = np.asarray(t["plain_ids"], dtype=np.int64)
        pb = paired_bits(ev, plain[None], list(LANGS), n_strata=args.budget, seed=seed)[
            0
        ]
        out["plain"].append(
            {
                "cipher": t["cipher"],
                "language": t["language"],
                "trial": t["trial"],
                "n_plain": len(plain),
                "n_cipher_symbols": len(t["symbols"]),
                "n_symbols": t["n_symbols"],
                "bits": {l: float(pb[j]) for j, l in enumerate(LANGS)},
            }
        )
    streams = {
        (t["cipher"], t["language"], t["trial"]): np.asarray(
            t["symbols"], dtype=np.int64
        )
        for t in texts
    }
    for r in sol["instances"]:
        seed = zlib.crc32(
            f"cross/{r['cipher']}/{r['language']}/{r['trial']}".encode()
        ) % (2**31)
        rows = np.stack(
            [np.asarray(c["decode"], dtype=np.int64) for c in r["candidates"]]
        )
        pb = paired_bits(ev, rows, [r["hypothesis"]], n_strata=args.budget, seed=seed)[
            :, 0
        ]
        best = int(np.argmin(pb))
        c = r["candidates"][best]
        m = np.asarray(c["map"], dtype=np.int64)
        sym = streams[(r["cipher"], r["language"], r["trial"])]
        if r["head"] == "homophonic":
            n_hom = np.bincount(m, minlength=A)
            cb = float(np.log2(np.maximum(n_hom[m[sym]], 1)).sum())
        else:
            cb = 0.0
        out["cross"].append(
            {
                **{k: r[k] for k in KEY},
                "bits": float(pb[best]),
                "n_plain": len(c["decode"]),
                "map": c["map"],
                "inner": c["inner"],
                "choice_bits": cb,
                "key_bits": key_bits(r["head"], n_symbols=len(m)),
            }
        )
        print(
            f"  {r['cipher']} {r['language']} t{r['trial']} head={r['head']} hyp={r['hypothesis']}: {pb[best]:.3f} bits/char",
            flush=True,
        )
    write_json_atomic(
        args.out_dir / "crosshead_scores.json",
        {"created_utc": datetime.now(UTC).isoformat(), **out},
    )


def _own_cells(out_dir: Path, per_language: int, offs) -> list[dict]:
    """The true-head cells from the rung artifacts: final decode bits under
    every hypothesis + penalty terms."""
    cells = []
    parser_w = NaibbeParser.CARD_WEIGHTS[False]
    for rung, kind in (
        ("rung1", "sub1to1"),
        ("rung2", "homophonic"),
        ("rung3", "naibbe"),
        ("rung4", "arithmetic"),
    ):
        p = out_dir / f"{rung}_scores.json"
        if not p.exists():
            continue
        sc = json.loads(p.read_text())["instances"]
        sol = (
            {
                (r["language"], r["trial"], r.get("length")): r
                for r in json.loads((out_dir / f"{rung}_solves.json").read_text())[
                    "instances"
                ]
            }
            if rung != "rung4"
            else None
        )
        for lang in LANGS:
            rows = [
                r
                for r in sc
                if r["language"] == lang and (rung != "rung1" or r["length"] == 700)
            ]
            for r in sorted(rows, key=lambda r: r["trial"])[:per_language]:
                for hyp, h in r["hypotheses"].items():
                    fin = h["final"]
                    n_plain = fin["n_plain"]
                    if kind == "sub1to1":
                        s = sol[(lang, r["trial"], r["length"])]
                        n_sym, kb, cb = len(s["cipher_ids"]), key_bits(kind), 0.0
                    elif kind == "homophonic":
                        s = sol[(lang, r["trial"], r["length"])]
                        m = np.asarray(
                            s["true_map"]
                        )  # the solved map is not stored in as_dict; use the class size
                        n_sym = len(s["cipher_ids"])
                        kb = key_bits(kind, n_symbols=len(m))
                        # choice bits need the solved map's homophone counts; use the
                        # true map's allocation (same class size; documented)
                        cb = choice_bits(
                            kind, np.asarray(s["plain_ids"]), sym_to_letter=m
                        )
                    elif kind == "naibbe":
                        s = sol[(lang, r["trial"], None)]
                        n_sym = sum(len(t) for t in s["tokens"])
                        kb = key_bits(kind)
                        cb = choice_bits(
                            kind,
                            np.zeros(n_plain),
                            card_weights=parser_w,
                            p_unigram=0.476,
                            n_tokens=len(s["tokens"]),
                        )
                    else:
                        n_sym = None
                        kb = key_bits(kind)
                        cb = choice_bits(kind, np.zeros(n_plain))
                    cells.append(
                        {
                            "cipher": kind,
                            "language": lang,
                            "trial": r["trial"],
                            "head": kind,
                            "hypothesis": hyp,
                            "bits": fin["bits"][hyp],
                            "calibrated_bits": calibrate_bits(
                                fin["bits"][hyp], hyp, offs
                            ),
                            "n_plain": n_plain,
                            "n_cipher_symbols": n_sym,
                            "key_bits": kb,
                            "choice_bits": cb,
                            "ser": fin.get("ser", fin.get("extra", {}).get("ser")),
                            "cross": False,
                        }
                    )
    return cells


def stage_report(args, root):
    table = CalibrationTable.load(args.primary, root)
    offs = table.additive_offsets()
    sc = json.loads((args.out_dir / "crosshead_scores.json").read_text())
    plain_info = {(p["cipher"], p["language"], p["trial"]): p for p in sc["plain"]}
    cells = _own_cells(args.out_dir, args.per_language, offs)
    # fill n_cipher_symbols for arithmetic from the plain records; cross cells
    for c in cells:
        pi = plain_info.get((c["cipher"], c["language"], c["trial"]))
        if c["n_cipher_symbols"] is None and pi:
            c["n_cipher_symbols"] = pi["n_cipher_symbols"]
    for r in sc["cross"]:
        pi = plain_info[(r["cipher"], r["language"], r["trial"])]
        cells.append(
            {
                **{k: r[k] for k in KEY},
                "bits": r["bits"],
                "calibrated_bits": calibrate_bits(r["bits"], r["hypothesis"], offs),
                "n_plain": r["n_plain"],
                "n_cipher_symbols": pi["n_cipher_symbols"],
                "key_bits": r["key_bits"],
                "choice_bits": r["choice_bits"],
                "ser": None,
                "cross": True,
            }
        )
    for c in cells:
        c["plain_bits_total"] = c["calibrated_bits"] * c["n_plain"]
        c["total_bits"] = c["plain_bits_total"] + c["key_bits"] + c["choice_bits"]
        c["total_per_cipher_symbol"] = c["total_bits"] / c["n_cipher_symbols"]
        c["penalty_per_plain_char"] = (c["key_bits"] + c["choice_bits"]) / c["n_plain"]
    # --- checks ------------------------------------------------------------
    by_text = {}
    for c in cells:
        by_text.setdefault((c["cipher"], c["language"], c["trial"]), []).append(c)
    same_scale, mdl_true, simplicity, lang_rank = [], [], [], []
    for key, cs in by_text.items():
        cipher, lang, _trial = key
        pi = plain_info.get(key)
        own = [c for c in cs if c["head"] == cipher]
        if pi and own:
            o = [c for c in own if c["hypothesis"] == lang]
            if o:
                same_scale.append(
                    {
                        "key": key,
                        "decode_bits": o[0]["calibrated_bits"],
                        "plain_bits": calibrate_bits(pi["bits"][lang], lang, offs),
                        "ser": o[0]["ser"],
                    }
                )
            best_lang = min(own, key=lambda c: c["calibrated_bits"])["hypothesis"]
            lang_rank.append(best_lang == lang)
        best_cell = min(cs, key=lambda c: c["total_per_cipher_symbol"])
        mdl_true.append(
            {
                "key": key,
                "winner_head": best_cell["head"],
                "true": best_cell["head"] == cipher,
                "margin_bits_per_symbol": (
                    sorted(
                        {(c["head"]): c["total_per_cipher_symbol"] for c in cs}.values()
                    )[1]
                    - best_cell["total_per_cipher_symbol"]
                    if len({c["head"] for c in cs}) > 1
                    else None
                ),
            }
        )
        if cipher == "sub1to1":
            r1 = [c for c in cs if c["head"] == "sub1to1" and c["hypothesis"] == lang]
            r2 = [
                c for c in cs if c["head"] == "homophonic" and c["hypothesis"] == lang
            ]
            if r1 and r2:
                simplicity.append(
                    {
                        "key": key,
                        "plain_bits_r1": r1[0]["calibrated_bits"],
                        "plain_bits_r2": r2[0]["calibrated_bits"],
                        "total_r1": r1[0]["total_per_cipher_symbol"],
                        "total_r2": r2[0]["total_per_cipher_symbol"],
                        "rung1_first": r1[0]["total_per_cipher_symbol"]
                        < r2[0]["total_per_cipher_symbol"],
                    }
                )
    good = [x for x in same_scale if x["ser"] is not None and x["ser"] < 0.05]
    acc = {
        "same_instrument_max_abs_gap_bits": (
            float(max(abs(s["decode_bits"] - s["plain_bits"]) for s in good))
            if good
            else None
        ),
        "same_instrument_n": len(good),
        "same_instrument_note": "decode vs true-plaintext bits on cells whose decode is the plaintext (SER < 5%); "
        "both are the frozen evaluator's letter-stream estimator at the same seed",
        "mdl_true_cipher_rate": float(np.mean([m["true"] for m in mdl_true])),
        "mdl_n": len(mdl_true),
        "mdl_margin_min_bits_per_symbol": float(
            min(
                m["margin_bits_per_symbol"]
                for m in mdl_true
                if m["margin_bits_per_symbol"] is not None
            )
        ),
        "simplicity_rung1_first_rate": (
            float(np.mean([s["rung1_first"] for s in simplicity]))
            if simplicity
            else None
        ),
        "simplicity_n": len(simplicity),
        "language_rank_within_true_head": (
            float(np.mean(lang_rank)) if lang_rank else None
        ),
        "language_rank_misses": (
            [
                f"{k[0]}/{k[1]}/t{k[2]}"
                for k, ok in zip(by_text.keys(), lang_rank)
                if not ok
            ]
            if lang_rank
            else []
        ),
    }
    acc["pass"] = bool(
        acc["mdl_true_cipher_rate"] >= 0.9
        and (acc["simplicity_rung1_first_rate"] or 0) >= 0.9
    )
    # table: mean total per cipher symbol, rows = ciphertext type, cols = head x hypothesis
    heads = ("sub1to1", "homophonic", "naibbe", "arithmetic")
    table_rows = {}
    for c in cells:
        d = table_rows.setdefault(c["cipher"], {})
        d.setdefault(f"{c['head']}/{c['hypothesis']}", []).append(
            c["total_per_cipher_symbol"]
        )
    table_mean = {
        k: {kk: float(np.mean(v)) for kk, v in d.items()} for k, d in table_rows.items()
    }
    plain_table = {}
    for c in cells:
        plain_table.setdefault(c["cipher"], {}).setdefault(
            f"{c['head']}/{c['hypothesis']}", []
        ).append(c["calibrated_bits"])
    plain_mean = {
        k: {kk: float(np.mean(v)) for kk, v in d.items()}
        for k, d in plain_table.items()
    }
    report = {
        "task": "5.6",
        "created_utc": datetime.now(UTC).isoformat(),
        "evaluator": sc["evaluator"],
        "budget": sc["budget"],
        "primary_calibration": args.primary,
        "cells": cells,
        "same_instrument": same_scale,
        "mdl": mdl_true,
        "simplicity": simplicity,
        "table_total_per_cipher_symbol": table_mean,
        "table_plain_bits_per_char": plain_mean,
        "acceptance": {
            "criterion": "cross-head scores on one scale: same instrument; MDL total picks the true cipher class; penalty orders equal decodes by simplicity",
            **acc,
        },
    }
    write_json_atomic(args.out_dir / "crosshead_report.json", report)
    md = [
        "### Cross-head uniform scale (task 5.6) — total description length per ciphertext symbol (plaintext + key + choice bits)",
        "",
        "| ciphertext | " + " | ".join(f"{h}/{l}" for h in heads for l in LANGS) + " |",
        "|---|" + "---|" * (len(heads) * len(LANGS)),
    ]
    for cipher in heads:
        if cipher not in table_mean:
            continue
        md.append(
            f"| {cipher} | "
            + " | ".join(
                f"{table_mean[cipher].get(f'{h}/{l}', float('nan')):.2f}"
                for h in heads
                for l in LANGS
            )
            + " |"
        )
    md += [
        "",
        "calibrated plaintext bits/char (the within-head language scale):",
        "",
        "| ciphertext | " + " | ".join(f"{h}/{l}" for h in heads for l in LANGS) + " |",
        "|---|" + "---|" * (len(heads) * len(LANGS)),
    ]
    for cipher in heads:
        if cipher not in plain_mean:
            continue
        md.append(
            f"| {cipher} | "
            + " | ".join(
                f"{plain_mean[cipher].get(f'{h}/{l}', float('nan')):.2f}"
                for h in heads
                for l in LANGS
            )
            + " |"
        )
    md += [
        "",
        (
            f"same instrument: decode vs true-plaintext bits, max |gap| {acc['same_instrument_max_abs_gap_bits']} over {acc['same_instrument_n']} cells; "
            f"MDL picks the true cipher class {acc['mdl_true_cipher_rate']:.0%} of {acc['mdl_n']}; rung-1 ranked first on 1:1 ciphertexts {acc['simplicity_rung1_first_rate']} of {acc['simplicity_n']}; "
            f"language rank within true head {acc['language_rank_within_true_head']} → **{'PASS' if acc['pass'] else 'FAIL'}**"
        ),
    ]
    md = "\n".join(md)
    (args.out_dir / "crosshead_report.md").write_text(md)
    print(md)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = data_root()
    p.add_argument("--stage", choices=["cross", "score", "report"], required=True)
    p.add_argument("--per-language", type=int, default=2)
    p.add_argument("--max-symbols", type=int, default=2000)
    p.add_argument("--restarts", type=int, default=16)
    p.add_argument("--sa-steps", type=int, default=60_000)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--fresh", action="store_true")
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--primary", default=CALIBRATION_VERSION)
    p.add_argument("--out-dir", type=Path, default=root / "analysis" / "phase5")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    {"cross": stage_cross, "score": stage_score, "report": stage_report}[args.stage](
        args, root
    )


if __name__ == "__main__":
    main()
