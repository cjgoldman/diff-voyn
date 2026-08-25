"""Confidence-masked judging probe — docs/confidence_mask_probe.md.

Would the diffusion judge do better on a decode whose low-confidence letters
are blanked (``MASK`` clamped for the whole trajectory) instead of shown as
possibly-wrong letters? Pre-registered stages:

    prepare   instance set: Phase-5 rung-1 (L=400, 700) and rung-2 (L=408)
              solved instances (their shortlists are the keys the inner
              search actually converged to), plus keys at controlled
              wrongness f via the control6a derangement; the stream's own
              held-out n-gram baseline (the uncovered-symbol charge).
    e1        does n-gram key sensitivity find the wrong symbols? (no judge)
    e2        every arm × coverage on the synthetics, all keys, 3 conditions
              (GPU; ``--shard i/n``)
    e3        Phase-6 control battery (positives / voynichesque /
              contamination / shuffled, sub1to1 head, per-hypothesis
              converged keys) through every arm; the ``full`` arm must
              reproduce the frozen Phase-6 numbers (GPU; ``--shard``)
    report    E1–E4 tables, the §6 decision rule, ``report.{json,md}``
    e5        VMS re-application (only if the rule adopts an arm)

All scoring on the frozen evaluator (sha256 checked against
``analysis/phase5/evaluator_freeze.json``), budget 64 strata × 4 replicate
seeds, one masking realization per (instance, seed) shared by every arm,
row and condition (CRN). Artifacts: ``DATA_ROOT/analysis/confidence_probe/``.

Usage:
    uv run python scripts/confidence_probe.py --stage prepare
    uv run python scripts/confidence_probe.py --stage e1
    uv run python scripts/confidence_probe.py --stage e2 --shard 0/2 --device cuda:0
    uv run python scripts/confidence_probe.py --stage e3 --shard 1/2 --device cuda:1
    uv run python scripts/confidence_probe.py --stage report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.data.loader import LANG_TO_INDEX
from diff_voyn.heads.confidence import (
    auroc,
    derange_key,
    freq_mask,
    ngram_sensitivity,
    oracle_mask,
    random_position_mask,
    random_symbol_mask,
    sensitivity_mask,
    shared_sensitivity,
    shuffle_within_mask,
    symbol_correct,
)
from diff_voyn.heads.ladder import load_done, write_json_atomic
from diff_voyn.heads.masked_bits import paired_bits_masked
from diff_voyn.heads.ngram import A, lm_dir, load_lm
from diff_voyn.heads.scale import choice_bits, key_bits
from diff_voyn.metrology import CalibrationTable
from diff_voyn.vms.apply import KEY as P6KEY
from diff_voyn.vms.apply import _job_seed, ciphertext_baselines

LANGS = tuple(LANG_TO_INDEX)
F_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.65)
COVERAGES = (0.9, 0.8, 0.7, 0.5)
ARMS = ("oracle", "sens_shared", "sens_perlang", "freq", "rand_sym", "rand_pos")
E2_KEY = ("instance", "key")
E3_KEY = ("instance", "hypothesis")


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("/".join(map(str, key)).encode()))


def _seed(*key) -> int:
    return zlib.crc32("/".join(map(str, key)).encode()) % (2**31)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def load_lms():
    return {lang: load_lm(lm_dir() / f"{lang}.npz") for lang in LANGS}


def load_evaluator(args, root: Path):
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator

    freeze = json.loads((root / "analysis/phase5/evaluator_freeze.json").read_text())
    want = freeze["frozen"]["evaluator"]["sha256"]
    h = hashlib.sha256()
    with open(args.ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        raise RuntimeError(f"{args.ckpt}: sha256 {h.hexdigest()} != frozen {want}")
    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    ev.meta["sha256"] = want
    return ev


# ---------------------------------------------------------------------------
# prepare


def stage_prepare(args, root: Path):
    p5 = root / "analysis" / "phase5"
    out = []
    r1 = json.load(open(p5 / "rung1_solves.json"))["instances"]
    r2 = json.load(open(p5 / "rung2_solves.json"))["instances"]
    for r in r1:
        L = int(r["length"])
        if L not in (400, 700):
            continue
        if L == 400 and int(r["trial"]) >= args.trials:
            continue
        if L == 700 and int(r["trial"]) >= args.trials_long:
            continue
        out.append(_instance("sub1to1", r, "perm", root))
    for r in r2:
        out.append(_instance("homophonic", r, "map", root))
    inst_path = args.out_dir / "instances.json"
    write_json_atomic(
        inst_path,
        {
            "created_utc": _now(),
            "f_grid": list(F_GRID),
            "coverages": list(COVERAGES),
            "source": "Phase-5 rung1_solves.json (L=400, 700) + rung2_solves.json (L=408)",
            "instances": out,
        },
    )
    print(f"{len(out)} instances -> {inst_path}")


def _instance(kind: str, r: dict, key_field: str, root: Path) -> dict:
    symbols = np.asarray(r["cipher_ids"], np.int64)
    plain = np.asarray(r["plain_ids"], np.int64)
    true_map = np.asarray(r["true_map"], np.int64)
    n_sym = A if kind == "sub1to1" else int(r["n_symbols"])
    name = f"{kind}/{r['language']}/L{r['length']}/t{r['trial']}"
    keys = {}
    for f in F_GRID:
        k = derange_key(true_map, symbols, f, _rng("derange", name, f))
        _, ok = symbol_correct(symbols, k, true_map)
        keys[f"f{f:g}"] = {
            "map": k.tolist(),
            "f": f,
            "symbol_wrong": float((~ok).mean()),
            "position_wrong": float((k[symbols] != plain).mean()),
        }
    for hyp in LANGS:
        best = r["hypotheses"][hyp]["shortlist"][0]
        k = np.asarray(best[key_field], np.int64)
        keys[f"search/{hyp}"] = {
            "map": k.tolist(),
            "hypothesis": hyp,
            "source": best.get("source"),
            "ser": float((k[symbols] != plain).mean()),
        }
    base = ciphertext_baselines(symbols, n_sym)
    return {
        "instance": name,
        "kind": kind,
        "language": r["language"],
        "length": int(r["length"]),
        "trial": int(r["trial"]),
        "n_symbols": n_sym,
        "symbols": symbols.tolist(),
        "plain_ids": plain.tolist(),
        "true_map": true_map.tolist(),
        "keys": keys,
        "uncovered_rate": min(v for k, v in base.items() if k.startswith("ngram")),
        "baselines": base,
    }


def load_instances(args) -> list[dict]:
    return json.load(open(args.out_dir / "instances.json"))["instances"]


# ---------------------------------------------------------------------------
# E1 — sensitivity vs symbol correctness (no judge)


def stage_e1(args, root: Path):
    lms = load_lms()
    insts = load_instances(args)
    per_key = []
    t0 = time.time()
    for inst in insts:
        symbols = np.asarray(inst["symbols"])
        true_map = np.asarray(inst["true_map"])
        counts = np.bincount(symbols, minlength=inst["n_symbols"]).astype(float)
        for kname, k in inst["keys"].items():
            key = np.asarray(k["map"])
            sens = {l: ngram_sensitivity(symbols, key, lm) for l, lm in lms.items()}
            shared = shared_sensitivity(sens)
            occ, ok = symbol_correct(symbols, key, true_map)
            # occurrence-weighted labels: repeat by count (what a position
            # mask actually acts on) — report both
            w = counts[occ].astype(int)
            rec = {
                "instance": inst["instance"],
                "kind": inst["kind"],
                "language": inst["language"],
                "length": inst["length"],
                "key": kname,
                "f": k.get("f"),
                "hypothesis": k.get("hypothesis"),
                "symbol_wrong": float((~ok).mean()),
                "n_wrong": int((~ok).sum()),
                "sens": {l: s.tolist() for l, s in sens.items()},
                "shared": shared.tolist(),
                "auroc": {
                    "sens_true_lang": auroc(sens[inst["language"]][occ], ok),
                    "sens_shared": auroc(shared[occ], ok),
                    "freq": auroc(counts[occ], ok),
                    "sens_true_lang_w": auroc(
                        np.repeat(sens[inst["language"]][occ], w), np.repeat(ok, w)
                    ),
                    "sens_shared_w": auroc(np.repeat(shared[occ], w), np.repeat(ok, w)),
                    "freq_w": auroc(np.repeat(counts[occ], w), np.repeat(ok, w)),
                    **{f"sens_{l}": auroc(sens[l][occ], ok) for l in LANGS},
                },
            }
            if k.get("hypothesis"):
                rec["auroc"]["sens_own_hyp"] = auroc(sens[k["hypothesis"]][occ], ok)
            per_key.append(rec)
        print(f"  {inst['instance']}  {time.time() - t0:.0f}s", flush=True)
    write_json_atomic(
        args.out_dir / "e1.json",
        {"created_utc": _now(), "records": per_key, "summary": _e1_summary(per_key)},
    )
    print(_e1_md(_e1_summary(per_key)))


def _agg(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"mean": None, "n": 0}
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(len(v)),
    }


def _e1_summary(recs):
    out = {}
    for kind in ("sub1to1", "homophonic"):
        for L in sorted({r["length"] for r in recs if r["kind"] == kind}):
            for kname in sorted({r["key"] for r in recs}):
                sub = [
                    r
                    for r in recs
                    if r["kind"] == kind and r["length"] == L and r["key"] == kname
                ]
                if not sub:
                    continue
                cell = {}
                for metric in (
                    "sens_true_lang",
                    "sens_shared",
                    "freq",
                    "sens_true_lang_w",
                    "sens_shared_w",
                    "freq_w",
                    "sens_own_hyp",
                ):
                    cell[metric] = {
                        "all": _agg(r["auroc"].get(metric) for r in sub),
                        **{
                            l: _agg(
                                r["auroc"].get(metric)
                                for r in sub
                                if r["language"] == l
                            )
                            for l in LANGS
                        },
                    }
                cell["symbol_wrong"] = _agg(r["symbol_wrong"] for r in sub)
                out[f"{kind}/L{L}/{kname}"] = cell
    # pass bar: AUROC >= 0.75 at f <= 0.3 in every language; latin within
    # 0.05 of the others (shared rule, the deployable arm)
    checks = []
    for cell_name, cell in out.items():
        kname = cell_name.split("/")[-1]
        if not kname.startswith("f") or float(kname[1:]) > 0.3 or kname == "f0":
            continue
        m = cell["sens_shared"]
        per = {l: m[l]["mean"] for l in LANGS}
        ok_floor = all(v is not None and v >= 0.75 for v in per.values())
        others = [per[l] for l in LANGS if l != "latin" and per[l] is not None]
        ok_latin = (
            per["latin"] is not None
            and bool(others)
            and abs(per["latin"] - float(np.mean(others))) <= 0.05
        )
        checks.append(
            {
                "cell": cell_name,
                "per_language": per,
                "floor_0.75": ok_floor,
                "latin_within_0.05": ok_latin,
                "pass": ok_floor and ok_latin,
            }
        )
    return {"cells": out, "pass_bar": checks, "e1_pass": all(c["pass"] for c in checks)}


def _fmt(x, d=3):
    return (
        "—"
        if x is None or (isinstance(x, float) and not np.isfinite(x))
        else f"{x:.{d}f}"
    )


def _e1_md(s):
    L = ["## E1 — AUROC of per-symbol confidence vs 'symbol correct'\n"]
    L.append(
        "| cell | wrong | sens (true LM) | sens shared | freq | sens own-hyp | shared la / it / de |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for name, c in s["cells"].items():
        L.append(
            f"| {name} | {_fmt(c['symbol_wrong']['mean'], 2)} | "
            f"{_fmt(c['sens_true_lang']['all']['mean'])} | "
            f"{_fmt(c['sens_shared']['all']['mean'])} | {_fmt(c['freq']['all']['mean'])} | "
            f"{_fmt(c['sens_own_hyp']['all']['mean'])} | "
            + " / ".join(_fmt(c["sens_shared"][l]["mean"], 2) for l in LANGS)
            + " |"
        )
    L.append(
        "\nPass bar (shared rule, f ≤ 0.3): AUROC ≥ 0.75 every language, latin within 0.05\n"
    )
    for c in s["pass_bar"]:
        L.append(
            f"- {c['cell']}: "
            + ", ".join(f"{l} {_fmt(v, 2)}" for l, v in c["per_language"].items())
            + f" → {'PASS' if c['pass'] else 'FAIL'}"
        )
    L.append(f"\n**E1 {'PASS' if s['e1_pass'] else 'FAIL'}**\n")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# masks / rows


def build_masks(
    symbols: np.ndarray,
    decode_by_hyp: dict[str, np.ndarray],
    plain: np.ndarray | None,
    shared_sens: np.ndarray,
    sens_by_lang: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> list[dict]:
    """[{arm, c, mask_lang, mask, info}] for one (instance, key)."""
    n = len(symbols)
    out = []
    for c in COVERAGES:
        m, info = sensitivity_mask(symbols, shared_sens, c)
        out.append(
            {"arm": "sens_shared", "c": c, "mask_lang": None, "mask": m, "info": info}
        )
        for l in LANGS:
            m, info = sensitivity_mask(symbols, sens_by_lang[l], c)
            out.append(
                {"arm": "sens_perlang", "c": c, "mask_lang": l, "mask": m, "info": info}
            )
        m, info = freq_mask(symbols, c)
        out.append({"arm": "freq", "c": c, "mask_lang": None, "mask": m, "info": info})
        m, info = random_symbol_mask(symbols, c, rng)
        out.append(
            {"arm": "rand_sym", "c": c, "mask_lang": None, "mask": m, "info": info}
        )
        m, info = random_position_mask(n, c, rng)
        out.append(
            {"arm": "rand_pos", "c": c, "mask_lang": None, "mask": m, "info": info}
        )
        if plain is not None:
            for l, dec in decode_by_hyp.items():
                m, info = oracle_mask(dec, plain, c, rng)
                out.append(
                    {"arm": "oracle", "c": c, "mask_lang": l, "mask": m, "info": info}
                )
    return out


def score_key(
    ev,
    symbols: np.ndarray,
    decode_by_hyp: dict[str, np.ndarray],
    masks: list[dict],
    *,
    conditions_for: dict[str, list[str]],
    seed: int,
    seeds: tuple[int, ...],
    budget: int,
    batch: int,
    full_shuffle: dict[str, np.ndarray] | None = None,
    window: tuple[int, int] | None = None,
) -> list[dict]:
    """Score every (decode, arm, c) pair — decode and its within-mask shuffle
    — under the conditions requested for that decode, one masking
    realization per seed shared by every row. Returns flat records."""
    recs = []
    rng = np.random.default_rng(seed + 424242)
    for hyp, dec in decode_by_hyp.items():
        a, b = window or (0, len(dec))
        d = dec[a:b]
        rows, obs, meta = _rows_for(
            d,
            masks,
            hyp,
            rng,
            full_shuffle=(full_shuffle or {}).get(hyp),
            window=(a, b),
        )
        conds = conditions_for[hyp]
        bits = np.zeros((len(seeds), len(rows), len(conds)))
        for si, sd in enumerate(seeds):
            bits[si] = paired_bits_masked(
                ev,
                rows,
                obs,
                conds,
                n_strata=budget,
                seed=seed + 1000 * sd,
                batch=batch,
            )
        recs += _records(rows, obs, meta, bits, conds, hyp, len(seeds))
    return recs


def _rows_for(d, masks, hyp, rng, *, full_shuffle=None, window=None):
    """Decode + within-mask shuffle rows for the full arm and every mask
    that applies to hypothesis ``hyp`` (``"*"`` = an f-key: every mask)."""
    a, b = window or (0, len(d))
    rows, obs, meta = [], [], []

    def add(arm, c, mask_lang, mask, info, shuffled=None):
        if shuffled is None:
            shuffled = shuffle_within_mask(d, mask, rng)
        for role, row in (("decode", d), ("shuffled", shuffled)):
            rows.append(row)
            obs.append(mask)
            meta.append(
                {
                    "arm": arm,
                    "c": c,
                    "mask_lang": mask_lang,
                    "role": role,
                    "coverage": float(mask.mean()),
                    "n_obs": int(mask.sum()),
                    "kept_symbols": info.get("kept_symbols") if info else None,
                    "purity": info.get("purity") if info else None,
                }
            )

    add("full", 1.0, None, np.ones(len(d), bool), {}, full_shuffle)
    for m in masks:
        if (
            hyp != "*"
            and m["arm"] in ("sens_perlang", "oracle")
            and m["mask_lang"] != hyp
        ):
            continue
        add(m["arm"], m["c"], m["mask_lang"], m["mask"][a:b], m["info"])
    return np.stack(rows), np.stack(obs), meta


def _records(rows, obs, meta, bits, conds, hyp, n_seeds):
    return [
        {
            **m,
            "hypothesis": hyp,
            "bits": {
                c: [float(bits[si, i, j]) for si in range(n_seeds)]
                for j, c in enumerate(conds)
            },
            "observed_decode": (
                rows[i][obs[i]].tolist() if m["role"] == "decode" else None
            ),
        }
        for i, m in enumerate(meta)
    ]


def pair_up(recs: list[dict]) -> list[dict]:
    """Merge decode/shuffled rows into one record with margins per condition."""
    out = {}
    for r in recs:
        k = (r["hypothesis"], r["arm"], r["c"], r["mask_lang"])
        e = out.setdefault(
            k,
            {
                "hypothesis": r["hypothesis"],
                "arm": r["arm"],
                "c": r["c"],
                "mask_lang": r["mask_lang"],
                "coverage": r["coverage"],
                "n_obs": r["n_obs"],
                "kept_symbols": r["kept_symbols"],
                "purity": r["purity"],
            },
        )
        e[r["role"]] = r["bits"]
        if r["role"] == "decode":
            e["observed_decode"] = r["observed_decode"]
    for e in out.values():
        e["margin"] = {
            c: [s - d for s, d in zip(e["shuffled"][c], e["decode"][c])]
            for c in e["decode"]
        }
    return list(out.values())


# ---------------------------------------------------------------------------
# E2 — synthetics


def stage_e2(args, root: Path):
    ev = load_evaluator(args, root)
    e1 = {
        (r["instance"], r["key"]): r
        for r in json.load(open(args.out_dir / "e1.json"))["records"]
    }
    insts = load_instances(args)
    si, sn = map(int, args.shard.split("/"))
    path = args.out_dir / f"e2_shard{si}of{sn}.json"
    done = {} if args.fresh else load_done(path, E2_KEY)
    jobs = []
    for i, inst in enumerate(insts):
        if i % sn != si or (args.long_too is False and inst["length"] > 500):
            continue
        for kname in inst["keys"]:
            if (inst["instance"], kname) not in done:
                jobs.append((inst, kname))
    print(f"shard {si}/{sn}: {len(jobs)} jobs to run, {len(done)} done", flush=True)
    recs = list(done.values())
    seeds = tuple(range(args.seeds))
    t0 = time.time()
    for n, (inst, kname) in enumerate(jobs):
        symbols = np.asarray(inst["symbols"])
        plain = np.asarray(inst["plain_ids"])
        k = inst["keys"][kname]
        key = np.asarray(k["map"])
        e1r = e1[(inst["instance"], kname)]
        sens = {l: np.asarray(e1r["sens"][l]) for l in LANGS}
        shared = np.asarray(e1r["shared"])
        seed = _seed("e2", inst["instance"])
        hyp = k.get("hypothesis") or "*"
        decode_by_hyp = {hyp: key[symbols]}
        masks = build_masks(
            symbols,
            decode_by_hyp,
            plain,
            shared,
            sens,
            _rng("masks", inst["instance"], kname),
        )
        if hyp == "*":
            # an f-key: ONE decode scored under all three conditions; every
            # per-language mask (and the oracle) applies to that decode
            raw = _score_fkey(ev, symbols, decode_by_hyp["*"], masks, seed, seeds, args)
        else:
            # a converged search key: its own hypothesis' decode, all conditions
            raw = score_key(
                ev,
                symbols,
                decode_by_hyp,
                masks,
                conditions_for={hyp: list(LANGS)},
                seed=seed,
                seeds=seeds,
                budget=args.budget,
                batch=args.batch,
            )
        rec = {
            "instance": inst["instance"],
            "key": kname,
            "kind": inst["kind"],
            "language": inst["language"],
            "length": inst["length"],
            "n_symbols": inst["n_symbols"],
            "f": k.get("f"),
            "hypothesis": k.get("hypothesis"),
            "position_wrong": float((key[symbols] != plain).mean()),
            "uncovered_rate": inst["uncovered_rate"],
            "key_bits": key_bits(inst["kind"], n_symbols=inst["n_symbols"]),
            "n_all": len(symbols),
            "seed": seed,
            "cells": pair_up(raw),
        }
        # choice bits per cell over the observed decode (homophonic)
        for c in rec["cells"]:
            if inst["kind"] == "homophonic" and c.get("observed_decode") is not None:
                c["choice_bits"] = float(
                    choice_bits(
                        "homophonic",
                        np.asarray(c["observed_decode"]),
                        sym_to_letter=key,
                    )
                )
            else:
                c["choice_bits"] = 0.0
            c.pop("observed_decode", None)
        recs.append(rec)
        done[(rec["instance"], rec["key"])] = rec
        write_json_atomic(
            path,
            {
                "created_utc": _now(),
                "evaluator": ev.meta,
                "budget": args.budget,
                "seeds": args.seeds,
                "instances": recs,
            },
        )
        el = time.time() - t0
        print(
            f"  [{n + 1}/{len(jobs)}] {inst['instance']} {kname}  {el:.0f}s  eta {el / (n + 1) * (len(jobs) - n - 1) / 60:.0f} min",
            flush=True,
        )


def _score_fkey(ev, symbols, decode, masks, seed, seeds, args):
    """An f-key has one decode scored under all three conditions; every
    per-language mask (and the oracle) is applied to that same decode."""
    rng = np.random.default_rng(seed + 424242)
    rows, obs, meta = _rows_for(decode, masks, "*", rng)
    conds = list(LANGS)
    bits = np.zeros((len(seeds), len(rows), len(conds)))
    for si, sd in enumerate(seeds):
        bits[si] = paired_bits_masked(
            ev,
            rows,
            obs,
            conds,
            n_strata=args.budget,
            seed=seed + 1000 * sd,
            batch=args.batch,
        )
    return _records(rows, obs, meta, bits, conds, "*", len(seeds))


# ---------------------------------------------------------------------------
# E3 — Phase-6 control battery


def controls_jobs(root: Path) -> list[dict]:
    """Phase-6 sub1to1 control cells: per instance the per-hypothesis
    converged key and the frozen full-arm numbers to reproduce."""
    cdir = root / "analysis/phase6/controls"
    manifest = {
        (m["name"], m["kind"]): m for m in json.load(open(cdir / "manifest.json"))
    }
    recs = []
    for f in sorted(cdir.glob("scores_shard*.json")):
        recs += json.load(open(f))["instances"]
    by_inst = {}
    for r in recs:
        if r["head"] != "sub1to1" or r["presentation"] != "eva":
            continue
        by_inst.setdefault(r["instance"], {})[r["hypothesis"]] = r
    jobs = []
    for name, hyps in sorted(by_inst.items()):
        m = manifest[(name, "eva")]
        inst = json.load(open(cdir / m["file"]))
        truth = inst["truth"]
        jobs.append(
            {
                "instance": name,
                "control": m["control"],
                "truth_language": truth.get("language"),
                "truth_family": truth.get("family"),
                "in_inventory": truth.get("in_inventory"),
                "symbols": inst["symbols"],
                "n_symbols": inst["n_symbols"],
                "plain_ids": truth.get("plain_ids"),
                "true_map": truth.get("sym_to_letter"),
                "keys": {h: r["final"]["key"]["map"] for h, r in hyps.items()},
                "phase6": {
                    h: {
                        "job_seed": _job_seed({k: r[k] for k in P6KEY}),
                        "window": r["full"]["windows"][0]["span"],
                        "seeds": r["full"]["windows"][0]["seeds"],
                        "key_bits": r["full"]["key_bits"],
                    }
                    for h, r in hyps.items()
                },
            }
        )
    return jobs


def stage_e3(args, root: Path):
    ev = load_evaluator(args, root)
    lms = load_lms()
    jobs = controls_jobs(root)
    si, sn = map(int, args.shard.split("/"))
    path = args.out_dir / f"e3_shard{si}of{sn}.json"
    done = {} if args.fresh else load_done(path, ("instance",))
    todo = [
        j for i, j in enumerate(jobs) if i % sn == si and (j["instance"],) not in done
    ]
    print(
        f"shard {si}/{sn}: {len(todo)} instances to run, {len(done)} done", flush=True
    )
    recs = list(done.values())
    seeds = tuple(range(args.seeds))
    t0 = time.time()
    for n, j in enumerate(todo):
        symbols = np.asarray(j["symbols"])
        plain = np.asarray(j["plain_ids"]) if j["plain_ids"] is not None else None
        keys = {h: np.asarray(k) for h, k in j["keys"].items()}
        sens_own = {h: ngram_sensitivity(symbols, keys[h], lms[h]) for h in LANGS}
        # the shared (fair) mask: min over hypotheses of each hypothesis'
        # key under its own LM — one mask for every cell of the instance
        shared = shared_sensitivity(sens_own)
        decode_by_hyp = {h: keys[h][symbols] for h in LANGS}
        masks = build_masks(
            symbols,
            decode_by_hyp,
            plain,
            shared,
            sens_own,
            _rng("masks-e3", j["instance"]),
        )
        a, b = j["phase6"][LANGS[0]]["window"]
        cells = []
        repro = {}
        for h in LANGS:
            seed = j["phase6"][h]["job_seed"]
            dec = decode_by_hyp[h][a:b]
            full_shuf = np.random.default_rng(seed).permutation(dec)  # Phase-6 draw
            raw = score_key(
                ev,
                symbols,
                {h: decode_by_hyp[h]},
                masks,
                conditions_for={h: [h]},
                seed=seed,
                seeds=seeds,
                budget=args.budget,
                batch=args.batch,
                full_shuffle={h: full_shuf},
                window=(a, b),
            )
            for c in pair_up(raw):
                c["choice_bits"] = 0.0
                c.pop("observed_decode", None)
                cells.append(c)
            full = [c for c in cells if c["hypothesis"] == h and c["arm"] == "full"][0]
            p6 = j["phase6"][h]["seeds"]
            repro[h] = {
                "decode_max_abs_diff": max(
                    abs(full["decode"][h][s] - p6[s]["decode"][h])
                    for s in range(len(seeds))
                ),
                "shuffled_max_abs_diff": max(
                    abs(full["shuffled"][h][s] - p6[s]["shuffled"][h])
                    for s in range(len(seeds))
                ),
            }
        rec = {
            "instance": j["instance"],
            "control": j["control"],
            "truth_language": j["truth_language"],
            "truth_family": j["truth_family"],
            "in_inventory": j["in_inventory"],
            "n_symbols": j["n_symbols"],
            "n_all": b - a,
            "window": [a, b],
            "key_bits": key_bits("sub1to1"),
            "uncovered_rate": None,
            "ser": {
                h: (
                    float((decode_by_hyp[h] != plain).mean())
                    if plain is not None
                    else None
                )
                for h in LANGS
            },
            "phase6_reproduction": repro,
            "cells": cells,
        }
        base = ciphertext_baselines(symbols[a:b], j["n_symbols"])
        rec["uncovered_rate"] = min(v for k, v in base.items() if k.startswith("ngram"))
        recs.append(rec)
        done[(rec["instance"],)] = rec
        write_json_atomic(
            path,
            {
                "created_utc": _now(),
                "evaluator": ev.meta,
                "budget": args.budget,
                "seeds": args.seeds,
                "instances": recs,
            },
        )
        el = time.time() - t0
        print(
            f"  [{n + 1}/{len(todo)}] {j['instance']} repro {max(v['decode_max_abs_diff'] for v in repro.values()):.2e}  {el:.0f}s  eta {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min",
            flush=True,
        )


# ---------------------------------------------------------------------------
# report


def _load_shards(out_dir: Path, prefix: str) -> list[dict]:
    recs = []
    for f in sorted(out_dir.glob(f"{prefix}_shard*.json")):
        recs += json.load(open(f))["instances"]
    return recs


def mdl_total(cell: dict, cond: str, rec: dict, seed_idx: int | None = None) -> float:
    """Phase-6 rule: bits_obs·n_obs + key + choice + (n_all − n_obs)·r_uncov."""
    b = cell["decode"][cond]
    bits = float(np.mean(b)) if seed_idx is None else b[seed_idx]
    return (
        bits * cell["n_obs"]
        + rec["key_bits"]
        + cell["choice_bits"]
        + (rec["n_all"] - cell["n_obs"]) * rec["uncovered_rate"]
    )


def _arm_cells(rec, arm, c):
    return [x for x in rec["cells"] if x["arm"] == arm and x["c"] == c]


def e2_analysis(e2: list[dict], table: CalibrationTable) -> dict:
    """Language recovery / margins / bias per arm × coverage × f."""
    arms = [("full", 1.0)] + [(a, c) for a in ARMS for c in COVERAGES]
    out = {"recovery": {}, "bias": {}, "e4": {}, "perlang_vs_shared": {}}
    fkeys = [r for r in e2 if r["f"] is not None]
    for arm, c in arms:
        key = f"{arm}@{c:g}"
        for f in F_GRID:
            sub = [r for r in fkeys if r["f"] == f]
            hits, flips, unres, margins = [], [], [], []
            for r in sub:
                if arm == "sens_perlang":
                    cells = {x["mask_lang"]: x for x in _arm_cells(r, arm, c)}
                    if len(cells) < 3:
                        continue
                    tot = {l: mdl_total(cells[l], l, r) for l in LANGS}
                    per_seed = [
                        {l: mdl_total(cells[l], l, r, s) for l in LANGS}
                        for s in range(len(cells[LANGS[0]]["decode"][LANGS[0]]))
                    ]
                else:
                    cells = _arm_cells(r, arm, c)
                    if not cells:
                        continue
                    x = cells[0]
                    tot = {l: mdl_total(x, l, r) for l in LANGS}
                    per_seed = [
                        {l: mdl_total(x, l, r, s) for l in LANGS}
                        for s in range(len(x["decode"][LANGS[0]]))
                    ]
                rank = sorted(tot, key=tot.get)
                hits.append(rank[0] == r["language"])
                margin = (tot[rank[1]] - tot[rank[0]]) / r["n_all"]
                margins.append(margin)
                unres.append(margin < table.margin_uncertainty_bits(rank[0], rank[1]))
                tops = [min(LANGS, key=ps.get) for ps in per_seed]
                flips.append(float(np.mean([t != rank[0] for t in tops])))
            if hits:
                out["recovery"].setdefault(key, {})[f"f{f:g}"] = {
                    "accuracy": float(np.mean(hits)),
                    "n": len(hits),
                    "flip_rate": float(np.mean(flips)),
                    "margin_unresolved": float(np.mean(unres)),
                    "margin_mean": float(np.mean(margins)),
                    "by_language": {
                        l: float(
                            np.mean(
                                [h for h, r in zip(hits, sub) if r["language"] == l]
                            )
                        )
                        for l in LANGS
                        if any(r["language"] == l for r in sub)
                    },
                }
        # search keys: cells (hyp, decode_hyp) ranked by mdl at own condition
        hits = []
        by_inst = {}
        for r in e2:
            if r["hypothesis"]:
                by_inst.setdefault(r["instance"], {})[r["hypothesis"]] = r
        for hyps in by_inst.values():
            if len(hyps) < 3:
                continue
            tot = {}
            for h, r in hyps.items():
                cells = [
                    x for x in _arm_cells(r, arm, c) if x["mask_lang"] in (None, h)
                ]
                if not cells:
                    break
                tot[h] = mdl_total(cells[0], h, r)
            if len(tot) < 3:
                continue
            rank = sorted(tot, key=tot.get)
            hits.append(rank[0] == next(iter(hyps.values()))["language"])
        if hits:
            out["recovery"].setdefault(key, {})["search"] = {
                "accuracy": float(np.mean(hits)),
                "n": len(hits),
            }
        # per-language shift of bits_obs(true) from full (f-keys, all f) — E2 bias
        shift = {l: [] for l in LANGS}
        for r in fkeys:
            full = _arm_cells(r, "full", 1.0)[0]
            cells = [
                x
                for x in _arm_cells(r, arm, c)
                if x["mask_lang"] in (None, r["language"], "*")
            ]
            if not cells:
                continue
            shift[r["language"]].append(
                np.mean(cells[0]["decode"][r["language"]])
                - np.mean(full["decode"][r["language"]])
            )
        out["bias"][key] = {l: _agg(v) for l, v in shift.items()}
        # E4: f = 0, bits_obs(true) under the rule vs rand_pos at the same c
        if arm in ("sens_shared", "freq") and c < 1:
            gap = {l: [] for l in LANGS}
            for r in fkeys:
                if r["f"] != 0.0:
                    continue
                a_ = _arm_cells(r, arm, c)
                b_ = _arm_cells(r, "rand_pos", c)
                if a_ and b_:
                    gap[r["language"]].append(
                        np.mean(a_[0]["decode"][r["language"]])
                        - np.mean(b_[0]["decode"][r["language"]])
                    )
            out["e4"][key] = {l: _agg(v) for l, v in gap.items()}
    # sens_perlang vs sens_shared ranking disagreement
    for c in COVERAGES:
        diff, n = 0, 0
        for r in fkeys:
            sh = _arm_cells(r, "sens_shared", c)
            pl = {x["mask_lang"]: x for x in _arm_cells(r, "sens_perlang", c)}
            if not sh or len(pl) < 3:
                continue
            rs = min(LANGS, key=lambda l: mdl_total(sh[0], l, r))
            rp = min(LANGS, key=lambda l: mdl_total(pl[l], l, r))
            diff += rs != rp
            n += 1
        out["perlang_vs_shared"][f"{c:g}"] = {
            "fraction_different": diff / n if n else None,
            "n": n,
        }
    return out


def _bias_shift(bias_entry: dict) -> dict:
    """Between-language differences of the mean shift (a uniform shift is
    not a ranking bias)."""
    m = {l: bias_entry[l]["mean"] for l in LANGS}
    pairs = {}
    for i, a in enumerate(LANGS):
        for b in LANGS[i + 1 :]:
            pairs[f"{a}-{b}"] = None if m[a] is None or m[b] is None else m[a] - m[b]
    return pairs


def e3_analysis(e3: list[dict], e2: list[dict]) -> dict:
    """Structure-margin separability per arm × c: positives vs negatives."""
    arms = (
        [("full", 1.0)]
        + [(a, c) for a in ARMS if a != "oracle" for c in COVERAGES]
        + [("oracle", c) for c in COVERAGES]
    )
    out = {}
    for arm, c in arms:
        pos, neg, pos_syn, detail = [], [], [], []
        for r in e3:
            # per hypothesis own-condition margin; the instance's margin is at
            # its MDL-top hypothesis under THIS arm (Phase-6 rule), also kept
            # at the full-arm top
            tot, marg = {}, {}
            for h in LANGS:
                cells = [
                    x
                    for x in _arm_cells(r, arm, c)
                    if x["hypothesis"] == h and x["mask_lang"] in (None, h)
                ]
                if not cells:
                    continue
                tot[h] = mdl_total(cells[0], h, r)
                marg[h] = float(np.mean(cells[0]["margin"][h]))
            if len(tot) < 3:
                continue
            top = min(tot, key=tot.get)
            full_tot = {
                h: mdl_total(
                    [x for x in _arm_cells(r, "full", 1.0) if x["hypothesis"] == h][0],
                    h,
                    r,
                )
                for h in LANGS
            }
            full_top = min(full_tot, key=full_tot.get)
            m = marg[top]
            d = {
                "instance": r["instance"],
                "control": r["control"],
                "top": top,
                "margin": m,
                "margin_at_full_top": marg[full_top],
                "margin_by_hyp": marg,
                "truth": r["truth_language"],
            }
            detail.append(d)
            if r["control"] == "positive":
                pos.append(m)
            elif r["control"] in ("voynichesque", "contamination", "shuffled"):
                neg.append((r["control"], m))
        # synthetic positives: E2 instances at f >= 0.5 (own condition margin, true language)
        for r in e2:
            if r["f"] is None or r["f"] < 0.5:
                continue
            cells = [
                x
                for x in _arm_cells(r, arm, c)
                if x["mask_lang"] in (None, r["language"], "*")
            ]
            if cells:
                pos_syn.append(float(np.mean(cells[0]["margin"][r["language"]])))
        negs = [m for _, m in neg]
        neg_main = [m for k, m in neg if k in ("voynichesque", "contamination")]
        res = {
            "n_pos": len(pos),
            "n_neg": len(negs),
            "n_pos_synthetic": len(pos_syn),
            "pos_min": float(min(pos)) if pos else None,
            "pos_max": float(max(pos)) if pos else None,
            "neg_max": float(max(neg_main)) if neg_main else None,
            "neg_max_all": float(max(negs)) if negs else None,
            "gap": (float(min(pos) - max(neg_main)) if pos and neg_main else None),
            "gap_with_synthetic_positives": (
                float(min(pos + pos_syn) - max(neg_main))
                if (pos or pos_syn) and neg_main
                else None
            ),
            "auroc": (
                auroc(
                    np.array(pos + negs),
                    np.array([True] * len(pos) + [False] * len(negs)),
                )
                if pos and negs
                else None
            ),
            "by_control": {
                k: _agg([m for kk, m in neg if kk == k])
                for k in ("voynichesque", "contamination", "shuffled")
            },
            "positives": _agg(pos),
            "synthetic_positives": _agg(pos_syn),
            "detail": detail,
        }
        res["threshold_band"] = (
            [res["neg_max"], res["pos_min"]]
            if res["gap"] is not None and res["gap"] > 0
            else None
        )
        out[f"{arm}@{c:g}"] = res
    return out


def decision(e1s, e2a, e3a, table) -> dict:
    full_gap = e3a["full@1"]["gap"]
    best = None
    rows = []
    for arm in ("sens_shared", "freq"):
        for c in COVERAGES:
            k = f"{arm}@{c:g}"
            gap = e3a[k]["gap"]
            rs = e3a[f"rand_sym@{c:g}"]["gap"]
            r2 = e2a["recovery"].get(k, {})
            full_rec = e2a["recovery"].get("full@1", {})
            rec_ok = all(
                (
                    r2.get(f"f{f:g}", {}).get("accuracy", 0)
                    >= full_rec.get(f"f{f:g}", {}).get("accuracy", 1) - 0.01
                )
                for f in F_GRID
                if f <= 0.3
            )
            pair_shift = _bias_shift(e2a["bias"][k])
            bias_ok = all(
                v is not None and abs(v) <= table.margin_uncertainty_bits(*p.split("-"))
                for p, v in pair_shift.items()
            )
            e4 = e2a["e4"].get(k, {})
            e4_eff = max(
                (abs(v["mean"]) for v in e4.values() if v.get("mean") is not None),
                default=None,
            )
            gain = None if gap is None or full_gap is None else gap - full_gap
            row = {
                "arm": k,
                "gap": gap,
                "gain_over_full": gain,
                "rand_sym_gap": rs,
                "beats_rand_sym_by": None if gap is None or rs is None else gap - rs,
                "rule2": gain is not None
                and gain >= 0.15
                and rs is not None
                and gap - rs >= 0.10,
                "rule3": rec_ok and bias_ok,
                "recovery_ok": rec_ok,
                "bias_ok": bias_ok,
                "pair_shift": pair_shift,
                "e4_selection_effect": e4_eff,
                "rule4": e4_eff is not None
                and gain is not None
                and e4_eff < 0.5 * gain,
            }
            row["adopt"] = bool(
                e1s["e1_pass"] and row["rule2"] and row["rule3"] and row["rule4"]
            )
            rows.append(row)
            if row["adopt"] and (
                best is None or row["gain_over_full"] > best["gain_over_full"]
            ):
                best = row
    return {
        "rule1_e1": e1s["e1_pass"],
        "full_gap": full_gap,
        "candidates": rows,
        "adopt": best is not None,
        "winner": best,
    }


def stage_report(args, root: Path):
    table = CalibrationTable.load(args.primary, root)
    e1 = json.load(open(args.out_dir / "e1.json"))
    e1s = e1["summary"]
    e2 = _load_shards(args.out_dir, "e2")
    e3 = _load_shards(args.out_dir, "e3")
    e2a = e2_analysis(e2, table) if e2 else {}
    e3a = e3_analysis(e3, e2) if e3 else {}
    dec = (
        decision(e1s, e2a, e3a, table)
        if (e2 and e3)
        else {"rule1_e1": e1s["e1_pass"], "adopt": False, "note": "E2/E3 incomplete"}
    )
    repro = [
        max(v["decode_max_abs_diff"] for v in r["phase6_reproduction"].values())
        for r in e3
    ]
    rep = {
        "created_utc": _now(),
        "n_e2_records": len(e2),
        "n_e3_instances": len(e3),
        "phase6_full_arm_reproduction_max_abs_diff": (
            float(max(repro)) if repro else None
        ),
        "e1": e1s,
        "e2": e2a,
        "e3": e3a,
        "decision": dec,
    }
    write_json_atomic(args.out_dir / "report.json", rep)
    md = report_md(rep)
    (args.out_dir / "report.md").write_text(md)
    print(md)


def report_md(rep: dict) -> str:
    L = [f"# Confidence-mask probe — report ({rep['created_utc']})\n"]
    L.append(_e1_md(rep["e1"]))
    if rep["e2"]:
        e2 = rep["e2"]
        L.append(
            "## E2 — language recovery by MDL total (f-keys; 'search' = per-hypothesis converged keys)\n"
        )
        fs = [f"f{f:g}" for f in F_GRID] + ["search"]
        L.append("| arm | " + " | ".join(fs) + " | unresolved f0/f0.3 | flip f0.3 |")
        L.append("|---|" + "---|" * (len(fs) + 2))
        for k, v in e2["recovery"].items():
            L.append(
                f"| {k} | "
                + " | ".join(_fmt(v.get(f, {}).get("accuracy"), 2) for f in fs)
                + f" | {_fmt(v.get('f0', {}).get('margin_unresolved'), 2)}/{_fmt(v.get('f0.3', {}).get('margin_unresolved'), 2)} | {_fmt(v.get('f0.3', {}).get('flip_rate'), 2)} |"
            )
        L.append(
            "\nPer-language shift of bits_obs(ℓ_true) from `full` (mean over all f-keys) and pairwise differences:\n"
        )
        L.append("| arm | latin | italian | german | la−it | la−de | it−de |")
        L.append("|---|---|---|---|---|---|---|")
        for k, v in e2["bias"].items():
            ps = _bias_shift(v)
            L.append(
                f"| {k} | "
                + " | ".join(_fmt(v[l]["mean"]) for l in LANGS)
                + " | "
                + " | ".join(_fmt(x) for x in ps.values())
                + " |"
            )
        L.append(
            "\nsens_perlang vs sens_shared ranking disagreement: "
            + ", ".join(
                f"c={c}: {_fmt(v['fraction_different'], 2)} (n={v['n']})"
                for c, v in e2["perlang_vs_shared"].items()
            )
            + "\n"
        )
        L.append(
            "## E4 — selection effect (f = 0, bits_obs(true) rule − rand_pos, same c)\n"
        )
        L.append("| arm | latin | italian | german |")
        L.append("|---|---|---|---|")
        for k, v in e2["e4"].items():
            L.append(
                f"| {k} | "
                + " | ".join(
                    f"{_fmt(v[l]['mean'])} ± {_fmt(v[l].get('sd'))}" for l in LANGS
                )
                + " |"
            )
    if rep["e3"]:
        L.append(
            f"\n## E3 — structure-margin separability (Phase-6 controls, sub1to1; full-arm reproduction max |Δ| = {rep['phase6_full_arm_reproduction_max_abs_diff']:.2e})\n"
        )
        L.append(
            "| arm | pos min–max (n) | voyn. mean/max | contam. mean/max | shuffled max | **gap** | gap w/ synth pos (n) | AUROC |"
        )
        L.append("|---|---|---|---|---|---|---|---|")
        for k, v in rep["e3"].items():
            bc = v["by_control"]
            L.append(
                f"| {k} | {_fmt(v['pos_min'], 2)}–{_fmt(v['pos_max'], 2)} ({v['n_pos']}) | {_fmt(bc['voynichesque']['mean'], 2)}/{_fmt(bc['voynichesque'].get('max'), 2)} | "
                f"{_fmt(bc['contamination']['mean'], 2)}/{_fmt(bc['contamination'].get('max'), 2)} | {_fmt(bc['shuffled'].get('max'), 2)} | **{_fmt(v['gap'], 3)}** | {_fmt(v['gap_with_synthetic_positives'], 3)} ({v['n_pos_synthetic']}) | {_fmt(v['auroc'], 3)} |"
            )
    d = rep["decision"]
    L.append("\n## §6 decision rule\n")
    L.append(f"- rule 1 (E1 pass): **{d.get('rule1_e1')}**")
    if "candidates" in d:
        L.append(f"- full-arm gap: {_fmt(d['full_gap'], 3)}")
        L.append(
            "\n| arm | gap | gain over full | beats rand_sym by | rule 2 | recovery ok | bias ok | E4 effect | rule 4 | adopt |"
        )
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in d["candidates"]:
            L.append(
                f"| {r['arm']} | {_fmt(r['gap'], 3)} | {_fmt(r['gain_over_full'], 3)} | {_fmt(r['beats_rand_sym_by'], 3)} | {r['rule2']} | {r['recovery_ok']} | {r['bias_ok']} | {_fmt(r['e4_selection_effect'], 3)} | {r['rule4']} | **{r['adopt']}** |"
            )
        L.append(
            f"\n**Adopt: {d['adopt']}**"
            + (f" — winner {d['winner']['arm']}" if d["adopt"] else "")
        )
    else:
        L.append(f"- {d.get('note')}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------


def main():
    root = data_root()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stage", required=True, choices=["prepare", "e1", "e2", "e3", "report"]
    )
    ap.add_argument(
        "--out-dir", type=Path, default=root / "analysis" / "confidence_probe"
    )
    ap.add_argument(
        "--trials", type=int, default=5, help="rung-1 L=400 trials per language"
    )
    ap.add_argument(
        "--trials-long", type=int, default=3, help="rung-1 L=700 trials per language"
    )
    ap.add_argument(
        "--long-too", action="store_true", help="E2: also score the L=700 instances"
    )
    ap.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--primary", default="v3-phase_c-ro")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    {
        "prepare": stage_prepare,
        "e1": stage_e1,
        "e2": stage_e2,
        "e3": stage_e3,
        "report": stage_report,
    }[args.stage](args, root)


if __name__ == "__main__":
    main()
