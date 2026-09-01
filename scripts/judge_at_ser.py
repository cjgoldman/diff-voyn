"""Language-judge sensitivity vs letter SER on the word-homophonic A-like
cells (docs/alt_loop_plan.md §8.5).

The hapax-wildcard diffusion-guided loop leaves the A-like wordhom cells at
letter SER 0.13 (German) / 0.17 (Latin) / 0.21–0.24 (Italian) (wildcard-only
finals, 2026-08-27; the anneal stage later brought these to 0.05 / 0.07 /
0.12 — docs/alt_loop_plan.md §8.6 — and this script scores those finals too,
via ``--run-tags``; the German and Latin anneal finals ARE called,
docs/project_status.md §5.8). Question: does
the frozen Phase-6 judge (per-language calibrated ELBO, letter-shuffle
structure margin, ``ABSTAIN_RULE``) call the language at those error
levels, and where between the recovered key (SER ≈ 0) and the stuck start
(0.64–0.77) does it stop?

Keys scored per cell (full stream, the exact Phase-6 scoring loop: paired
decode/shuffled windows × 4 replicate seeds × 3 language conditions,
budget 64):
  truth                   — the generating key (SER 0 up to rule collisions)
  stuck                   — the solve's n-gram MDL pick the loop starts from
  wild:*                  — the loop's recorded final maps (runs_wild32/96)
  uni@p / rare@p          — truth with a fraction of TYPES reassigned to a
                            random wrong unit: uniformly over types, or the
                            rarest types first (the search's error profile:
                            wrong-rate monotone decreasing in occurrence)

Artifacts: DATA_ROOT/analysis/altloop/judge_at_ser.json, report .md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.ladder import write_json_atomic
from diff_voyn.heads.two_tier import paired_bits
from diff_voyn.heads.wordhom import (
    UnitTargets,
    adjacency,
    expand_units,
    hypothesis_targets,
    project_key,
    rule_violations,
    unit_ser,
    units_suffix,
)
from diff_voyn.metrology import CALIBRATION_VERSION, CalibrationTable
from diff_voyn.vms.apply import (
    ABSTAIN_RULE,
    KEY,
    LANGS,
    WORDHOM,
    _choice_total,
    _wordhom_choice_params,
    cell_from_score,
    head_key_bits,
)

CELLS = [("positive/german/Alike", "german"), ("positive/italian/Alike", "italian"),
         ("positive/latin/Alike", "latin")]
WILD_FILES = ["runs_wild32map.json", "runs_wild96.json", "runs_anneal.json", "runs_anneal_de.json"]


def load_cell(root, name, lang, wild_files=WILD_FILES, prefix="wild", units=None):
    wd = root / "analysis/wordhom"
    bat = wd / "battery/wordtypesall"
    man = {}
    if (bat / "manifest.json").exists():
        man = {m["name"]: m for m in json.loads((bat / "manifest.json").read_text())}
    if name in man:
        inst = json.loads((bat / man[name]["file"]).read_text())
    else:
        inst = json.loads(
            (wd / "controls/wordtypesall" / (name.replace("/", "_") + "_wordtypesall.json")).read_text()
        )
    solves = []
    suf = units_suffix(units)
    files = [f"battery/battery_solves{suf}.json"] + ([] if suf else ["controls_solves.json"])
    for fn in files:
        if (wd / fn).exists():
            solves += json.loads((wd / fn).read_text())["instances"]
    rec = next(s for s in solves if s["instance"] == name and s["hypothesis"] == lang)
    stuck = max(rec["candidates"], key=lambda c: c["inner"])["map"]
    finals = {}
    for fn in wild_files:
        fp = root / "analysis/altloop" / fn
        if not fp.exists():
            continue
        for r in json.loads(fp.read_text()):
            if r["cell"] == f"wh/{name}/{lang}" and r["start"] == "stuck" and "final_map" in r:
                finals[f"{prefix}:{fn[4:-5]}/s{r['seed']}"] = r["final_map"]
    return inst, np.asarray(stuck, dtype=np.int64), finals


def corrupt(true_map, occ, frac, rng, n_units, rare_first):
    m = true_map.copy()
    n = len(m)
    k = int(round(frac * n))
    if rare_first:
        # rarest types first; ties broken at random
        order = np.lexsort((rng.random(n), occ))
        idx = order[:k]
    else:
        idx = rng.choice(n, size=k, replace=False)
    new = rng.integers(0, n_units - 1, size=k)
    new = new + (new >= m[idx])  # any unit but the current one
    m[idx] = new
    return m


def score_map(ev, table, inst, meta, m, targets, hyp, *, budget, seeds, score_windows, seed):
    sym = np.asarray(inst["symbols"], dtype=np.int64)
    letters = expand_units(m[sym], targets)
    n_plain = len(letters)
    params = _wordhom_choice_params(inst, m, targets, 0, len(sym))
    cb = _choice_total(WORDHOM, letters, params)
    kb = head_key_bits(WORDHOM, int(inst["n_symbols"]), n_units=targets.n)
    W = ev.window
    cuts = [(s, s + W) for s in range(0, max(n_plain - W + 1, 1), W)] or [(0, n_plain)]
    if len(cuts) > score_windows:
        cuts = [cuts[i] for i in np.linspace(0, len(cuts) - 1, score_windows).astype(int)]
    rng = np.random.default_rng(seed)
    wins = []
    for wi, (s, e) in enumerate(cuts):
        dec = letters[s:e]
        rows = np.stack([dec, rng.permutation(dec)])
        entry = {"span": [int(s), int(e)], "seeds": []}
        for sd in seeds:
            pb = paired_bits(ev, rows, list(LANGS), n_strata=budget, seed=seed + 1000 * sd + 17 * wi)
            entry["seeds"].append(
                {"decode": {l: float(pb[0, j]) for j, l in enumerate(LANGS)},
                 "shuffled": {l: float(pb[1, j]) for j, l in enumerate(LANGS)}}
            )
        wins.append(entry)
    rec = {
        "instance": inst["name"], "presentation": "wordtypesall", "head": WORDHOM,
        "window": "full", "hypothesis": hyp, "window_span": [0, len(sym)],
        "final": {"source": "given", "window_bits": {l: float("nan") for l in LANGS}},
        "full": {"n_plain": n_plain, "n_cipher_covered": int(inst["coverage"]["n_kept_chars"]),
                 "key_bits": kb, "choice_bits": float(cb), "n_windows_scored": len(cuts),
                 "windows": wins},
    }
    cell = cell_from_score(rec, table, meta)
    return {k: cell[k] for k in (
        "plain_bits", "plain_bits_sem", "plain_bits_by_condition", "structure_margin",
        "structure_margin_by_condition", "language_rank_of_decode", "top_language_of_decode",
        "top_margin_bits", "top_margin_uncertainty_bits", "replicate_flip_rate",
        "window_vote_for_top", "total_per_all_symbols", "language_like", "n_windows",
    )}


def main():
    p = argparse.ArgumentParser()
    root = data_root()
    p.add_argument("--stage", choices=["run", "report"], default="run")
    p.add_argument("--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--fracs", type=float, nargs="*",
                   default=[0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45])
    p.add_argument("--rare-fracs", type=float, nargs="*",
                   default=[0.30, 0.45, 0.55, 0.65, 0.75, 0.85])
    p.add_argument("--corrupt-seeds", type=int, default=1)
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--score-windows", type=int, default=16)
    p.add_argument("--primary", default=CALIBRATION_VERSION)
    p.add_argument("--tag", default="")
    p.add_argument(
        "--battery",
        nargs="*",
        default=None,
        help="NAME:HYP cells from the wordhom battery / controls; keys = truth (if the "
        "hypothesis is the generating language), stuck, and the finals of --run-tags",
    )
    p.add_argument("--run-tags", nargs="*", default=["_bat_wild", "_bat_anneal"])
    p.add_argument("--units", default=None, help="wordhom unit-set spec (d5 default, d5b20)")
    args = p.parse_args()
    out_dir = root / "analysis/altloop"
    path = out_dir / f"judge_at_ser{args.tag}.json"
    if args.stage == "report":
        report(path, out_dir / f"judge_at_ser{args.tag}.md")
        return
    torch.set_float32_matmul_precision("high")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from wordhom_study import _inst_meta  # noqa: E402
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator

    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    table = CalibrationTable.load(args.primary, root)
    res = json.loads(path.read_text()) if path.exists() else []
    done = {(r["cell"], r.get("hypothesis", r["truth_language"]), r["key"]) for r in res}
    if args.battery is not None:
        cells = [tuple(c.rsplit(":", 1)) for c in args.battery]
        wild_files = [f"runs{t}.json" for t in args.run_tags]
        prefix = "final"
        args.fracs, args.rare_fracs = [], []
    else:
        cells, wild_files, prefix = CELLS, WILD_FILES, "wild"
    from diff_voyn.vms.apply import build_ngram_evaluator

    ng = build_ngram_evaluator()
    for name, lang in cells:
        if args.only and not any(o in name for o in args.only):
            continue
        inst, stuck, finals = load_cell(root, name, lang, wild_files, prefix, args.units)
        meta = _inst_meta(inst)
        tr = inst.get("truth", {})
        has_truth = tr.get("kind") == "wordhom" and tr.get("language") == lang
        plain = np.asarray(tr["plain_ids"], dtype=np.int64) if "plain_ids" in tr else None
        # the decoder's unit space (instance override / --units spec); the
        # truth key is projected into it (identity when the spaces coincide)
        targets = hypothesis_targets(ng, lang, units=args.units, inst=inst)
        true_map = (
            project_key(tr["sym_to_unit"], UnitTargets.from_list(tr["bigrams"]), targets)
            if has_truth
            else None
        )
        sym = np.asarray(inst["symbols"], dtype=np.int64)
        adj = adjacency(sym, np.asarray(inst["token_pos"], dtype=np.int64))
        occ = np.bincount(sym, minlength=int(inst["n_symbols"]))
        keys = {"stuck": stuck}
        if has_truth:
            keys["truth"] = true_map
        keys.update({k: np.asarray(v, dtype=np.int64) for k, v in finals.items()})
        for cs in range(args.corrupt_seeds):
            rng = np.random.default_rng(1000 + cs)
            for f in args.fracs:
                keys[f"uni@{f:.2f}/s{cs}"] = corrupt(true_map, occ, f, rng, targets.n, False)
            for f in args.rare_fracs:
                keys[f"rare@{f:.2f}/s{cs}"] = corrupt(true_map, occ, f, rng, targets.n, True)
        for kname, m in keys.items():
            if (name, lang, kname) in done:
                continue
            t0 = time.time()
            dec = expand_units(m[sym], targets)
            wrong = (m != true_map) if has_truth else np.zeros(len(m), bool)
            r = {
                "cell": name, "truth_language": tr.get("language"), "hypothesis": lang,
                "control": inst.get("control", "positive"), "key": kname,
                **({"units": args.units} if args.units else {}),
                "ser": float(unit_ser(dec, plain)) if plain is not None else None,
                "map_err_types": float(wrong.mean()) if has_truth else None,
                "map_err_occ": float((occ * wrong).sum() / occ.sum()) if has_truth else None,
                "violations": int(rule_violations(m[sym], sym, adj)),
                "n_plain": int(len(dec)),
            }
            r.update(score_map(ev, table, inst, meta, m, targets, lang, budget=args.budget,
                               seeds=tuple(range(args.seeds)), score_windows=args.score_windows,
                               seed=0))
            r["called"] = bool(r["language_like"] and r["top_language_of_decode"] == lang)
            r["seconds"] = round(time.time() - t0, 1)
            res.append(r)
            write_json_atomic(path, res)
            print(
                f"{name}/{lang} {kname:14s} ser={-1 if r['ser'] is None else r['ser']:.3f} plain={r['plain_bits']:.3f} "
                f"margin={r['structure_margin']:.2f} rank={r['language_rank_of_decode']} "
                f"top_margin={r['top_margin_bits']:.3f}±{r['top_margin_uncertainty_bits']:.3f} "
                f"flip={r['replicate_flip_rate']:.2f} like={r['language_like']} "
                f"called={r['called']} {r['seconds']:.0f}s",
                flush=True,
            )
    report(path, out_dir / f"judge_at_ser{args.tag}.md")


def report(path, md):
    res = json.loads(path.read_text())
    lines = [
        "# Language judge vs letter SER — word-homophonic A-like cells",
        "",
        f"Abstain rule: plain ≤ {ABSTAIN_RULE['max_plain_bits']} bits/char AND structure margin ≥ "
        f"{ABSTAIN_RULE['min_structure_margin']}. `called` = language-like AND top language is the truth.",
        "",
        "| cell | key | SER | map err (occ) | plain bits | structure margin | rank | top margin ± unc | flip | window vote | MDL/sym | like | called |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    fs = lambda x: "n/a" if x is None else f"{x:.3f}"
    for r in sorted(res, key=lambda r: (r["cell"], r.get("hypothesis", ""), -1 if r["ser"] is None else r["ser"])):
        lines.append(
            f"| {r['cell']}/{r.get('hypothesis', r['truth_language'])} | {r['key']} | {fs(r['ser'])} | {fs(r['map_err_occ'])} | {r['plain_bits']:.3f} | "
            f"{r['structure_margin']:.2f} | {'>'.join(l[:2] for l in r['language_rank_of_decode'])} | "
            f"{r['top_margin_bits']:.3f} ± {r['top_margin_uncertainty_bits']:.3f} | {r['replicate_flip_rate']:.2f} | "
            f"{r['window_vote_for_top']:.2f} | {r['total_per_all_symbols']:.3f} | {'yes' if r['language_like'] else 'no'} | "
            f"{'yes' if r['called'] else 'no'} |"
        )
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
