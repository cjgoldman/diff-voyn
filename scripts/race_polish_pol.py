"""Proof-of-life for the racing polish (docs/race_polish_plan.md §3, abridged):
greedy ``elbo_polish`` vs ``race_polish`` on (a) rung-2 synthetic cells with
ground truth — a null control started AT the true key, a repaired-pick cell
and a 2-symbol corruption — and (b) one Borg Latin window from its recorded
MDL-pick key (the Phase-6 failure case). Frozen evaluator, few sweeps.
Artifacts: DATA_ROOT/analysis/race_polish/pol.json"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.diffusion_eval import DiffusionEvaluator
from diff_voyn.heads.ladder import elbo_polish, race_polish, write_json_atomic
from diff_voyn.heads.scale import choice_bits
from diff_voyn.vms.apply import load_instance


def choice_fn(m, dec):
    return choice_bits("homophonic", dec, sym_to_letter=m) / max(len(dec), 1)


def ser_map(m, true_map, cipher):
    return float((m[cipher] != true_map[cipher]).mean())


def run_both(ev, cipher, start, lang, *, sweeps, seed, pair_swaps, log, race_kw):
    out = {}
    t0 = time.time()
    g, gi = elbo_polish(
        ev,
        cipher,
        start,
        language=lang,
        choice_term_in_polish=True,  # deliberately the Phase-6 objective (doc §7)
        choice_fn=choice_fn,
        sweeps=sweeps,
        budget=8,
        confirm_budget=64,
        seed=seed,
        pair_swaps=pair_swaps,
    )
    out["greedy"] = {
        "map": g.tolist(),
        "accepted": gi["accepted"],
        "n_changed": gi["n_changed"],
        "n_sweeps": len(gi["trace"]),
        "seconds": time.time() - t0,
    }
    log(
        f"  greedy: accepted={gi['accepted']} changed={gi['n_changed']} sweeps={len(gi['trace'])} {time.time()-t0:.0f}s"
    )
    t0 = time.time()
    r, ri = race_polish(
        ev,
        cipher,
        start,
        language=lang,
        choice_term_in_polish=True,  # deliberately the Phase-6 objective (doc §7)
        choice_fn=choice_fn,
        sweeps=sweeps,
        confirm_budget=64,
        seed=seed,
        pair_swaps=pair_swaps,
        **race_kw,
    )
    out["race"] = {
        "map": r.tolist(),
        "accepted": ri["accepted"],
        "n_changed": ri["n_changed"],
        "n_moves": ri["n_moves"],
        "draws_total": ri["draws_total"],
        "trace": ri["trace"],
        "seconds": time.time() - t0,
    }
    stages = [
        [(s["n_alive_in"], s["n_alive_out"]) for s in e["stages"]] for e in ri["trace"]
    ]
    log(
        f"  race:   accepted={ri['accepted']} changed={ri['n_changed']} moves={ri['n_moves']} stages={stages} draws={ri['draws_total']} {time.time()-t0:.0f}s"
    )
    return out


def main():
    p = argparse.ArgumentParser()
    root = data_root()
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--sweeps", type=int, default=2)
    p.add_argument("--skip-borg", action="store_true")
    p.add_argument("--borg-chars", type=int, default=4000)
    args = p.parse_args()
    out_dir = root / "analysis/race_polish"
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = (out_dir / "pol.log").open("a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    torch.set_float32_matmul_precision("high")
    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    race_kw = {"budgets": (4, 16, 64, 128), "max_survivors": (None, 64, 12, 4)}
    results = {"rung2": [], "borg": []}

    # -- rung 2 --------------------------------------------------------------
    solves = json.loads((root / "analysis/phase5/rung2_solves.json").read_text())[
        "instances"
    ]
    scores = json.loads((root / "analysis/phase5/rung2_scores.json").read_text())[
        "instances"
    ]
    by = {(i["language"], i["trial"]): i for i in solves}
    sc_by = {(i["language"], i["trial"]): i for i in scores}
    cells = [
        ("latin", 2, "null"),
        ("german", 0, "null"),
        ("latin", 1, "pick"),
        ("italian", 3, "pick"),
        ("latin", 2, "corrupt2"),
        ("german", 0, "corrupt2"),
    ]
    for lang, trial, kind in cells:
        inst = by[(lang, trial)]
        cipher = np.asarray(inst["cipher_ids"], dtype=np.int64)
        true_map = np.asarray(inst["true_map"], dtype=np.int64)
        rng = np.random.default_rng(trial)
        if kind == "null":
            start = true_map.copy()
        elif kind == "pick":
            sl = inst["hypotheses"][lang]["shortlist"]
            # the diffusion pick is the best-ELBO shortlist entry; take the recorded SER match
            want = sc_by[(lang, trial)]["hypotheses"][lang]["diffusion"]["ser"]
            start = np.asarray(
                next(e for e in sl if abs(e["ser"] - want) < 1e-9)["map"],
                dtype=np.int64,
            )
        else:
            start = true_map.copy()
            occ = np.unique(cipher)
            for s in rng.choice(occ, size=2, replace=False):
                start[s] = (start[s] + 1 + rng.integers(0, 24)) % 25
        log(
            f"rung2 {lang} t{trial} {kind}: start SER {ser_map(start, true_map, cipher):.4f}"
        )
        res = run_both(
            ev,
            cipher,
            start,
            lang,
            sweeps=args.sweeps,
            seed=trial,
            pair_swaps=True,
            log=log,
            race_kw=race_kw,
        )
        for k in ("greedy", "race"):
            res[k]["ser"] = ser_map(np.asarray(res[k]["map"]), true_map, cipher)
        res |= {
            "language": lang,
            "trial": trial,
            "kind": kind,
            "ser_start": ser_map(start, true_map, cipher),
        }
        log(
            f"  SER start {res['ser_start']:.4f} -> greedy {res['greedy']['ser']:.4f} | race {res['race']['ser']:.4f}"
        )
        results["rung2"].append(res)
        write_json_atomic(out_dir / "pol.json", results)

    # -- Borg ----------------------------------------------------------------
    if args.skip_borg:
        return
    from anchors import _borg_ser

    adir = root / "analysis/phase6/anchors"
    binst = load_instance(adir / "borg_eva.json")
    bsolves = json.loads((adir / "borg_solves.json").read_text())["instances"]
    bscores = []
    for sp in sorted(adir.glob("borg_scores*.json")):
        bscores += json.loads(sp.read_text())["instances"]
    for hyp, window in [("latin", 0)]:
        sc = next(
            r for r in bscores if r["hypothesis"] == hyp and r["window"] == window
        )
        so = next(
            r for r in bsolves if r["hypothesis"] == hyp and r["window"] == window
        )
        a, b = sc["window_span"]
        b = min(b, a + args.borg_chars)
        sym = np.asarray(binst["symbols"][a:b], dtype=np.int64)
        start = np.asarray(
            next(c for c in so["candidates"] if c["source"] == sc["pick_mdl_source"])[
                "map"
            ],
            dtype=np.int64,
        )
        s0 = _borg_ser(binst, start)
        s_rec = _borg_ser(binst, np.asarray(sc["final"]["key"]["map"]))
        log(
            f"borg {hyp} w{window} [{a},{b}): start SER {s0['ser_weighted']:.4f} (median page {s0['ser_median_page']:.4f}); recorded greedy final {s_rec['ser_weighted']:.4f} ({s_rec['ser_median_page']:.4f})"
        )
        res = run_both(
            ev,
            sym,
            start,
            hyp,
            sweeps=args.sweeps,
            seed=0,
            pair_swaps=False,
            log=log,
            race_kw=race_kw,
        )
        for k in ("greedy", "race"):
            res[k]["ser"] = _borg_ser(binst, np.asarray(res[k]["map"]))
            log(
                f"  {k}: SER {res[k]['ser']['ser_weighted']:.4f} median page {res[k]['ser']['ser_median_page']:.4f}"
            )
        res |= {
            "hypothesis": hyp,
            "window": window,
            "span": [a, b],
            "ser_start": s0,
            "ser_recorded_final": s_rec,
        }
        results["borg"].append(res)
        write_json_atomic(out_dir / "pol.json", results)


if __name__ == "__main__":
    main()
