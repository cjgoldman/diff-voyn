"""Hapax-as-wildcard objective (docs/alt_loop_plan.md §8.4).

Types with <= ``--hapax-max`` occurrences are frozen at the start key and
their letters are charged a constant + reset the n-gram context
(``WordHomophonicHead.wild_types``), so the n-gram objective ranks keys by
the frequent types alone. Reports, per wordhom cell: the stuck-vs-truth
gap under the standard and the wildcard objective, and SA from the stuck
start / the truth / a cold init under the wildcard objective.

Artifacts: DATA_ROOT/analysis/altloop/hapax_wildcard{tag}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from altloop_pol import _build_ngram_evaluator, build_cells, data_root

from diff_voyn.heads.ladder import write_json_atomic
from diff_voyn.heads.wordhom import unit_ser

NO_WILD = False  # --no-wild: same SA under the standard objective (control)


def summarize(c, m, head):
    w = c.wrong(m)
    nh = ~c.rare_type
    dec = c.decode(m)
    return {
        "ser": float(unit_ser(dec, c.plain)),
        "map_err_occ": float((c.occ * w).sum() / c.occ.sum()),
        "map_err_occ_nonhapax": float((c.occ * w)[nh].sum() / c.occ[nh].sum()),
        "type_err_nonhapax": float(w[nh].mean()),
        "type_err_hapax": float(w[~nh].mean()) if (~nh).any() else None,
        "obj_std": float(std_objective(c, m, head)),
        "obj_wild": float(wild_objective(c, m, head)),
    }


def std_objective(c, m, head):
    saved, head.wild_types = head.wild_types, None
    try:
        return head.objective(m, c.symbols, c.adj, c.lang, c.targets)
    finally:
        head.wild_types = saved


def wild_objective(c, m, head):
    saved, head.wild_types = head.wild_types, c.rare_type
    try:
        return head.objective(m, c.symbols, c.adj, c.lang, c.targets)
    finally:
        head.wild_types = saved


def run_sa(c, head, m0, rng, steps, t_start, t_end):
    head.wild_types = None if NO_WILD else c.rare_type
    try:
        out, sc, _ = head.sa_phase(
            c.symbols,
            c.adj,
            m0.copy(),
            c.lang,
            c.targets,
            rng,
            steps=steps,
            t_start=t_start,
            t_end=t_end,
        )
    finally:
        head.wild_types = None
    return out, float(sc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=Path,
        default=data_root() / "runs/phase_c-85m-seed0/ckpt_final.pt",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", nargs="*", default=["wh/"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hapax-max", type=int, default=1)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--t-start", type=float, default=2.0)
    p.add_argument("--t-end", type=float, default=0.3)
    p.add_argument("--cold-t-start", type=float, default=15.0)
    p.add_argument("--cold-t-end", type=float, default=0.5)
    p.add_argument("--stretch", action="store_true")
    p.add_argument("--deep", action="store_true")
    p.add_argument("--n-draws", type=int, default=16)
    p.add_argument("--mask-rate", type=float, default=0.3)
    p.add_argument("--hapax-mask-rate", type=float, default=1.0)
    p.add_argument("--no-wild", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    global NO_WILD
    NO_WILD = args.no_wild

    torch.set_float32_matmul_precision("high")
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator

    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    ng = _build_ngram_evaluator()
    cells = [
        c for c in build_cells(args, ng, ev, stretch=args.stretch) if c.tag == "wh"
    ]
    out_dir = data_root() / "analysis/altloop"
    path = out_dir / f"hapax_wildcard{args.tag}.json"
    res = json.loads(path.read_text()) if path.exists() else []
    done = {(r["cell"], r["start"], r["seed"]) for r in res}

    for c in cells:
        head = c.head
        c.rare_type = c.occ <= args.hapax_max
        nh = int(c.rare_type.sum())
        print(
            f"\n== {c.name}: {nh}/{len(c.occ)} wildcard types "
            f"({c.rare_type[c.symbols].mean():.1%} of tokens)",
            flush=True,
        )
        stuck = summarize(c, c.start, head)
        truth = summarize(c, c.true_map, head)
        print(
            f"  stuck: std {stuck['obj_std']:.1f} wild {stuck['obj_wild']:.1f} ser {stuck['ser']:.3f}"
        )
        print(f"  truth: std {truth['obj_std']:.1f} wild {truth['obj_wild']:.1f}")
        print(
            f"  gap truth-stuck: std {truth['obj_std']-stuck['obj_std']:+.1f}  wild {truth['obj_wild']-stuck['obj_wild']:+.1f}",
            flush=True,
        )
        # the truth with hapaxes frozen at the *stuck* assignment: what the
        # wildcard search could reach at best
        reach = c.true_map.copy()
        reach[c.rare_type] = c.start[c.rare_type]
        reach_s = summarize(c, reach, head)
        starts = {
            "stuck": (c.start, args.t_start, args.t_end),
            "truth": (reach, args.t_start, args.t_end),
            "cold": (None, args.cold_t_start, args.cold_t_end),
        }
        for seed in range(args.seeds):
            for name, (m0, ts, te) in starts.items():
                if (c.name, name, seed) in done:
                    continue
                rng = np.random.default_rng(args.seed + 1000 * seed + 7)
                if m0 is None:
                    m0 = head.frequency_init(c.symbols, len(c.occ), c.lang, c.targets)
                    m0[c.rare_type] = c.start[c.rare_type]
                t0 = time.time()
                out, _sc = run_sa(c, head, m0, rng, args.steps, ts, te)
                fin = summarize(c, out, head)
                rec = {
                    "cell": c.name,
                    "start": name,
                    "seed": seed,
                    "steps": args.steps,
                    "hapax_max": args.hapax_max,
                    "n_wild": nh,
                    "stuck": stuck,
                    "truth": truth,
                    "reachable_truth": reach_s,
                    "final": fin,
                    "seconds": time.time() - t0,
                }
                res.append(rec)
                write_json_atomic(path, res)
                print(
                    f"  {name:5s} s{seed}: ser {fin['ser']:.3f} nonhapax-err {fin['map_err_occ_nonhapax']:.3f} "
                    f"wild {fin['obj_wild']:.1f} std {fin['obj_std']:.1f} ({rec['seconds']:.0f}s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
