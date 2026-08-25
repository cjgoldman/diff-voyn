"""Proof of life for the alternating n-gram ↔ diffusion key search
(docs/alt_loop_plan.md). Stages:

  pol1  — PoL-1: at each stuck start key, how precise is the denoiser's
          disagreement set against the truth (and how large is it at the
          true key)? Minutes; go/no-go.
  run   — the arms (posterior k=4/8/all, random k=8, race, none, null) on
          the rung-2 and word-homophonic cells.
  report— tables → analysis/altloop/report.md

Artifacts: DATA_ROOT/analysis/altloop/{pol1,runs}.json
"""

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
from rung2_diffusion import _build_ngram_evaluator

from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.altloop import alternate
from diff_voyn.heads.ladder import race_polish, write_json_atomic
from diff_voyn.heads.posterior import (
    A,
    disagreements,
    position_posterior,
    symbol_scores,
    unit_scores,
)
from diff_voyn.heads.rung2_homophonic import HomophonicHead
from diff_voyn.heads.wordhom import (
    WordHomophonicHead,
    adjacency,
    expand_units,
    rule_violations,
    unit_ser,
)

# -- cells --------------------------------------------------------------------

# (language, trial, shortlist index): index 0 is the recorded MDL pick;
# the bank entries are non-basin local optima with SER ~0.2-0.4 whose
# instance's recorded winner sits in the true basin.
R2_CELLS = [
    ("latin", 5, 0, "t5-pick"),
    ("latin", 5, 4, "t5-oracle"),
    ("latin", 0, 2, "bank"),
    ("latin", 3, 1, "bank"),
    ("italian", 0, 3, "bank"),
    ("italian", 4, 3, "bank"),
    ("german", 0, 3, "bank"),
    ("german", 1, 4, "bank"),
]
# deeper non-basin optima (SER 0.5-0.6) for the cold-SA configuration
R2_DEEP = [
    ("latin", 0, 5, "deep"),
    ("latin", 3, 3, "deep"),
    ("italian", 0, 10, "deep"),
    ("italian", 4, 11, "deep"),
    ("german", 0, 13, "deep"),
    ("german", 1, 15, "deep"),
]
WH_CELLS = [("positive/german/t0", "german"), ("positive/italian/t0", "italian")]
WH_STRETCH = [("positive/german/Alike", "german")]


def log_to(path):
    f = path.open("a")

    def log(s):
        print(s, flush=True)
        f.write(s + "\n")
        f.flush()

    return log


class R2Cell:
    def __init__(self, inst, idx, tag, ng, ev, args):
        self.tag = tag
        self.lang = inst["language"]
        self.name = f"r2/{self.lang}/t{inst['trial']}/sl{idx}/{tag}"
        self.cipher = np.asarray(inst["cipher_ids"], dtype=np.int64)
        self.plain = np.asarray(inst["plain_ids"], dtype=np.int64)
        self.true_map = np.asarray(inst["true_map"], dtype=np.int64)
        sl = inst["hypotheses"][self.lang]["shortlist"][idx]
        self.start = np.asarray(sl["map"], dtype=np.int64)
        self.n_sym = int(inst["n_symbols"])
        self.head = HomophonicHead(ng, seed=inst["trial"])
        self.ev = ev
        self.args = args

    def objective(self, m):
        return self.head._objective(m[self.cipher], self.lang)

    def short_sa(self, m, rng):
        out, sc, _ = self.head._sa_phase(
            self.cipher,
            m.copy(),
            self.lang,
            rng,
            steps=self.args.r2_sa_steps,
            t_start=self.args.t_start,
            t_end=self.args.t_end,
        )
        return out, sc

    def scores(self, m):
        P = position_posterior(
            self.ev,
            m[self.cipher],
            self.lang,
            n_draws=self.args.n_draws,
            mask_rate=self.args.mask_rate,
            seed=self.args.seed,
        )
        return symbol_scores(P, self.cipher, self.n_sym)

    def wrong(self, m):
        return m != self.true_map

    def random_unit(self, s, cur, rng):
        u = int(rng.integers(A))
        return u if u != cur else (u + 1) % A

    def race(self, m):
        out, _ = race_polish(
            self.ev, self.cipher, m, language=self.lang, sweeps=2, seed=self.args.seed
        )
        return out

    def metrics(self, m):
        return {
            "ser": float((m[self.cipher] != self.plain).mean()),
            "n_wrong_symbols": int(self.wrong(m).sum()),
            "elbo_bits": float(
                self.ev.score_stream(m[self.cipher], language=self.lang, n_strata=64)
            ),
        }


class WHCell:
    def __init__(self, inst, hyp, ng, ev, args, start_map):
        self.name = f"wh/{inst['name']}/{hyp}"
        self.tag = "wh"
        self.lang = hyp
        self.symbols = np.asarray(inst["symbols"], dtype=np.int64)
        self.pos = np.asarray(inst["token_pos"], dtype=np.int64)
        self.adj = adjacency(self.symbols, self.pos)
        tr = inst["truth"]
        self.plain = np.asarray(tr["plain_ids"], dtype=np.int64)
        self.true_map = np.asarray(tr["sym_to_unit"], dtype=np.int64)
        self.n_sym = int(inst["n_symbols"])
        self.head = WordHomophonicHead(ng, seed=args.seed)
        self.targets = self.head.targets_for(hyp)
        assert self.targets.as_list() == tr["bigrams"], "targets mismatch"
        self.start = np.asarray(start_map, dtype=np.int64)
        self.ev = ev
        self.args = args
        self.occ = np.bincount(self.symbols, minlength=self.n_sym)

    def decode(self, m):
        return expand_units(m[self.symbols], self.targets)

    def objective(self, m):
        return self.head.objective(m, self.symbols, self.adj, self.lang, self.targets)

    def short_sa(self, m, rng):
        out, sc, _ = self.head.sa_phase(
            self.symbols,
            self.adj,
            m.copy(),
            self.lang,
            self.targets,
            rng,
            steps=self.args.wh_sa_steps,
            t_start=self.args.t_start,
            t_end=self.args.t_end,
        )
        return out, sc

    def scores(self, m):
        P = position_posterior(
            self.ev,
            self.decode(m),
            self.lang,
            n_draws=self.args.n_draws,
            mask_rate=self.args.mask_rate,
            seed=self.args.seed,
        )
        return unit_scores(P, self.symbols, m, self.targets)

    def wrong(self, m):
        return m != self.true_map

    def random_unit(self, s, cur, rng):
        if cur < A:
            u = int(rng.integers(A))
            return u if u != cur else (u + 1) % A
        nb = self.targets.n - A
        u = A + int(rng.integers(nb))
        return u if u != cur else A + (u - A + 1) % nb

    def race(self, m):
        raise NotImplementedError("race arm is rung-2 only (symbol→letter interface)")

    def metrics(self, m):
        dec = self.decode(m)
        w = self.wrong(m)
        return {
            "ser": float(unit_ser(dec, self.plain)),
            "map_err_occ": float((self.occ * w).sum() / self.occ.sum()),
            "n_wrong_types": int(w.sum()),
            "violations": int(rule_violations(m[self.symbols], self.symbols, self.adj)),
            "elbo_bits": float(
                self.ev.score_stream(dec, language=self.lang, n_strata=64)
            ),
        }


def build_cells(args, ng, ev, *, stretch=False):
    root = data_root()
    cells = []
    r2 = json.loads((root / "analysis/phase5/rung2_solves.json").read_text())
    by = {(i["language"], i["trial"]): i for i in r2["instances"]}
    for lang, t, idx, tag in R2_CELLS + (R2_DEEP if args.deep else []):
        cells.append(R2Cell(by[(lang, t)], idx, tag, ng, ev, args))
    wd = root / "analysis/wordhom"
    solves = json.loads((wd / "controls_solves.json").read_text())["instances"]
    for name, hyp in WH_CELLS + (WH_STRETCH if stretch else []):
        fname = name.replace("/", "_") + "_wordtypesall.json"
        inst = json.loads((wd / "controls/wordtypesall" / fname).read_text())
        rec = next(
            s for s in solves if s["instance"] == name and s["hypothesis"] == hyp
        )
        best = max(rec["candidates"], key=lambda c: c["inner"])
        cells.append(WHCell(inst, hyp, ng, ev, args, best["map"]))
    if args.only:
        cells = [c for c in cells if any(o in c.name for o in args.only)]
    return cells


# -- stages -------------------------------------------------------------------


def precision_at(D, wrong, true_map, ks=(4, 8, 16, None)):
    out = {}
    for k in ks:
        take = D if k is None else D[:k]
        if not take:
            out[str(k)] = None
            continue
        hits = [wrong[s] for s, _, _ in take]
        exact = [true_map[s] == u for s, u, _ in take]
        out[str(k)] = {
            "n": len(take),
            "precision_wrong": float(np.mean(hits)),
            "precision_exact": float(np.mean(exact)),
        }
    return out


def stage_pol1(args, cells, log, out_dir):
    res = []
    for c in cells:
        t0 = time.time()
        wrong = c.wrong(c.start)
        D = disagreements(c.scores(c.start), c.start)
        D_true = disagreements(c.scores(c.true_map), c.true_map)
        rec = {
            "cell": c.name,
            "n_symbols": c.n_sym,
            "n_wrong_at_start": int(wrong.sum()),
            "start_metrics": c.metrics(c.start),
            "n_disagree_start": len(D),
            "precision": precision_at(D, wrong, c.true_map),
            "n_disagree_at_truth": len(D_true),
            "disagree_at_truth_top": [
                (s, int(c.true_map[s]), u, round(m, 2)) for s, u, m in D_true[:8]
            ],
            "seconds": time.time() - t0,
        }
        res.append(rec)
        p = rec["precision"]
        log(
            f"{c.name}: wrong {rec['n_wrong_at_start']}/{c.n_sym}  |D|={len(D)} "
            f"prec@8 wrong={p['8'] and p['8']['precision_wrong']:.2f} exact={p['8'] and p['8']['precision_exact']:.2f} "
            f"prec@all wrong={p['None'] and p['None']['precision_wrong']:.2f}  |D(truth)|={len(D_true)}  "
            f"ser={rec['start_metrics']['ser']:.3f}  {rec['seconds']:.0f}s"
        )
        write_json_atomic(out_dir / "pol1.json", res)


ARMS = {
    "post-k4": ("posterior", 4),
    "post-k8": ("posterior", 8),
    "post-all": ("posterior", None),
    "psamp-k8": ("posterior_sample", 8),
    "psamp-all": ("posterior_sample", None),
    "rand-k8": ("random", 8),
    "rand-k512": ("random", 512),
    "none": ("none", None),
    "race": ("race", None),
}


def stage_run(args, cells, log, out_dir):
    path = out_dir / f"runs{args.tag}.json"
    res = json.loads(path.read_text()) if path.exists() else []
    done = {(r["cell"], r["arm"], r["start"], r["seed"]): r for r in res}
    for c in cells:
        for arm in args.arms:
            mech, k = ARMS[arm]
            if mech == "race" and c.tag == "wh":
                continue
            for start_name in ("stuck", "null"):
                if start_name == "null" and arm not in (
                    "post-k8",
                    "post-all",
                    "rand-k8",
                    "rand-k512",
                ):
                    continue
                for seed in range(args.seeds):
                    key = (c.name, arm, start_name, seed)
                    if key in done:
                        continue
                    start = c.true_map if start_name == "null" else c.start
                    t0 = time.time()
                    _, info = alternate(
                        start,
                        mechanism=mech,
                        objective=c.objective,
                        short_sa=c.short_sa,
                        scores_fn=c.scores,
                        race_fn=c.race,
                        random_unit=c.random_unit,
                        metrics=c.metrics,
                        occ=np.bincount(
                            c.cipher if c.tag != "wh" else c.symbols, minlength=c.n_sym
                        ),
                        k=k,
                        rounds=args.rounds,
                        seed=args.seed + 101 * seed,
                    )
                    rec = {
                        "cell": c.name,
                        "arm": arm,
                        "start": start_name,
                        "seed": seed,
                        "seconds": time.time() - t0,
                        **info,
                    }
                    res.append(rec)
                    done[key] = rec
                    sm, fm = info["start_metrics"], info["final_metrics"]
                    log(
                        f"{c.name} {arm} {start_name} s{seed}: ser {sm['ser']:.3f}→{fm['ser']:.3f} "
                        f"obj {info['start_obj']:.1f}→{info['final_obj']:.1f} "
                        f"acc {info['n_accepted']}/{info['n_rounds']} {rec['seconds']:.0f}s"
                    )
                    write_json_atomic(path, res)


def stage_report(args, out_dir):
    res = []
    for f in sorted(out_dir.glob("runs*.json")):
        res += json.loads(f.read_text())
    lines = [
        "# Alternating loop — proof of life",
        "",
        "| cell | arm | start | seeds | SER start → final (per seed) | n-gram obj Δ | accepted rounds |",
        "|---|---|---|---|---|---|---|",
    ]
    groups = {}
    for r in res:
        groups.setdefault((r["cell"], r["arm"], r["start"]), []).append(r)
    for (cell, arm, start), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["seed"])
        sers = ", ".join(
            f"{r['start_metrics']['ser']:.3f}→{r['final_metrics']['ser']:.3f}"
            for r in rs
        )
        dobj = ", ".join(f"{r['final_obj']-r['start_obj']:+.1f}" for r in rs)
        acc = ", ".join(f"{r['n_accepted']}/{r['n_rounds']}" for r in rs)
        lines.append(
            f"| {cell} | {arm} | {start} | {len(rs)} | {sers} | {dobj} | {acc} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    root = data_root()
    p.add_argument("--stage", choices=["pol1", "run", "report"], required=True)
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--n-draws", type=int, default=16)
    p.add_argument("--mask-rate", type=float, default=0.3)
    p.add_argument("--r2-sa-steps", type=int, default=50_000)
    p.add_argument("--wh-sa-steps", type=int, default=200_000)
    p.add_argument("--t-start", type=float, default=2.0)
    p.add_argument("--t-end", type=float, default=0.3)
    p.add_argument("--stretch", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--deep", action="store_true")
    args = p.parse_args()
    out_dir = root / "analysis/altloop"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = log_to(out_dir / f"{args.stage}{args.tag}.log")
    if args.stage == "report":
        stage_report(args, out_dir)
        return
    torch.set_float32_matmul_precision("high")
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator

    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    ng = _build_ngram_evaluator()
    cells = build_cells(args, ng, ev, stretch=args.stretch)
    log(
        f"== {args.stage} {time.strftime('%FT%T')} cells={[c.name for c in cells]} args={vars(args)}"
    )
    {"pol1": stage_pol1, "run": stage_run}[args.stage](args, cells, log, out_dir)


if __name__ == "__main__":
    main()
