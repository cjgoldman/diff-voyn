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
WH_STRETCH = [
    ("positive/german/Alike", "german"),
    ("positive/italian/Alike", "italian"),
    ("positive/latin/Alike", "latin"),
]


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
        tr = inst.get("truth", {})
        self.n_sym = int(inst["n_symbols"])
        self.head = WordHomophonicHead(ng, seed=args.seed)
        self.targets = self.head.targets_for(hyp)
        # battery cells (scripts/wordhom_battery.py): negatives carry no key
        # and cross-language cells a key in another hypothesis' unit space —
        # type-level truth only when the hypothesis is the generating one
        self.has_truth = (
            tr.get("kind") == "wordhom" and tr.get("language") == hyp
        )
        self.plain = (
            np.asarray(tr["plain_ids"], dtype=np.int64) if "plain_ids" in tr else None
        )
        if self.has_truth:
            self.true_map = np.asarray(tr["sym_to_unit"], dtype=np.int64)
            assert self.targets.as_list() == tr["bigrams"], "targets mismatch"
        else:
            self.true_map = None
        self.start = np.asarray(start_map, dtype=np.int64)
        self.ev = ev
        self.args = args
        self.occ = np.bincount(self.symbols, minlength=self.n_sym)
        # hapax-masking proposer (arms ``*-hm``): letter positions of types
        # with <= hapax_max occurrences are withheld from the denoiser
        self.hapax_mask = False
        # no-hapax proposer (arms ``*-nh``): rare types never enter the
        # disagreement set (their score row is -inf → no finite margin)
        self.drop_hapax = False
        self.rare_type = self.occ <= args.hapax_max
        self.rare_tok = self.rare_type[self.symbols]
        self.wild = None
        # --wild: hapax types are wildcards in the n-gram objective
        # (docs/alt_loop_plan.md §8.4) — frozen out of SA and proposals
        if getattr(args, "wild", False):
            self.set_wild(self.rare_type)
            self.drop_hapax = True

    def set_wild(self, mask):
        """Current wildcard set: charged a constant in the n-gram objective,
        frozen out of SA/polish and (``drop_hapax``) of the proposals."""
        self.wild = np.asarray(mask, dtype=bool)
        self.head.wild_types = self.wild if self.wild.any() else None

    def wild_schedule(self, start, end, seed):
        """§8.6 anneal: from round ``start`` to ``end`` the hapax types are
        re-admitted to the objective (and to the search) in equal batches,
        in a fixed seeded random order, so the objective is the wildcard one
        before ``start`` and the standard one from ``end`` on. Returns the
        ``schedule`` callable for :func:`alternate` (a dict on the rounds
        where the set shrinks, None otherwise)."""
        rare = np.flatnonzero(self.rare_type)
        order = np.random.default_rng(seed).permutation(rare)
        n_steps = end - start + 1

        def schedule(r):
            if r < start or r > end + 1:
                return None
            frac = min(1.0, (r - start + 1) / n_steps) if r <= end else 1.0
            n_admit = round(frac * len(order))
            mask = self.rare_type.copy()
            mask[order[:n_admit]] = False
            if r > start and int(mask.sum()) == int(self.wild.sum()):
                return None
            self.set_wild(mask)
            return {"n_wild": int(mask.sum()), "n_admitted": n_admit}

        return schedule

    def decode(self, m):
        return expand_units(m[self.symbols], self.targets)

    def rare_positions(self, m):
        """(L,) bool over the decode's letter positions: emitted by a rare type."""
        isbig = self.targets.second[m[self.symbols]] >= 0
        return np.repeat(self.rare_tok, 1 + isbig)

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
        dec = self.decode(m)
        P = position_posterior(
            self.ev,
            dec,
            self.lang,
            n_draws=self.args.n_draws,
            mask_rate=self.args.mask_rate,
            seed=self.args.seed,
            force_mask=self.rare_positions(m) if self.hapax_mask else None,
            force_rate=self.args.hapax_mask_rate,
        )
        S = unit_scores(P, self.symbols, m, self.targets)
        if self.drop_hapax:
            S[self.wild if self.wild is not None else self.rare_type] = -np.inf
        return S

    def wrong(self, m):
        if self.true_map is None:
            return np.zeros(len(m), dtype=bool)
        return m != self.true_map

    def judge_bits(self, m, seed):
        return float(
            self.ev.score_stream(
                self.decode(m), language=self.lang, n_strata=64, seed=seed
            )
        )

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
            "ser": float(unit_ser(dec, self.plain)) if self.plain is not None else None,
            "map_err_occ": (
                float((self.occ * w).sum() / self.occ.sum()) if self.has_truth else None
            ),
            "n_wrong_types": int(w.sum()) if self.has_truth else None,
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
    if getattr(args, "battery", False):
        cells = build_battery_cells(args, ng, ev)
    if args.only:
        cells = [c for c in cells if any(o in c.name for o in args.only)]
    return cells


def build_battery_cells(args, ng, ev):
    """``--cells NAME:HYP`` from the wordhom battery
    (``analysis/wordhom/battery/wordtypesall`` + ``battery_solves.json``),
    falling back to the Phase-6 wordhom controls (positives) for names not
    in the battery; the start is the solve's n-gram MDL pick as always."""
    root = data_root()
    wd = root / "analysis/wordhom"
    bat = wd / "battery/wordtypesall"
    man = {m["name"]: m for m in json.loads((bat / "manifest.json").read_text())}
    solves = []
    for fn in ("battery/battery_solves.json", "controls_solves.json"):
        if (wd / fn).exists():
            solves += json.loads((wd / fn).read_text())["instances"]
    cells = []
    for spec in args.cells:
        name, hyp = spec.rsplit(":", 1)
        if name in man:
            inst = json.loads((bat / man[name]["file"]).read_text())
        else:
            fname = name.replace("/", "_") + "_wordtypesall.json"
            inst = json.loads((wd / "controls/wordtypesall" / fname).read_text())
        rec = next(
            (s for s in solves if s["instance"] == name and s["hypothesis"] == hyp),
            None,
        )
        if rec is None:
            print(f"no solve for {name}:{hyp}, skipped", flush=True)
            continue
        best = max(rec["candidates"], key=lambda c: c["inner"])
        cells.append(WHCell(inst, hyp, ng, ev, args, best["map"]))
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
    # small-commitment arms (docs/alt_loop_plan.md §9): commit the 1-2
    # most-supported symbols per round, re-read the posterior, repeat
    "post-k1": ("posterior", 1),
    "psamp-k1": ("posterior_sample", 1),
    "psamp-k2": ("posterior_sample", 2),
    "psamp-k64": ("posterior_sample", 64),
    "rand-k2": ("random", 2),
    # hapax-masked variants (wordhom cells only): rare types' letters are
    # withheld from the denoiser before the posterior is read
    "post-all-hm": ("posterior", None),
    "psamp-all-hm": ("posterior_sample", None),
    "post-k8-hm": ("posterior", 8),
    # no-hapax variants: rare types are dropped from the proposal set
    "post-all-nh": ("posterior", None),
    "psamp-all-nh": ("posterior_sample", None),
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
            if c.tag == "wh":
                c.hapax_mask = arm.endswith("-hm")
                c.drop_hapax = arm.endswith("-nh") or bool(args.wild)
            elif arm.endswith(("-hm", "-nh")):
                continue
            for start_name in ("stuck", "null"):
                if start_name == "null" and args.no_null:
                    continue
                if start_name == "null" and arm not in (
                    "psamp-k1",
                    "post-k8",
                    "post-all",
                    "rand-k8",
                    "rand-k512",
                    "post-all-hm",
                    "psamp-all-hm",
                    "post-all-nh",
                    "psamp-all-nh",
                ):
                    continue
                for seed in range(args.seeds):
                    key = (c.name, arm, start_name, seed)
                    if key in done:
                        continue
                    if start_name == "null" and getattr(c, "true_map", None) is None:
                        continue
                    start = c.true_map if start_name == "null" else c.start
                    if args.start_from:
                        prev = json.loads(
                            (out_dir / f"runs{args.start_from}.json").read_text()
                        )
                        pv = next(
                            (
                                r
                                for r in prev
                                if r["cell"] == c.name
                                and r["seed"] == seed
                                and r["start"] == start_name
                                and "final_map" in r
                            ),
                            None,
                        )
                        if pv is None:
                            log(
                                f"{c.name} {arm} {start_name} s{seed}: no final_map in runs{args.start_from}.json, skipped"
                            )
                            continue
                        start = np.asarray(pv["final_map"], dtype=np.int64)
                    accept_fn = None
                    if args.judge_accept is not None:
                        margin = args.judge_accept

                        def accept_fn(cur, new, r, c=c, margin=margin, seed=seed):
                            s_ = (
                                7919 * (seed + 1) + r
                            )  # CRN within a round, fresh masks across rounds
                            b_cur = c.judge_bits(cur, s_)
                            b_new = c.judge_bits(new, s_)
                            return b_new < b_cur - margin, {
                                "bits_cur": b_cur,
                                "bits_new": b_new,
                            }

                    schedule = None
                    if args.wild and args.wild_anneal and c.tag == "wh":
                        c.set_wild(c.rare_type)  # the schedule mutates it
                        a0, a1 = args.wild_anneal
                        schedule = c.wild_schedule(a0, a1, args.seed + 101 * seed)
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
                        patience=args.patience,
                        seed=args.seed + 101 * seed,
                        accept_fn=accept_fn,
                        schedule=schedule,
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
                    fs = lambda x: "n/a" if x is None else f"{x:.3f}"
                    log(
                        f"{c.name} {arm} {start_name} s{seed}: ser {fs(sm['ser'])}→{fs(fm['ser'])} "
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
        fs = lambda x: "n/a" if x is None else f"{x:.3f}"
        sers = ", ".join(
            f"{fs(r['start_metrics']['ser'])}→{fs(r['final_metrics']['ser'])}"
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
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--n-draws", type=int, default=16)
    p.add_argument("--mask-rate", type=float, default=0.3)
    p.add_argument(
        "--hapax-max",
        type=int,
        default=1,
        help="-hm arms: mask types with <= this many occurrences",
    )
    p.add_argument(
        "--hapax-mask-rate",
        type=float,
        default=1.0,
        help="-hm arms: per-draw probability a rare position is masked",
    )
    p.add_argument("--no-null", action="store_true")
    p.add_argument("--wild", action="store_true")
    p.add_argument(
        "--wild-anneal",
        type=lambda v: tuple(int(x) for x in v.split(",")),
        default=None,
        help="START,END rounds over which the wildcard (hapax) set is re-admitted "
        "to the objective in equal batches (docs/alt_loop_plan.md §8.6)",
    )
    p.add_argument(
        "--start-from",
        default=None,
        help="tag of a runs file whose final_map seeds the start",
    )
    p.add_argument(
        "--judge-accept",
        type=float,
        default=None,
        help="accept a round iff the frozen judge's bits/char drop by more than this margin (CRN-paired)",
    )
    p.add_argument("--r2-sa-steps", type=int, default=50_000)
    p.add_argument("--wh-sa-steps", type=int, default=200_000)
    p.add_argument("--t-start", type=float, default=2.0)
    p.add_argument("--t-end", type=float, default=0.3)
    p.add_argument("--stretch", action="store_true")
    p.add_argument(
        "--battery",
        action="store_true",
        help="cells from --cells NAME:HYP over the wordhom battery (scripts/wordhom_battery.py)",
    )
    p.add_argument("--cells", nargs="*", default=[])
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
