"""Diffusion-guided n-gram key search on the manuscript, with controls in
every cell (``docs/altloop_vms_plan.md``).

  run    — one head (--head wordhom|homophonic|sub1to1): every cell runs the
           three arms (none → rand → psamp, so the control gate can be read
           the moment psamp reports), per-round metrics stream to ClearML
           and to analysis/altloop_vms/events.log; NOTABLE-or-better rounds
           rewrite promising.json.
  report — analysis/altloop_vms/report.md from the runs JSONs.

Nothing here touches the Phase-6 record (ABSTAIN_RULE, controls, the 87
cells); this is a new search on top of it, reported separately. The
"promising" tiers are fixed in ``heads.altloop`` (§5) before any manuscript
number is read.
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
from diff_voyn.heads.altloop import (
    ABSTAIN_MAX_PLAIN,
    ABSTAIN_MIN_MARGIN,
    NOTABLE_ABOVE_CONTROLS,
    NOTABLE_MIN,
    REF_TRUE_MIN,
    REF_VMS_CEILING,
    alternate,
    classify_tier,
)
from diff_voyn.heads.ladder import write_json_atomic
from diff_voyn.heads.posterior import A, position_posterior, symbol_scores, unit_scores
from diff_voyn.heads.two_tier import paired_bits
from diff_voyn.metrology import CalibrationTable, calibrate_bits
from diff_voyn.vms.apply import KEY as APPLY_KEY
from diff_voyn.vms.apply import LANGS, build_ngram_evaluator, head_key_bits

HEADS = ("wordhom", "homophonic", "sub1to1")
# arm -> (mechanism, k) per head (§3)
ARMS = {
    "wordhom": {
        "none": ("none", None),
        "rand": ("random", 512),
        "post": ("posterior", None),
        "psamp": ("posterior_sample", None),
    },
    "homophonic": {
        "none": ("none", None),
        "rand": ("random", 8),
        "psamp": ("posterior_sample", 8),
    },
    "sub1to1": {
        "none": ("none", None),
        "rand": ("random_swap", 4),
        "psamp": ("pair_swap", 4),
    },
}
ARM_ORDER = ("none", "rand", "psamp")  # controls first: the gate reads on the spot
CONTROLS = ("none", "rand")


def is_treatment(arm: str) -> bool:
    """Judge-driven arms (post/psamp) — the ones the §5 tiers are read on."""
    return arm not in CONTROLS


RUN_KEY = ("cell", "arm", "seed")
INSTANCES = ("IT2a/A", "RF1b/A", "IT2a/B", "RF1b/B")  # A before B (§2)
BOXER = ("boxer20/A", "boxer20/B")


def log_to(path: Path):
    f = path.open("a")

    def log(s):
        print(s, flush=True)
        f.write(s + "\n")
        f.flush()

    return log


def cell_seed(name: str) -> int:
    return zlib.crc32(name.encode()) % (2**31)


# -- cells --------------------------------------------------------------------


class Cell:
    """One (head × presentation × dialect × window × language) manuscript
    cell: the recorded window, its start key, the n-gram objective and
    short SA of the head, the judge's proposer scores, and the §4 metrics
    (own-condition ELBO of the window decode and of a letter-shuffled copy,
    same masks for every arm and round of the cell — CRN)."""

    def __init__(self, head, inst, rec, start_map, ng, ev, args, offs, n_cipher_all):
        self.head = head
        self.lang = rec["hypothesis"]
        self.inst_name = rec["instance"]
        self.window = int(rec["window"])
        self.name = (
            f"{head}/{rec['presentation']}:{rec['instance']}/w{self.window}/{self.lang}"
        )
        a, b = rec["window_span"]
        self.span = (int(a), int(b))
        self.start = np.asarray(start_map, dtype=np.int64)
        self.n_sym = int(inst["n_symbols"])
        self.ev, self.ng, self.args, self.offs = ev, ng, args, offs
        self.seed = cell_seed(self.name)
        self.n_cipher_all = n_cipher_all
        if head == "wordhom":
            from diff_voyn.heads.wordhom import WordHomophonicHead, adjacency

            self.symbols = np.asarray(inst["symbols"][a:b], dtype=np.int64)
            pos = np.asarray(inst["token_pos"][a:b], dtype=np.int64)
            self.adj = adjacency(self.symbols, pos)
            # unit set: d5 (Phase-6 hypothesis) or --units d5b20 (doubles +
            # top-20 bigrams variant, 2026-08-30); the start key must come
            # from a solve in the same space (vms_solves<_units>.json)
            self.units = getattr(args, "units", None)
            self.hd = WordHomophonicHead(ng, seed=self.seed, units=self.units)
            self.targets = self.hd.targets_for(self.lang)
            assert len(rec["candidates"][0]["bigrams"]) == len(self.targets.bigrams), (
                self.name, "solve/hypothesis unit-set mismatch")
            self.n_cipher_window = int(sum(len(t) for t in inst["all_tokens"][a:b]))
            self.key_len = self.n_sym
            # hapax-as-wildcard objective (docs/alt_loop_plan.md §8.4/8.6):
            # rare types are charged a constant in the n-gram objective,
            # frozen out of SA/polish and dropped from the judge's proposals
            occ = np.bincount(self.symbols, minlength=self.n_sym)
            self.rare_type = occ <= int(getattr(args, "hapax_max", 1))
            self.wild = None
            if getattr(args, "wild", False):
                self.set_wild(self.rare_type)
        else:
            self.symbols = np.asarray(inst["symbols"][a:b], dtype=np.int64)
            self.n_cipher_window = len(self.symbols)
            if head == "homophonic":
                from diff_voyn.heads.rung2_homophonic import HomophonicHead

                self.hd = HomophonicHead(ng, seed=self.seed)
                self.key_len = self.n_sym
            else:
                from diff_voyn.heads.rung1_sinkhorn import SinkhornSubstitutionHead

                self.hd = SinkhornSubstitutionHead(ng, seed=self.seed)
                self.key_len = A  # injective 25-slot key (Phase-6 convention)
        assert len(self.start) == self.key_len, (
            self.name,
            len(self.start),
            self.key_len,
        )
        self.occ = np.bincount(self.symbols, minlength=self.key_len)
        self.key_bits = head_key_bits(head, self.n_sym)

    # -- wildcard set (wordhom only) ---------------------------------------

    def set_wild(self, mask):
        self.wild = np.asarray(mask, dtype=bool)
        self.hd.wild_types = self.wild if self.wild.any() else None

    def wild_schedule(self, start, end, seed):
        """§8.6 anneal: rounds ``start..end`` re-admit the rare types to the
        objective (and the search) in equal batches, seeded random order;
        standard objective from ``end`` on. Returns ``alternate``'s
        ``schedule`` callable (a dict on rounds where the set shrinks)."""
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
            return {"n_wild": int(mask.sum()), "n_admitted": int(n_admit)}

        return schedule

    # -- head interface ------------------------------------------------------

    def decode(self, m):
        if self.head == "wordhom":
            from diff_voyn.heads.wordhom import expand_units

            return expand_units(m[self.symbols], self.targets)
        return m[self.symbols]

    def objective(self, m):
        if self.head == "wordhom":
            return self.hd.objective(m, self.symbols, self.adj, self.lang, self.targets)
        if self.head == "homophonic":
            return self.hd._objective(m[self.symbols], self.lang)
        return self.ng.score_hard(
            m[self.symbols], language=self.lang, order=self.hd.rescore_order
        )

    def short_sa(self, m, rng):
        a = self.args
        if self.head == "wordhom":
            out, sc, _ = self.hd.sa_phase(
                self.symbols,
                self.adj,
                m.copy(),
                self.lang,
                self.targets,
                rng,
                steps=a.wh_sa_steps,
                t_start=a.t_start,
                t_end=a.t_end,
            )
            return out, sc
        if self.head == "homophonic":
            out, sc, _ = self.hd._sa_phase(
                self.symbols,
                m.copy(),
                self.lang,
                rng,
                steps=a.sym_sa_steps,
                t_start=a.t_start,
                t_end=a.t_end,
            )
            return out, sc
        # sub1to1: the head's own local search (2-swap climb + a few ILS kicks)
        out, sc, _ = self.hd._ils(
            self.symbols, m.copy(), self.lang, rng, kicks=a.ils_kicks
        )
        return out, sc

    def scores(self, m):
        P = position_posterior(
            self.ev,
            self.decode(m),
            self.lang,
            n_draws=self.args.n_draws,
            mask_rate=self.args.mask_rate,
            seed=self.seed,
        )
        if self.head == "wordhom":
            S = unit_scores(P, self.symbols, m, self.targets)
            if self.wild is not None and self.wild.any():
                S[self.wild] = -np.inf  # wildcards never enter the proposal set
            return S
        return symbol_scores(P, self.symbols, self.key_len)

    def random_unit(self, s, cur, rng):
        if self.head == "wordhom" and cur >= A:
            nb = self.targets.n - A
            u = A + int(rng.integers(nb))
            return u if u != cur else A + (u - A + 1) % nb
        u = int(rng.integers(A))
        return u if u != cur else (u + 1) % A

    def choice_bits(self, m, dec):
        if self.head == "sub1to1":
            return 0.0
        if self.head == "homophonic":
            from diff_voyn.heads.scale import choice_bits

            return float(choice_bits("homophonic", dec, sym_to_letter=m))
        from diff_voyn.heads.wordhom import choice_bits_total, repeat_positions

        return float(
            choice_bits_total(
                m,
                self.symbols,
                repeat_positions(self.symbols, self.adj),
                self.targets.n,
            )
        )

    # -- §4 metrics ----------------------------------------------------------

    def _cuts(self, n):
        W = self.ev.window
        cuts = [(s, s + W) for s in range(0, max(n - W + 1, 1), W)]
        if n < W:
            cuts = [(0, n)]
        if len(cuts) > self.args.score_windows:
            idx = np.linspace(0, len(cuts) - 1, self.args.score_windows).astype(int)
            cuts = [cuts[i] for i in idx]
        return cuts

    def metrics(self, m, *, conditions=None, seeds=(0,)):
        conds = list(conditions or [self.lang])
        dec = np.asarray(self.decode(m), dtype=np.int64)
        rng = np.random.default_rng(self.seed)  # same shuffles every call (CRN)
        vals = {c: [] for c in conds}
        shuf_vals = {c: [] for c in conds}
        for wi, (s, e) in enumerate(self._cuts(len(dec))):
            d = dec[s:e]
            rows = np.stack([d, rng.permutation(d)])
            for sd in seeds:
                pb = paired_bits(
                    self.ev,
                    rows,
                    conds,
                    n_strata=self.args.budget,
                    seed=self.seed + 1000 * sd + 17 * wi,
                )
                for j, c in enumerate(conds):
                    vals[c].append(calibrate_bits(float(pb[0, j]), c, self.offs))
                    shuf_vals[c].append(calibrate_bits(float(pb[1, j]), c, self.offs))
        plain = float(np.mean(vals[self.lang]))
        margin = float(np.mean(shuf_vals[self.lang]) - plain)
        cb = self.choice_bits(m, dec)
        out = {
            "plain_bits": plain,
            "structure_margin": margin,
            "choice_bits": cb,
            "n_plain": len(dec),
            "mdl_total_per_symbol": (plain * len(dec) + self.key_bits + cb)
            / max(self.n_cipher_window, 1),
        }
        if len(conds) > 1:
            by = {c: float(np.mean(vals[c])) for c in conds}
            rank = sorted(by, key=by.get)
            out["plain_bits_by_condition"] = by
            out["language_rank_of_decode"] = rank
            out["top_margin_bits"] = by[rank[1]] - by[rank[0]]
            if len(seeds) > 1:
                tops = []
                for si in range(len(seeds)):
                    tops.append(
                        min(conds, key=lambda c: np.mean(vals[c][si :: len(seeds)]))
                    )
                out["replicate_flip_rate"] = float(
                    np.mean([t != rank[0] for t in tops])
                )
        return out


def build_cells(head, ng, ev, args, offs):
    root = data_root()
    cells = []
    if head == "wordhom":
        from diff_voyn.heads.wordhom import units_suffix

        wd = root / "analysis/wordhom"
        suf = units_suffix(getattr(args, "units", None))
        solves = json.loads((wd / f"vms_solves{suf}.json").read_text())["instances"]
        meta = json.loads((root / "analysis/phase6/vms_report.json").read_text())[
            "instances"
        ]
        for inst_name in INSTANCES:
            fname = inst_name.replace("/", "_") + "_wordtypesall.json"
            inst = json.loads((wd / "presentations" / fname).read_text())
            n_all = meta[f"{inst_name}/eva"]["n_cipher_all"]
            for rec in sorted(
                (r for r in solves if r["instance"] == inst_name),
                key=lambda r: (r["window"], LANGS.index(r["hypothesis"])),
            ):
                best = max(rec["candidates"], key=lambda c: c["inner"])
                cells.append(
                    Cell(head, inst, rec, best["map"], ng, ev, args, offs, n_all)
                )
    else:
        pd = root / "analysis/phase6"
        scores = []
        for sh in sorted(pd.glob("vms_scores_shard*.json")):
            scores += json.loads(sh.read_text())["instances"]
        solves = {
            tuple(r[k] for k in APPLY_KEY): r
            for r in json.loads((pd / "vms_solves.json").read_text())["instances"]
        }
        meta = json.loads((pd / "vms_report.json").read_text())["instances"]
        order = [(i, "eva") for i in INSTANCES] + [(b, "boxer") for b in BOXER]
        for inst_name, pres in order:
            inst = json.loads(
                (
                    pd
                    / "presentations"
                    / (inst_name.replace("/", "_") + f"_{pres}.json")
                ).read_text()
            )
            n_all = meta[f"{inst_name}/{pres}"]["n_cipher_all"]
            recs = [
                r for r in scores if r["instance"] == inst_name and r["head"] == head
            ]
            for rec in sorted(
                recs, key=lambda r: (r["window"], LANGS.index(r["hypothesis"]))
            ):
                # start = the MDL-pick n-gram candidate of the recorded solve
                # (plan §2). NOT the Phase-6 polished key: the ELBO polish
                # moved it ~6500 nats below the n-gram optimum (smoke run),
                # so a loop that accepts on the n-gram objective would leave
                # it in its first round whatever the arm.
                sol = solves[tuple(rec[k] for k in APPLY_KEY)]
                cand = next(
                    c
                    for c in sol["candidates"]
                    if c["source"] == rec["pick_mdl_source"]
                )
                cells.append(
                    Cell(head, inst, rec, cand["map"], ng, ev, args, offs, n_all)
                )
    if args.only:
        cells = [c for c in cells if any(o in c.name for o in args.only)]
    if getattr(args, "langs", None):
        cells = [c for c in cells if c.lang in args.langs]
    return cells


# -- awareness (§6, §7) ----------------------------------------------------


class Reporter:
    def __init__(self, args, out_dir, log, config):
        self.log = log
        self.out_dir = out_dir
        self.events = out_dir / "events.log"
        self.task = None
        self.best_margin = {}  # head -> running max over psamp cells
        if not args.no_clearml:
            try:
                from diff_voyn.infra.clearml_task import init_analysis_task

                self.task = init_analysis_task(
                    f"altloop-vms-{datetime.now(UTC):%Y-%m-%d}",
                    ["phase6-followup", "altloop", args.head],
                    config,
                )
                lg = self.task.get_logger()
                for it in (0, 6):
                    lg.report_scalar(
                        "best_structure_margin", "ref_vms_ceiling", REF_VMS_CEILING, it
                    )
                    lg.report_scalar(
                        "best_structure_margin", "ref_true_min", REF_TRUE_MIN, it
                    )
                log(f"clearml task {self.task.id}")
            except Exception as e:  # noqa: BLE001
                log(f"WARNING clearml unavailable: {e!r}")

    def scalar(self, cell, arm, it, metrics):
        if self.task is None:
            return
        try:
            from diff_voyn.infra.clearml_task import report_cell_round

            report_cell_round(self.task, cell, arm, it, metrics)
        except Exception as e:  # noqa: BLE001
            self.log(f"WARNING clearml scalar failed: {e!r}")

    def summary(self, title, series, value, it=0):
        if self.task is None:
            return
        try:
            self.task.get_logger().report_scalar(title, series, float(value), it)
        except Exception as e:  # noqa: BLE001
            self.log(f"WARNING clearml summary failed: {e!r}")

    def tag(self, tag):
        if self.task is None:
            return
        try:
            from diff_voyn.infra.clearml_task import add_tag

            add_tag(self.task, tag)
        except Exception as e:  # noqa: BLE001
            self.log(f"WARNING clearml tag failed: {e!r}")

    def event(self, tier, cell, rnd, margin, rand, none, plain):
        fm = lambda v: "na" if v is None else f"{v:.3f}"
        line = (
            f"EVENT {tier} {cell} round={rnd} margin={fm(margin)} rand={fm(rand)} "
            f"none={fm(none)} plain={fm(plain)}"
        )
        with self.events.open("a") as f:
            f.write(f"{datetime.now(UTC):%FT%T} {line}\n")
        self.log(line)

    def heartbeat(self, text):
        with self.events.open("a") as f:
            f.write(f"{datetime.now(UTC):%FT%T} CELL {text}\n")

    def upload(self, path: Path):
        if self.task is None:
            return
        try:
            self.task.upload_artifact(path.stem, str(path), wait_on_upload=False)
        except Exception as e:  # noqa: BLE001
            self.log(f"WARNING clearml upload failed: {e!r}")

    def sample(self, name, text):
        if self.task is None:
            return
        try:
            self.task.get_logger().report_text(f"{name}\n{text}")
        except Exception as e:  # noqa: BLE001
            self.log(f"WARNING clearml text failed: {e!r}")


def letters_text(ids) -> str:
    from diff_voyn.vocab import LETTER_IDS, decode

    return decode([LETTER_IDS[int(i)] for i in ids])


def control_best(res, cell_name, arm):
    """Best structure margin the control arm reached on the cell (start and
    every accepted round, all seeds); None if the arm has not reported."""
    vals = []
    for r in res:
        if r["cell"] != cell_name or r["arm"] != arm:
            continue
        vals.append(r["start_metrics"]["structure_margin"])
        for t in r["trace"]:
            if t.get("accepted"):
                vals.append(t["metrics_after_sa"]["structure_margin"])
    return max(vals) if vals else None


def control_language_like(res, cell_name):
    for r in res:
        if r["cell"] == cell_name and r["arm"] in ("rand", "none"):
            fm = r["final_metrics"]
            if (
                fm["plain_bits"] <= ABSTAIN_MAX_PLAIN
                and fm["structure_margin"] >= ABSTAIN_MIN_MARGIN
            ):
                return True
    return False


# -- stages -------------------------------------------------------------------


class RunState:
    """Bookkeeping shared by the rounds of one (cell × arm × seed) run."""

    def __init__(
        self, cell, arm, seed, res, rep, log, tier_counts, promising, prom_path
    ):
        self.c, self.arm, self.seed = cell, arm, seed
        self.res, self.rep, self.log = res, rep, log
        self.tier_counts, self.promising, self.prom_path = (
            tier_counts,
            promising,
            prom_path,
        )
        self.start_metrics = cell.metrics(cell.start)
        self.last = self.start_metrics
        self.key = cell.start.tolist()
        self.sa_key = None

    def short_sa(self, m, rng):
        out, sc = self.c.short_sa(m, rng)
        self.sa_key = np.asarray(out).tolist()
        return out, sc

    def on_round(self, info):
        c, arm, seed = self.c, self.arm, self.seed
        ms, mp = info.get("metrics_after_sa"), info.get("metrics_proposed")
        if info.get("accepted"):
            self.key, self.last = self.sa_key, ms
        cur_m = self.last  # the accepted key (unchanged if rejected)
        sc = {
            "ngram_obj": info["obj_out"],
            "ngram_obj_proposed": info.get("obj_proposed"),
            "ngram_obj_after_sa": info.get("obj_after_sa"),
            "accepted": int(bool(info.get("accepted"))),
            "n_changed": info.get("n_changed_total"),
            "seconds": info["seconds"],
            "plain_bits": cur_m["plain_bits"],
            "structure_margin": cur_m["structure_margin"],
            "mdl_total_per_symbol": cur_m["mdl_total_per_symbol"],
        }
        if ms:
            sc["plain_bits_after_sa"] = ms["plain_bits"]
            sc["structure_margin_after_sa"] = ms["structure_margin"]
            sc["mdl_total_per_symbol_after_sa"] = ms["mdl_total_per_symbol"]
        if mp:
            sc["structure_margin_proposed"] = mp["structure_margin"]
        self.rep.scalar(c.name, f"{arm}-s{seed}", info["round"] + 1, sc)
        self.log(
            f"  {c.name} {arm} s{seed} r{info['round']}: obj {info['obj_in']:.1f}→{info['obj_out']:.1f} "
            f"acc={info.get('accepted')} chg={info.get('n_changed_total')} "
            f"margin={cur_m['structure_margin']:.3f} plain={cur_m['plain_bits']:.3f} {info['seconds']:.0f}s"
        )
        if not is_treatment(arm):
            return
        rb = control_best(self.res, c.name, "rand")
        nb = control_best(self.res, c.name, "none")
        cb = None if rb is None or nb is None else max(rb, nb)
        tier = classify_tier(
            cur_m["structure_margin"], cur_m["plain_bits"], cb, flip_rate=None
        )
        self.rep.event(
            tier,
            c.name,
            info["round"],
            cur_m["structure_margin"],
            rb,
            nb,
            cur_m["plain_bits"],
        )
        if tier in self.tier_counts:
            self.tier_counts[tier] += 1
            self.rep.summary("promising_count", tier, self.tier_counts[tier])
        if tier in ("NOTABLE", "PROMISING", "LANGUAGE-LIKE"):
            self.rep.tag(tier)
            text = letters_text(c.decode(np.asarray(self.key, dtype=np.int64)))[:400]
            self.promising.append(
                {
                    "utc": datetime.now(UTC).isoformat(),
                    "tier": tier,
                    "cell": c.name,
                    "arm": arm,
                    "seed": seed,
                    "round": info["round"],
                    "metrics": cur_m,
                    "controls": {"rand": rb, "none": nb},
                    "decode_sample": text,
                    "key": self.key,
                }
            )
            write_json_atomic(self.prom_path, self.promising)
            self.rep.upload(self.prom_path)
            self.rep.sample(f"{tier} {c.name} r{info['round']}", text)


def start_key_from(args, c, arm, seed):
    """``--start-from TAG``: the final key of the same (cell, arm, seed) run
    in runs_<head><TAG>.json (stage 2 of the wildcard → anneal pipeline)."""
    path = (
        data_root() / "analysis/altloop_vms" / f"runs_{args.head}{args.start_from}.json"
    )
    prev = json.loads(path.read_text())["runs"]
    for r in prev:
        if r["cell"] == c.name and r["arm"] == arm and r["seed"] == seed:
            return np.asarray(r["final_key"], dtype=np.int64)
    return None


def run_one(args, c, arm, seed, res, rep, log, tier_counts, promising, prom_path):
    mech, k = ARMS[args.head][arm]
    t0 = time.time()
    if args.start_from:
        sk = start_key_from(args, c, arm, seed)
        if sk is None:
            log(
                f"{c.name} {arm} s{seed}: no run in runs_{args.head}{args.start_from}.json, skipped"
            )
            return None
        c.start = sk
    schedule = None
    if args.wild and c.head == "wordhom":
        c.set_wild(c.rare_type)  # the schedule mutates it; reset per run
        if args.wild_anneal:
            a0, a1 = args.wild_anneal
            schedule = c.wild_schedule(a0, a1, c.seed + 101 * seed)
    st = RunState(c, arm, seed, res, rep, log, tier_counts, promising, prom_path)
    rep.scalar(
        c.name,
        f"{arm}-s{seed}",
        0,
        {
            "plain_bits": st.start_metrics["plain_bits"],
            "structure_margin": st.start_metrics["structure_margin"],
            "mdl_total_per_symbol": st.start_metrics["mdl_total_per_symbol"],
            "ngram_obj": c.objective(c.start),
        },
    )
    final_key, info = alternate(
        c.start,
        mechanism=mech,
        objective=c.objective,
        short_sa=st.short_sa,
        scores_fn=c.scores,
        random_unit=c.random_unit,
        metrics=c.metrics,
        occ=c.occ,
        k=k,
        rounds=args.rounds,
        patience=args.patience,
        seed=c.seed + 101 * seed,
        on_round=st.on_round,
        schedule=schedule,
    )
    info["start_metrics"] = st.start_metrics
    # end-of-run reading: all three conditions, 4 scoring seeds (flip-rate)
    fm = c.metrics(final_key, conditions=list(LANGS), seeds=(0, 1, 2, 3))
    fm["language_like"] = bool(
        fm["plain_bits"] <= ABSTAIN_MAX_PLAIN
        and fm["structure_margin"] >= ABSTAIN_MIN_MARGIN
    )
    rank = fm["language_rank_of_decode"]
    fm["top_margin_uncertainty_bits"] = args.table.margin_uncertainty_bits(
        rank[0], rank[1]
    )
    info["final_metrics"] = fm
    rec = {
        "cell": c.name,
        "head": args.head,
        "tag": args.tag,
        "units": getattr(c, "units", None),
        "start_from": args.start_from,
        "wild": bool(args.wild),
        "hapax_max": int(args.hapax_max),
        "wild_anneal": list(args.wild_anneal) if args.wild_anneal else None,
        "n_wild_start": (
            int(c.rare_type.sum()) if args.wild and c.head == "wordhom" else 0
        ),
        "instance": c.inst_name,
        "window": c.window,
        "window_span": list(c.span),
        "language": c.lang,
        "arm": arm,
        "mechanism": mech,
        "seed": seed,
        "seconds": time.time() - t0,
        "final_key": np.asarray(final_key).tolist(),
        "n_cipher_window": c.n_cipher_window,
        "n_cipher_all": c.n_cipher_all,
        **{k2: v for k2, v in info.items() if k2 != "k"},
        "k": k,
    }
    sm = st.start_metrics
    log(
        f"{c.name} {arm} s{seed}: margin {sm['structure_margin']:.3f}→{fm['structure_margin']:.3f} "
        f"plain {sm['plain_bits']:.3f}→{fm['plain_bits']:.3f} "
        f"obj {info['start_obj']:.1f}→{info['final_obj']:.1f} acc {info['n_accepted']}/{info['n_rounds']} "
        f"top={rank[0]} ({fm['top_margin_bits']:.3f}±{fm['top_margin_uncertainty_bits']:.3f}) "
        f"{rec['seconds']:.0f}s"
    )
    return rec


def stage_run(args, cells, ng, ev, log, out_dir, config):
    path = out_dir / f"runs_{args.head}{args.tag}.json"
    res = json.loads(path.read_text())["runs"] if path.exists() else []
    done = {tuple(r[k] for k in RUN_KEY) for r in res}
    rep = Reporter(args, out_dir, log, config)
    prom_path = out_dir / "promising.json"
    promising = json.loads(prom_path.read_text()) if prom_path.exists() else []
    tier_counts = {t: 0 for t in ("NOISE", "NOTABLE", "PROMISING", "LANGUAGE-LIKE")}
    best = max(
        [r["final_metrics"]["structure_margin"] for r in res if is_treatment(r["arm"])]
        + [-1.0]
    )
    n_done_cells = 0
    for c in cells:
        for seed in range(args.seeds):
            for arm in args.arms:
                if (c.name, arm, seed) in done:
                    continue
                rec = run_one(
                    args, c, arm, seed, res, rep, log, tier_counts, promising, prom_path
                )
                if rec is None:
                    continue
                res.append(rec)
                done.add((c.name, arm, seed))
                write_json_atomic(
                    path,
                    {
                        "created_utc": datetime.now(UTC).isoformat(),
                        "config": config,
                        "runs": res,
                    },
                )
                if is_treatment(arm):
                    fm = rec["final_metrics"]
                    best = max(best, fm["structure_margin"])
                    rep.summary("best_structure_margin", args.head, best, len(res))
                    rb = control_best(res, c.name, "rand")
                    nb = control_best(res, c.name, "none")
                    if rb is not None and nb is not None:
                        rep.summary(
                            "best_delta_vs_controls",
                            args.head,
                            fm["structure_margin"] - max(rb, nb),
                            len(res),
                        )
        n_done_cells += 1
        rep.summary("cells_done", args.head, n_done_cells)
        rep.upload(path)
        fin = {
            a: [
                r["final_metrics"]["structure_margin"]
                for r in res
                if r["cell"] == c.name and r["arm"] == a
            ]
            for a in args.arms
        }
        rep.heartbeat(
            f"{c.name} done: "
            + " ".join(f"{a}={','.join(f'{v:.3f}' for v in fin[a])}" for a in args.arms)
        )
    log(f"== {args.head} complete: {len(res)} runs, {n_done_cells} cells this launch")


def stage_report(out_dir):
    runs = []
    for f in sorted(out_dir.glob("runs_*.json")):
        runs += json.loads(f.read_text())["runs"]
    lines = [
        "# Alternating loop on the manuscript — controls in every cell",
        "",
        f"Generated {datetime.now(UTC):%F %T} UTC from {len(runs)} runs. Plan: `docs/altloop_vms_plan.md`.",
        "",
        "**What this can and cannot say (plan §1, verbatim).** Can say: whether the",
        "diffusion-guided loop finds, in any cipher × language × dialect cell, a key that",
        "(a) the n-gram objective prefers, (b) the frozen judge scores as more language-like",
        "than Phase 6's best (structure margin above the manuscript's recorded 1.25",
        "ceiling), and (c) the random and SA-alone arms on the *same cell* do **not**",
        'reach. Only (a)+(b)+(c) together is "promising". Cannot say: that a null result',
        'means "nothing to find" — the proposer is blind on mostly-wrong keys (PoL-1 exact',
        "precision 0) and the synthetic battery never tested the manuscript's 3.0–4.6",
        'tokens-per-type regime. A null here is "the method did not find it", not evidence',
        "of absence — and the Phase-6 abstention stands regardless. This does not change",
        "the Phase-6 record. Not run: `naibbe` and `arithmetic` (scope, not evidence).",
        "",
        f"Tiers (§5): NOTABLE margin ≥ {NOTABLE_MIN} and ≥ {NOTABLE_ABOVE_CONTROLS} above both controls' best;",
        f"PROMISING margin ≥ {REF_TRUE_MIN} and both controls < {NOTABLE_MIN}; LANGUAGE-LIKE = ABSTAIN_RULE",
        "(plain ≤ 3.0, margin ≥ 1.5) + flip-rate 0 + controls fail the rule. Anything a control",
        "also reaches is NOISE whatever the number.",
        "",
    ]
    by_head = {}
    for r in runs:
        by_head.setdefault(r["head"] + r.get("tag", ""), {}).setdefault(
            r["cell"], []
        ).append(r)
    for head, cells in by_head.items():
        arms_here = [
            a
            for a in ("none", "rand", "post", "psamp")
            if any(r["arm"] == a for rs in cells.values() for r in rs)
        ]
        treat = [a for a in arms_here if is_treatment(a)]
        r0 = next(iter(cells.values()))[0]
        wild_note = (
            f" — wildcard objective (types with ≤ {r0.get('hapax_max')} occurrences, "
            f"{r0.get('n_wild_start')} wild at start), anneal {r0.get('wild_anneal')}, "
            f"start-from `{r0.get('start_from')}`"
            if r0.get("wild")
            else ""
        )
        lines += [
            f"## {head} — {len(cells)} cells{wild_note}",
            "",
            "| cell | start margin / plain | none (final margin per seed) | rand | "
            + " | ".join(treat)
            + " | treatment best-round margin | Δ treatment − best control | n-gram obj Δ (treatment) | accepted (treatment) | top lang (treatment) | tier |",
            "|---|---|---|---|" + "---|" * len(treat) + "---|---|---|---|---|---|",
        ]
        n_tier = {}
        for cell, rs in cells.items():
            arms = {
                a: sorted([r for r in rs if r["arm"] == a], key=lambda r: r["seed"])
                for a in ("none", "rand", "post", "psamp")
            }
            sm = rs[0]["start_metrics"]

            def fin(a, arms=arms):
                return (
                    ", ".join(
                        f"{r['final_metrics']['structure_margin']:.3f}" for r in arms[a]
                    )
                    or "—"
                )

            ps = [r for a in treat for r in arms[a]]
            best_ps = max(
                [r["final_metrics"]["structure_margin"] for r in ps]
                + [
                    t["metrics_after_sa"]["structure_margin"]
                    for r in ps
                    for t in r["trace"]
                    if t.get("accepted")
                ]
                + [-9]
            )
            rb, nb = control_best(rs, cell, "rand"), control_best(rs, cell, "none")
            cb = None if rb is None or nb is None else max(rb, nb)
            plain_best = (
                max(ps, key=lambda r: r["final_metrics"]["structure_margin"])[
                    "final_metrics"
                ]["plain_bits"]
                if ps
                else float("nan")
            )
            flips = [r["final_metrics"].get("replicate_flip_rate", 0.0) for r in ps]
            tops = {r["final_metrics"]["language_rank_of_decode"][0] for r in ps}
            flip = (
                None
                if not ps
                else (0.0 if len(tops) == 1 and all(f == 0 for f in flips) else 1.0)
            )
            tier = (
                classify_tier(
                    best_ps,
                    plain_best,
                    cb,
                    flip_rate=flip,
                    controls_language_like=control_language_like(rs, cell),
                )
                if ps
                else "PENDING"
            )
            n_tier[tier] = n_tier.get(tier, 0) + 1
            dobj = ", ".join(f"{r['final_obj']-r['start_obj']:+.1f}" for r in ps) or "—"
            acc = ", ".join(f"{r['n_accepted']}/{r['n_rounds']}" for r in ps) or "—"
            top = (
                ", ".join(
                    f"{r['final_metrics']['language_rank_of_decode'][0]} {r['final_metrics']['top_margin_bits']:.3f}±{r['final_metrics']['top_margin_uncertainty_bits']:.3f}"
                    for r in ps
                )
                or "—"
            )
            delta = "—" if cb is None or not ps else f"{best_ps - cb:+.3f}"
            lines.append(
                f"| {cell} | {sm['structure_margin']:.3f} / {sm['plain_bits']:.3f} | {fin('none')} | {fin('rand')} | "
                + " | ".join(fin(a) for a in treat)
                + " | "
                f"{best_ps:.3f} | {delta} | {dobj} | {acc} | {top} | **{tier}** |"
            )
        lines += [
            "",
            "Tier counts: " + ", ".join(f"{k} {v}" for k, v in sorted(n_tier.items())),
            "",
        ]
        m_all = [
            r["final_metrics"]["structure_margin"]
            for rs in cells.values()
            for r in rs
            if is_treatment(r["arm"])
        ]
        if m_all:
            lines.append(
                f"treatment final margins: min {min(m_all):.3f}, max {max(m_all):.3f} (manuscript ceiling {REF_VMS_CEILING}, lowest true decipherment {REF_TRUE_MIN}); "
                f"language_like: {sum(r['final_metrics']['language_like'] for rs in cells.values() for r in rs)} of {sum(len(rs) for rs in cells.values())} runs."
            )
            lines.append("")
    prom = out_dir / "promising.json"
    if prom.exists():
        p = json.loads(prom.read_text())
        lines += [f"## promising.json — {len(p)} entries", ""]
        for e in p:
            lines.append(
                f"- {e['tier']} {e['cell']} s{e['seed']} r{e['round']}: margin {e['metrics']['structure_margin']:.3f} plain {e['metrics']['plain_bits']:.3f} controls {e['controls']}"
            )
    else:
        lines += [
            "No NOTABLE-or-better round was recorded (promising.json absent).",
            "",
        ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    root = data_root()
    p.add_argument("--stage", choices=["run", "report"], required=True)
    p.add_argument("--head", choices=HEADS)
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument(
        "--arms",
        nargs="+",
        default=list(ARM_ORDER),
        choices=("none", "rand", "post", "psamp"),
    )
    p.add_argument("--tag", default="", help="suffix of runs_<head><tag>.json")
    p.add_argument(
        "--wild",
        action="store_true",
        help="hapax-as-wildcard n-gram objective (wordhom)",
    )
    p.add_argument("--hapax-max", type=int, default=1)
    p.add_argument(
        "--wild-anneal",
        type=lambda v: tuple(int(x) for x in v.split(",")),
        default=None,
        help="START,END rounds over which the wildcard set is re-admitted (docs/alt_loop_plan.md §8.6)",
    )
    p.add_argument(
        "--start-from",
        default=None,
        help="tag of a runs file whose final_key seeds each (cell, arm, seed)",
    )
    p.add_argument(
        "--units",
        default=None,
        help="wordhom unit-set spec (d5 default; d5b20 = doubles + top-20 bigrams) — "
        "starts come from analysis/wordhom/vms_solves<_units>.json",
    )
    p.add_argument(
        "--langs",
        nargs="*",
        default=None,
        help="restrict wordhom cells to these hypothesis languages",
    )
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--n-draws", type=int, default=16)
    p.add_argument("--mask-rate", type=float, default=0.3)
    p.add_argument("--wh-sa-steps", type=int, default=200_000)
    p.add_argument("--sym-sa-steps", type=int, default=50_000)
    p.add_argument("--ils-kicks", type=int, default=3)
    p.add_argument("--t-start", type=float, default=2.0)
    p.add_argument("--t-end", type=float, default=0.3)
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--score-windows", type=int, default=8)
    p.add_argument("--no-clearml", action="store_true")
    args = p.parse_args()
    out_dir = root / "analysis/altloop_vms"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        stage_report(out_dir)
        return
    if not args.head:
        p.error("--head is required for --stage run")
    log = log_to(out_dir / f"run_{args.head}.log")
    torch.set_float32_matmul_precision("high")
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator

    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    args.table = CalibrationTable.load(ev.meta["calibration_version"], root)
    offs = ev.calibration_offsets_bits
    ng = build_ngram_evaluator()
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    cells = build_cells(args.head, ng, ev, args, offs)
    config = {
        "plan": "docs/altloop_vms_plan.md",
        "head": args.head,
        "cells": [c.name for c in cells],
        "arms": {a: {"mechanism": m, "k": k} for a, (m, k) in ARMS[args.head].items()},
        "tiers": {
            "NOTABLE": {
                "min_margin": NOTABLE_MIN,
                "above_controls": NOTABLE_ABOVE_CONTROLS,
            },
            "PROMISING": {"min_margin": REF_TRUE_MIN, "controls_below": NOTABLE_MIN},
            "LANGUAGE-LIKE": {
                "max_plain": ABSTAIN_MAX_PLAIN,
                "min_margin": ABSTAIN_MIN_MARGIN,
                "flip_rate": 0,
            },
            "ref_vms_ceiling": REF_VMS_CEILING,
            "ref_true_min": REF_TRUE_MIN,
        },
        "evaluator": str(args.ckpt),
        "evaluator_sha256": hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
        "calibration_version": ev.meta["calibration_version"],
        "calibration_policy": "report-only (offsets measured, not subtracted)",
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
            if k != "table"
        },
    }
    log(
        f"== run {args.head} {time.strftime('%FT%T')} {len(cells)} cells: {[c.name for c in cells]}"
    )
    log(
        f"   evaluator sha256 {config['evaluator_sha256'][:12]} calibration {config['calibration_version']}"
    )
    stage_run(args, cells, ng, ev, log, out_dir, config)


if __name__ == "__main__":
    main()
