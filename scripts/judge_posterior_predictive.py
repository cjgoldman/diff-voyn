"""Posterior-predictive judging of the rare key entries (R2 of
``docs/bayesian_perspective_review.md`` §3; protocol and PRE-REGISTERED
readings in ``docs/judge_posterior_predictive.md`` — read that first).

The solver of record commits every hapax type of a word-homophonic key to
one unit before the frozen judge sees the decode, and > 95 % of the anneal
finals' residual errors sit on those rare types (``docs/alt_loop_plan.md``
§8.6, error-spread note). This script judges a key *with the hapax entries
integrated out* instead: the hapax positions are withheld from the denoiser
(``position_posterior(force_mask=…)``), one unit per rare type is sampled
from the type's summed posterior (same length class, temperature ``--temp``),
the decode is rebuilt with that fill and scored by the exact Phase-6 judge
(``judge_at_ser.score_map``: paired decode / letter-shuffled windows × 4
replicate seeds × 3 language conditions, budget 64), and the judge's
statistics are averaged over ``--fills`` independent fills. The shuffled
reference is re-drawn per fill from the fill's own letters.

Keys judged (from the recorded ``judge_at_ser*.json`` rows, so every
posterior-predictive number sits next to the committed key's recorded
number): the A-like anneal finals (``wild:anneal*``), every battery anneal
final (``final:_bat_anneal*`` — B-like positives, negatives, cross-language,
dirty, mixed, nodouble, revdouble) and, with ``--include-truth``, the truth
keys (the fill's own cost on a perfect key). The d5b20 (``big*``) files are
a different hypothesis space and are skipped, like ``evidence_odds.py``.

Nothing here is a call rule; ``ABSTAIN_RULE`` stays the judge of record.

Artifacts: DATA_ROOT/analysis/altloop/judge_posterior_predictive<tag>.{json,md}
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.ladder import write_json_atomic
from diff_voyn.heads.posterior import position_posterior, unit_scores
from diff_voyn.heads.wordhom import (
    UnitTargets,
    adjacency,
    expand_units,
    hypothesis_targets,
    project_key,
    rule_violations,
    unit_ser,
)
from diff_voyn.metrology import CALIBRATION_VERSION, CalibrationTable
from diff_voyn.vms.apply import ABSTAIN_RULE

DEFAULT_KEY_RE = r"^(wild:_?anneal|final:_bat_anneal)"


# -- key / instance resolution --------------------------------------------------


def source_files(root: Path, sources: list[str] | None) -> list[Path]:
    if sources:
        return [
            Path(s) if Path(s).is_absolute() else root / "analysis/altloop" / s
            for s in sources
        ]
    out = [root / "analysis/altloop/judge_at_ser.json"]
    for fn in sorted(
        glob.glob(str(root / "analysis/altloop/judge_at_ser_battery_*.json"))
    ):
        if Path(fn).name.startswith("judge_at_ser_battery_big"):
            continue  # d5b20 decoder — a different hypothesis space
        out.append(Path(fn))
    return [p for p in out if p.exists()]


def load_instance(root: Path, name: str) -> dict:
    wd = root / "analysis/wordhom"
    bat = wd / "battery/wordtypesall"
    if (bat / "manifest.json").exists():
        man = {m["name"]: m for m in json.loads((bat / "manifest.json").read_text())}
        if name in man:
            return json.loads((bat / man[name]["file"]).read_text())
    return json.loads(
        (
            wd
            / "controls/wordtypesall"
            / (name.replace("/", "_") + "_wordtypesall.json")
        ).read_text()
    )


class RunFiles:
    """Final maps of the recorded loop runs, keyed as ``judge_at_ser`` keys them:
    ``wild:<name>/s<seed>`` → ``runs_<name>.json`` (the A-like study; the
    leading underscore was dropped when those rows were written) and
    ``final:<tag>/s<seed>`` → ``runs<tag>.json`` (battery tags)."""

    def __init__(self, root: Path):
        self.dir = root / "analysis/altloop"
        self.cache: dict[str, list] = {}

    def _runs(self, fn: str) -> list:
        if fn not in self.cache:
            p = self.dir / fn
            self.cache[fn] = json.loads(p.read_text()) if p.exists() else []
        return self.cache[fn]

    def final_map(self, key: str, name: str, lang: str) -> np.ndarray | None:
        m = re.match(r"^(wild|final):(.+)/s(\d+)$", key)
        if not m:
            return None
        kind, part, seed = m.group(1), m.group(2), int(m.group(3))
        fn = f"runs_{part.lstrip('_')}.json" if kind == "wild" else f"runs{part}.json"
        for r in self._runs(fn):
            if (
                r["cell"] == f"wh/{name}/{lang}"
                and r["start"] == "stuck"
                and int(r["seed"]) == seed
                and "final_map" in r
            ):
                return np.asarray(r["final_map"], dtype=np.int64)
        return None


def truth_map(inst: dict, lang: str, targets: UnitTargets) -> np.ndarray | None:
    tr = inst.get("truth", {})
    if tr.get("kind") != "wordhom" or tr.get("language") != lang:
        return None
    return project_key(tr["sym_to_unit"], UnitTargets.from_list(tr["bigrams"]), targets)


# -- the fill ------------------------------------------------------------------


def letter_positions_of_types(symbols, m, targets, type_mask) -> np.ndarray:
    """(L,) bool over the expanded decode: the letter positions emitted by the
    occurrences of the types in ``type_mask`` (both letters of a bigram unit)."""
    units = m[symbols]
    isbig = targets.second[units] >= 0
    starts = np.concatenate([[0], np.cumsum(1 + isbig)[:-1]])
    L = len(units) + int(isbig.sum())
    mask = np.zeros(L, dtype=bool)
    sel = type_mask[symbols]
    mask[starts[sel]] = True
    mask[starts[sel & isbig] + 1] = True
    return mask


def sample_fill(ev, inst, m, targets, lang, rare, *, mask_rate, temp, seed, rng):
    """One posterior-predictive fill of the rare types: withhold their
    positions from the denoiser, sample one unit per type (same length
    class) from the type's summed log-posterior at temperature ``temp``."""
    symbols = np.asarray(inst["symbols"], dtype=np.int64)
    dec = expand_units(m[symbols], targets)
    force = letter_positions_of_types(symbols, m, targets, rare)
    P = position_posterior(
        ev, dec, lang, n_draws=1, mask_rate=mask_rate, seed=seed, force_mask=force
    )
    S = unit_scores(P, symbols, m, targets)  # (n_types, n_units), -inf off-class
    out = m.copy()
    ent = []
    for t in np.flatnonzero(rare):
        row = S[t]
        ok = np.isfinite(row)
        z = row[ok] / temp
        p = np.exp(z - z.max())
        p /= p.sum()
        out[t] = np.flatnonzero(ok)[rng.choice(ok.sum(), p=p)]
        ent.append(float(-(p * np.log(np.maximum(p, 1e-300))).sum()))
    return out, float(np.mean(ent)) if ent else 0.0, int(force.sum())


# -- main ----------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = data_root()
    p.add_argument("--stage", choices=["run", "report"], default="run")
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="judge_at_ser*.json files to take keys from (default: all non-big)",
    )
    p.add_argument(
        "--key-re", default=DEFAULT_KEY_RE, help="regex over the recorded key names"
    )
    p.add_argument(
        "--include-truth", action="store_true", help="also fill the truth keys"
    )
    p.add_argument(
        "--only", nargs="*", default=None, help="substrings of the cell name"
    )
    p.add_argument(
        "--fills", type=int, default=4, help="K posterior-predictive fills per key"
    )
    p.add_argument(
        "--committed-rescore",
        type=int,
        default=1,
        help="re-score the committed key with this many fresh shuffle draws "
        "(shuffle-draw noise reference, paired with the fills)",
    )
    p.add_argument("--hapax-max", type=int, default=1, help="rare = occurrence ≤ this")
    p.add_argument("--mask-rate", type=float, default=0.3)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--score-windows", type=int, default=16)
    p.add_argument("--primary", default=CALIBRATION_VERSION)
    p.add_argument(
        "--units", default=None, help="unit-set spec (must match the sources)"
    )
    p.add_argument("--tag", default="")
    p.add_argument("--out-dir", type=Path, default=root / "analysis/altloop")
    p.add_argument(
        "--dry-run", action="store_true", help="list the keys and rare counts; no GPU"
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"judge_posterior_predictive{args.tag}.json"
    md = args.out_dir / f"judge_posterior_predictive{args.tag}.md"
    if args.stage == "report":
        report(path, md)
        return

    # recorded rows → the key list
    key_re = re.compile(args.key_re)
    rows = []
    for fp in source_files(root, args.sources):
        for r in json.loads(fp.read_text()):
            if r.get("units") != args.units:
                continue
            if not (
                key_re.search(r["key"]) or (args.include_truth and r["key"] == "truth")
            ):
                continue
            if args.only and not any(o in r["cell"] for o in args.only):
                continue
            rows.append((fp.name, r))
    seen: set = set()
    rows = [
        (src, r)
        for src, r in rows
        if (
            r["key"] != "truth"
            or (r["cell"], r.get("hypothesis"), r["key"]) not in seen
        )
        and not seen.add((r["cell"], r.get("hypothesis"), r["key"]))
    ]
    print(
        f"{len(rows)} recorded keys from {len(source_files(root, args.sources))} files",
        flush=True,
    )

    from wordhom_study import _inst_meta

    from diff_voyn.vms.apply import build_ngram_evaluator

    ng = build_ngram_evaluator()
    runs = RunFiles(root)
    res = json.loads(path.read_text()) if path.exists() else []
    done = {(r["source"], r["cell"], r["hypothesis"], r["key"]) for r in res}
    ev = table = None
    if not args.dry_run:
        from judge_at_ser import score_map

        from diff_voyn.heads.diffusion_eval import DiffusionEvaluator

        torch.set_float32_matmul_precision("high")
        ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
        table = CalibrationTable.load(args.primary, root)

    insts: dict[str, dict] = {}
    for src, r in rows:
        name, lang, key = r["cell"], r.get("hypothesis", r["truth_language"]), r["key"]
        if (src, name, lang, key) in done:
            continue
        inst = insts.setdefault(name, load_instance(root, name))
        targets = hypothesis_targets(ng, lang, units=args.units, inst=inst)
        tmap = truth_map(inst, lang, targets)
        m = tmap if key == "truth" else runs.final_map(key, name, lang)
        if m is None:
            print(f"  ! no map for {name}/{lang} {key} — skipped", flush=True)
            continue
        symbols = np.asarray(inst["symbols"], dtype=np.int64)
        occ = np.bincount(symbols, minlength=int(inst["n_symbols"]))
        rare = occ <= args.hapax_max
        adj = adjacency(symbols, np.asarray(inst["token_pos"], dtype=np.int64))
        plain = (
            np.asarray(inst["truth"]["plain_ids"], dtype=np.int64)
            if tmap is not None and "plain_ids" in inst["truth"]
            else None
        )
        n_rare_pos = int(letter_positions_of_types(symbols, m, targets, rare).sum())
        rec = {
            "source": src,
            "cell": name,
            "truth_language": inst.get("truth", {}).get("language"),
            "hypothesis": lang,
            "control": inst.get("control", r.get("control", "positive")),
            "key": key,
            "n_types": len(m),
            "n_rare_types": int(rare.sum()),
            "n_rare_letters": n_rare_pos,
            "n_plain": len(expand_units(m[symbols], targets)),
            "hapax_max": args.hapax_max,
            "mask_rate": args.mask_rate,
            "temp": args.temp,
            "committed": {
                k: r.get(k)
                for k in (
                    "ser",
                    "map_err_occ",
                    "plain_bits",
                    "structure_margin",
                    "language_rank_of_decode",
                    "top_margin_bits",
                    "top_margin_uncertainty_bits",
                    "language_like",
                    "called",
                    "violations",
                )
            },
            "committed_rescore": [],
            "fills": [],
        }
        if args.dry_run:
            print(
                f"{name}/{lang} {key:26s} types {len(m):5d} rare {rare.sum():5d} "
                f"({100*n_rare_pos/rec['n_plain']:.1f} % of letters) committed margin "
                f"{r['structure_margin']:.2f} called={r['called']}",
                flush=True,
            )
            continue
        meta = _inst_meta(inst)
        t0 = time.time()

        def judge(
            mm,
            shuffle_seed,
            *,
            lang=lang,
            inst=inst,
            meta=meta,
            targets=targets,
            symbols=symbols,
            tmap=tmap,
            plain=plain,
            occ=occ,
            rare=rare,
            adj=adj,
        ):
            dec = expand_units(mm[symbols], targets)
            wrong = (mm != tmap) if tmap is not None else None
            out = {
                "shuffle_seed": int(shuffle_seed),
                "ser": float(unit_ser(dec, plain)) if plain is not None else None,
                "map_err_occ": (
                    float((occ * wrong).sum() / occ.sum())
                    if wrong is not None
                    else None
                ),
                "rare_err_types": (
                    float(wrong[rare].mean())
                    if wrong is not None and rare.any()
                    else None
                ),
                "violations": int(rule_violations(mm[symbols], symbols, adj)),
            }
            out.update(
                score_map(
                    ev,
                    table,
                    inst,
                    meta,
                    mm,
                    targets,
                    lang,
                    budget=args.budget,
                    seeds=tuple(range(args.seeds)),
                    score_windows=args.score_windows,
                    seed=0,
                    shuffle_seed=shuffle_seed,
                )
            )
            out["called"] = bool(
                out["language_like"] and out["top_language_of_decode"] == lang
            )
            return out

        for k in range(args.committed_rescore):
            rec["committed_rescore"].append(judge(m, 1 + k))
        for k in range(args.fills):
            rng = np.random.default_rng(10_000 + k)
            mm, ent, _ = sample_fill(
                ev,
                inst,
                m,
                targets,
                lang,
                rare,
                mask_rate=args.mask_rate,
                temp=args.temp,
                seed=10_000 + k,
                rng=rng,
            )
            f = judge(mm, 100 + k)
            f["fill"], f["mean_entropy_nats"] = k, ent
            rec["fills"].append(f)
        rec["summary"] = summarize(rec)
        rec["seconds"] = round(time.time() - t0, 1)
        res.append(rec)
        write_json_atomic(path, res)
        s = rec["summary"]
        fs = lambda x: "n/a" if x is None else f"{x:.3f}"
        print(
            f"{name}/{lang} {key:26s} committed margin {rec['committed']['structure_margin']:.2f} "
            f"(rescore {s['rescore_margin']:.2f}) → pp {s['pp_margin']:.2f} ± {s['pp_margin_sd']:.2f} "
            f"Δ {s['delta_margin']:+.2f}; plain {rec['committed']['plain_bits']:.3f} → {s['pp_plain']:.3f}; "
            f"ser {fs(rec['committed']['ser'])} → {fs(s['pp_ser'])}; called {s['pp_called_frac']:.2f} "
            f"{rec['seconds']:.0f}s",
            flush=True,
        )
    if not args.dry_run:
        report(path, md)


def summarize(rec: dict) -> dict:
    fills = rec["fills"]
    mg = np.array([f["structure_margin"] for f in fills])
    pl = np.array([f["plain_bits"] for f in fills])
    rs = rec["committed_rescore"]
    ser = [f["ser"] for f in fills if f["ser"] is not None]
    return {
        "pp_margin": float(mg.mean()),
        "pp_margin_sd": float(mg.std(ddof=1)) if len(mg) > 1 else 0.0,
        "pp_plain": float(pl.mean()),
        "pp_plain_sd": float(pl.std(ddof=1)) if len(pl) > 1 else 0.0,
        "pp_ser": float(np.mean(ser)) if ser else None,
        "pp_called_frac": float(np.mean([f["called"] for f in fills])),
        "pp_top_language": max(
            {f["top_language_of_decode"] for f in fills},
            key=[f["top_language_of_decode"] for f in fills].count,
        ),
        "rescore_margin": (
            float(np.mean([r["structure_margin"] for r in rs])) if rs else float("nan")
        ),
        "rescore_plain": (
            float(np.mean([r["plain_bits"] for r in rs])) if rs else float("nan")
        ),
        "delta_margin": float(mg.mean() - rec["committed"]["structure_margin"]),
        "delta_plain": float(pl.mean() - rec["committed"]["plain_bits"]),
        "pp_language_like": bool(
            pl.mean() <= ABSTAIN_RULE["max_plain_bits"]
            and mg.mean() >= ABSTAIN_RULE["min_structure_margin"]
        ),
    }


def report(path, md):
    res = json.loads(path.read_text())
    lines = [
        "# Posterior-predictive judging of the rare key entries (R2)",
        "",
        (
            "Protocol and pre-registered readings: `docs/judge_posterior_predictive.md`. "
            f"Abstain rule: plain ≤ {ABSTAIN_RULE['max_plain_bits']} AND margin ≥ "
            f"{ABSTAIN_RULE['min_structure_margin']}. `committed` = the recorded judge row of the key; "
            "`rescore` = the committed key with a fresh shuffle draw; `pp` = mean over K fills."
        ),
        "",
        "| cell | key | control | rare types (% letters) | SER committed → pp | margin committed / rescore → pp ± sd (Δ) | plain committed → pp | called committed → pp frac | pp like |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    fs = lambda x: "n/a" if x is None else f"{x:.3f}"
    for r in sorted(
        res, key=lambda r: (r["control"], r["cell"], r["hypothesis"], r["key"])
    ):
        s, c = r["summary"], r["committed"]
        lines.append(
            f"| {r['cell']}/{r['hypothesis']} | {r['key']} | {r['control']} | "
            f"{r['n_rare_types']} ({100*r['n_rare_letters']/r['n_plain']:.1f} %) | "
            f"{fs(c['ser'])} → {fs(s['pp_ser'])} | "
            f"{c['structure_margin']:.2f} / {s['rescore_margin']:.2f} → {s['pp_margin']:.2f} ± "
            f"{s['pp_margin_sd']:.2f} ({s['delta_margin']:+.2f}) | "
            f"{c['plain_bits']:.3f} → {s['pp_plain']:.3f} | "
            f"{'yes' if c['called'] else 'no'} → {s['pp_called_frac']:.2f} | "
            f"{'yes' if s['pp_language_like'] else 'no'} |"
        )
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
