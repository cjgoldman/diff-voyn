"""Control experiment 6a — can one model be both a fair judge and a search
target?  (docs/project_goals_and_progress.md §9, item 6a)

The Phase-2 curriculum showed the diffusion evaluator text under a wrong key
(20–50 % of positions) labelled with its true language so that its language
call would not move as the key goes from 0 % to 50 % wrong.  The paper's
argument (§2.7) is that *slope toward the key* (what a search target needs)
and *insensitivity to key errors* (what a fair judge needs) are contradictory
requirements for one score — mechanistic, never tested.  The ablation is the
"more simply" arm of the plan: re-run the Phase-B curriculum WITHOUT the
wrong-key family (``scripts/train.py --noise-families segmentation
transcription``) and compare, on the same size tier and paired texts,

    phase_a   clean pretraining only (no curriculum)
    phase_b   the full curriculum (wrong key + parse + transcription)
    nowk      the curriculum without the wrong-key family

on three measurements:

  (i)   key-search success from random / partially wrong starts with the
        model's own NELBO as the objective (this script, ``--stage search``):
        steepest-ascent pair-swap search over bijective keys on rung-1
        synthetic ciphers, the same move set and start keys for every
        objective, the pentagram n-gram as the reference target;
  (ii)  stability of the language call as the key is corrupted 0 → 50 %
        (``scripts/ngram_robustness.py --rank-ckpt`` on the 2.6 grid) and the
        own-language cost slope (``scripts/robustness_curve.py``);
  (iii) the per-language held-out canary (``--stage canary``).

``--stage report`` folds all three into
``DATA_ROOT/analysis/control6/control6a_report.{json,md}``.

Usage (25M tier; ``--tier 85m`` for the 85M trio):
    uv run python scripts/control6a_judge_vs_target.py --stage search --tier 25m
    uv run python scripts/control6a_judge_vs_target.py --stage canary --tier 25m
    uv run python scripts/control6a_judge_vs_target.py --stage report --tier 25m
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX, NULL_LANG_INDEX, CorpusWindows
from diff_voyn.heads.ladder import load_done, wilson, write_json_atomic
from diff_voyn.heads.ngram import A, lm_dir, load_lm
from diff_voyn.infra.nelbo import estimate_nelbo_bits_per_char

LANGS = tuple(LANG_TO_INDEX)
KEY = ("model", "language", "length", "trial", "start")
NG_REF = "ng5"


def tier_models(root: Path, tier: str) -> dict[str, Path]:
    runs = root / "runs"
    m = {
        "phase_a": runs / f"phase_a-{tier}-seed0" / "ckpt_final.pt",
        "phase_b": runs / f"phase_b-{tier}-seed0" / "ckpt_final.pt",
        "nowk": runs / f"phase_b-{tier}-nowk-seed0" / "ckpt_final.pt",
    }
    if tier == "85m":
        m["phase_c"] = runs / "phase_c-85m-seed0" / "ckpt_final.pt"
    return m


# ---------------------------------------------------------------------------
# (i) search


def _starts(
    true_inverse: np.ndarray, kinds: list[str], seed: int
) -> dict[str, np.ndarray]:
    """Start keys (cipher symbol -> letter): random permutations and
    partially wrong keys with a fraction f of the symbols' assignments
    permuted among themselves (self-consistent, still bijective)."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in kinds:
        if k.startswith("random"):
            out[k] = rng.permutation(A)
        elif k.startswith("wrong"):
            f = int(k[5:]) / 100.0
            n = max(2, round(f * A))
            key = true_inverse.copy()
            idx = rng.choice(A, size=n, replace=False)
            while True:  # derangement of the chosen subset
                p = rng.permutation(n)
                if (p != np.arange(n)).all():
                    break
            key[idx] = true_inverse[idx[p]]
            out[k] = key
        else:
            raise ValueError(k)
    return out


def _climb(objective, cipher, plain, start, sweeps, trace, n_evals):
    cur = start.copy()
    best_bits = None
    for _ in range(sweeps):
        cands = [cur]
        for s1 in range(A):
            for s2 in range(s1 + 1, A):
                m = cur.copy()
                m[s1], m[s2] = m[s2], m[s1]
                cands.append(m)
        bits = objective(np.stack([m[cipher] for m in cands]))
        n_evals[0] += len(cands)
        k = int(np.argmin(bits))
        best_bits = float(bits[k])
        trace.append(
            {
                "bits": float(bits[0]),
                "ser": float(np.mean(cur[cipher] != plain)),
                "gain": float(bits[0] - bits[k]),
            }
        )
        if k == 0:
            break
        cur = cands[k]
    return cur, best_bits


def swap_search(
    objective,
    cipher: np.ndarray,
    plain: np.ndarray,
    start: np.ndarray,
    sweeps: int,
    kicks: int = 0,
    rng: np.random.Generator | None = None,
):
    """Steepest-ascent search over pair swaps of a bijective key with ILS
    kicks (the Phase-5 inner search's move set).  ``objective`` maps stacked
    decodes (n, L) -> bits per row (lower is better); every sweep scores the
    current key and all C(25,2) swaps in one paired batch and takes the best
    move until no swap improves; then ``kicks`` times: perturb the incumbent
    by three random swaps, re-climb, keep the result if it is better."""
    trace: list = []
    n_evals = [0]
    best, best_bits = _climb(objective, cipher, plain, start, sweeps, trace, n_evals)
    rng = rng or np.random.default_rng(0)
    for _ in range(kicks):
        cand = best.copy()
        for _k in range(3):
            i, j = rng.integers(0, A, size=2)
            cand[i], cand[j] = cand[j], cand[i]
        cand, cb = _climb(objective, cipher, plain, cand, sweeps, trace, n_evals)
        if cb < best_bits:
            best, best_bits = cand, cb
    return best, trace, n_evals[0]


def _diffusion_objective(ev, language: str, budget: int, seed: int, batch: int):
    from diff_voyn.heads.two_tier import paired_bits

    def f(rows):
        return paired_bits(
            ev, rows, [language], n_strata=budget, seed=seed, batch=batch
        )[:, 0]

    return f


def _ngram_objective(lm, order: int = 5):
    def f(rows):
        return np.array(
            [-lm.score_ids(r, order) / (len(r) * np.log(2.0)) for r in rows]
        )

    return f


def stage_search(args, root):
    import rung1_diffusion as r1

    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator

    torch.set_float32_matmul_precision("high")
    suite = [
        j
        for j in r1.build_suite(root, args.trials, args.lengths, args.seed)
        if j["trial"] < args.trials
    ]
    models = tier_models(root, args.tier)
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}
    lms = {lang: load_lm(lm_dir() / f"{lang}.npz") for lang in LANGS}
    path = args.out_dir / f"control6a_search_{args.tier}.json"
    done = load_done(path, KEY) if not args.fresh else {}
    results = list(done.values())
    settings = {
        "tier": args.tier,
        "models": {k: str(v) for k, v in models.items()},
        "trials": args.trials,
        "lengths": list(args.lengths),
        "starts": list(args.starts),
        "sweeps": args.sweeps,
        "kicks": args.kicks,
        "budget": args.budget,
        "objective": "own-language NELBO bits/char (fixed masks per instance) / pentagram bits",
        "move_set": "all C(25,2) pair swaps, steepest ascent + 3-swap ILS kicks",
    }

    def flush():
        write_json_atomic(
            path,
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "settings": settings,
                "instances": results,
            },
        )

    t0 = time.time()
    order = [NG_REF] + list(models)
    for name in order:
        ev = None
        if name != NG_REF:
            if not models[name].exists():
                print(f"skip {name}: {models[name]} missing", flush=True)
                continue
            ev = DiffusionEvaluator.from_checkpoint(models[name], device=args.device)
        todo = []
        for j in suite:
            for st in args.starts:
                if (name, j["language"], j["length"], j["trial"], st) not in done:
                    todo.append((j, st))
        print(f"{name}: {len(todo)} searches", flush=True)
        for n, (j, st) in enumerate(todo, 1):
            cipher = np.asarray(j["cipher_ids"], dtype=np.int64)
            plain = np.asarray(j["plain_ids"], dtype=np.int64)
            true_inv = np.asarray(j["true_map"], dtype=np.int64)
            seed = zlib.crc32(
                f"6a/{j['language']}/{j['length']}/{j['trial']}".encode()
            ) % (2**31)
            start = _starts(true_inv, [st], seed + 17 * (1 + args.starts.index(st)))[st]
            if name == NG_REF:
                obj = _ngram_objective(lms[j["language"]])
            else:
                obj = _diffusion_objective(
                    ev, j["language"], args.budget, seed, args.batch
                )
            t1 = time.time()
            found, trace, n_evals = swap_search(
                obj,
                cipher,
                plain,
                start,
                args.sweeps,
                kicks=args.kicks,
                rng=np.random.default_rng(seed + 1),
            )
            true_bits, found_bits, start_bits = obj(
                np.stack([true_inv[cipher], found[cipher], start[cipher]])
            )
            rec = {
                "model": name,
                "language": j["language"],
                "length": j["length"],
                "trial": j["trial"],
                "start": st,
                "start_ser": float(np.mean(start[cipher] != plain)),
                "final_ser": float(np.mean(found[cipher] != plain)),
                "sweeps_used": len(trace),
                "n_evals": n_evals,
                "bits_start": float(start_bits),
                "bits_found": float(found_bits),
                "bits_true": float(true_bits),
                "true_key_is_better": bool(true_bits < found_bits - 1e-9),
                "trace_ser": [t["ser"] for t in trace],
                "trace_bits": [t["bits"] for t in trace],
                "seconds": round(time.time() - t1, 1),
            }
            results.append(rec)
            done[(name, j["language"], j["length"], j["trial"], st)] = rec
            if n % 10 == 0 or n == len(todo):
                print(
                    f"  {name} {n}/{len(todo)} {j['language']} L={j['length']} {st}: "
                    f"SER {rec['start_ser']:.2f} → {rec['final_ser']:.2f} in {len(trace)} sweeps "
                    f"({rec['seconds']}s, total {time.time()-t0:.0f}s)",
                    flush=True,
                )
                flush()
        del ev
        if args.device == "cuda":
            torch.cuda.empty_cache()
    flush()
    print(f"written {path}")


# ---------------------------------------------------------------------------
# (iii) canary


def stage_canary(args, root):
    from diff_voyn.infra.checkpoint import load_backbone

    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    heldout = CorpusWindows(
        corpus_dir,
        {
            lang: [d["doc_id"] for d in sp["heldout"]]
            for lang, sp in splits["languages"].items()
        },
    )
    models = tier_models(root, args.tier)
    out = {
        "tier": args.tier,
        "windows": args.canary_windows,
        "strata": args.canary_strata,
        "models": {},
    }
    for name, ckpt in models.items():
        if not ckpt.exists():
            print(f"skip {name}: missing")
            continue
        model, meta = load_backbone(ckpt, args.device, ema=True)
        seq_len = 1024
        res = {}
        for lang in LANGS:
            rng = np.random.default_rng(12345 + LANG_TO_INDEX[lang])
            ids = torch.from_numpy(
                np.stack(
                    [
                        heldout.sample_window(lang, seq_len, rng)
                        for _ in range(args.canary_windows)
                    ]
                )
            ).long()
            cond = estimate_nelbo_bits_per_char(
                model,
                ids,
                LANG_TO_INDEX[lang],
                n_strata=args.canary_strata,
                seed=0,
                device=args.device,
            )
            unc = estimate_nelbo_bits_per_char(
                model,
                ids,
                NULL_LANG_INDEX,
                n_strata=args.canary_strata,
                seed=0,
                device=args.device,
            )
            res[lang] = {"cond": float(cond), "uncond": float(unc)}
            print(f"  {name:8s} {lang:8s} cond {cond:.4f} uncond {unc:.4f}", flush=True)
        out["models"][name] = {
            "ckpt": str(ckpt),
            "step": meta.get("step"),
            "nelbo": res,
        }
        del model
        torch.cuda.empty_cache()
    path = args.out_dir / f"control6a_canary_{args.tier}.json"
    write_json_atomic(path, out)
    print(f"written {path}")


# ---------------------------------------------------------------------------
# report


def _search_summary(instances: list[dict]) -> dict:
    out = {}
    models = sorted({r["model"] for r in instances}, key=lambda m: (m != NG_REF, m))
    for m in models:
        rs = [r for r in instances if r["model"] == m]
        by = {}
        for st in sorted({r["start"] for r in rs}):
            for L in sorted({r["length"] for r in rs}):
                sel = [r for r in rs if r["start"] == st and r["length"] == L]
                if not sel:
                    continue
                k = sum(r["final_ser"] < 0.05 for r in sel)
                by[f"{st}/L{L}"] = {
                    "n": len(sel),
                    "solved": k / len(sel),
                    "solved_ci95": list(wilson(k, len(sel))),
                    "final_ser_median": float(np.median([r["final_ser"] for r in sel])),
                    "final_ser_mean": float(np.mean([r["final_ser"] for r in sel])),
                    "start_ser_mean": float(np.mean([r["start_ser"] for r in sel])),
                    "sweeps_mean": float(np.mean([r["sweeps_used"] for r in sel])),
                    "true_key_better_rate": float(
                        np.mean([r["true_key_is_better"] for r in sel])
                    ),
                    "by_language": {
                        l: float(
                            np.mean(
                                [
                                    r["final_ser"] < 0.05
                                    for r in sel
                                    if r["language"] == l
                                ]
                            )
                        )
                        for l in LANGS
                        if any(r["language"] == l for r in sel)
                    },
                }
        out[m] = by
    return out


def _load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def stage_report(args, root):
    tier = args.tier
    search = _load_json(args.out_dir / f"control6a_search_{tier}.json")
    canary = _load_json(args.out_dir / f"control6a_canary_{tier}.json")
    rank = _load_json(
        root / "analysis" / "phase2" / f"robustness_control6a-{tier}.json"
    )
    curves = {
        name: _load_json(root / "analysis" / "phase2" / f"robustness_{tag}.json")
        for name, tag in (
            ("phase_a", f"phase_a-{tier}"),
            ("phase_b", f"phase_b-{tier}"),
            ("nowk", f"phase_b_nowk-{tier}"),
        )
    }
    rep = {
        "created_utc": datetime.now(UTC).isoformat(),
        "tier": tier,
        "search": _search_summary(search["instances"]) if search else None,
        "search_settings": search["settings"] if search else None,
        "canary": canary["models"] if canary else None,
        "judge_accuracy": None,
        "cost_curves": {},
    }
    if rank:
        ja = rank["judge_accuracy"]
        rep["judge_accuracy"] = {
            k: {
                "severities": v["severities"],
                "top1_raw": v["top1_raw"],
                "n": v["n_windows"],
            }
            for k, v in ja.items()
            if not k.startswith("ng")
        }
        rep["judge_accuracy_ngram"] = {
            k: {
                "severities": v["severities"],
                "top1_excess": v["top1_excess"],
                "n": v["n_windows"],
            }
            for k, v in ja.items()
            if k.startswith("ng5")
        }
    for name, c in curves.items():
        if not c:
            continue
        for k, v in c["curves"].items():
            _, lang, fam = k.split("/")
            if fam != "substitution":
                continue
            grid = v["severities"] if "severities" in v else v.get("grid")
            means = v["means"] if "means" in v else v.get("mean")
            ctrl = c["controls"].get(k.rsplit("/", 1)[0], {})
            rep["cost_curves"][f"{name}/{lang}"] = {
                "grid": grid,
                "means": means,
                "shuffled": (
                    ctrl.get("shuffled", {}).get("mean")
                    if isinstance(ctrl.get("shuffled"), dict)
                    else ctrl.get("shuffled")
                ),
                "stats": v.get("stats"),
            }
    out = args.out_dir / f"control6a_report_{tier}.json"
    write_json_atomic(out, rep)
    md = render_md(rep)
    (args.out_dir / f"control6a_report_{tier}.md").write_text(md)
    print(md)
    print(f"written {out}")


def render_md(rep: dict) -> str:
    L = [f"# Control 6a — one model as judge and search target ({rep['tier']})\n"]
    if rep["search"]:
        s = rep["search_settings"]
        L.append(
            f"## (i) Key search under the model's own score\n\n{len(s['lengths'])} lengths × {s['trials']} trials × 3 languages, "
            f"starts {s['starts']}, ≤{s['sweeps']} steepest-ascent pair-swap sweeps + {s.get('kicks', 0)} ILS kicks, diffusion budget {s['budget']} strata (fixed masks).\n"
        )
        cells = sorted({c for m in rep["search"].values() for c in m})
        L.append(
            "| objective | "
            + " | ".join(f"{c} solved / median SER / true-key-better" for c in cells)
            + " |"
        )
        L.append("|---|" + "---|" * len(cells))
        for m, by in rep["search"].items():
            L.append(
                f"| {m} | "
                + " | ".join(
                    (
                        f"{by[c]['solved']:.2f} / {by[c]['final_ser_median']:.2f} / {by[c]['true_key_better_rate']:.2f}"
                        if c in by
                        else "—"
                    )
                    for c in cells
                )
                + " |"
            )
        L.append("\nPer language (solved fraction):\n")
        L.append("| objective | cell | latin | italian | german |")
        L.append("|---|---|---|---|---|")
        for m, by in rep["search"].items():
            for c, v in by.items():
                bl = v["by_language"]
                L.append(
                    f"| {m} | {c} | "
                    + " | ".join(f"{bl.get(l, float('nan')):.2f}" for l in LANGS)
                    + " |"
                )
    if rep["judge_accuracy"]:
        L.append(
            "\n## (ii) Language-call stability under a wrong key (top-1 of the true language, raw NELBO)\n"
        )
        L.append(
            "| judge | lang | "
            + " | ".join(
                f"{s:g}"
                for s in next(iter(rep["judge_accuracy"].values()))["severities"]
            )
            + " |"
        )
        L.append(
            "|---|---|"
            + "---|" * len(next(iter(rep["judge_accuracy"].values()))["severities"])
        )
        for k, v in rep["judge_accuracy"].items():
            judge, lang, fam = k.split("/")
            if fam != "substitution":
                continue
            L.append(
                f"| {judge} | {lang} | "
                + " | ".join(f"{a:.2f}" for a in v["top1_raw"])
                + " |"
            )
        L.append("\nParse / transcription families (top-1 at each grid point):\n")
        for k, v in rep["judge_accuracy"].items():
            judge, lang, fam = k.split("/")
            if fam == "substitution":
                continue
            L.append(
                f"- {judge} {lang} {fam}: "
                + ", ".join(
                    f"{s:g}→{a:.2f}" for s, a in zip(v["severities"], v["top1_raw"])
                )
            )
    if rep["cost_curves"]:
        L.append(
            "\n## (ii′) Own-language cost vs wrong-key severity (bits/char; slope = search signal)\n"
        )
        for k, v in rep["cost_curves"].items():
            if v["grid"] and v["means"]:
                L.append(
                    f"- {k}: "
                    + ", ".join(f"{g:g}:{m:.2f}" for g, m in zip(v["grid"], v["means"]))
                    + (
                        f"; shuffled {v['shuffled']:.2f}"
                        if isinstance(v["shuffled"], (int, float))
                        else ""
                    )
                )
    if rep["canary"]:
        L.append(
            "\n## (iii) Per-language held-out canary (EMA weights, bits/char, cond | uncond)\n"
        )
        L.append("| model | step | latin | italian | german |")
        L.append("|---|---|---|---|---|")
        for m, v in rep["canary"].items():
            L.append(
                f"| {m} | {v['step']} | "
                + " | ".join(
                    f"{v['nelbo'][l]['cond']:.3f} \\| {v['nelbo'][l]['uncond']:.3f}"
                    for l in LANGS
                )
                + " |"
            )
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------


def main():
    root = data_root()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", required=True, choices=["search", "canary", "report"])
    ap.add_argument("--tier", default="25m", choices=["25m", "85m"])
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="subset of phase_a phase_b nowk phase_c",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="rung-1 suite seed (Phase-5 texts)"
    )
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--lengths", type=int, nargs="+", default=[200, 400])
    ap.add_argument(
        "--starts", nargs="+", default=["random1", "random2", "wrong20", "wrong50"]
    )
    ap.add_argument("--sweeps", type=int, default=60)
    ap.add_argument("--kicks", type=int, default=8, help="ILS kicks after convergence")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--canary-windows", type=int, default=32)
    ap.add_argument("--canary-strata", type=int, default=64)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=root / "analysis" / "control6")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    {"search": stage_search, "canary": stage_canary, "report": stage_report}[
        args.stage
    ](args, root)


if __name__ == "__main__":
    main()
