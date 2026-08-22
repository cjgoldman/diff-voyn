"""Task 3.2 — sample-budget study: how many stratified timestep draws until
the language *ranking* is stable, per window length {50 … 700}.

For each length, a fixed set of held-out windows (per language) is scored
under every conditioning language with CRN at budgets B ∈ {4 … 128} draws
(``n_strata = B``, one sample per stratum), ``R`` times with fresh seeds.
The reported *flip-rate* at a budget is the probability that two independent
replicates disagree on the top-1 language of a window — pure Monte-Carlo
ranking noise (ties in the true margin show up as flips too, which is the
honest reading: such windows are not rankable at that budget). Also reported:
disagreement with the consensus ranking at the largest budget, and the
s.e.m. of the between-language difference.

Chosen budget. Flips on a window whose *true* margin is ~0 are not
Monte-Carlo noise — a tie flips 50% of the time at any budget — and the
clean-text regime is full of them: the same-text conditioning margin shrinks
as ~1/L (≈30 bits per window in total), so at ≥400 chars the median margin
is below 0.1 bits/char and the all-window flip-rate plateaus at ~2% (Phase-B
85M, budgets 4…128). The study therefore reports the flip-rate twice: over
all windows, and over the windows whose consensus margin (pooled replicates
at the largest budget) exceeds ``--margin-floor`` (default 0.05 bits/char —
the order of the close-pair same-text conditioning margin of task 3.6, i.e.
the population the instrument reports as *unresolved* rather than ranks).
The chosen budget = the smallest with resolvable-window flip-rate < 1% at
every length; the all-window criterion is recorded alongside
(``chosen_budget_all_windows``). ``--restat`` recomputes every statistic from
the saved per-window arrays without the GPU.

Usage:
    uv run python scripts/sample_budget.py --ckpt DATA_ROOT/runs/phase_b-85m-seed0/ckpt_final.pt
    uv run python scripts/sample_budget.py --restat [--margin-floor 0.05]
Writes DATA_ROOT/analysis/phase3/sample_budget.{json,npz,png}; ClearML ``task3.2``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX, CorpusWindows
from diff_voyn.infra.checkpoint import load_backbone
from diff_voyn.metrology import ScoreSettings, score_conditions

LANGS = tuple(LANG_TO_INDEX)
LENGTHS = (50, 100, 200, 400, 700)
BUDGETS = (4, 8, 16, 32, 64, 128)
FLIP_TARGET = 0.01
MARGIN_FLOOR = 0.05  # bits/char; see module docstring
MARGIN_FLOOR_GRID = (0.02, 0.05, 0.1, 0.2)


def sample_windows(
    heldout: CorpusWindows, lang: str, n: int, length: int, rng
) -> np.ndarray:
    return np.stack([heldout.sample_window(lang, length, rng) for _ in range(n)])


def flip_per_window(top1: np.ndarray) -> np.ndarray:
    """top1 [R, N] -> [N] pairwise replicate disagreement per window."""
    R = top1.shape[0]
    pairs = list(itertools.combinations(range(R), 2))
    dis = np.stack([top1[a] != top1[b] for a, b in pairs])  # [P, N]
    return dis.mean(axis=0)


def flip_rate(top1: np.ndarray) -> float:
    """top1 [R, N] -> mean over windows of pairwise replicate disagreement."""
    return float(flip_per_window(top1).mean())


def consensus_margin(arr_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scores at the largest budget [R, N, C], replicates pooled -> (top-1
    language [N], margin to the runner-up in bits/char [N])."""
    cons = arr_max.mean(axis=0)
    srt = np.sort(cons, axis=1)
    return cons.argmin(axis=1), srt[:, 1] - srt[:, 0]


def budget_stats(
    scores_by_budget: dict[int, np.ndarray],
    truth: np.ndarray,
    margin_floor: float,
    seconds: dict[int, float] | None = None,
) -> dict[int, dict]:
    """Per-budget statistics for one length. ``scores_by_budget[B]`` is
    [R, N, C] bits/char; ``truth`` [N] the true language index."""
    n_lang = next(iter(scores_by_budget.values())).shape[2]
    ref, margin = consensus_margin(scores_by_budget[max(scores_by_budget)])
    resolvable = margin > margin_floor
    out = {}
    for B, arr in sorted(scores_by_budget.items()):
        top1 = arr.argmin(axis=2)  # [R, N]
        fpw = flip_per_window(top1)
        diffs = np.stack(
            [
                arr[:, :, a] - arr[:, :, b]
                for a, b in itertools.combinations(range(n_lang), 2)
            ]
        )
        out[B] = {
            "flip_rate": float(fpw.mean()),
            "flip_rate_resolvable": (
                float(fpw[resolvable].mean()) if resolvable.any() else None
            ),
            "flip_rate_by_margin_floor": {
                str(f): {
                    "flip_rate": (
                        float(fpw[margin > f].mean()) if (margin > f).any() else None
                    ),
                    "fraction_kept": float((margin > f).mean()),
                }
                for f in MARGIN_FLOOR_GRID
            },
            "top1_true_rate": float((top1 == truth[None, :]).mean()),
            "diff_sem_mean": float(diffs.std(axis=1, ddof=1).mean()),
            "disagree_with_consensus": float((top1 != ref[None, :]).mean()),
        }
        if seconds and B in seconds:
            out[B]["seconds"] = seconds[B]
    out["consensus"] = {
        "budget": int(max(scores_by_budget)),
        "margin_floor": margin_floor,
        "fraction_resolvable": float(resolvable.mean()),
        "margin_quantiles_bits": {
            q: float(np.quantile(margin, float(q) / 100))
            for q in ("5", "10", "25", "50")
        },
        "margin_median_x_length_bits": None,  # filled by the caller
        "top1_true_rate": float((ref == truth).mean()),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_b-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--windows", type=int, default=64, help="per language")
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    p.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--merge",
        action="store_true",
        help="add the given --budgets to an existing report (same windows/seeds: "
        "the window sampler and replicate seeds are deterministic), re-deriving "
        "the consensus and the chosen budget over the union",
    )
    p.add_argument(
        "--restat",
        action="store_true",
        help="recompute every statistic (and the chosen budget) from the saved "
        "per-window arrays; no scoring, no GPU",
    )
    p.add_argument("--margin-floor", type=float, default=MARGIN_FLOOR)
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.restat:
        args.merge, args.lengths = True, []
    if args.device == "cuda" and not args.restat:
        torch.set_float32_matmul_precision("high")

    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    heldout = CorpusWindows(
        corpus_dir,
        {
            l: [d["doc_id"] for d in sp["heldout"]]
            for l, sp in splits["languages"].items()
        },
    )
    model, meta = (None, None) if args.restat else load_backbone(args.ckpt, args.device)
    rng = np.random.default_rng(args.seed)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "3.2",
        "backbone": meta,
        "settings": vars(args) | {"ckpt": str(args.ckpt)},
        "flip_target": FLIP_TARGET,
        "margin_floor": args.margin_floor,
        "by_length": {},
    }
    arrays = {}
    out_dir = root / "analysis" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_report, prev_arrays = None, {}
    if args.merge:
        prev_report = json.loads((out_dir / "sample_budget.json").read_text())
        with np.load(out_dir / "sample_budget_scores.npz") as npz:
            prev_arrays = {k: npz[k] for k in npz.files if "/" in k}
        report["by_length"] = prev_report["by_length"]
        report["merged_from"] = prev_report["created_utc"]
        if args.restat:
            report["backbone"] = prev_report["backbone"]
            report["settings"] = prev_report["settings"] | {
                "restat": True,
                "margin_floor": args.margin_floor,
            }
    for length in list(LENGTHS) if args.merge else args.lengths:
        if args.restat:
            n_per_lang = int(prev_report["settings"]["windows"])
        else:
            ids = np.concatenate(
                [sample_windows(heldout, l, args.windows, length, rng) for l in LANGS]
            )
            n_per_lang = args.windows
        truth = np.repeat(np.arange(len(LANGS)), n_per_lang)
        seconds: dict[int, float] = {}
        scores_by_budget = {}
        if args.merge:
            for k, arr in prev_arrays.items():
                if k.startswith(f"L{length}/B"):
                    B = int(k.split("/B")[1])
                    scores_by_budget[B] = arr
                    prev = prev_report["by_length"][str(length)]["by_budget"][str(B)]
                    if "seconds" in prev:
                        seconds[B] = prev["seconds"]
                    arrays[k] = arr
        t0 = time.time()
        for B in [] if length not in args.lengths else args.budgets:
            runs = []
            for r in range(args.reps):
                st = ScoreSettings(
                    n_strata=B, seed=args.seed + 100_000 * (r + 1), batch=args.batch
                )
                runs.append(
                    score_conditions(model, ids, LANGS, settings=st, device=args.device)
                )
            arr = np.stack(runs)  # [R, N, C]
            scores_by_budget[B] = arr
            seconds[B] = round(time.time() - t0, 1)
            arrays[f"L{length}/B{B}"] = arr
        stats = budget_stats(scores_by_budget, truth, args.margin_floor, seconds)
        consensus = stats.pop("consensus")
        consensus["margin_median_x_length_bits"] = (
            consensus["margin_quantiles_bits"]["50"] * length
        )
        report["by_length"][str(length)] = {
            "n_windows": len(truth),
            "consensus": consensus,
            "by_budget": {str(B): v for B, v in sorted(stats.items())},
        }
        print(
            f"L={length} (resolvable {consensus['fraction_resolvable']:.0%} at "
            f"margin > {args.margin_floor} bits/char): "
            + "  ".join(
                f"B{B}: flip {v['flip_rate']:.3%} / resolvable "
                f"{(v['flip_rate_resolvable'] or 0):.3%} acc {v['top1_true_rate']:.1%}"
                for B, v in sorted(stats.items())
            ),
            flush=True,
        )

    all_lengths = sorted(int(L) for L in report["by_length"])
    all_budgets = sorted(
        {int(B) for d in report["by_length"].values() for B in d["by_budget"]}
    )

    def smallest_passing(key: str) -> int | None:
        for B in all_budgets:
            if all(
                str(B) in report["by_length"][str(L)]["by_budget"]
                and report["by_length"][str(L)]["by_budget"][str(B)][key] is not None
                and report["by_length"][str(L)]["by_budget"][str(B)][key] < FLIP_TARGET
                for L in all_lengths
            ):
                return B
        return None

    chosen = smallest_passing("flip_rate_resolvable")
    chosen_all = smallest_passing("flip_rate")
    report["chosen_budget"] = chosen
    report["chosen_budget_criterion"] = (
        f"smallest budget with replicate flip-rate < {FLIP_TARGET:.0%} at every length "
        f"over windows whose consensus margin exceeds {args.margin_floor} bits/char "
        "(resolvable windows; ties are not Monte-Carlo noise — see module docstring)"
    )
    report["chosen_budget_note"] = (
        f"budget {chosen}: resolvable-window flip-rate < {FLIP_TARGET:.0%} at every length"
        if chosen is not None
        else f"no budget ≤ {max(all_budgets)} reaches resolvable-window flip-rate "
        f"< {FLIP_TARGET:.0%} at every length"
    )
    report["chosen_budget_all_windows"] = chosen_all
    report["chosen_budget_all_windows_note"] = (
        f"budget {chosen_all}: all-window flip-rate < {FLIP_TARGET:.0%} at every length"
        if chosen_all is not None
        else f"no budget ≤ {max(all_budgets)} reaches all-window flip-rate < "
        f"{FLIP_TARGET:.0%} at every length (near-tie windows)"
    )
    args.lengths, args.budgets = all_lengths, all_budgets
    out = out_dir / "sample_budget.json"
    out.write_text(json.dumps(report, indent=2))
    np.savez(out_dir / "sample_budget_scores.npz", ids_note="see json", **arrays)
    print(
        f"chosen budget (resolvable windows): {chosen}; all windows: {chosen_all}; "
        f"written {out}"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, key, title in (
            (axes[0], "flip_rate", "all windows"),
            (
                axes[1],
                "flip_rate_resolvable",
                f"windows with margin > {args.margin_floor} bits/char",
            ),
        ):
            for L in args.lengths:
                ys = [
                    report["by_length"][str(L)]["by_budget"][str(B)][key]
                    for B in args.budgets
                ]
                ys = [np.nan if y is None else y for y in ys]
                ax.plot(args.budgets, ys, marker="o", label=f"L={L}")
            ax.axhline(FLIP_TARGET, color="k", ls="--", lw=0.8)
            if chosen:
                ax.axvline(chosen, color="gray", ls=":", lw=0.8)
            ax.set_xscale("log", base=2)
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_xlabel("sample budget (timestep draws per window)")
            ax.set_title(title, fontsize=9)
        axes[0].set_ylabel("top-1 flip-rate between replicates")
        axes[0].legend(fontsize=8)
        fig.suptitle("Task 3.2 — ranking stability vs budget (Phase-B 85M)")
        fig.tight_layout()
        fig.savefig(out_dir / "sample_budget.png", dpi=120)
    except (ImportError, OSError, ValueError) as e:  # plotting is a convenience
        print(f"(plot skipped: {e})")

    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        task = init_task(
            RunConfig(run_name="sample-budget", phase="phase3"), root, tags=["task3.2"]
        )
        task.connect_configuration(report, name="sample_budget")
        logger = task.get_logger()
        for L, res in report["by_length"].items():
            for B, v in res["by_budget"].items():
                logger.report_scalar("flip_rate", f"L{L}", v["flip_rate"], int(B))
                if v.get("flip_rate_resolvable") is not None:
                    logger.report_scalar(
                        "flip_rate_resolvable",
                        f"L{L}",
                        v["flip_rate_resolvable"],
                        int(B),
                    )
                logger.report_scalar(
                    "top1_true_rate", f"L{L}", v["top1_true_rate"], int(B)
                )
        if (out_dir / "sample_budget.png").exists():
            logger.report_image(
                "sample_budget",
                "flip_rate",
                local_path=str(out_dir / "sample_budget.png"),
                iteration=0,
            )
        logger.flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()


if __name__ == "__main__":
    main()
