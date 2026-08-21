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

The chosen budget = the smallest with flip-rate < 1% at every length; it is
written to the report and consumed by ``scripts/language_recovery.py``.

Usage:
    uv run python scripts/sample_budget.py --ckpt DATA_ROOT/runs/phase_b-85m-seed0/ckpt_final.pt
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


def sample_windows(
    heldout: CorpusWindows, lang: str, n: int, length: int, rng
) -> np.ndarray:
    return np.stack([heldout.sample_window(lang, length, rng) for _ in range(n)])


def flip_rate(top1: np.ndarray) -> float:
    """top1 [R, N] -> mean over windows of pairwise replicate disagreement."""
    R = top1.shape[0]
    pairs = list(itertools.combinations(range(R), 2))
    dis = np.stack([top1[a] != top1[b] for a, b in pairs])  # [P, N]
    return float(dis.mean())


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
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.device == "cuda":
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
    model, meta = load_backbone(args.ckpt, args.device)
    rng = np.random.default_rng(args.seed)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "3.2",
        "backbone": meta,
        "settings": vars(args) | {"ckpt": str(args.ckpt)},
        "flip_target": FLIP_TARGET,
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
    for length in list(LENGTHS) if args.merge else args.lengths:
        ids = np.concatenate(
            [sample_windows(heldout, l, args.windows, length, rng) for l in LANGS]
        )
        truth = np.repeat(np.arange(len(LANGS)), args.windows)
        per_budget = {}
        scores_by_budget = {}
        if args.merge:
            for k, arr in prev_arrays.items():
                if k.startswith(f"L{length}/B"):
                    B = int(k.split("/B")[1])
                    scores_by_budget[B] = arr
                    per_budget[B] = prev_report["by_length"][str(length)]["by_budget"][
                        str(B)
                    ]
                    arrays[k] = arr
        if length not in args.lengths:
            report["by_length"][str(length)] = {
                "n_windows": len(ids),
                "by_budget": {str(B): v for B, v in sorted(per_budget.items())},
            }
            continue
        t0 = time.time()
        for B in args.budgets:
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
            top1 = arr.argmin(axis=2)  # [R, N]
            diffs = np.stack(
                [
                    arr[:, :, a] - arr[:, :, b]
                    for a, b in itertools.combinations(range(len(LANGS)), 2)
                ]
            )
            per_budget[B] = {
                "flip_rate": flip_rate(top1),
                "top1_true_rate": float((top1 == truth[None, :]).mean()),
                "diff_sem_mean": float(diffs.std(axis=1, ddof=1).mean()),
                "seconds": round(time.time() - t0, 1),
            }
            arrays[f"L{length}/B{B}"] = arr
        ref = (
            scores_by_budget[max(scores_by_budget)].mean(axis=0).argmin(axis=1)
        )  # consensus
        for B, arr in scores_by_budget.items():
            top1 = arr.argmin(axis=2)
            per_budget[B]["disagree_with_consensus"] = float(
                (top1 != ref[None, :]).mean()
            )
        report["by_length"][str(length)] = {
            "n_windows": len(ids),
            "by_budget": {str(B): v for B, v in sorted(per_budget.items())},
        }
        print(
            f"L={length}: "
            + "  ".join(
                f"B{B}: flip {v['flip_rate']:.3%} acc {v['top1_true_rate']:.1%}"
                for B, v in per_budget.items()
            ),
            flush=True,
        )

    all_lengths = sorted(int(L) for L in report["by_length"])
    all_budgets = sorted(
        {int(B) for d in report["by_length"].values() for B in d["by_budget"]}
    )
    chosen = None
    for B in all_budgets:
        if all(
            str(B) in report["by_length"][str(L)]["by_budget"]
            and report["by_length"][str(L)]["by_budget"][str(B)]["flip_rate"]
            < FLIP_TARGET
            for L in all_lengths
        ):
            chosen = B
            break
    report["chosen_budget"] = chosen
    report["chosen_budget_note"] = (
        f"smallest budget with replicate flip-rate < {FLIP_TARGET:.0%} at every length"
        if chosen is not None
        else f"no budget ≤ {max(all_budgets)} reaches flip-rate < {FLIP_TARGET:.0%} at every length"
    )
    args.lengths, args.budgets = all_lengths, all_budgets
    out = out_dir / "sample_budget.json"
    out.write_text(json.dumps(report, indent=2))
    np.savez(out_dir / "sample_budget_scores.npz", ids_note="see json", **arrays)
    print(f"chosen budget: {chosen}; written {out}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for L in args.lengths:
            ys = [
                report["by_length"][str(L)]["by_budget"][str(B)]["flip_rate"]
                for B in args.budgets
            ]
            ax.plot(args.budgets, ys, marker="o", label=f"L={L}")
        ax.axhline(FLIP_TARGET, color="k", ls="--", lw=0.8)
        if chosen:
            ax.axvline(chosen, color="gray", ls=":", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("sample budget (timestep draws per window)")
        ax.set_ylabel("top-1 flip-rate between replicates")
        ax.set_title("Task 3.2 — ranking stability vs budget")
        ax.legend(fontsize=8)
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
