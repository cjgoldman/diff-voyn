"""Noise severity sweeps / robustness curve — tasks 2.1 (acceptance sweep),
2.5 (NULL-slot probe) and 2.6 (robustness curve), feeding Gate G2.

For each checkpoint given, the same fixed set of held-out source windows
(evenly spaced over the tiled held-out split, per language) is corrupted by
each noise family at each severity of a grid — with a fixed noise seed, so
every checkpoint scores *identical* noised texts (common random numbers
across models), and with a fixed masking seed (CRN across severities and
conditions). Scored: own-language per-window NELBO (bits/char, EMA weights).

Reported per (checkpoint, family, language):

- the curve: mean ± s.e.m. over windows at each severity;
- **monotone**: each paired increment (window-wise NELBO difference between
  adjacent severities) has mean > −2 s.e.m. — degradation never reverses;
- **no cliff** (operational definition for G2): no single grid increment of
  ≤0.05 severity raises the mean NELBO by more than ``--cliff-bits`` (0.5
  bits/char) and no increment carries more than half of the total rise over
  the grid — the curve climbs gradually instead of jumping to the
  "no structure" level (shuffled-letters and uniform-random references are
  scored alongside as the ceiling).

2.5 probe: clean text on the 2N-slot NULL frame — per-position NELBO split
into NULL slots and letter slots, next to the clean bits/char of the same
source text. Before Phase B the NULL slots are catastrophic (a never-seen
token); after Phase B they should cost about the parse-pattern entropy
(H(0.476) ≈ 1 bit) and the letter slots about the clean rate.

Usage:
    uv run python scripts/robustness_curve.py --tag phase_a \\
        --ckpt 85m=DATA_ROOT/runs/phase_a-85m-seed0/ckpt_final.pt \\
        --ckpt 25m=DATA_ROOT/runs/phase_a-25m-seed0/ckpt_final.pt

Writes DATA_ROOT/analysis/phase2/robustness_<tag>.json (+ .npz per-window
arrays, + .png plot) and a ClearML task tagged ``task2.6``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX, CorpusWindows
from diff_voyn.data.noise import (
    EVAL_NOISE_SEED,
    LETTER_BASE,
    N_LETTERS,
    SegmentationNoise,
    SubstitutionNoise,
    TranscriptionNoise,
    frame_with_nulls,
)
from diff_voyn.infra.checkpoint import load_backbone
from diff_voyn.infra.nelbo import per_position_nelbo_bits, per_window_nelbo_bits
from diff_voyn.vocab import NULL_ID

GRIDS = {
    "substitution": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0],
    "segmentation": [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
    "transcription": [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
}
FAMILIES = {
    "substitution": SubstitutionNoise,
    "segmentation": SegmentationNoise,
    "transcription": TranscriptionNoise,
}
SOURCE_MARGIN = 1.5


def parse_ckpts(items: list[str]) -> dict[str, Path]:
    out = {}
    for it in items:
        name, _, path = it.partition("=")
        if not path:
            raise SystemExit(f"--ckpt expects name=path, got {it!r}")
        out[name] = Path(path)
    return out


def select_sources(
    windows: CorpusWindows, lang: str, n: int, src_len: int, rng: np.random.Generator
) -> np.ndarray:
    """``n`` evenly spaced source windows of ``src_len`` chars from the tiled
    held-out split (deterministic; covers every document)."""
    tiled = windows.tiled_windows(lang, src_len)
    idx = np.linspace(0, len(tiled) - 1, num=min(n, len(tiled))).round().astype(int)
    return tiled[np.unique(idx)]


def build_variants(
    sources: np.ndarray, seq_len: int, noise_seed: int
) -> dict[str, dict[float, np.ndarray]]:
    """{family: {severity: [n, seq_len] uint8}} plus 'framed', 'shuffled',
    'uniform' controls under key ('control', name). Deterministic."""
    out: dict = {}
    for fi, (fam, grid) in enumerate(GRIDS.items()):
        out[fam] = {}
        for sev in grid:
            rng = np.random.default_rng([noise_seed, fi, int(sev * 1000)])
            rows = []
            for src in sources:
                ids, _ = FAMILIES[fam](sev)(src, rng)
                if len(ids) < seq_len:
                    ids = np.tile(ids, int(np.ceil(seq_len / len(ids))))
                rows.append(ids[:seq_len])
            out[fam][sev] = np.stack(rows)
    rng = np.random.default_rng([noise_seed, 7])
    out["framed"] = np.stack(
        [frame_with_nulls(src, rng)[0][:seq_len] for src in sources]
    )
    rng = np.random.default_rng([noise_seed, 8])
    out["shuffled"] = np.stack([rng.permutation(src[:seq_len]) for src in sources])
    out["uniform"] = (
        rng.integers(0, N_LETTERS, size=(len(sources), seq_len)) + LETTER_BASE
    ).astype(np.uint8)
    return out


@torch.no_grad()
def score(
    model, ids_np: np.ndarray, lang_idx: int, strata: int, device: str, batch: int
):
    ids = torch.from_numpy(ids_np.astype(np.int64))
    out = []
    for ci, i in enumerate(range(0, len(ids), batch)):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(
                per_window_nelbo_bits(
                    model,
                    ids[i : i + batch],
                    lang_idx,
                    n_strata=strata,
                    seed=ci,
                    device=device,
                ).numpy()
            )
    return np.concatenate(out)


@torch.no_grad()
def score_positions(model, ids_np, lang_idx, strata, device, batch):
    ids = torch.from_numpy(ids_np.astype(np.int64))
    out = []
    for ci, i in enumerate(range(0, len(ids), batch)):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(
                per_position_nelbo_bits(
                    model,
                    ids[i : i + batch],
                    lang_idx,
                    n_strata=strata,
                    seed=ci,
                    device=device,
                ).numpy()
            )
    return np.concatenate(out)


def curve_stats(grid: list[float], per_window: np.ndarray, cliff_bits: float) -> dict:
    """per_window [n_sev, n_windows] → means, sems, monotone/cliff verdicts.

    *monotone*: every paired increment has mean > −2 s.e.m.
    *no cliff*: degradation never **accelerates** — the slope (bits per unit
    severity) of each increment is ≤ 1.5× the steepest earlier slope (a
    gracefully degrading curve is concave: the first errors cost the most,
    then it saturates toward the no-structure ceiling; a catastrophic cliff
    is a late, sudden jump) — and no single grid increment exceeds
    ``cliff_bits`` or carries more than half of the total rise.
    *sensitivity*: the rise over the first grid step, and its share of the
    total rise — the quantity Phase B is meant to reduce.
    """
    means = per_window.mean(1)
    sems = per_window.std(1, ddof=1) / np.sqrt(per_window.shape[1])
    incs = np.diff(per_window, axis=0)  # paired increments [n_sev-1, n_windows]
    inc_mean = incs.mean(1)
    inc_sem = incs.std(1, ddof=1) / np.sqrt(incs.shape[1])
    monotone = bool((inc_mean > -2 * inc_sem).all())
    dsev = np.diff(np.array(grid))
    slopes = inc_mean / dsev
    total_rise = float(means[-1] - means[0])
    share = inc_mean / total_rise if total_rise > 0 else np.zeros_like(inc_mean)
    accel = np.zeros(len(slopes), dtype=bool)
    for i in range(1, len(slopes)):
        accel[i] = slopes[i] > 1.5 * max(slopes[:i].max(), 1e-9)
    no_cliff = bool(
        (inc_mean <= cliff_bits).all() and (share <= 0.5).all() and not accel.any()
    )
    return {
        "severities": grid,
        "mean_bits": means.tolist(),
        "sem_bits": sems.tolist(),
        "increment_mean": inc_mean.tolist(),
        "increment_sem": inc_sem.tolist(),
        "slopes_bits_per_unit_severity": slopes.tolist(),
        "monotone": monotone,
        "total_rise_bits": total_rise,
        "max_increment_bits": float(inc_mean.max()),
        "max_increment_share": float(share.max()) if total_rise > 0 else 0.0,
        "accelerating_increments": [int(i) for i in np.flatnonzero(accel)],
        "first_step_rise_bits": float(inc_mean[0]),
        "first_step_share": float(share[0]) if total_rise > 0 else 0.0,
        "no_cliff": no_cliff,
    }


def restat(root: Path, tag: str, cliff_bits: float) -> None:
    """Recompute the curve verdicts of an existing report from its saved
    per-window arrays (no GPU)."""
    out_dir = root / "analysis" / "phase2"
    json_path = out_dir / f"robustness_{tag}.json"
    report = json.loads(json_path.read_text())
    arrays = np.load(out_dir / f"robustness_{tag}_windows.npz")
    all_ok = True
    for key in list(report["curves"]):
        fam = key.split("/")[-1]
        st = curve_stats(GRIDS[fam], arrays[key], cliff_bits)
        report["curves"][key] = st
        all_ok &= st["monotone"] and st["no_cliff"]
        print(
            f"  {key:28s} {'monotone' if st['monotone'] else 'NON-MONOTONE'}, "
            f"{'no cliff' if st['no_cliff'] else 'CLIFF'}; first step "
            f"+{st['first_step_rise_bits']:.3f} bits ({st['first_step_share']:.0%} "
            f"of rise {st['total_rise_bits']:.3f})"
        )
    report["all_curves_monotone_no_cliff"] = bool(all_ok)
    report["settings"]["cliff_bits"] = cliff_bits
    json_path.write_text(json.dumps(report, indent=2))
    print(f"re-judged: {json_path}  all ok: {all_ok}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", action="append", default=[], help="name=path (repeatable)")
    p.add_argument("--tag", required=True, help="output name, e.g. phase_a / phase_b")
    p.add_argument("--windows", type=int, default=48, help="per language")
    p.add_argument("--strata", type=int, default=16)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--noise-seed", type=int, default=EVAL_NOISE_SEED)
    p.add_argument("--cliff-bits", type=float, default=1.0)
    p.add_argument(
        "--restat",
        action="store_true",
        help="re-judge an existing --tag report from its saved per-window arrays",
    )
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    root = data_root()
    out_dir = root / "analysis" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.restat:
        restat(root, args.tag, args.cliff_bits)
        return
    if not args.ckpt:
        p.error("--ckpt name=path is required (or --restat)")
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")

    ckpts = parse_ckpts(args.ckpt)
    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    heldout = CorpusWindows(
        corpus_dir,
        {
            lang: [d["doc_id"] for d in sp["heldout"]]
            for lang, sp in splits["languages"].items()
        },
    )
    seq_len = 1024
    src_len = int(seq_len * SOURCE_MARGIN)
    rng = np.random.default_rng(0)
    sources = {
        lang: select_sources(heldout, lang, args.windows, src_len, rng)
        for lang in LANG_TO_INDEX
    }
    variants = {
        lang: build_variants(sources[lang], seq_len, args.noise_seed)
        for lang in LANG_TO_INDEX
    }

    report: dict = {
        "created_utc": datetime.now(UTC).isoformat(),
        "tag": args.tag,
        "settings": {
            "windows_per_language": {l: len(s) for l, s in sources.items()},
            "strata": args.strata,
            "noise_seed": args.noise_seed,
            "seq_len": seq_len,
            "source_margin": SOURCE_MARGIN,
            "grids": GRIDS,
            "cliff_bits": args.cliff_bits,
            "crn": "same noised texts for every checkpoint; masking seed = chunk index for every variant/condition",
        },
        "checkpoints": {},
        "curves": {},
        "controls": {},
        "null_frame": {},
    }
    arrays: dict[str, np.ndarray] = {}
    all_ok = True
    for name, path in ckpts.items():
        model, meta = load_backbone(path, args.device, ema=True)
        report["checkpoints"][name] = meta
        print(f"== {name}: {path.name} step {meta['step']} ({meta['weights']} weights)")
        for lang, li in LANG_TO_INDEX.items():
            clean = None
            for fam, grid in GRIDS.items():
                rows = []
                for sev in grid:
                    rows.append(
                        score(
                            model,
                            variants[lang][fam][sev],
                            li,
                            args.strata,
                            args.device,
                            args.batch,
                        )
                    )
                per_window = np.stack(rows)
                arrays[f"{name}/{lang}/{fam}"] = per_window
                stats = curve_stats(grid, per_window, args.cliff_bits)
                report["curves"][f"{name}/{lang}/{fam}"] = stats
                clean = per_window[0]
                verdict = (
                    ("monotone" if stats["monotone"] else "NON-MONOTONE")
                    + ", "
                    + ("no cliff" if stats["no_cliff"] else "CLIFF")
                )
                all_ok &= stats["monotone"] and stats["no_cliff"]
                curve = "  ".join(
                    f"{s:.2f}:{m:.3f}" for s, m in zip(grid, stats["mean_bits"])
                )
                print(
                    f"  {lang:8s} {fam:13s} {curve}   [{verdict}; first step "
                    f"+{stats['first_step_rise_bits']:.3f} bits = "
                    f"{stats['first_step_share']:.0%} of rise]",
                    flush=True,
                )
            ctrl = {}
            for cname in ("shuffled", "uniform"):
                v = score(
                    model,
                    variants[lang][cname],
                    li,
                    args.strata,
                    args.device,
                    args.batch,
                )
                ctrl[cname] = {
                    "mean_bits": float(v.mean()),
                    "sem_bits": float(v.std(ddof=1) / np.sqrt(len(v))),
                }
            ctrl["clean"] = {
                "mean_bits": float(clean.mean()),
                "sem_bits": float(clean.std(ddof=1) / np.sqrt(len(clean))),
            }
            report["controls"][f"{name}/{lang}"] = ctrl
            print(
                f"  {lang:8s} controls: clean {ctrl['clean']['mean_bits']:.3f}  shuffled {ctrl['shuffled']['mean_bits']:.3f}  uniform {ctrl['uniform']['mean_bits']:.3f}"
            )
            # 2.5 NULL-frame probe
            framed = variants[lang]["framed"]
            pos = score_positions(
                model, framed, li, args.strata, args.device, args.batch
            )
            is_null = framed == NULL_ID
            null_bits = float(pos[is_null].mean())
            letter_bits = float(pos[~is_null].mean())
            per_win_letter = (pos * ~is_null).sum(1) / (~is_null).sum(1)
            nf = {
                "null_slot_bits": null_bits,
                "letter_slot_bits": letter_bits,
                "letter_slot_sem": float(
                    per_win_letter.std(ddof=1) / np.sqrt(len(per_win_letter))
                ),
                "frame_bits_per_slot": float(pos.mean()),
                "null_fraction": float(is_null.mean()),
                "clean_bits_per_char": float(clean.mean()),
                "letter_over_clean": letter_bits / float(clean.mean()),
            }
            report["null_frame"][f"{name}/{lang}"] = nf
            print(
                f"  {lang:8s} NULL frame: null-slot {null_bits:.3f} bits  letter-slot {letter_bits:.3f}"
                f" (clean {clean.mean():.3f}, ratio {nf['letter_over_clean']:.3f})  per-slot {pos.mean():.3f}"
            )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    report["all_curves_monotone_no_cliff"] = bool(all_ok)

    json_path = out_dir / f"robustness_{args.tag}.json"
    json_path.write_text(json.dumps(report, indent=2))
    np.savez_compressed(out_dir / f"robustness_{args.tag}_windows.npz", **arrays)
    print(f"report: {json_path}")

    # plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fams = list(GRIDS)
        fig, axes = plt.subplots(1, len(fams), figsize=(5 * len(fams), 4), sharey=True)
        for ax, fam in zip(axes, fams):
            for name in ckpts:
                for lang in LANG_TO_INDEX:
                    st = report["curves"][f"{name}/{lang}/{fam}"]
                    ax.errorbar(
                        st["severities"],
                        st["mean_bits"],
                        yerr=st["sem_bits"],
                        marker="o",
                        ms=3,
                        label=f"{name} {lang}",
                    )
            ax.set_title(f"{fam} noise")
            ax.set_xlabel("severity")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("held-out NELBO (bits/char, own language)")
        axes[-1].legend(fontsize=7)
        fig.suptitle(f"Robustness curve — {args.tag}")
        fig.tight_layout()
        png = out_dir / f"robustness_{args.tag}.png"
        fig.savefig(png, dpi=120)
        print(f"plot: {png}")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")

    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        cfg = RunConfig(run_name=f"robustness-{args.tag}", phase="phase2")
        task = init_task(cfg, root, tags=["task2.6", "task2.1", args.tag])
        task.connect_configuration(report["settings"], name="settings")
        task.connect_configuration(
            {k: v for k, v in report.items() if k != "curves"}, name="summary"
        )
        logger = task.get_logger()
        for key, st in report["curves"].items():
            name, lang, fam = key.split("/")
            for s, m in zip(st["severities"], st["mean_bits"]):
                logger.report_scalar(
                    f"robustness/{fam}", f"{name}/{lang}", m, round(s * 100)
                )
        for key, nf in report["null_frame"].items():
            logger.report_scalar(
                "null_frame/null_slot_bits", key, nf["null_slot_bits"], 0
            )
            logger.report_scalar(
                "null_frame/letter_slot_bits", key, nf["letter_slot_bits"], 0
            )
        if (out_dir / f"robustness_{args.tag}.png").exists():
            logger.report_image(
                "robustness",
                args.tag,
                local_path=str(out_dir / f"robustness_{args.tag}.png"),
                iteration=0,
            )
        logger.flush(wait=True)
        print(f"ClearML: {task.get_output_log_web_page()}")
        task.close()


if __name__ == "__main__":
    main()
