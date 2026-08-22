"""Language-ID head evaluation — task 4.2 acceptance ("LID accuracy vs
noise-severity curves") and task 4.3 acceptance ("abstain class triggers on
negative controls at >95%"), re-run at end of Phase C for task 4.5 / G4.

For a (backbone, head) pair:

1. **Severity curves** — per noise family (wrong key / wrong parse /
   transcription, ``diff_voyn.data.noise``) × severity grid × window length
   × language: held-out source windows (evenly spaced over the tiled held-out
   split, identical for every checkpoint = common random numbers across
   models) are corrupted with a fixed noise seed and classified; reported
   are top-1 accuracy (language), abstain rate, and mean true-class
   probability. The NULL-framed clean text is one more point.
2. **Negative controls** — voynichesque (held-out pool), shuffled language
   windows, uniform-random letters, per length: abstain rate.
3. **Decipherment inputs** — the rung-1 decipherments of the 3.6 suite
   (``recovery_solves.json``): the true-hypothesis decipherment (label = the
   cipher's language) and the wrong-hypothesis decipherments (what the head
   says about them: abstain / the hypothesis language / the true language) —
   the head's behaviour on exactly what Phase 5/6 will feed it.

Usage:
    uv run python scripts/lid_eval.py --tag phase_b_85m \\
        --ckpt DATA_ROOT/runs/phase_b-85m-seed0/ckpt_final.pt \\
        --head DATA_ROOT/runs/lid_head-85m-seed0/lid_head_final.pt
    uv run python scripts/lid_eval.py --tag phase_c_85m \\
        --ckpt DATA_ROOT/runs/phase_c-85m-seed0/ckpt_final.pt   # joint: head inside
Writes DATA_ROOT/analysis/phase4/lid_eval_<tag>.{json,md,png}; ClearML tag
``task4.2``.
"""

from __future__ import annotations

import argparse
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
from diff_voyn.data.abstain import (
    ABSTAIN_LABEL,
    load_or_build_voynichesque_pool,
    sample_from_pool,
    shuffled_window,
    uniform_random_letters,
)
from diff_voyn.data.loader import LANG_TO_INDEX, CorpusWindows
from diff_voyn.data.noise import (
    LETTER_BASE,
    P_UNIGRAM_NAIBBE,
    SegmentationNoise,
    SubstitutionNoise,
    TranscriptionNoise,
    frame_with_nulls,
)
from diff_voyn.infra.checkpoint import load_backbone, load_lid_head
from diff_voyn.model.lid_head import LID_CLASSES, predict

LANGS = tuple(LANG_TO_INDEX)
FAMILIES = {
    "substitution": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75],
    "segmentation": [0.0, 0.02, 0.05, 0.1, 0.2, 0.3],
    "transcription": [0.0, 0.02, 0.05, 0.1, 0.2, 0.3],
}
NOISE_SEED = 4242


def apply_family(
    ids: np.ndarray, family: str, severity: float, rng: np.random.Generator
) -> np.ndarray:
    if severity <= 0.0:
        return ids
    if family == "substitution":
        out, _ = SubstitutionNoise(severity)(ids, rng)
    elif family == "segmentation":
        out, _ = SegmentationNoise(severity, P_UNIGRAM_NAIBBE)(ids, rng)
    elif family == "transcription":
        out, _ = TranscriptionNoise(severity)(ids, rng)
    else:
        raise ValueError(family)
    return out


def crop(ids: np.ndarray, length: int) -> np.ndarray:
    if len(ids) < length:
        ids = np.tile(ids, int(np.ceil(length / len(ids))))
    return ids[:length]


def source_windows(
    heldout: CorpusWindows, lang: str, n: int, length: int
) -> np.ndarray:
    """``n`` evenly spaced held-out windows of ``length`` (deterministic)."""
    tiles = heldout.tiled_windows(lang, length)
    idx = np.linspace(0, len(tiles) - 1, n).round().astype(int)
    return tiles[idx]


def summarize(probs: torch.Tensor, label: int) -> dict:
    top = probs.argmax(-1)
    return {
        "n": len(probs),
        "acc": float((top == label).float().mean()),
        "abstain_rate": float((top == ABSTAIN_LABEL).float().mean()),
        "p_true_mean": float(probs[:, label].mean()),
        "pred_hist": {
            c: float((top == i).float().mean()) for i, c in enumerate(LID_CLASSES)
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    p.add_argument("--ckpt", type=Path, required=True, help="backbone checkpoint")
    p.add_argument(
        "--head",
        type=Path,
        default=None,
        help="head checkpoint (default: inside --ckpt)",
    )
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=48, help="windows per language and cell")
    p.add_argument("--lengths", type=int, nargs="+", default=[100, 200, 400, 1024])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--calibrated", action="store_true", help="use the fitted temperature"
    )
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = args.device
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    backbone, bmeta = load_backbone(args.ckpt, device)
    head, hmeta = load_lid_head(args.head or args.ckpt, device)
    print(
        f"backbone {bmeta['run_name']} step {bmeta['step']} ({bmeta['weights']}); "
        f"head {hmeta['run_name']} step {hmeta['step']} ({hmeta['weights']}, "
        f"T={hmeta['temperature']:.3f}, levels {hmeta['lid_head_config']['mask_levels']})"
    )
    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    heldout = CorpusWindows(
        corpus_dir,
        {
            l: [d["doc_id"] for d in sp["heldout"]]
            for l, sp in splits["languages"].items()
        },
    )
    pool = load_or_build_voynichesque_pool(
        root, heldout, "heldout", n_encryptions=120, seed=1
    )
    pred = lambda ids: predict(
        backbone,
        head,
        torch.from_numpy(np.asarray(ids).astype(np.int64)),
        batch=args.batch,
        seed=0,
        device=device,
        calibrated=args.calibrated,
    )
    t0 = time.time()

    # 1. severity curves (+ NULL frame)
    curves: dict = {}
    for L in args.lengths:
        src_len = int(L * 1.6)
        for lang in LANGS:
            li = LANG_TO_INDEX[lang]
            src = source_windows(heldout, lang, args.n, src_len)
            for fi, (fam, grid) in enumerate(FAMILIES.items()):
                for sev in grid:
                    rng = np.random.default_rng([NOISE_SEED, li, L, fi])
                    xs = np.stack(
                        [crop(apply_family(s, fam, sev, rng), L) for s in src]
                    )
                    curves[f"{fam}/L{L}/{lang}/{sev}"] = summarize(pred(xs), li)
            rng = np.random.default_rng([NOISE_SEED, li, L, 999])
            xs = np.stack([crop(frame_with_nulls(s, rng)[0], L) for s in src])
            curves[f"framed/L{L}/{lang}/0.0"] = summarize(pred(xs), li)
        print(f"  curves L={L} done ({time.time()-t0:.0f}s)", flush=True)

    # 2. negative controls
    controls: dict = {}
    for L in args.lengths:
        rng = np.random.default_rng([NOISE_SEED, 7, L])
        n_abs = args.n * len(LANGS)
        sets = {
            "voynichesque": np.stack(
                [sample_from_pool(pool, L, rng) for _ in range(n_abs)]
            ),
            "shuffled": np.stack(
                [
                    shuffled_window(heldout.sample_window(LANGS[i % 3], L, rng), rng)
                    for i in range(n_abs)
                ]
            ),
            "uniform_random": np.stack(
                [uniform_random_letters(L, rng) for _ in range(n_abs)]
            ),
        }
        for name, xs in sets.items():
            controls[f"{name}/L{L}"] = summarize(pred(xs), ABSTAIN_LABEL)
    print(f"  controls done ({time.time()-t0:.0f}s)", flush=True)

    # 3. decipherment inputs (3.6 suite)
    decipher: dict = {}
    solves_path = root / "analysis" / "phase3" / "recovery_solves.json"
    if solves_path.exists():
        inst = json.loads(solves_path.read_text())["instances"]
        cells: dict = {}
        for r in inst:
            cells.setdefault((r["language"], r["length"]), []).append(r)
        for (lang, L), rs in sorted(cells.items()):
            li = LANG_TO_INDEX[lang]
            entry = {"n": len(rs)}
            for hyp in LANGS:
                xs = np.array([r["decipherments"][hyp] for r in rs]) + LETTER_BASE
                s = summarize(pred(xs), li)
                s["pred_hypothesis_rate"] = s["pred_hist"][hyp]
                s["ser_mean"] = float(
                    np.mean(
                        [
                            np.mean(
                                np.asarray(r["decipherments"][hyp])
                                != np.asarray(r["plain_ids"])
                            )
                            for r in rs
                        ]
                    )
                )
                entry[
                    "true_hypothesis" if hyp == lang else f"wrong_hypothesis/{hyp}"
                ] = s
            decipher[f"{lang}/L{L}"] = entry
        print(f"  decipherments done ({time.time()-t0:.0f}s)", flush=True)

    # aggregate acceptances
    long_L = max(args.lengths)
    clean_long = float(
        np.mean([curves[f"substitution/L{long_L}/{l}/0.0"]["acc"] for l in LANGS])
    )
    abstain_rates = {k: v["abstain_rate"] for k, v in controls.items()}
    dec_true = {
        k: v["true_hypothesis"]["acc"]
        for k, v in decipher.items()
        if int(k.split("/L")[1]) >= 200
    }
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "4.2/4.3",
        "tag": args.tag,
        "backbone": bmeta,
        "head": hmeta,
        "settings": {
            "n_per_cell": args.n,
            "lengths": args.lengths,
            "families": FAMILIES,
            "noise_seed": NOISE_SEED,
            "calibrated": args.calibrated,
        },
        "curves": curves,
        "controls": controls,
        "decipherments": decipher,
        "summary": {
            "clean_long_acc": clean_long,
            "abstain_rates_controls": abstain_rates,
            "true_decipherment_acc_ge200": dec_true,
            "true_decipherment_acc_ge200_mean": (
                float(np.mean(list(dec_true.values()))) if dec_true else None
            ),
        },
        "acceptance": {
            "4.1 clean long text > 99%": {
                "value": clean_long,
                "pass": clean_long > 0.99,
            },
            "4.3 abstain on negative controls > 95%": {
                "value": abstain_rates,
                "pass": all(v > 0.95 for v in abstain_rates.values()),
            },
        },
    }
    out_dir = root / "analysis" / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"lid_eval_{args.tag}.json"
    out.write_text(json.dumps(report, indent=1))
    md = render_markdown(report)
    (out_dir / f"lid_eval_{args.tag}.md").write_text(md)
    print(md)
    try:
        plot(report, out_dir / f"lid_eval_{args.tag}.png")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")
    print(f"written {out}")
    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        task = init_task(
            RunConfig(run_name=f"lid-eval-{args.tag}", phase="phase4"),
            root,
            tags=["task4.2", "task4.3"],
        )
        task.connect_configuration(report["summary"], name="summary")
        logger = task.get_logger()
        for k, v in curves.items():
            fam, L, lang, sev = k.split("/")
            logger.report_scalar(
                f"lid_acc_{fam}_{L}", lang, v["acc"], round(float(sev) * 100)
            )
        for k, v in controls.items():
            name, L = k.split("/")
            logger.report_scalar(
                "lid_abstain_rate", name, v["abstain_rate"], int(L[1:])
            )
        logger.flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()


def render_markdown(rep: dict) -> str:
    lines = [
        (
            f"### LID head evaluation — `{rep['tag']}` (backbone step {rep['backbone']['step']}, "
            f"head step {rep['head']['step']}, T={rep['head']['temperature']:.3f})"
        ),
        "",
        "**Severity curves** (top-1 language accuracy; abstain rate in parentheses):",
        "",
    ]
    lengths = rep["settings"]["lengths"]
    for fam, grid in rep["settings"]["families"].items():
        lines.append(f"*{fam}* — columns: severity")
        lines.append("| L / lang | " + " | ".join(str(s) for s in grid) + " |")
        lines.append("|---|" + "---|" * len(grid))
        for L in lengths:
            for lang in LANGS:
                cells = [rep["curves"][f"{fam}/L{L}/{lang}/{s}"] for s in grid]
                lines.append(
                    f"| L{L} {lang} | "
                    + " | ".join(
                        f"{c['acc']:.2f} ({c['abstain_rate']:.2f})" for c in cells
                    )
                    + " |"
                )
        lines.append("")
    lines.append(
        "*NULL frame (clean text on the 2N-slot frame)*: "
        + ", ".join(
            f"L{L} {lang} {rep['curves'][f'framed/L{L}/{lang}/0.0']['acc']:.2f}"
            for L in lengths
            for lang in LANGS
        )
    )
    lines += [
        "",
        "**Negative controls** (abstain rate):",
        "",
        "| control | " + " | ".join(f"L{L}" for L in lengths) + " |",
        "|---|" + "---|" * len(lengths),
    ]
    for name in ("voynichesque", "shuffled", "uniform_random"):
        lines.append(
            f"| {name} | "
            + " | ".join(
                f"{rep['controls'][f'{name}/L{L}']['abstain_rate']:.3f}"
                for L in lengths
            )
            + " |"
        )
    if rep["decipherments"]:
        lines += [
            "",
            (
                "**Rung-1 decipherments (3.6 suite)** — true hypothesis: accuracy (abstain); "
                "wrong hypotheses: abstain / predicted-as-hypothesis / predicted-as-truth"
            ),
            "",
            "| cell | true hyp. | wrong hyps. |",
            "|---|---|---|",
        ]
        for k, v in rep["decipherments"].items():
            t = v["true_hypothesis"]
            wrong = [
                (h.split("/")[1], s)
                for h, s in v.items()
                if h.startswith("wrong_hypothesis")
            ]
            lines.append(
                f"| {k} | {t['acc']:.2f} ({t['abstain_rate']:.2f}) | "
                + "; ".join(
                    f"{h}: {s['abstain_rate']:.2f} / {s['pred_hypothesis_rate']:.2f} / {s['acc']:.2f}"
                    for h, s in wrong
                )
                + " |"
            )
    a = rep["acceptance"]
    lines += ["", "**Acceptance**:", ""]
    for k, v in a.items():
        lines.append(
            f"- {k}: {'PASS' if v['pass'] else 'FAIL'} — {json.dumps(v['value']) if isinstance(v['value'], dict) else f'{v['value']:.4f}'}"
        )
    return "\n".join(lines)


def plot(rep: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = list(rep["settings"]["families"])
    lengths = rep["settings"]["lengths"]
    fig, axes = plt.subplots(1, len(fams) + 1, figsize=(5 * (len(fams) + 1), 4))
    for ax, fam in zip(axes, fams):
        grid = rep["settings"]["families"][fam]
        for L in lengths:
            acc = [
                np.mean([rep["curves"][f"{fam}/L{L}/{l}/{s}"]["acc"] for l in LANGS])
                for s in grid
            ]
            ax.plot(grid, acc, marker="o", label=f"L={L}")
        ax.set_title(f"{fam}: LID accuracy vs severity")
        ax.set_xlabel("severity")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("top-1 language accuracy")
    axes[0].legend()
    ax = axes[-1]
    for name in ("voynichesque", "shuffled", "uniform_random"):
        ax.plot(
            lengths,
            [rep["controls"][f"{name}/L{L}"]["abstain_rate"] for L in lengths],
            marker="s",
            label=name,
        )
    ax.axhline(0.95, ls="--", c="gray")
    ax.set_title("abstain rate on negative controls")
    ax.set_xlabel("window length")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle(rep["tag"])
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
