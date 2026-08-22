"""Robustness / degradation profile of the prototype n-gram language judges
(CH.0 ``NgramLM``), measured on the SAME noised texts as the diffusion
judges' task-2.6 robustness curves — so the two instruments can be compared
window-for-window.

Reuses ``scripts/robustness_curve.py`` verbatim for source selection and
noise generation (48 evenly spaced held-out windows per language, fixed noise
seed ``EVAL_NOISE_SEED``): the uint8 id arrays scored here are bit-identical
to the ones scored by the Phase-A/B checkpoints (common random numbers across
judges). Scored per window: own-language bits/char under each n-gram order
(1, 2, 3 = the DP order, 5 = the anchor order), for every noise family and
severity of the 2.6 grids plus the shuffled / uniform controls.

Reported, mirroring 2.6 so the numbers line up with ``docs/phase2_status.md``:

- the absolute curves (mean ± s.e.m.) with the 2.6 ``curve_stats`` verdicts
  (monotone, no cliff, first-step rise, total rise);
- the clean → 20 %-wrong-key margin (the G2.2 discriminability quantity);
- a **scale-free profile**: the fraction of the clean → shuffled-letters gap
  consumed at each severity, ``(m(s) − m(0)) / (m_shuffled − m(0))``. A
  pentagram LM and a diffusion NELBO have different clean rates and
  different no-structure ceilings, so absolute bits are not comparable
  across instruments; this normalization is. The same profile is computed
  from the saved diffusion per-window arrays (``--diffusion tag=name``).
- **judge accuracy**: at each severity, the fraction of windows whose true
  language ranks first when all three LMs score the noised text with the
  CH.1 calibration offsets (−held-out bits/char of that order). This is the
  property that matters for a *language* judge; the absolute curves only
  say how much a wrong key costs under the true language. The saved
  diffusion arrays hold own-language scores only; ``--rank-ckpt name=path``
  scores the same variants under all three language conditions with a
  diffusion checkpoint (GPU, reduced grid, CRN masking across conditions)
  to draw the diffusion analogue.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/ngram_robustness.py \\
        --diffusion phase_b-85m=85m --diffusion phase_a-85m=85m \\
        --rank-ckpt phase_c-85m=DATA_ROOT/runs/phase_c-85m-seed0/ckpt_final.pt

Writes DATA_ROOT/analysis/phase2/robustness_ngram.{json,_windows.npz,png}.
The n-gram part is CPU only, single-threaded (the trainers own the cores).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import torch

torch.set_num_threads(1)

from robustness_curve import (
    GRIDS,
    SOURCE_MARGIN,
    build_variants,
    curve_stats,
    select_sources,
)

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX, CorpusWindows
from diff_voyn.data.noise import EVAL_NOISE_SEED, LETTER_BASE
from diff_voyn.heads.ngram import NgramLM, lm_dir, load_lm

ORDERS = [1, 2, 3, 5]
MARGIN_SEVERITY = 0.20  # the G2.2 clean→20%-wrong-key margin


def lm_bits(lm: NgramLM, ids_np: np.ndarray, order: int) -> np.ndarray:
    """[n, L] letter ids → [n] bits/char under ``lm`` at ``order``."""
    letters = ids_np.astype(np.int64) - LETTER_BASE
    return np.array([lm.bits_per_char(row, order) for row in letters])


def relative_profile(means: np.ndarray, ceiling: float) -> list[float]:
    """Undefined (empty) when the clean→shuffled gap is negligible — the
    unigram LM, whose rate shuffling cannot change."""
    gap = ceiling - means[0]
    return ((means - means[0]) / gap).tolist() if gap > 0.1 else []


# Reduced grid for the (GPU) diffusion cross-language ranking — a few points
# per family are enough for "where does the judge start misnaming the
# language", and the trainers own the GPUs.
RANK_GRIDS = {
    "substitution": [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0],
    "segmentation": [0.0, 0.10, 0.20, 0.30],
    "transcription": [0.0, 0.05, 0.10, 0.20],
}


def diffusion_ranking(
    ckpt: Path, variants: dict, n_windows: int, strata: int, batch: int, device: str
) -> tuple[dict, dict]:
    """Score each noised variant under ALL three language conditions (CRN:
    masking seed = chunk index, identical across conditions). Returns
    ``{"lang/fam": {severity: [n, 3] raw NELBO bits/char}}`` (columns in
    ``LANG_TO_INDEX`` order) and the checkpoint meta."""
    from robustness_curve import score

    from diff_voyn.infra.checkpoint import load_backbone

    model, meta = load_backbone(ckpt, device, ema=True)
    arrays: dict = {}
    langs = list(LANG_TO_INDEX)
    for lang in langs:
        for fam, grid in RANK_GRIDS.items():
            arrays[f"{lang}/{fam}"] = {}
            for sev in grid:
                ids = variants[lang][fam][sev]
                sel = np.linspace(0, len(ids) - 1, num=min(n_windows, len(ids)))
                ids = ids[np.unique(sel.round().astype(int))]
                arrays[f"{lang}/{fam}"][sev] = np.stack(
                    [
                        score(model, ids, LANG_TO_INDEX[l2], strata, device, batch)
                        for l2 in langs
                    ],
                    1,
                )
            print(f"  scored {lang:8s} {fam}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return arrays, meta


def judge_accuracy(rank_arrays: dict) -> tuple[dict, dict]:
    """``rank_arrays[judge]["lang/fam"][severity] = [n, 3]`` raw bits/char →
    per (judge, lang, fam): top-1 accuracy of the true language under two
    conventions, identical for every judge:

    - *raw*: lowest bits/char wins (the adopted Phase-3 policy for
      decipherment rankings, offsets zero);
    - *excess*: each language's score minus that judge's own clean mean for
      the language (severity 0, same windows) — the CH.1 "−held-out
      bits/char" convention the n-gram judges used for prototyping; without
      it the lowest-entropy language (German) wins any direct text ranking.

    Plus the mean excess margin (runner-up − true; negative = misranked).

    Each judge has ONE meaningful convention. The n-gram judges are three
    separate densities, so only excess bits rank languages (CH.1). The
    diffusion judge is one multilingual density whose language conditioning
    is a ~0.01–0.03 bits/char nudge on the same text; subtracting
    per-language clean means (spread ≈ 0.45 bits) swamps it and the ranking
    becomes "which language has the highest clean rate" — the Phase-3 finding
    (applying AR-gap offsets drops recovery to ~70 %) in another form. The
    plot uses excess for ``ng*`` and raw for diffusion checkpoints."""
    langs = list(LANG_TO_INDEX)
    out, offsets = {}, {}
    for judge, blocks in rank_arrays.items():
        clean = np.array(
            [blocks[f"{l}/substitution"][0.0][:, i].mean() for i, l in enumerate(langs)]
        )
        offsets[judge] = {l: -float(c) for l, c in zip(langs, clean)}
        for key, by_sev in blocks.items():
            lang = key.split("/")[0]
            tc = langs.index(lang)
            sev = sorted(by_sev)
            raw = np.stack([by_sev[s] for s in sev])  # [n_sev, n, 3]
            exc = raw - clean[None, None, :]
            true = exc[..., tc]
            others = np.delete(exc, tc, axis=2).min(2)
            out[f"{judge}/{key}"] = {
                "severities": sev,
                "top1_raw": (raw.argmin(2) == tc).mean(1).tolist(),
                "top1_excess": (exc.argmin(2) == tc).mean(1).tolist(),
                "margin_excess_bits": (others - true).mean(1).tolist(),
                "n_windows": int(raw.shape[1]),
            }
    return out, offsets


def diffusion_profiles(root: Path, items: list[str]) -> dict:
    """Scale-free profiles + margins from saved 2.6 per-window arrays."""
    out = {}
    for it in items:
        tag, _, name = it.partition("=")
        d = root / "analysis" / "phase2"
        rep = json.loads((d / f"robustness_{tag}.json").read_text())
        arr = np.load(d / f"robustness_{tag}_windows.npz")
        for lang in LANG_TO_INDEX:
            ctrl = rep["controls"][f"{name}/{lang}"]
            for fam, grid in GRIDS.items():
                pw = arr[f"{name}/{lang}/{fam}"]
                means = pw.mean(1)
                st = {
                    "severities": grid,
                    "mean_bits": means.tolist(),
                    "clean_bits": float(means[0]),
                    "shuffled_bits": ctrl["shuffled"]["mean_bits"],
                    "uniform_bits": ctrl["uniform"]["mean_bits"],
                    "relative_profile": relative_profile(
                        means, ctrl["shuffled"]["mean_bits"]
                    ),
                    "first_step_rise_bits": float(means[1] - means[0]),
                }
                if fam == "substitution":
                    j = grid.index(MARGIN_SEVERITY)
                    st["margin_20pct_bits"] = float(means[j] - means[0])
                    rp = st["relative_profile"]
                    st["margin_20pct_relative"] = rp[j] if rp else None
                out[f"{tag}/{lang}/{fam}"] = st
    return out


def plot_report(report: dict, args, out_dir: Path) -> None:
    """The 3×3 figure: absolute curves / normalized profiles / judge accuracy."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fams = list(GRIDS)
        fig, axes = plt.subplots(3, len(fams), figsize=(5 * len(fams), 12))
        colors = {1: "#bbbbbb", 2: "#999999", 3: "#e6842a", 5: "#c0392b"}
        dcolors = ["#1f77b4", "#17becf", "#2ca02c"]
        for ci, fam in enumerate(fams):
            ax_abs, ax_rel, ax_acc = axes[0, ci], axes[1, ci], axes[2, ci]
            for k in args.orders:
                for lang, ls in zip(LANG_TO_INDEX, ["-", "--", ":"]):
                    st = report["curves"][f"ng{k}/{lang}/{fam}"]
                    lab = f"ng{k} {lang}"
                    ax_abs.plot(
                        st["severities"],
                        st["mean_bits"],
                        ls,
                        color=colors.get(k, "k"),
                        marker="o",
                        ms=3,
                        label=lab,
                    )
                    if st["relative_profile"]:
                        ax_rel.plot(
                            st["severities"],
                            st["relative_profile"],
                            ls,
                            color=colors.get(k, "k"),
                            marker="o",
                            ms=3,
                            label=lab,
                        )
                    acc = report["judge_accuracy"][f"ng{k}/{lang}/{fam}"]
                    ax_acc.plot(
                        acc["severities"],
                        acc["top1_excess"],
                        ls,
                        color=colors.get(k, "k"),
                        marker="o",
                        ms=3,
                        label=lab,
                    )
            for name in report["diffusion_checkpoints"]:
                for lang, ls in zip(LANG_TO_INDEX, ["-", "--", ":"]):
                    acc = report["judge_accuracy"][f"{name}/{lang}/{fam}"]
                    ax_acc.plot(
                        acc["severities"],
                        acc["top1_raw"],
                        ls,
                        color=dcolors[
                            list(report["diffusion_checkpoints"]).index(name)
                            % len(dcolors)
                        ],
                        marker="s",
                        ms=4,
                        lw=2,
                        label=f"{name} {lang}",
                    )
            dtags = [it.partition("=")[0] for it in args.diffusion]
            for key, st in report["diffusion"].items():
                if not key.endswith(fam):
                    continue
                tag, lang, _ = key.split("/")
                ls = {"latin": "-", "italian": "--", "german": ":"}[lang]
                col = dcolors[dtags.index(tag) % len(dcolors)]
                ax_abs.plot(
                    st["severities"],
                    st["mean_bits"],
                    ls,
                    color=col,
                    marker="s",
                    ms=3,
                    label=f"{tag} {lang}",
                )
                ax_rel.plot(
                    st["severities"],
                    st["relative_profile"],
                    ls,
                    color=col,
                    marker="s",
                    ms=3,
                    label=f"{tag} {lang}",
                )
            ax_abs.set_title(f"{fam} noise — own-language bits/char")
            ax_rel.set_title(f"{fam} — fraction of clean→shuffled gap consumed")
            ax_acc.set_title(f"{fam} — judge top-1 language accuracy")
            for ax in (ax_abs, ax_rel, ax_acc):
                ax.set_xlabel("severity")
                ax.grid(alpha=0.3)
            ax_rel.set_ylim(-0.05, 1.15)
            ax_acc.set_ylim(-0.05, 1.05)
        axes[0, -1].legend(fontsize=6, ncol=2)
        axes[2, -1].legend(fontsize=6, ncol=2, loc="lower left")
        fig.suptitle(
            "n-gram judges vs diffusion judges — robustness (same noised texts)"
        )
        fig.tight_layout()
        png = out_dir / f"robustness_{args.tag}.png"
        fig.savefig(png, dpi=120)
        print(f"plot: {png}")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--windows", type=int, default=48, help="per language (2.6: 48)")
    p.add_argument("--noise-seed", type=int, default=EVAL_NOISE_SEED)
    p.add_argument("--cliff-bits", type=float, default=1.0)
    p.add_argument("--orders", type=int, nargs="+", default=ORDERS)
    p.add_argument(
        "--diffusion",
        action="append",
        default=[],
        help="tag=ckpt-name of a saved 2.6 report to co-plot (repeatable)",
    )
    p.add_argument(
        "--rank-ckpt",
        action="append",
        default=[],
        help="name=path of a diffusion checkpoint to score under all three "
        "language conditions (GPU; reduced grid) for a diffusion judge-accuracy curve",
    )
    p.add_argument("--rank-windows", type=int, default=24)
    p.add_argument("--rank-strata", type=int, default=8)
    p.add_argument("--rank-batch", type=int, default=24)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="ngram")
    p.add_argument(
        "--replot", action="store_true", help="redraw the figure from the saved report"
    )
    args = p.parse_args()
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")

    root = data_root()
    out_dir = root / "analysis" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.replot:
        report = json.loads((out_dir / f"robustness_{args.tag}.json").read_text())
        args.orders = report["settings"]["orders"]
        plot_report(report, args, out_dir)
        return

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

    lms = {lang: load_lm(lm_dir() / f"{lang}.npz") for lang in LANG_TO_INDEX}
    summary = json.loads((lm_dir() / "summary.json").read_text())
    # CH.1 calibration offsets: −held-out bits/char of the scoring order, so
    # cross-language rankings compare excess bits (R1 in n-gram form).
    offsets = {
        k: {lang: -summary[lang]["per_order_bits"][str(k)] for lang in LANG_TO_INDEX}
        for k in args.orders
    }

    report: dict = {
        "created_utc": datetime.now(UTC).isoformat(),
        "tag": args.tag,
        "settings": {
            "windows_per_language": {l: len(s) for l, s in sources.items()},
            "noise_seed": args.noise_seed,
            "seq_len": seq_len,
            "source_margin": SOURCE_MARGIN,
            "grids": GRIDS,
            "cliff_bits": args.cliff_bits,
            "orders": args.orders,
            "ngram_lm_version": lms["latin"].meta.get("ngram_lm_version"),
            "ch1_heldout_offsets_bits": offsets,
            "crn": "identical noised texts to robustness_curve.py (same sources, same noise seed)",
        },
        "curves": {},
        "controls": {},
        "judge_accuracy": {},
        "diffusion": diffusion_profiles(root, args.diffusion),
    }
    arrays: dict[str, np.ndarray] = {}
    rank_arrays: dict[str, dict] = {}

    for k in args.orders:
        print(f"== n-gram order {k}")
        rank_arrays[f"ng{k}"] = {}
        for lang in LANG_TO_INDEX:
            name = f"ng{k}"
            shuffled = lm_bits(lms[lang], variants[lang]["shuffled"], k)
            uniform = lm_bits(lms[lang], variants[lang]["uniform"], k)
            clean = None
            for fam, grid in GRIDS.items():
                # own-language curve + all-language scores for the ranking
                own_rows = []
                rank_arrays[name][f"{lang}/{fam}"] = {}
                for sev in grid:
                    ids = variants[lang][fam][sev]
                    stacked = np.stack(
                        [lm_bits(lms[l2], ids, k) for l2 in LANG_TO_INDEX], 1
                    )
                    rank_arrays[name][f"{lang}/{fam}"][sev] = stacked
                    own_rows.append(stacked[:, list(LANG_TO_INDEX).index(lang)])
                pw = np.stack(own_rows)
                if clean is None:
                    clean = pw[0]
                key = f"{name}/{lang}/{fam}"
                arrays[key] = pw
                st = curve_stats(grid, pw, args.cliff_bits)
                st["clean_bits"] = float(pw[0].mean())
                st["shuffled_bits"] = float(shuffled.mean())
                st["uniform_bits"] = float(uniform.mean())
                st["relative_profile"] = relative_profile(pw.mean(1), shuffled.mean())
                if fam == "substitution":
                    j = grid.index(MARGIN_SEVERITY)
                    st["margin_20pct_bits"] = float((pw[j] - pw[0]).mean())
                    rp = st["relative_profile"]
                    st["margin_20pct_relative"] = rp[j] if rp else None
                report["curves"][key] = st
                print(
                    f"  {lang:8s} {fam:13s} clean {st['clean_bits']:.3f} → "
                    f"{st['mean_bits'][-1]:.3f} (shuffled {st['shuffled_bits']:.3f}); "
                    f"first step +{st['first_step_rise_bits']:.3f} "
                    f"({st['first_step_share']:.0%} of rise); "
                    f"{'monotone' if st['monotone'] else 'NON-MONOTONE'}, "
                    f"{'no cliff' if st['no_cliff'] else 'CLIFF'}"
                )
            report["controls"][f"{name}/{lang}"] = {
                "clean": {
                    "mean_bits": float(clean.mean()),
                    "sem_bits": float(clean.std(ddof=1) / np.sqrt(len(clean))),
                },
                "shuffled": {
                    "mean_bits": float(shuffled.mean()),
                    "sem_bits": float(shuffled.std(ddof=1) / np.sqrt(len(shuffled))),
                },
                "uniform": {
                    "mean_bits": float(uniform.mean()),
                    "sem_bits": float(uniform.std(ddof=1) / np.sqrt(len(uniform))),
                },
            }

    report["settings"]["rank"] = {
        "diffusion_grids": RANK_GRIDS,
        "diffusion_windows": args.rank_windows,
        "diffusion_strata": args.rank_strata,
        "conventions": "raw = lowest bits/char; excess = minus the judge's own clean mean per language",
    }
    report["diffusion_checkpoints"] = {}
    for it in args.rank_ckpt:
        name, _, path = it.partition("=")
        print(f"== diffusion ranking: {name} ({path})")
        rank_arrays[name], report["diffusion_checkpoints"][name] = diffusion_ranking(
            Path(path),
            variants,
            args.rank_windows,
            args.rank_strata,
            args.rank_batch,
            args.device,
        )
    report["judge_accuracy"], report["settings"]["excess_offsets_bits"] = (
        judge_accuracy(rank_arrays)
    )
    for judge, blocks in rank_arrays.items():
        for key, by_sev in blocks.items():
            for sev, v in by_sev.items():
                arrays[f"rank/{judge}/{key}/{sev}"] = v
    for key, st in report["judge_accuracy"].items():
        print(
            f"  {key:32s} top-1 excess "
            + " ".join(f"{v:.2f}" for v in st["top1_excess"])
            + "   raw "
            + " ".join(f"{v:.2f}" for v in st["top1_raw"])
        )

    json_path = out_dir / f"robustness_{args.tag}.json"
    json_path.write_text(json.dumps(report, indent=2))
    np.savez_compressed(out_dir / f"robustness_{args.tag}_windows.npz", **arrays)
    print(f"report: {json_path}")

    plot_report(report, args, out_dir)


if __name__ == "__main__":
    main()
