"""Task 4.7 — seed replication: Phase A → B → (head) → C at 25M with extra
seeds, and a ranking-stability report across seeds.

``--run --seed N`` drives the whole chain for one seed as subprocesses,
skipping every stage whose artifact already exists (so seed 0 — the main
25M chain — only gets the stages it lacks, and a killed run resumes from
its ``ckpt_last.pt`` — Phase A's main run and its EMA tail are told apart
by the step recorded in ``ckpt_final.pt``):

    train.py --phase phase_a --model 25m --seed N      (20k steps)
    train.py --phase phase_a ... --resume --steps 23000 --schedule-total 20000
             --ema-reset --ema-decay 0.999             (the G1 EMA tail)
    train.py --phase phase_b --init-from <phase_a final>
    train_lid_head.py --ckpt <phase_b final> --seed N
    train.py --phase phase_c --init-from <phase_b final> --lid-head <head>
    calibrate.py (AR v3, version v3-phase_c-25m-seedN)
    language_recovery.py --stage score (Phase-3 solves, budget 64 × 4 reps)
    lid_eval.py --tag phase_c_25m_seedN

``--report`` collects what exists: per seed the tiled held-out NELBO and
offsets (from the calibration table), the clean-text LID top-1, the 1:1
recovery accuracy per cell, and the pairwise instance-level agreement of
the ELBO winner between seeds (the ranking-stability statistic). Writes
DATA_ROOT/analysis/phase4/seed_replication.{json,md}; ``complete`` is
true once every requested seed has scored the suite.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/seed_replication.py --run --seed 1
    uv run python scripts/seed_replication.py --report --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from diff_voyn.ciphers.external import data_root
from diff_voyn.data.loader import LANG_TO_INDEX
from diff_voyn.metrology import CalibrationTable, rank_languages

LANGS = tuple(LANG_TO_INDEX)
REPO = Path(__file__).resolve().parent.parent


def sh(cmd: list[str], log: Path) -> None:
    print(f"$ {' '.join(cmd)}  (log {log})", flush=True)
    with log.open("a") as f:
        f.write(f"\n== {datetime.now(UTC).isoformat()} $ {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(
            cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=False
        )
    if r.returncode:
        raise SystemExit(f"stage failed ({r.returncode}): {' '.join(cmd)}")


def _ckpt_step(path: Path) -> int:
    import torch

    return int(torch.load(path, map_location="cpu", weights_only=False)["step"])


def run_seed(seed: int, root: Path, out_dir: Path) -> None:
    py = [sys.executable]
    runs = root / "runs"
    log = out_dir / f"seed{seed}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    pa = runs / f"phase_a-25m-seed{seed}"
    pb = runs / f"phase_b-25m-seed{seed}"
    lh = runs / f"lid_head-25m-seed{seed}"
    pc = runs / f"phase_c-25m-seed{seed}"
    train = py + ["scripts/train.py", "--model", "25m", "--seed", str(seed)]
    if not (pa / "ckpt_final.pt").exists():
        if (pa / "ckpt_last.pt").exists():
            sh(train + ["--phase", "phase_a", "--resume"], log)
        else:
            sh(train + ["--phase", "phase_a"], log)
        sh(
            train
            + [
                "--phase",
                "phase_a",
                "--resume",
                "--steps",
                "23000",
                "--schedule-total",
                "20000",
                "--ema-reset",
                "--ema-decay",
                "0.999",
            ],
            log,
        )
    elif _ckpt_step(pa / "ckpt_final.pt") < 23000:  # interrupted EMA tail
        sh(
            train
            + [
                "--phase",
                "phase_a",
                "--resume",
                "--steps",
                "23000",
                "--schedule-total",
                "20000",
                "--ema-reset",
                "--ema-decay",
                "0.999",
            ],
            log,
        )
    if not (pb / "ckpt_final.pt").exists():
        sh(
            train + ["--phase", "phase_b", "--init-from", str(pa / "ckpt_final.pt")],
            log,
        )
    if not (lh / "lid_head_final.pt").exists():
        sh(
            py
            + [
                "scripts/train_lid_head.py",
                "--ckpt",
                str(pb / "ckpt_final.pt"),
                "--seed",
                str(seed),
            ],
            log,
        )
    if not (pc / "ckpt_final.pt").exists():
        sh(
            train
            + [
                "--phase",
                "phase_c",
                "--init-from",
                str(pb / "ckpt_final.pt"),
                "--lid-head",
                str(lh / "lid_head_final.pt"),
                "--lid-batch",
                "12",
            ],
            log,
        )
    version = f"v3-phase_c-25m-seed{seed}" if seed else "v3-phase_c-25m"
    if not CalibrationTable.file_for(version, root).exists():
        sh(
            py
            + [
                "scripts/calibrate.py",
                "--ckpt",
                str(pc / "ckpt_final.pt"),
                "--ar-dir",
                str(root / "ar_reference" / "v3"),
                "--phase",
                "phase_c",
                "--version",
                version,
                "--batch",
                "16",
                "--strata",
                "32",
                "--seed",
                "0",
            ],
            log,
        )
    seed_dir = out_dir / f"seed{seed}"
    if not (seed_dir / "recovery_scores.json").exists():
        sh(
            py
            + [
                "scripts/language_recovery.py",
                "--stage",
                "score",
                "--ckpt",
                str(pc / "ckpt_final.pt"),
                "--budgets",
                "64",
                "--out-dir",
                str(seed_dir),
            ],
            log,
        )
    tag = f"phase_c_25m_seed{seed}" if seed else "phase_c_25m"
    if not (root / "analysis" / "phase4" / f"lid_eval_{tag}.json").exists():
        sh(
            py
            + [
                "scripts/lid_eval.py",
                "--tag",
                tag,
                "--ckpt",
                str(pc / "ckpt_final.pt"),
            ],
            log,
        )
    print(f"seed {seed} complete", flush=True)


def winners(scores_path: Path) -> dict:
    """Uncalibrated (= report-only) ELBO winner per instance."""
    out = {}
    for r in json.loads(scores_path.read_text())["instances"]:
        mean_bits = {h: float(np.mean(v)) for h, v in r["diffusion_bits"].items()}
        out[(r["language"], r["length"], r["trial"])] = rank_languages(mean_bits, {})[
            0
        ][0]
    return out


def report(seeds: list[int], root: Path, out_dir: Path) -> dict:
    per_seed = {}
    win = {}
    for seed in seeds:
        version = f"v3-phase_c-25m-seed{seed}" if seed else "v3-phase_c-25m"
        entry: dict = {
            "seed": seed,
            "calibration": None,
            "recovery": None,
            "lid_eval": None,
        }
        try:
            t = CalibrationTable.load(version, root)
            raw = json.loads(Path(t.path).read_text())
            entry["calibration"] = {
                "version": version,
                "nelbo_bits": t.nelbo_bits,
                "offsets_bits": t.offsets_bits,
                "spread_bits": t.spread_bits,
                "lid_top1_acc": {
                    l: raw["languages"][l].get("lid_top1_acc") for l in LANGS
                },
            }
        except FileNotFoundError:
            pass
        sp = out_dir / f"seed{seed}" / "recovery_scores.json"
        if sp.exists():
            w = winners(sp)
            win[seed] = w
            cells: dict = {}
            for (lang, L, _trial), v in w.items():
                c = cells.setdefault(f"{lang}/L{L}", [0, 0])
                c[0] += int(v == lang)
                c[1] += 1
            entry["recovery"] = {
                "cells": {k: v[0] / v[1] for k, v in cells.items()},
                "ge200_language_acc": float(
                    np.mean(
                        [
                            v[0] / v[1]
                            for k, v in cells.items()
                            if int(k.split("/L")[1]) >= 200
                        ]
                    )
                ),
            }
        tag = f"phase_c_25m_seed{seed}" if seed else "phase_c_25m"
        le = root / "analysis" / "phase4" / f"lid_eval_{tag}.json"
        if le.exists():
            entry["lid_eval"] = json.loads(le.read_text())["summary"]
        per_seed[str(seed)] = entry
    pairs = {}
    done = sorted(win)
    for i, a in enumerate(done):
        for b in done[i + 1 :]:
            keys = [k for k in win[a] if k in win[b]]
            long_keys = [k for k in keys if k[1] >= 200]
            pairs[f"{a}-{b}"] = {
                "n": len(keys),
                "winner_agreement": float(
                    np.mean([win[a][k] == win[b][k] for k in keys])
                ),
                "winner_agreement_ge200": float(
                    np.mean([win[a][k] == win[b][k] for k in long_keys])
                ),
            }
    complete = all(per_seed[str(s)]["recovery"] is not None for s in seeds)
    ge200 = {
        s: e["recovery"]["ge200_language_acc"]
        for s, e in per_seed.items()
        if e["recovery"]
    }
    nelbo = {
        s: e["calibration"]["nelbo_bits"]
        for s, e in per_seed.items()
        if e["calibration"]
    }
    summary = (
        f"seeds scored {sorted(ge200)}: ≥200 language accuracy "
        + ", ".join(f"seed{s} {v:.1%}" for s, v in ge200.items())
        + "; pairwise ≥200 winner agreement "
        + (
            ", ".join(
                f"{k} {v['winner_agreement_ge200']:.1%}" for k, v in pairs.items()
            )
            or "n/a"
        )
        + "; held-out NELBO by seed "
        + "; ".join(
            f"seed{s} " + "/".join(f"{v[l]:.3f}" for l in LANGS)
            for s, v in nelbo.items()
        )
    )
    rep = {
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "4.7",
        "seeds_requested": seeds,
        "complete": complete,
        "per_seed": per_seed,
        "pairwise_winner_agreement": pairs,
        "summary_line": summary,
    }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    (root / "analysis" / "phase4" / "seed_replication.json").write_text(
        json.dumps(rep, indent=1)
    )
    md = [
        "### Seed replication at 25M (task 4.7)",
        "",
        f"Complete: {complete}. {summary}",
        "",
        "| seed | held-out NELBO latin/italian/german | offsets | clean LID top-1 | ≥200 recovery | clean long LID | abstain (voy./shuf. L1024) |",
        "|---|---|---|---|---|---|---|",
    ]
    for s, e in per_seed.items():
        c, r, le = e["calibration"], e["recovery"], e["lid_eval"]
        md.append(
            f"| {s} | "
            + ("/".join(f"{c['nelbo_bits'][l]:.3f}" for l in LANGS) if c else "—")
            + " | "
            + ("/".join(f"{c['offsets_bits'][l]:+.3f}" for l in LANGS) if c else "—")
            + " | "
            + (
                "/".join(f"{(c['lid_top1_acc'][l] or float('nan')):.2f}" for l in LANGS)
                if c
                else "—"
            )
            + " | "
            + (f"{r['ge200_language_acc']:.1%}" if r else "—")
            + " | "
            + (f"{le['clean_long_acc']:.3f}" if le else "—")
            + " | "
            + (
                f"{le['abstain_rates_controls'].get('voynichesque/L1024', float('nan')):.2f}/"
                f"{le['abstain_rates_controls'].get('shuffled/L1024', float('nan')):.2f}"
                if le
                else "—"
            )
            + " |"
        )
    if pairs:
        md += ["", "Pairwise instance-level ELBO-winner agreement:", ""]
        for k, v in pairs.items():
            md.append(
                f"- seeds {k}: {v['winner_agreement']:.1%} all lengths, {v['winner_agreement_ge200']:.1%} at ≥200 (n={v['n']})"
            )
    (root / "analysis" / "phase4" / "seed_replication.md").write_text("\n".join(md))
    print("\n".join(md))
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    root = data_root()
    out_dir = root / "analysis" / "phase4" / "seed_replication"
    if args.run:
        run_seed(args.seed, root, out_dir)
    if args.report or not args.run:
        report(args.seeds, root, out_dir)


if __name__ == "__main__":
    main()
