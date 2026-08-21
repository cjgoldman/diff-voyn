"""Gate G2 verification (end of Phase 2 — noise curriculum).

Gate wording (task breakdown): *noised-input NELBO degrades smoothly, not
catastrophically; clean-text NELBO has not drifted from its G1 value
(calibration anchor intact).* Plus the task acceptance criteria feeding it:
2.4 clean-text NELBO within 1% of Phase A; 2.5 NULL slots in-distribution.

Checks, in order, for each Phase-B model (85M, 25M):

1. **Clean anchor (2.4 / G2):** full tiled held-out NELBO of the Phase-B final
   EMA weights, own-language condition, scored *exactly* as the G1 anchor was
   (``scripts/calibrate.py``: 1024-char tiles, 32 strata, masking seed = chunk
   index, batch 16, bf16) and compared to the G1 value stored in the
   calibration tables (``calibration_v1.json`` for 85M,
   ``calibration_25m-arv2.json`` for 25M). Criterion: |drift| < 1% per
   language (hard). Raw-vs-EMA gap of the final checkpoint is reported (the
   G1 EMA-lag lesson; WARN above 0.5%).
2. **Smooth degradation (2.4 / 2.6 / G2):** from the robustness report
   ``scripts/robustness_curve.py --tag phase_b`` (run first; this script
   refuses to judge without it): every (family × language) curve monotone and
   cliff-free (hard); before/after comparison with the ``phase_a-*`` reports:
   first-step sensitivity and the clean→severity-0.2 margin (the instrument
   must still discriminate: margin ≥ 0.1 bits/char, hard).
3. **NULL slots in-distribution (2.5):** from the same report, on clean text
   laid on the 2N-slot frame: NULL-slot NELBO ≤ 1.5 bits, and the **frame
   overhead per plaintext letter** — total frame bits per letter (all slots,
   NULL included) minus the clean bits/char of the same text — ≤ 1.0 bit
   (hard). The unavoidable floor is the parse-pattern information itself,
   H(0.476) ≈ 1.0 bit per token = 0.656 bits per letter when the
   unigram/bigram pattern is unpredictable; the pattern's cost lands partly on
   the NULL slots and partly on letter slots (a masked slot 2 pays
   −log P(not NULL) before −log P(letter | not NULL)), which is why the
   letter-slot ratio alone is not the right metric — it is still reported.
   In Phase 5 the pattern is the head's own ``w_t`` and identical across
   language hypotheses, so this overhead cancels in the ranking. Phase-A
   values are shown alongside (there the overhead is ~11 bits/letter).
4. **Training-log sanity:** clean canary, noised canary and NULL-slot series
   parsed from ``DATA_ROOT/runs/phase_b-<size>.log`` — start vs end.

Run: ``uv run python scripts/g2_check.py [--no-clearml] [--device cpu]``
Report: ``DATA_ROOT/runs/g2_report.json``; ClearML task tagged ``g2``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX, CorpusWindows
from diff_voyn.infra.checkpoint import load_backbone
from diff_voyn.infra.nelbo import per_window_nelbo_bits

RUNS = {"85m": "phase_b-85m-seed0", "25m": "phase_b-25m-seed0"}
LOGS = {"85m": "phase_b-85m.log", "25m": "phase_b-25m.log"}
ANCHORS = {"85m": "calibration_v1.json", "25m": "calibration_25m-arv2.json"}
ROBUST_A = {"85m": "phase_a-85m", "25m": "phase_a-25m"}

DRIFT_THRESHOLD = 0.01
NULL_SLOT_MAX_BITS = 1.5
FRAME_OVERHEAD_MAX_BITS = 1.0  # per plaintext letter; floor ≈ 0.656 (pattern entropy)
MIN_MARGIN_AT_02 = 0.10
# Monotonicity is judged on the range where the severity scale is a scale of
# *inconsistency*: for substitution noise, severity 1.0 is a fully consistent
# relabeling (every letter remapped, none kept) — a more lawful text than the
# mixed alphabets at 0.5–0.75 — and a noise-trained model may legitimately
# score it lower. The 1.0 point is reported separately with its margin.
MONOTONE_MAX_SEVERITY = {"substitution": 0.75}

CLEAN_RE = re.compile(r"step\s+(\d+)\s+heldout NELBO \(EMA, cond\|uncond\): (.*)$")
NOISE_RE = re.compile(r"step\s+(\d+)\s+heldout noise canary \(EMA\): (.*)$")
LANG_RE = re.compile(r"(\w+) ([\d.]+)\|([\d.]+)u")
NOISE_LANG_RE = re.compile(
    r"(\w+) clean ([\d.]+) noised ([\d.]+) null ([\d.]+)/letter ([\d.]+)"
)

FAILURES: list[str] = []
WARNINGS: list[str] = []


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        (WARNINGS if warn_only else FAILURES).append(name)


@torch.no_grad()
def tiled_nelbo(
    model, ids: torch.Tensor, lang_idx: int, device: str, strata: int, batch: int
):
    """Own-language per-window NELBO on tiled windows, the calibrate.py way
    (seed = chunk index → identical masking draws to the G1 anchor run)."""
    out = np.zeros(len(ids), dtype=np.float64)
    for ci, i in enumerate(range(0, len(ids), batch)):
        chunk = ids[i : i + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out[i : i + len(chunk)] = per_window_nelbo_bits(
                model, chunk, lang_idx, n_strata=strata, seed=ci, device=device
            ).numpy()
    return out


def parse_log(path: Path) -> tuple[dict, dict]:
    clean: dict[int, dict[str, float]] = {}
    noise: dict[int, dict[str, dict[str, float]]] = {}
    if not path.exists():
        return clean, noise
    for line in path.read_text().splitlines():
        m = CLEAN_RE.search(line)
        if m:
            clean[int(m.group(1))] = {
                lang: float(c) for lang, c, _ in LANG_RE.findall(m.group(2))
            }
        m = NOISE_RE.search(line)
        if m:
            noise[int(m.group(1))] = {
                lang: {
                    "clean_ref": float(c),
                    "noised": float(n),
                    "null_slot": float(nu),
                    "letter_slot": float(le),
                }
                for lang, c, n, nu, le in NOISE_LANG_RE.findall(m.group(2))
            }
    return clean, noise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--strata", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-windows", type=int, default=None, help="debug cap")
    p.add_argument("--sizes", nargs="+", default=list(RUNS), choices=list(RUNS))
    args = p.parse_args()
    root = data_root()
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
    report: dict = {"created_utc": datetime.now(UTC).isoformat(), "models": {}}

    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    heldout = CorpusWindows(
        corpus_dir,
        {
            lang: [d["doc_id"] for d in sp["heldout"]]
            for lang, sp in splits["languages"].items()
        },
    )

    for size in args.sizes:
        rep: dict = {}
        report["models"][size] = rep
        ckpt = root / "runs" / RUNS[size] / "ckpt_final.pt"
        if not ckpt.exists():
            check(f"{size}: Phase-B final checkpoint {ckpt}", False, "missing")
            continue
        anchor = json.loads((root / "calibration" / ANCHORS[size]).read_text())
        rep["anchor_source"] = {
            "file": ANCHORS[size],
            "checkpoint": anchor["backbone"]["path"],
            "step": anchor["backbone"]["step"],
        }

        # -------------------------------------------------------------- G2.1
        print(
            f"G2.1 [{size}] clean-text anchor: tiled held-out NELBO vs G1 (<{DRIFT_THRESHOLD:.0%} drift)"
        )
        ema, meta = load_backbone(ckpt, args.device, ema=True)
        raw, _ = load_backbone(ckpt, args.device, ema=False)
        rep["checkpoint"] = meta
        rep["clean_anchor"] = {}
        for lang, li in LANG_TO_INDEX.items():
            ids = torch.from_numpy(
                heldout.tiled_windows(lang, meta["model"]["seq_len"]).astype(np.int64)
            )
            if args.max_windows:
                ids = ids[: args.max_windows]
            v_ema = tiled_nelbo(ema, ids, li, args.device, args.strata, args.batch)
            v_raw = tiled_nelbo(raw, ids, li, args.device, args.strata, args.batch)
            g1 = anchor["languages"][lang]["nelbo_bits"]
            now = float(v_ema.mean())
            drift = (now - g1) / g1
            sem = float(v_ema.std(ddof=1) / np.sqrt(len(v_ema)))
            raw_gap = (float(v_raw.mean()) - now) / now
            check(
                f"{size} {lang}: G1 {g1:.4f} → Phase B {now:.4f} ± {sem:.4f} ({drift:+.2%}, {len(ids)} windows)",
                abs(drift) < DRIFT_THRESHOLD,
            )
            check(
                f"{size} {lang}: raw vs EMA gap {raw_gap:+.2%}",
                abs(raw_gap) < 0.005,
                warn_only=True,
            )
            rep["clean_anchor"][lang] = {
                "g1_bits": g1,
                "phase_b_bits": now,
                "sem": sem,
                "rel_drift": drift,
                "raw_bits": float(v_raw.mean()),
                "raw_vs_ema_rel_gap": raw_gap,
                "n_windows": len(ids),
            }
        del ema, raw
        if args.device == "cuda":
            torch.cuda.empty_cache()

        # -------------------------------------------------------------- G2.2
        print(
            f"G2.2 [{size}] noised-input degradation: monotone, no cliff, still discriminative"
        )
        rb_path = root / "analysis" / "phase2" / f"robustness_phase_b-{size}.json"
        ra_path = root / "analysis" / "phase2" / f"robustness_{ROBUST_A[size]}.json"
        if not rb_path.exists():
            check(
                f"{size}: robustness report {rb_path.name}",
                False,
                "run scripts/robustness_curve.py --tag phase_b-<size> --ckpt <size>=<phase_b ckpt_final>",
            )
            continue
        rb = json.loads(rb_path.read_text())
        ra = json.loads(ra_path.read_text()) if ra_path.exists() else None
        same = (
            rb["checkpoints"][size]["path"] == str(ckpt)
            and rb["checkpoints"][size]["step"] == meta["step"]
        )
        check(
            f"{size}: robustness report scored this checkpoint (step {meta['step']})",
            same,
        )
        rep["robustness"] = {"phase_b": {}, "phase_a": {}}
        for lang in LANG_TO_INDEX:
            for fam in ("substitution", "segmentation", "transcription"):
                st = rb["curves"][f"{size}/{lang}/{fam}"]
                smax = MONOTONE_MAX_SEVERITY.get(fam, 1.0)
                sev = st["severities"]
                core = [i for i in range(len(sev) - 1) if sev[i + 1] <= smax]
                mono = all(
                    st["increment_mean"][i] > -2 * st["increment_sem"][i] for i in core
                )
                st["monotone_core"] = mono
                st["monotone_max_severity"] = smax
                tail = ""
                if smax < sev[-1]:
                    tail = (
                        f"; tail {sev[-1]}: {st['mean_bits'][-1]:.3f} "
                        f"(+{st['mean_bits'][-1] - st['mean_bits'][0]:.3f} over clean)"
                    )
                check(
                    f"{size} {lang} {fam}: monotone(≤{smax})={mono} no_cliff={st['no_cliff']} "
                    f"(first step +{st['first_step_rise_bits']:.3f} bits, rise {st['total_rise_bits']:.3f}){tail}",
                    mono and st["no_cliff"],
                )
                rep["robustness"]["phase_b"][f"{lang}/{fam}"] = st
                if ra is not None:
                    sa = ra["curves"][f"{size}/{lang}/{fam}"]
                    rep["robustness"]["phase_a"][f"{lang}/{fam}"] = sa
                    print(
                        f"        before→after: first step {sa['first_step_rise_bits']:.3f}→{st['first_step_rise_bits']:.3f} bits; "
                        f"rise {sa['total_rise_bits']:.3f}→{st['total_rise_bits']:.3f}"
                    )
            st = rb["curves"][f"{size}/{lang}/substitution"]
            i02 = st["severities"].index(0.2)
            margin = st["mean_bits"][i02] - st["mean_bits"][0]
            check(
                f"{size} {lang}: clean→20%-wrong-key margin {margin:.3f} bits/char (≥ {MIN_MARGIN_AT_02})",
                margin >= MIN_MARGIN_AT_02,
            )
            rep["robustness"][f"margin_at_0.2/{lang}"] = margin
        ctrl = {k: v for k, v in rb["controls"].items() if k.startswith(size)}
        rep["robustness"]["controls"] = ctrl

        # -------------------------------------------------------------- G2.3
        print(f"G2.3 [{size}] NULL slots in-distribution (2N-slot frame on clean text)")
        rep["null_frame"] = {}

        def overhead(nf: dict) -> float:
            per_letter = nf["frame_bits_per_slot"] / (1.0 - nf["null_fraction"])
            return per_letter - nf["clean_bits_per_char"]

        for lang in LANG_TO_INDEX:
            nf = dict(rb["null_frame"][f"{size}/{lang}"])
            nf["frame_overhead_bits_per_letter"] = overhead(nf)
            before = ra["null_frame"][f"{size}/{lang}"] if ra is not None else None
            b = ""
            if before:
                before = dict(before)
                before["frame_overhead_bits_per_letter"] = overhead(before)
                b = (
                    f" (Phase A: NULL {before['null_slot_bits']:.1f} bits, overhead "
                    f"{before['frame_overhead_bits_per_letter']:.1f} bits/letter)"
                )
            check(
                f"{size} {lang}: NULL-slot {nf['null_slot_bits']:.3f} bits (≤ {NULL_SLOT_MAX_BITS}); "
                f"frame overhead {nf['frame_overhead_bits_per_letter']:.3f} bits/letter "
                f"(≤ {FRAME_OVERHEAD_MAX_BITS}; floor 0.656); letter-slot "
                f"{nf['letter_slot_bits']:.3f} = {nf['letter_over_clean']:.2f}× clean{b}",
                nf["null_slot_bits"] <= NULL_SLOT_MAX_BITS
                and nf["frame_overhead_bits_per_letter"] <= FRAME_OVERHEAD_MAX_BITS,
            )
            rep["null_frame"][lang] = {"phase_b": nf, "phase_a": before}

        # -------------------------------------------------------------- G2.4
        print(f"G2.4 [{size}] training-log canaries (start → end)")
        clean, noise = parse_log(root / "runs" / LOGS[size])
        if clean and noise:
            s0, s1 = min(clean), max(clean)
            for lang in LANG_TO_INDEX:
                c0, c1 = clean[s0][lang], clean[s1][lang]
                n0, n1 = noise[min(noise)][lang], noise[max(noise)][lang]
                print(
                    f"  {size} {lang}: clean canary {c0:.3f}→{c1:.3f} (steps {s0}→{s1}); "
                    f"noised {n0['noised']:.3f}→{n1['noised']:.3f}; NULL-slot {n0['null_slot']:.2f}→{n1['null_slot']:.2f}; "
                    f"letter-slot {n0['letter_slot']:.3f}→{n1['letter_slot']:.3f}"
                )
            rep["log"] = {
                "clean_first": clean[s0],
                "clean_last": clean[s1],
                "noise_first": noise[min(noise)],
                "noise_last": noise[max(noise)],
                "steps": [s0, s1],
            }
        else:
            check(
                f"{size}: training log with canaries",
                False,
                "not found / no eval lines",
                warn_only=True,
            )

    (root / "runs" / "g2_report.json").write_text(json.dumps(report, indent=2))
    print(f"report: {root / 'runs' / 'g2_report.json'}")

    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        cfg = RunConfig(run_name="g2-check", phase="phase2")
        task = init_task(cfg, root, tags=["g2"])
        task.connect_configuration(report, name="g2_report")
        logger = task.get_logger()
        for size, rep in report["models"].items():
            for lang, d in rep.get("clean_anchor", {}).items():
                logger.report_scalar(
                    "g2_clean_anchor_drift", f"{size}/{lang}", d["rel_drift"], 0
                )
                logger.report_scalar(
                    "g2_clean_tiled_nelbo", f"{size}/{lang}", d["phase_b_bits"], 0
                )
            for lang, d in rep.get("null_frame", {}).items():
                logger.report_scalar(
                    "g2_null_slot_bits",
                    f"{size}/{lang}",
                    d["phase_b"]["null_slot_bits"],
                    0,
                )
        logger.flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()

    print()
    if WARNINGS:
        print(f"G2: {len(WARNINGS)} warning(s): {WARNINGS}")
    if FAILURES:
        print(f"G2: {len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("G2: all hard checks passed")


if __name__ == "__main__":
    main()
