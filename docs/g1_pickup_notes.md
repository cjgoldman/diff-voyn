# G1 pickup notes — RESOLVED 2026-08-21

Gate G1 **passed**; the full account (plateau via EMA-reset tail, calibration
table v1 and its caveats, per-document Latin heterogeneity) is in
`docs/phase1_status.md` → "Gate G1 — verification" and "G1 verdict".

> **Record status (banner extended 2026-09-01):** interruption record, 2026-08-20/21 (G1 PASS
> 2026-08-21). The artifact locations below are still where those artifacts live; the "Next"
> and "Open follow-ups" items are all done (inline note below). The Phase-A checkpoints named
> here were superseded as the evaluator by `phase_c-85m-seed0` (G4, 2026-08-22).
> **Current project position: `docs/project_status.md`.**

Artifacts:
- Final backbones: `DATA_ROOT/runs/phase_a-{85m,25m}-seed0/ckpt_final.pt`
  (step 23000, EMA decay 0.999 tail); step-20000 originals kept as
  `ckpt_step20000.pt`.
- AR references: `DATA_ROOT/ar_reference/v1/` (6000-step), `v2/` (20000-step
  latin/german + v1 italian) with `summary.json` selection records; logs
  `train_v1.log`, `train_v1_italian_variants.log`, `train_v2.log`.
- Calibration: `DATA_ROOT/calibration/calibration_v1.json` (+ `_windows.npz`
  per-window arrays for the 3.5 audit), `calibration_v1-arv1.json`,
  `calibration_25m-arv2.json`, `calibration_ngram_provisional.json`.
- Reports: `DATA_ROOT/runs/g1_report.json`; ClearML tasks tagged `g1`
  (`2ac6897c…`), `task3.4` (calibration `d8982aff…`, 25M `a1ea5c40…`).

Next: Phase 2 — tasks 2.1 (structured substitution noise), 2.2 (segmentation
noise), 2.3 (transcription noise) as CPU-testable generators with severity
sweeps against the pilot/25M model; G2 anchor = tiled held-out NELBO
(85M: latin 2.3496, italian 2.5538, german 1.8997 bits/char).

Open follow-ups (not blocking): matched-capacity AR reference if the 3.5
audit shows the offset slack matters; per-document scoring (3.3) before any
further Latin-held-out conclusions; the `diff_voyn/data/` package and the
rest of this session's work are uncommitted (see `git status`).
*[All three resolved: the matched-capacity reference became the multilingual AR
reference v3 and the report-only policy (`docs/phase3_status.md` §3.4, 2026-08-21);
per-document scoring is `docs/phase3_status.md` §3.3; the work was committed
2026-08-21 ("phase 1 complete").]*
