# G1 pickup notes (written 2026-08-20, before container restart for GPU recovery)

Both Phase-A runs COMPLETED all 20k steps cleanly before the GPUs went
offline (final checkpoints + manifests in `DATA_ROOT/runs/phase_a-{85m,25m}-seed0/`,
full canary history in `DATA_ROOT/runs/phase_a-{85m,25m}.log`). Final EMA
canary (bits/char, cond|uncond @ step 20000):

- 85M: latin 2.414|2.413, italian 2.697|2.705, german 2.143|2.143
- 25M: latin 2.613|2.619, italian 2.830|2.839, german 2.297|2.304

All three beat the 5-gram AR reference (latin 2.955 / italian 2.842 /
german 2.405, `DATA_ROOT/ngram_lms/v1/summary.json`).

## State of the G1 gate check (in progress, interrupted)

New, **uncommitted** code from this session (git status shows it; /workspace
is a bind mount so it survives restart):

- `diff_voyn/infra/nelbo.py` — added `per_window_nelbo_bits()` (per-window
  scores, CRN; task 3.3 groundwork), exported in `infra/__init__.py`.
- `tests/test_nelbo.py` — two new tests for it (pass; lint clean).
- `scripts/g1_check.py` — full G1 verification, **written but NOT yet run**.
- `docs/phase1_status.md` needs a final G1-verdict section once g1_check runs.

## Next actions after restart (in order)

1. Verify GPUs: `uv run python -c "import torch; print(torch.cuda.is_available())"`.
2. `uv run python scripts/g1_check.py` (auto-uses GPU; add `--no-clearml` to
   dry-run). It performs: plateau check from run logs, raw-vs-EMA scoring of
   final ckpts, interference check, 25M/85M ranking-agreement probe (task
   1.6), writes `DATA_ROOT/calibration/calibration_v1.json` +
   `DATA_ROOT/runs/g1_report.json`, and registers a ClearML task tagged `g1`.
3. **The open G1 judgement call:** the EMA canary had NOT met the <0.5%/1000-step
   plateau criterion at step 20000 (85M latin improved ~2.2% over the last
   800 steps). Working hypothesis: EMA lag (decay 0.9999 ⇒ ~10k-step time
   constant), not genuine model improvement — train NELBO was flat over the
   last 2000 steps. The g1_check raw-vs-EMA section decides it:
   - If raw ≪ EMA (gap ≳1–2%) and raw is flat → residual slope is EMA
     catch-up. Cleanest fix: resume both runs briefly at the LR floor to let
     EMA converge, e.g. `uv run python scripts/train.py --phase phase_a
     --model 85m --resume --steps 24000` (cosine already at its 10% floor;
     scheduler horizon change is cosmetic at the floor). Same for 25m
     (`CUDA_VISIBLE_DEVICES=1`).
   - If raw itself is still improving → not plateaued; resume for longer
     (e.g. --steps 30000) and re-check.
4. Calibration v1 written by g1_check is **provisional** (n-gram reference;
   negative offsets are expected and NOT usable as ranking offsets). Task 3.4
   proper — small per-language char-AR transformers on identical data — still
   to be built/trained (GPU), then calibration v2 replaces it.
5. Update `docs/phase1_status.md` with the G1 verdict; then Phase 2 (noise
   curriculum, tasks 2.1–2.3 generators first — CPU-friendly, unit-testable).

## Cross-session context

- A parallel session advanced the cipher-head track (commits `03af643`,
  `5188ddf`; see `docs/cipher_heads_status.md` and memory
  `cipher-heads-early-track`) and added `Backbone.forward_soft()` (soft
  mixture inputs, design §8). Don't duplicate that work.
- Phase-1 code was committed by that session in `a235206`; only the files
  listed above plus `docs/interim_status_2026-08-19.md`,
  `scripts/doubling_*.py` are uncommitted.
- Watch item: at convergence the 85M conditional and unconditional NELBOs are
  equal to 3 decimals for latin/german (language inferable from content;
  conditioning gain → 0). Expected, but the ranking probe (g1_check G1.3)
  should confirm conditional scoring still separates languages per window.
