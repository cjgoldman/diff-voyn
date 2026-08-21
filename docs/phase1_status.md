# Phase 1 — Backbone pretraining: status

Status record for Phase 1 of the [task breakdown](../reference_docs/Diffusion%20Model%20Training%20-%20Task%20Breakdown.md).
Written 2026-08-18. Companion to `docs/phase0_decisions.md`.

## 1.1 — MDLM diffusion core (DONE)

`diff_voyn/model/diffusion.py`: continuous-time masked (absorbing-state)
diffusion, log-linear schedule (α_t = 1−t, so the masking probability at time
t is exactly t — matching the Phase-0 `MaskingSampler`), SUBS parameterization
(MASK logit −inf, carry-over terms vanish from the loss), Rao-Blackwellized
NELBO with per-draw integrand `(1/t)·Σ_masked −log p_θ(x_i|z_t, L)`,
normalized per character of the window so the training loss is an upper bound
on nats/char.

Acceptance — "loss matches a reference MDLM implementation to numerical
tolerance" (`tests/test_diffusion.py`):

- **Exact reference**: for a tiny backbone and short sequences, the NELBO is
  computed with *no sampling at all* — the inner expectation enumerated over
  all 2^L mask patterns, the time integral by 256-point midpoint quadrature.
  Both the stratified scoring estimator (`infra/nelbo.py`) and a Monte-Carlo
  average of the training-loss integrand match it within 5%.
- **Discrete-time reference**: an independently coded T-step MDLM bound
  converges to the continuous-time value (T=512 within 1%, monotone in T) —
  our loss is the T→∞ limit of the discrete bound, as in the MDLM paper.
- Analytic anchors: zero logits ⇒ exactly log₂(32)=5 bits/char; partial-mask
  weighting exact; unmasked examples contribute exactly 0.

**Open item (recorded, not hidden):** the text8-scale bits/char anchor
(≈1.4–1.5, the D3PM-era number) has not been run — it needs a dedicated
text8 training run on a 27-symbol vocab, which is outside the frozen v1 vocab
and data pipeline. The exact-reference tests plus the in-corpus NELBO
trajectories are the correctness evidence for now; the anchor run can be
scheduled as a side experiment if reviewers want it.

## 1.2 — Encoder backbone (DONE)

`diff_voyn/model/backbone.py`: encoder-only, bidirectional (no causal mask),
RMSNorm pre-norm, SwiGLU, RoPE, no biases, no time conditioning. Presets
(`model_preset` in `infra/config.py`):

| preset | layers | d_model | heads | d_ffn | params |
|---|---|---|---|---|---|
| `85m` | 12 | 768 | 12 | 2048 | 85.4M |
| `25m` | 6 | 512 | 8 | 1408 | 19.3M |

Language conditioning reuses the Phase-0 `LanguageConditioning` (additive
per-position embedding, 10% dropout to the learned NULL-language embedding).
The forward output is SUBS-parameterized at the source, so the training loss,
the canary, and the future Phase-3 harness all see the same distribution.

Acceptance: conditional vs unconditional NELBO differ in the expected
direction on pilot data (see 1.3 numbers below — Δ(uncond−cond) > 0 for every
language, on both EMA and raw weights); realized conditioning-dropout rate is
logged every 50 steps (`train/lang_cond_dropout_rate`, observed 0.09–0.11)
and unit-tested at 0.10±0.02.

## 1.3 — Pilot run (DONE — weights discarded as planned)

Run `pilot-25m-seed0` (ClearML task `477e13510c7745c899673a51130bd82a`,
project `diff-voyn`): 25M preset, 2000 steps × 32×1024 chars, all three
languages jointly, bf16, ~200k chars/s on one RTX 3090, **zero non-finite
losses** (the loop hard-aborts and dumps a debug checkpoint on the first one).

- Train NELBO: 5.0 (random) → 2.97 bits/char.
- Held-out canary (EMA weights, CRN seed 0, logged to ClearML every 250
  steps) declined smoothly for all languages; at step 2000:
  latin 4.06|4.17u, italian 4.16|4.21u, german 4.14|4.15u bits/char
  (cond|uncond). EMA at decay 0.9999 retains ~82% of the random init after
  2000 steps, so the EMA canary lags the model — expected in a short pilot,
  irrelevant at Phase-A length (0.9999^20000 ≈ 0.14).
- `scripts/score_checkpoint.py` scored the checkpoints with no manual steps
  (loads config from the checkpoint, EMA or raw weights, same fixed windows
  and CRN seed as the training canary). Final raw weights:

  | language | cond | uncond | Δ(u−c) |
  |---|---|---|---|
  | latin | 3.187 | 3.201 | +0.014 |
  | italian | 3.242 | 3.341 | +0.099 |
  | german | 2.765 | 2.769 | +0.004 |

## 1.4 — Phase A full pretraining (LAUNCHED 2026-08-18)

Both models train in parallel, one per RTX 3090, design §7.5 settings
(AdamW β₂=0.98, peak LR 3e-4, 2k warmup, cosine→10%, batch 16×32×1024 ≈ 0.5M
chars, dropout 0.1, EMA 0.9999), 20k optimizer steps ≈ 10.5B training chars:

- `phase_a-85m-seed0` — GPU 0, log `DATA_ROOT/runs/phase_a-85m.log`
- `phase_a-25m-seed0` — GPU 1, log `DATA_ROOT/runs/phase_a-25m.log`

Canary + unconditional NELBO log to ClearML every 200 steps; checkpoints to
`DATA_ROOT/runs/<run>/ckpt_last.pt` every 200 steps. Resume after any
interruption with the same command + `--resume`. The G1 plateau criterion
(<0.5% improvement over a trailing window, *every* language, none stalling —
tasks 1.4/1.5) is judged from the ClearML `heldout_nelbo_bits_per_char`
curves; if a language stalls while others improve, adjust the sampling
temperature τ, not the schedule (non-negotiable #3).

## Remaining before Gate G1

- 1.4 plateau + 1.5 interference watch: monitor the two ClearML tasks.
- 1.6 (P2): 25M/85M ranking-agreement probe on a clean-text sample once both
  runs finish.
- Calibration table v1 comes from task 3.4 (Phase 3) — G1 lists it, so the
  small per-language char-AR reference models should be built while Phase A
  trains (metrology must not slip behind training).

## Gate G1 — verification (2026-08-20, in progress; see "G1 verdict" below when final)

Both runs completed 20k steps cleanly (2026-08-19 for 25M, 2026-08-20 for
85M). `scripts/g1_check.py` (dry run, step-20000 checkpoints) gave:

| check | result |
|---|---|
| G1.2 interference (trailing-half improvement, no language < ½ median) | PASS, all 6 cells (85M: latin +34.7%, italian +25.4%, german +38.7%; 25M: +28.5/+24.1/+34.3%) |
| G1.3 ranking agreement 25M vs 85M (24 clean 512-char windows, CRN) | top-1 agreement 100%, full ranking 96%; both models 100% top-1 = true language |
| G1.1 EMA canary plateau (<0.5% over trailing 1000 steps) | **not met at 20k**: +1.3…+2.9% per language on both models |
| G1.1b raw vs EMA on the step-20000 checkpoints | raw **6–14% better** than EMA (85M: latin 2.149 vs 2.420, italian 2.528 vs 2.696, german 1.876 vs 2.138 on the canary windows, 64 strata; 25M: 9–10% gap) |

**Diagnosis.** Train NELBO was flat (2.00–2.02 bits/char) over the last 2000
steps at the LR floor, so the residual canary slope is not model improvement:
it is EMA lag. With decay 0.9999 and no EMA warm-up, the time constant is
10k steps — at step 20000 the shadow still carried 14% weight on the random
init and ~22% on the first 5k (warmup / high-LR) steps. That EMA is not a fit
frozen evaluator; the raw weights are.

**Resolution (the EMA tail).** Rather than re-run 2–3 time constants at the
floor (days, for no model gain), both runs were resumed from step 20000 for
3000 steps with:

- LR held at the cosine floor (3e-5): `--schedule-total 20000 --steps 23000`
  (the cosine horizon is decoupled from the step budget; re-running with a
  longer horizon would otherwise have bumped the LR back to ~5e-5);
- EMA shadow **reset to the raw weights** at step 20000 and decay **0.999**
  for the tail (`--ema-reset --ema-decay 0.999`): 3000 steps = 3 time
  constants, so the final EMA is a proper average of late, floor-LR weights
  with <5% weight on the reset snapshot. The decay change applies to the tail
  only and is recorded in the checkpoint (`extra.schedule`), the run manifest,
  and the ClearML task (tag `resume`). It is symmetric across languages, so
  it touches no R1 fairness property;
- raw-weight canary logged next to the EMA canary every eval (new
  `heldout_nelbo_bits_per_char_raw` series) so EMA lag can never again
  masquerade as training progress;
- data stream re-seeded from the resume step (no replay of the first windows).

Step-20000 artifacts are preserved as `ckpt_step20000.pt` /
`run_manifest_step20000.json` in each run dir. Tail ClearML tasks:
85M `b57b04214cbe4477b84c243534e5fd05`, 25M `8ec5972581f44784bd7597c00d1c0b9b`.
The plateau criterion is then judged on the tail's EMA canary (steps
22000→23000) by `g1_check.py` (hard check; raw canary reported alongside).

Also fixed on the way: `load_checkpoint` crashed when fewer GPUs were visible
at resume than at save (CUDA RNG state list); and `.gitignore`'s bare `data`
pattern had been ignoring the `diff_voyn/data/` package (loader.py was never
committed) — now anchored to `/data/`.

### Calibration table v1 (task 3.4, G1 checklist)

`g1_check.py`'s n-gram table is only a stand-in
(`calibration_ngram_provisional.json`; offsets mix bound gap with n-gram
deficiency and are negative — not usable). Task 3.4 proper is now
implemented: `diff_voyn/model/ar_reference.py` (`CharARLM`: the backbone's
RMSNorm/SwiGLU/RoPE blocks with a causal mask, ~10M params, BOS-shifted so
every one of the 1024 window chars is scored exactly as the NELBO averages
over them), `scripts/train_ar_reference.py` (one model per language on the
backbone's own train split, identical budget; best- and final-held-out
checkpoints kept) and `scripts/calibrate.py` (full tiled held-out split,
NELBO under all language conditions with CRN + unconditional, NLL_AR, paired
offset ± s.e.m. per language, per-window arrays kept for the 3.5 fairness
audit; writes `calibration_v1.json`). Both are unit/smoke tested; training
runs on GPU 1 once the 25M tail frees it, calibration on the 85M tail's final
EMA checkpoint.

## G1 verdict (2026-08-21) — **PASSED**

`scripts/g1_check.py` on the tail-final checkpoints (step 23000; ClearML task
`2ac6897c6c274e7fb75c8639e4b3338d`, tag `g1`; report
`DATA_ROOT/runs/g1_report.json`):

| check | result |
|---|---|
| 1.4 plateau, EMA canary, steps 22000→23000 (<0.5%) | PASS all 6 cells: 85M +0.09/+0.04/+0.16%, 25M +0.08/+0.04/+0.10% (latin/italian/german); raw canary the same within 0.2% |
| raw vs EMA on final checkpoints | gap ≤0.14% everywhere — the EMA is now the model |
| 1.5 interference (trailing-half improvement ≥ ½ median) | PASS; no language stalled (85M +38.5/+26.9/+42.5%, 25M +31.7/+24.6/+35.0% since step 11600) |
| 1.6 ranking agreement, 24 clean 512-char windows, CRN | 85M/25M top-1 agree 96%, full ranking 79%; top-1 = true language 92% (85M) / 96% (25M) |
| calibration table v1 (3.4) | produced, `DATA_ROOT/calibration/calibration_v1.json` (below) |

Final held-out NELBO (EMA, bits/char). Canary = the 8 fixed 1024-char windows
per language used during training; **tiled = the full held-out split cut into
consecutive 1024-char windows (481 / 515 / 799 windows)** — the tiled numbers
are the clean-text anchor that G2 must reproduce ("clean-text NELBO has not
drifted from its G1 value"):

| model | latin canary / tiled | italian canary / tiled | german canary / tiled |
|---|---|---|---|
| 85M (step 23000) | 2.148 / **2.3496 ± 0.014** | 2.516 / **2.5538 ± 0.005** | 1.875 / **1.8997 ± 0.005** |
| 25M (step 23000) | 2.380 / 2.5530 ± 0.013 | 2.670 / 2.7008 ± 0.004 | 2.089 / 2.0964 ± 0.005 |

The canary windows are not representative of the split (Latin canary 2.148 vs
tiled 2.350): future gate comparisons use the tiled set (`CorpusWindows.
tiled_windows`), the canary remains a cheap in-training trend signal.

### Calibration table v1 (task 3.4) and its caveats

Char-AR reference models (`DATA_ROOT/ar_reference/`): `CharARLM` 10.6M
params, causal version of the backbone blocks, one model per language on the
backbone's own train split. v1 budget 6000 steps × 64 × 1024 chars; Italian
(3.6M train chars) overfits that budget (best 2.656 @ step 750, final 3.216),
so its reference was selected by held-out among three candidates (short
1500-step schedule **2.6365**, dropout 0.3 2.655, main run 2.656; recorded in
`v1/summary.json`). Latin and German were compute-limited at 6000 steps, so a
v2 at 20000 steps was trained for them (Latin 2.2589 best @ 14000 of 20000,
German 1.7507 still improving at 20000). The calibration table uses the v2
reference (`calibration_v1.json`; the v1-reference table is kept as
`calibration_v1-arv1.json`):

| lang | NELBO 85M (tiled) | NLL_AR (v2) | **offset = NELBO − NLL_AR** | s.e.m. | same with AR v1 |
|---|---|---|---|---|---|
| latin | 2.3496 | 2.2589 | **+0.0906** | 0.0065 | +0.0042 |
| italian | 2.5538 | 2.6365 | **−0.0827** | 0.0030 | −0.0827 |
| german | 1.8997 | 1.7507 | **+0.1489** | 0.0022 | +0.0260 |

Offset spread 0.23 bits/char (AR v2) vs 0.11 (AR v1). Read honestly:

- `NELBO − NLL_AR` = (bound gap) − (how much better the diffusion model is
  than the AR reference as a density model). With a 10M reference trained on
  ≤1.3B chars against an 85M instrument trained on 10.5B, the second term is
  not negligible and is **reference-dependent** (the offsets moved by +0.09
  and +0.12 bits from AR v1 to v2 for Latin and German). The Italian offset is
  negative because its reference is data-limited, not because the Italian
  bound is tighter. So the offsets are *lower bounds* on the bound gaps, with
  a language-dependent slack that tracks corpus size — exactly the
  correlation the 3.5 fairness audit is meant to catch. They are stored and
  versioned as the design requires, but **must not be read as proof of
  comparable bound tightness**; the G3 synthetic-recovery suite (3.6) is the
  actual test of the calibrated ranking. A matched-capacity reference
  (≥25M-class AR, backbone-scale char budget) is the upgrade path if 3.5 finds
  the slack matters.
- The offsets apply to the **own-language score of a candidate plaintext**
  (Phase 5: each language hypothesis decodes its own plaintext, scored under
  its own condition, then `NELBO − offset_L` compared). They are *not* a
  correction for scoring one fixed text under the three conditioning
  languages — for that probe the between-condition margins are only
  0.02–0.035 bits/char median (conditioning gain → 0 at convergence, the
  watch item from the pickup notes), an order of magnitude below the offsets.
- Per-window LID by conditioning (85M, 1024-char clean windows, uncalibrated):
  latin 87.5%, italian 99.0%, german 98.1% top-1. The Latin misses are
  document-level: `apothecary_ellis_1854` (33/49 windows; mean NELBO 2.98 vs
  2.11 for Roger Bacon's 217 windows), Cato (14/85), Cicero's Aratea (6/28).
  The Latin held-out split is heterogeneous (2.11–2.98 bits/char by document)
  — reinforcing task 3.3's per-document mean + spread reporting.

### EMA lesson recorded for Phases B/C

EMA decay 0.9999 with no warm-up was mismatched to a 20k-step run (time
constant 10k steps; 14% of the shadow was still the random init at 20k).
Any later phase must either warm the EMA (decay ramp), choose the decay from
the planned step count (time constant ≲ 1/5 of the run), or end with an EMA
tail as here. `train.py` now logs the raw canary beside the EMA canary so the
gap is visible on the dashboard.

**→ Phase 2 (noise curriculum) may start: tasks 2.1–2.3 generators first.**
Frozen evaluator candidates: `DATA_ROOT/runs/phase_a-85m-seed0/ckpt_final.pt`
(EMA, step 23000) and the 25M sibling.
