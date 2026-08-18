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
