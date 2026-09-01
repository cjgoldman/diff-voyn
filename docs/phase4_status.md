# Phase 4 — Language-ID head (delayed, then joint): status

> **Record status (banner added 2026-09-01):** Phase 4 / LID head and Phase-C joint fine-tune,
> 2026-08-22 (G4 PASS ~09:00 UTC). **Still current as of 2026-09-01:** the evaluator
> `phase_c-85m-seed0/ckpt_final.pt` and calibration table `v3-phase_c-ro` named in the verdict
> (frozen for Phase 5, unchanged since); the head's role as a short-text cross-check, not a
> wrong-key abstention instrument; task 4.7 (25M seed replication, P2) is **still paused** at
> the checkpoints listed in §4.7. Superseded or discharged items in the carry-overs and the
> post-hoc assessment carry inline notes. **Current project position:
> `docs/project_status.md`.**

Status record for Phase 4 of the [task breakdown](../reference_docs/Diffusion%20Model%20Training%20-%20Task%20Breakdown.md)
(design §6, §7.2; requirement R1 via the G4 canary). Started 2026-08-22 after
Gate G3 (`docs/phase3_status.md`). Code: `diff_voyn/model/lid_head.py`,
`diff_voyn/data/abstain.py`, `Backbone.hidden()` in `diff_voyn/model/backbone.py`,
checkpoint keys `lid_head` / `lid_ema` (`diff_voyn/infra/checkpoint.py`,
`load_lid_head`); scripts named per task below; artifacts under
`DATA_ROOT/analysis/phase4/` and `DATA_ROOT/runs/{lid_head,phase_c}-*`;
ClearML project `diff-voyn`, tags `task4.x`, `g4`.

## 4.1 — Head architecture (DONE)

`LIDHead` (`diff_voyn/model/lid_head.py`): mean-pool of the backbone's
final-layer hidden states (after the final RMSNorm, `Backbone.hidden()` —
no new backbone parameters, every earlier checkpoint still loads) computed
at masking levels **(0, 0.15, 0.30, 0.50)** with common random numbers and
averaged → LayerNorm → Linear(d, 512) → GELU → Dropout(0.1) → Linear(512, 4)
→ softmax over {latin, italian, german, **abstain**}. 266k parameters on the
25M backbone (d = 512), 397k on the 85M (d = 768). A `log_temperature`
buffer (task 4.6) divides the logits at inference.

Decisions recorded here:

- **The feature pass is unconditional** (NULL-language embedding). A head
  fed features computed under the true language's conditioning embedding
  would read the label off the conditioning signal, not the text — and at
  deployment the language is the unknown. The head classifies *text*; the
  per-language ELBO classifies *(text, condition)* pairs. This is what
  makes the head a cross-check rather than a restatement of the ELBO.
- **Masking levels** are the same at training and inference (no
  train/test feature shift); level 0 is included because it carries the
  most evidence and the backbone sees near-unmasked inputs all through
  training (t → 0).
- **Stop-gradient switch**: `pooled_features(..., stop_gradient=True)`
  computes the backbone pass under `torch.no_grad` and detaches — the
  backbone gradient is *exactly* zero, not merely unused
  (`tests/test_lid_head.py::test_stop_gradient_leaves_backbone_grads_exactly_zero`;
  the released mode is tested to reach the first attention layer).
- **λ schedule** `lambda_schedule(step, ramp, cap)`: linear 0 → cap; the
  cap is what the Phase-C guards halve.

Acceptance: trains to >99% on clean long text — see 4.2 (100.0% at 1024
chars for both sizes); stop-gradient verified by the test above.

## 4.3 — Abstain-class data (DONE, one addition to the design's list)

`diff_voyn/data/abstain.py`. Negative controls for the fourth class:

- **voynichesque** — the pinned `voynichesque.py` (greshko/naibbe-cipher @
  df3d074, wrapped in `diff_voyn.ciphers.controls.Voynichesque`). Its
  EVA-style glyph stream is ordinary Latin letters, so after the shared
  normalizer it is in-vocabulary text (18 distinct letters) — it has to be
  rejected on *structure*. Pools are generated once and cached
  (`DATA_ROOT/abstain/voynichesque_v1_{train,heldout}.npz`): 400 encryptions
  of 600-char train-split windows (content destroyed by construction) /
  120 of held-out windows; the upstream parameter sampler is infeasible for
  ~24% of seeds (“not enough options for alphabet 1”), those seeds are
  skipped and counted (125 of 525 attempts for the train pool).
- **shuffled** — a language window with its letters permuted (unigram
  statistics of a real language, no sequential structure).
- **uniform** (*Phase-4 addition, 15% of abstain examples*) — i.i.d.
  uniform letters. The first Phase-B head, trained on exactly the design's
  two controls, abstained on voynichesque and shuffled text at 97–100% but
  labelled uniform-random letters as a language **100% of the time** at
  every length (`analysis/phase4/lid_eval_phase_b_25m_mixv1.md`, control
  `uniform_random`: abstain 0.7% at L100, 0.0% at L200–1024). Both design
  controls keep unigram structure, so "no structure at all" was simply out
  of distribution. A small share of uniform noise fixes this without
  changing what the class is for (structured non-language); the mix is
  versioned in every run manifest (`LIDDataConfig`).

Training stream (`LIDExampleStream`): 75% language windows drawn with the
τ-balanced language sampler and corrupted by the Phase-B
`NoiseMixture` (clean / noised / NULL-framed / both, i.e. "the same
corruption distribution as deployment", design §6), labelled with their
language; 25% abstain (45% voynichesque / 40% shuffled / 15% uniform).
Window lengths 128 / 256 / 512 / 1024 (20/25/25/30%), one length per batch.

## 4.2 — Phase-B attachment behind the stop-gradient (DONE)

`scripts/train_lid_head.py`: frozen Phase-B EMA backbone (`requires_grad`
off, eval mode), head only, 3000 steps × 32 windows, AdamW 1e-3 cosine,
head EMA 0.999. Held-out canary every 250 steps on fixed sets (clean /
noised / framed × L ∈ {128, 256, 1024}, voynichesque / shuffled / uniform).

Results (EMA head, held-out canary sets, 32 windows per language; ClearML
`lid_head-{85m,25m}-seed0`, tags `task4.2 task4.3`):

| head | clean L128 / L256 / L1024 | noised (20% key + 5% + 5%) L128 / L256 / L1024 | framed L1024 | voynichesque / shuffled / uniform (L1024) |
|---|---|---|---|---|
| 85M | 0.99 / 1.00 / **1.00** | 0.90 / 1.00 / 0.99 | 1.00 | 1.00 / 1.00 / 1.00 |
| 25M | 1.00 / 1.00 / **1.00** | 0.84 / 0.89 / 0.96 | 1.00 | 1.00 / 1.00 / 1.00 |

Both heads reach 99–100% train accuracy within ~300 steps — the "nearly
linear problem" of design §7.2 — and the acceptances hold: 4.1 clean long
text 100% (both), 4.3 abstain >95% on every control at every length
(lowest: 85M shuffled/uniform at L100, 97.9%).

`scripts/lid_eval.py` produces the acceptance curves: per noise family ×
severity × length × language on evenly spaced held-out windows (fixed
noise seed → identical inputs for every checkpoint), the NULL-frame point,
the negative controls, and the head's behaviour on the rung-1
decipherments of the 3.6 suite.

Severity curves (`analysis/phase4/lid_eval_phase_b_{85m,25m}.{md,png}`;
top-1 language accuracy averaged over the three languages, 48 held-out
windows per language and cell):

| head | L | wrong key 20% / 30% / 50% / 75% | parse edits 10% / 20% / 30% | transcription 10% / 20% / 30% | NULL frame |
|---|---|---|---|---|---|
| 85M | 200 | 0.99 / 0.95 / 0.78 / 0.53 | 1.00 / 0.99 / 0.84 | 1.00 / 0.99 / 0.90 | 1.00 |
| 85M | 1024 | 1.00 / 1.00 / 0.90 / 0.59 | 1.00 / 1.00 / 0.97 | 1.00 / 1.00 / 0.96 | 1.00 |
| 25M | 200 | 0.91 / 0.83 / 0.71 / 0.50 | 1.00 / 0.97 / 0.78 | 1.00 / 0.97 / 0.81 | 0.99 |
| 25M | 1024 | 1.00 / 0.92 / 0.76 / 0.50 | 1.00 / 0.99 / 0.76 | 1.00 / 1.00 / 0.83 | 1.00 |

Degradation is graceful and monotone in severity at every length; the
85M head is uniformly more robust than the 25M one (the features, not the
head, carry the robustness — same head architecture and data). Italian is
the language lost first under a wrong key (at 50% wrong key and L1024:
Italian 0.73, Latin 0.96, German 1.00 for the 85M): the Romance pair is
the close one, as in Phase 3.

Behaviour on the rung-1 decipherments of the 3.6 suite (the inputs Phase
5/6 will actually feed it): on the *true-hypothesis* decipherment the 85M
head recovers the language at 100% for L ≥ 100 (98% at latin/L700), 90–98%
at L = 50. On *wrong-hypothesis* decipherments it does **not** abstain
(2–18%): a wrong-hypothesis solve of a German cipher under the Latin
hypothesis is, by construction, a maximally Latin-like letter stream (the
n-gram inner search made it so), and the head calls it Latin 58–70% of the
time; and at L ≥ 200 a wrong-hypothesis solve of a Latin/Italian cipher
mostly *is* the true text (the Phase-3 finding), which the head labels
with the truth 72–100%. The head is therefore a check on "which language
is this text", not an abstention instrument for wrong keys — that remains
the ELBO margin (a wrong decipherment sits 1–2 bits/char above the true
one and only 0.5–1.9 below its shuffled text, `docs/phase3_status.md`).

## 4.4 / 4.5 — Phase C joint fine-tune and canary (DONE)

`scripts/train.py --phase phase_c --init-from <Phase-B final> --lid-head
<Phase-B head>`: `L = L_NELBO + λ·L_LID`, 2000 steps at the Phase-B batch
(16 × 32 × 1024 chars), peak LR 3e-5 (≈3× the Phase-B floor) cosine to
3e-6, EMA 0.9975 (time constant 400 steps ≤ 1/5 of the run — the G1
lesson), **the Phase-B noise mixture and clean fraction retained**
(design §7.3). The LID term has its own batch stream (12 windows per
optimizer step, the 4.3 mix); abstain text never reaches the diffusion
loss. Implementation of the joint gradient: the head always sees the
unscaled features and loss, the backbone receives the LID gradient scaled
by λ (`feats = λ·f + (1−λ)·stop_grad(f)`), so at λ = 0 the backbone
gradient is exactly zero and the head keeps training.

Guards (both logged to ClearML `lid/*`, `phase_c_canary_degradation/*`
and recorded in `<run_dir>/lambda_schedule.json`):

1. **Gradient-norm rule** (design §7.2: LID gradient under ~10% of the
   diffusion gradient): the LID backward runs first in each accumulation
   window so its backbone gradient is measured alone; every 50 steps the
   diffusion gradient is obtained by difference and the ratio
   ‖λ∇L_LID‖/‖∇L_NELBO‖ is logged; above 0.10 the λ cap is halved.
2. **Canary abort rule** (task 4.5): the per-language held-out NELBO (EMA
   *and* raw weights, same windows and masking seed as the Phase-B canary)
   is compared with its value at the start of Phase C every 100 steps; a
   relative degradation above 1% on any language halves the λ cap.

**Runs** (`DATA_ROOT/runs/phase_c-{85m,25m}-seed0`, ClearML tag `task4.4`;
85M 5.5 h on one 3090 at 58k chars/s — the LID term costs ~8%):

| | 85M | 25M |
|---|---|---|
| λ cap halvings (grad-ratio rule) | steps 250, 300, 500, 700 → cap **0.0031** | steps 150, 300, 450, 700 → cap **0.0031** |
| final λ; last-10 window ratios | 0.0031; 0.017–0.061 | 0.0031; 0.035–0.064 |
| canary breaches (>1%) | **none** (worst mid-run +0.48% italian, end +0.06 / +0.24 / +0.00%) | **none** (end +0.02 / +0.00 / −0.03%) |
| tiled held-out NELBO, Phase B → C (latin / italian / german) | 2.3554 → 2.3559 (+0.02%) / 2.5584 → 2.5635 (+0.20%) / 1.9077 → 1.9073 (−0.02%) | 2.5573 → 2.5571 / 2.7168 → 2.7177 / 2.1109 → 2.1100 (all < 0.04%) |
| calibration offsets, Phase B → C | +0.138 / +0.008 / +0.206 → +0.138 / +0.013 / +0.205 (spread 0.198 → 0.193) | +0.340 / +0.166 / +0.409 → +0.340 / +0.167 / +0.408 |
| held-out LID (EMA) at end, clean / noised / abstain, L1024 | 1.00 / 1.00 / 1.00 | 1.00 / 0.96 / 1.00 |

**Finding — the 10% rule sets λ, not the "~0.05" default.** With a
12-window LID batch the backbone gradient of the LID term is ~4–7× the
diffusion gradient's norm at equal weight, so the ratio crosses 10% at
λ ≈ 0.007 and the rule halves the cap four times on both sizes; the joint
phase ends at λ = 0.0031 (≈1/16 of the design's informed default) with the
LID gradient at 2–6% of the diffusion gradient. The ratio is judged on
50-step window means: the single-step snapshot ratio (also logged) spikes
to 0.3–0.8 on hard LID batches and would otherwise drive λ to zero.

**Finding — joint training changed nothing measurable in the instrument.**
The canary never moved more than 0.5% on either size; the full tiled
held-out NELBO moved by < 0.2% per language (4.5 criterion 1%) and the
calibration offsets by ≤ 0.005 bits/char. What the head gained from the
released gradient is small but positive: mean Δ accuracy over the 4.2
severity grid **+0.3 pp** (85M; largest gains segmentation 30% at L400
latin +23 pp, transcription 30% at L1024 latin +12 pp; largest losses
segmentation 30% at L100/L200 italian −8 to −12 pp) and +0.8 pp (25M);
clean long text 100%, abstain controls ≥ 95.1% at L100 and 100% at
L ≥ 200 (85M; the 25M joint head dips to 92–94% on shuffled/uniform at
L100 — below its shortest training length of 128 — and stays ≥ 96.5% at
L ≥ 200; `lid_eval_phase_c_{85m,25m}.md`).

**Synthetic ranking end-B → end-C (task 4.5 / G4)** — the 3.6 suite
re-scored with the Phase-C 85M weights on the Phase-3 solves (budget 64 ×
4 replicate seeds; `analysis/phase4/recovery_{scores,report}.json`,
ranking under `v3-phase_c-ro`, report-only like `v3-ro`):

| cell | end-B | end-C | | cell | end-B | end-C |
|---|---|---|---|---|---|---|
| german L50 / L100 | 78% / 94% | 78% / 94% | | german L200 / L400 / L700 | 100 / 100 / 100% | 100 / 100 / 100% |
| italian L50 / L100 | 90% / 100% | 88% / 100% | | italian L200 / L400 / L700 | 100 / 100 / 100% | 100 / 100 / 100% |
| latin L50 / L100 | 58% / 88% | 56% / 88% | | latin L200 / L400 / L700 | 94 / 96 / 96% | 94 / **98 / 98%** |

≥ 200 chars: language **98.4% → 98.9%**, family 98.7% → 99.1% (bar 97.1%).
11 of 15 cells are identical; the 4 changed cells moved by one instance
each. At the instance level 6 of 750 winners changed (2 of 450 at ≥ 200
chars) and every one of them is a same-text near-tie (margin below the
calibration's precision in one or both phases) — no change with a resolved
margin, so nothing to flag as a framework red flag. The replicate
flip-rates are unchanged (latin L700 4.3% → 3.3%, still the same-text
near-tie floor).

## 4.6 — Head calibration and head-vs-ELBO agreement (DONE)

`scripts/head_calibration.py`: temperature fitted by NLL on the
true-hypothesis rung-1 decipherments of the 3.6 suite (held-out plaintext)
plus a shuffled copy of each plaintext (abstain), even trials fit / odd
trials test; written to `<run_dir>/lid_head_calibrated.pt` (the joint
checkpoint is never edited). Agreement matrix: head class × ELBO winner
under the adopted table, per length.

Results (`analysis/phase4/head_calibration.{json,md}`; 750 true
decipherments + 750 shuffled controls, even/odd trials):

- **T = 1.648** (the Phase-B head is over-confident): test NLL 0.111 →
  0.096, test ECE 0.017 → 0.010, accuracy 0.967 (the residual errors are
  the L = 50 decipherments, 85% — where the rung-1 solve itself is poor).
  Per length the single temperature trades off: it fixes the L50 ECE
  (0.088 → 0.043) and makes the long inputs slightly *under*-confident
  (L200–700 ECE 0.001–0.004 → 0.004–0.014, where the head is 100% right).
  A length-dependent temperature would be the next refinement if the head
  is ever used quantitatively; the calibrated head lives in
  `runs/phase_c-85m-seed0/lid_head_calibrated.pt`.
  *[Still not done as of 2026-09-01 (`docs/project_status.md` §6). The head was never
  used quantitatively; the 2026-08-29 learned-judge probe confirmed that its abstain
  class never fires on wrong-key decodes (memory `learned-judge-probe`,
  `docs/judge_alternatives.md`), consistent with the 4.2 finding above.]*
- **Head vs ELBO agreement** on the true-hypothesis decipherments
  (ELBO winner under `v3-phase_c-ro`):

| L | 50 | 100 | 200 | 400 | 700 | all |
|---|---|---|---|---|---|---|
| agreement | 0.733 | 0.940 | 0.973 | 0.993 | 0.993 | 0.927 |
| head right / ELBO right | 0.967 / 0.740 | 1.000 / 0.940 | 0.993 / 0.980 | 1.000 / 0.993 | 1.000 / 0.993 | 0.992 / 0.929 |

  Above 200 chars the two rankings agree at 97–99% and disagree only on
  the Latin near-ties (the ELBO's 2–6 Latin misses per cell, which the
  head gets right: rows of the matrix show the ELBO naming *italian* for
  20 head-latin instances and *german* for 11). Below 100 chars the head is
  clearly the stronger classifier (96.7% vs 74.0% at L50) — it pools
  evidence over a whole window while the ELBO margin on 50 chars is a few
  bits. The head abstained on 3 of 750 true decipherments (all L = 50).
  This is the diagnostic design §6 asks for: the head corroborates the
  ELBO where the ELBO is resolvable and is the better short-text
  cross-check where it is not; the ELBO remains the primary metric.

## 4.7 — Seed replication at 25M (P2, PAUSED — resumable)

`scripts/seed_replication.py --run --seed N` drives Phase A (20k + the
3k EMA tail) → B → head → C → calibration (`v3-phase_c-25m-seedN`) →
recovery scoring → `lid_eval` for one seed, skipping stages whose
artifacts exist and resuming interrupted training from `ckpt_last.pt`;
`--report` aggregates whatever seeds have finished into
`analysis/phase4/seed_replication.{json,md}` (held-out NELBO and offsets
per seed, recovery accuracy per cell, pairwise instance-level agreement of
the ELBO winner).

State at close-out (2026-08-22 19:30 UTC): seed 0 (the main 25M chain) is
scored — ≥ 200 recovery 98.9%, held-out NELBO 2.557 / 2.718 / 2.110,
offsets +0.340 / +0.167 / +0.408, clean LID top-1 0.97 / 1.00 / 1.00.
Seeds 1 and 2 were **paused** to free both GPUs for Phase 5 (decision:
reach the remaining high-risk questions — search fairness — sooner):

| seed | paused at | remaining |
|---|---|---|
| 1 | Phase A EMA tail, step 20 200 / 23 000 (`runs/phase_a-25m-seed1/ckpt_last.pt`) | ~2 800 tail steps (~2 h), then B (~4.5 h), head, C (~2 h), scoring |
| 2 | Phase A, step 14 200 / 20 000 (`runs/phase_a-25m-seed2/ckpt_last.pt`) | ~5 800 + 3 000 steps (~6.5 h), then B, head, C, scoring |

Resume (one GPU each, any time; every stage is idempotent):

```sh
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts/seed_replication.py --run --seed 1 > DATA_ROOT/analysis/phase4/seed_replication_seed1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts/seed_replication.py --run --seed 2 > DATA_ROOT/analysis/phase4/seed_replication_seed2.log 2>&1 &
uv run python scripts/seed_replication.py --report && uv run python scripts/g4_check.py   # clears the 4.7 WARN
```

Nothing in the G4 gate wording depends on 4.7 (P2); the ranking-stability
statistic it will add is an error bar for the Phase-6 report, not a
precondition for Phase 5.
*[Status check 2026-09-01: still paused exactly as tabled — `analysis/phase4/
seed_replication.md` reads "Complete: False, seeds scored ['0']", the seed-1/2
`ckpt_last.pt` files are dated 2026-08-22, no `phase_b/phase_c-25m-seed{1,2}` runs
exist. Phase 6 (2026-08-24) reported without the 4.7 error bar.
`docs/project_status.md` §6.]*

## Gate G4 — verdict: **PASS** (2026-08-22 ~09:00 UTC; 4.7 paused, resumable)

`scripts/g4_check.py` → `DATA_ROOT/runs/g4_report.json`, ClearML tag `g4`
(2026-08-22 ~09:00 UTC, `CALIBRATION_VERSION = "v3-phase_c-ro"`):

| check | status | value |
|---|---|---|
| 4.1 head > 99% on clean long text; stop-gradient verified | PASS | 100.0%; `test_stop_gradient_leaves_backbone_grads_exactly_zero` |
| 4.2 head converged, severity curves produced | PASS | 20% wrong key: L100 0.97, L200 0.99, L400 0.99, L1024 1.00 |
| 4.3 abstain > 95% on negative controls (Phase-B head / joint model) | PASS / PASS | lowest 97.9% / 95.1% (L100 uniform); 100% at L ≥ 200 |
| 4.4 joint fine-tune finished, λ schedule logged; LID grad < 10% | PASS / PASS | 40 log points, 4 halvings, λ 0.0031; last-10 ratios 0.017–0.061 |
| 4.4 Phase-C calibration table produced and adopted | PASS | `v3-phase_c-ro` (report-only) |
| 4.5 / G4 per-language held-out NELBO not degraded > 1% | PASS | +0.02 / +0.20 / −0.02% |
| 4.5 in-run canary | PASS | 0 breaches |
| G4 synthetic ranking ≥ 200 chars unchanged or improved | PASS | 98.4% → 98.9% language, 98.7% → 99.1% family |
| 4.5 per-cell / per-instance changes | PASS | 11/15 cells identical; 6/750 winners changed, all same-text near-ties; no red flag |
| 4.6 head calibrated; agreement matrix | PASS | T = 1.648; agreement ≥ 200 chars 97–99% |
| G4 head and ELBO rankings agree on clean synthetics (≥ 95% at ≥ 200) | PASS | 0.973 / 0.993 / 0.993 |
| audit table == applied table == recovery primary; no un-escalated finding | PASS | `v3-phase_c-ro`; escalated: 3 × reference-dependence, language-dependence (`docs/phase4_fairness_audit.md`) |
| 4.7 seed replication (P2) | WARN | seeds 1, 2 paused at resumable checkpoints (§4.7) |

**Verdict: PASS.** The evaluator for Phase 5 is the joint Phase-C 85M EMA
checkpoint `DATA_ROOT/runs/phase_c-85m-seed0/ckpt_final.pt` (head inside;
calibrated head in `lid_head_calibrated.pt`), with the 25M sibling
`phase_c-25m-seed0` for restart-heavy search.

### Carry-overs

- **Phase 5 (frozen evaluator)**: freeze `phase_c-85m-seed0/ckpt_final.pt`
  (EMA) — `load_backbone` ignores the head keys, so every Phase-3 consumer
  works unchanged. `CALIBRATION_VERSION = "v3-phase_c-ro"`; margin
  uncertainty latin–italian 0.125, latin–german 0.067, italian–german
  0.192 bits/char.
- **Head usage**: `load_lid_head(<run>/lid_head_calibrated.pt)` +
  `lid_head.predict(..., calibrated=True)`. Use it as the short-text
  cross-check and the "is this any trained language" channel for
  voynichesque-like input; do **not** use it to reject wrong-key
  decipherments (4.2 finding) — that is the ELBO's shuffled-text margin.
  Below 128 chars its abstain rate on negative controls falls to ~95%;
  a length-dependent temperature is the next refinement.
  *[Not done as of 2026-09-01; see the note under §4.6.]*
- **λ for any future joint phase**: start the cap at 0.006, not 0.05 — the
  10% rule lands at 0.003 on both sizes with a 12-window LID batch.
- **4.7 is paused, not dropped**: resume per §4.7 when a GPU is free
  (≈ 9 h and ≈ 14 h of 25M compute remain), then `--report`, `g4_check.py`,
  and add the ranking-stability numbers to the Phase-6 error bars.
- **Uncommitted state**: the Phase-3 and Phase-4 work is in the working
  tree (the Phase-3 files were modified but not committed before Phase 4
  began). *[Committed 2026-08-22 ("phase 4 complete"); no longer applies.]*

## Assessment after the n-gram judge side study (2026-08-22)

Written after [ngram_judge_robustness.md](ngram_judge_robustness.md) (the
CH.0 n-gram judges scored on the task-2.6 noised windows, with cross-language
ranking curves for the Phase-A and Phase-C diffusion judges). What those
findings say about the merits and goals of the project so far.

### What the findings support

- **The R1 fairness emphasis is vindicated empirically.** The n-gram judge's
  failure under noise is not "more error" but a *directional* bias: every
  corrupted text drifts toward the language with the most forgiving LM
  (German), and the drift grows with severity, so no static offset can fix
  it. That is the bound-tightness asymmetry the design worried about, shown
  on a simpler instrument. One jointly trained multilingual density with a
  small conditioning margin compared on the *same text* is what makes
  cross-language comparison legitimate; the Phase-3 report-only decision and
  this study are the same finding seen twice.
- **The Phase-5 architecture is confirmed from both sides.** The n-gram
  objective keeps slope far from the key (good search signal, bad judge);
  the diffusion ELBO is flat far from the key (bad search signal, good
  near-key judge). "Cheap DP inside the search, ELBO on shortlists" is the
  right division of labour for the two degradation shapes, and explains the
  CH track's empirical success with n-gram-driven search.
- **The Phase-B noise curriculum is the most valuable training decision.**
  A language call that stays flat to a 50 %-wrong key is what lets a judge
  rank languages *during* a partial decipherment — the premise of LID by
  trial decipherment. Phase 2 was framed as smoothness vs discriminability;
  it was actually buying the core capability.

### What the findings complicate

- **The architecture's share of the credit is smaller than the framing
  implies.** The clean Phase-A diffusion model fails as a language judge as
  fast as a trigram LM; the robustness came from training on labelled noised
  text. The honest claim is "a noise-curriculum-trained, symmetric
  multilingual density plus a differentiable evaluator", not "diffusion
  beats n-grams". The diffusion choice is justified by properties this study
  did not test — bidirectional any-position scoring, soft expected-embedding
  inputs, the NULL frame, gradients reaching head parameters — all validated
  in the CH track. Attribute robustness to the curriculum and the
  differentiable-heads result to the architecture. A cheap control (retrain
  the n-gram LMs on the Phase-B mixture; optionally give the AR reference v3
  the same curriculum) would make the attribution rigorous.
  *[Not run as of 2026-09-01 (`docs/ngram_judge_robustness.md` §3 records it as
  unmeasured; `docs/project_status.md` §6 open register).]*
- **The plateau cuts both ways for Phase 6.** Scoring a half-wrong key the
  same as a fully wrong one is fine for shortlists, but on the VMS, if no
  cipher hypothesis gets near a key, every (cipher × language) cell sits on
  that plateau and the language ranking among them is noise — and the
  plateau is only ~0.8–1 bit below shuffled text, so it can look like
  partial structure. The Phase-3 abstention machinery (shuffled-text margin,
  per-document spread, flip-rate) is the defence; the final table must lead
  with those error bars, not the point estimates.
- **Search fairness is the main unmeasured risk.** The cross-language
  ranking is only as fair as the per-language inner search: if the
  n-gram-driven solve hits the true basin less often for Latin than German
  (the CH track saw this — Latin basin-hit rate, the high-entropy Latin
  documents), "couldn't find the key under Latin" becomes "not Latin". The
  judge is symmetric; the search is not yet proven to be. Phase-5 acceptance
  should include per-language solve success on matched synthetic
  difficulty, not only aggregate recovery.
  *[Partially discharged: Phase 5 (G5, 2026-08-23) reported per-language solve
  behaviour at every rung and recorded "Latin is the hard language for the n-gram inner
  search at every rung" (`docs/phase5_status.md`). Later qualified: Latin held-out
  document 0 is a 4.66-bits/char drug-recipe list carrying 10 % of the held-out weight
  (2026-08-30, `docs/alt_loop_plan.md` §10; memory `latin-pharmacopoeia-doc`), so part of
  "Latin is hard" is the split, not the language — check the Latin plaintext bpc before
  reading a Latin solve failure as search asymmetry. `docs/project_status.md` §6.]*

### Net

The core claim — language recoverable through trial decipherment, robustly
across partial keys, without bias toward any language — is in better shape
than before: the naive approach's failure mode is characterized and the
instrument demonstrably avoids it. What needs adjusting is where the credit
goes (curriculum and symmetric design → robustness; diffusion architecture →
differentiable heads) and, in Phase 5, closing the search-fairness gap.
