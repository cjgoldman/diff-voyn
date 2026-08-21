# Phase 2 — Noise curriculum: status

Status record for Phase 2 of the [task breakdown](../reference_docs/Diffusion%20Model%20Training%20-%20Task%20Breakdown.md)
(design §7.3 and §8, requirement R2). Started 2026-08-21 after Gate G1
(`docs/phase1_status.md`). Companion docs: `phase0_decisions.md`,
`phase1_status.md`.

## What Phase B trains on — and why it is still a likelihood

Noise is applied to the **data**, never to the masking process: a noised
window is the training *target*, so the loss remains the MDLM NELBO of
whatever text the model is shown. At deployment a cipher head's partially
decrypted plaintext is scored the same way; Phase B just makes such text
in-distribution so its score degrades gradually (R2) instead of collapsing to
the shuffled-text ceiling. The trade-off is deliberate and measured, not
assumed: the robustness curve (2.6) reports both the *smoothness* Phase B
buys and the *discriminability* it must not destroy (the clean→20%-wrong-key
margin is a hard G2 check, ≥0.1 bits/char).

## 2.1 — Structured substitution noise (DONE)

`diff_voyn/data/noise.py::SubstitutionNoise`. A **self-consistent many-to-one
wrong key**: letters are drawn in random order into the wrong-key set until
their frequency *in the window* covers the severity; each wrong-key letter is
remapped to one fixed, uniformly random other letter (targets may collide ⇒
many-to-one, like a wrong homophonic key). `severity` = expected fraction of
positions altered; with `exact_rate` (default) the map is applied to each
occurrence with probability `severity/covered` so the realized rate is
unbiased — the "partially wrong homophonic key" case (wrong target fixed per
letter, only some occurrences hit). `exact_rate=False` gives the fully
consistent key (every occurrence remapped). Not i.i.d. flip noise: the unit
test `test_substitution_is_self_consistent_not_iid` asserts that every
changed occurrence of a letter goes to the same target and untouched letters
are never altered.

**Acceptance — severity sweep on the Phase-A models** (`scripts/robustness_curve.py`,
48 tiled held-out windows/language, own-language NELBO, 16 strata, CRN):
monotone in all 6 (model × language) cells, both models. Curves (bits/char):

| model | lang | 0 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 | 1.0 | shuffled | uniform |
|---|---|---|---|---|---|---|---|---|---|---|
| 85M | latin | 2.374 | 2.879 | 3.216 | 3.738 | 4.000 | 4.322 | 4.590 | 4.656 | 5.438 |
| 85M | italian | 2.541 | 3.057 | 3.541 | 4.077 | 4.360 | 4.775 | 5.220 | 5.066 | 5.820 |
| 85M | german | 1.901 | 2.537 | 2.983 | 3.589 | 3.891 | 4.366 | 4.686 | 4.650 | 5.431 |
| 25M | latin | 2.575 | 3.024 | 3.325 | 3.816 | 4.059 | 4.438 | 4.818 | 4.632 | 5.561 |
| 25M | italian | 2.690 | 3.151 | 3.588 | 4.135 | 4.487 | 4.913 | 5.510 | 5.150 | 6.106 |
| 25M | german | 2.098 | 2.677 | 3.090 | 3.634 | 3.923 | 4.468 | 4.850 | 4.599 | 5.502 |

"Hard but lawful": the degradation is strictly monotone and saturates at the
no-structure ceiling (a 100%-wrong key ≈ shuffled letters), but the clean
Phase-A model is *front-loaded*: the first 5% of wrong-key errors costs
0.45–0.64 bits/char (16–23% of the whole rise), and at 20–30% wrong the text
is already scored within ~0.6 bits of shuffled text. That is the regime a
cipher head's early search lives in, and it is what Phase B is meant to
flatten without erasing the margin.

## 2.2 — Segmentation noise (DONE)

`SegmentationNoise`: letter-stream parse errors modelled on the Naibbe token
stream. The stream is parsed into 1-/2-letter tokens with the **measured**
Naibbe unigram rate `P_UNIGRAM_NAIBBE = 0.476` (pinned naibbe_v2 @ df3d074 on
20k chars per language: latin 0.479, italian 0.477, german 0.473 — i.e. 0.656
tokens per letter, matching `data/ciphers/acceptance_stats.json`'s 0.65). Each
token is misparsed with probability `severity / tokens_per_letter`: a bigram
read as a unigram **drops** its second letter; a unigram read as a bigram
**gains** one — a **duplicate** of itself (p=0.5) or a letter drawn from the
window's own letter distribution. So insertions/deletions balance at
p/(1−p) ≈ 0.91 — a property of the parse model, unit-tested. `severity` =
expected edits per source letter. No space noise exists (task 0.3 stripped all
whitespace); the output is letters only and the test decodes it and asserts
no whitespace. Sweep: monotone, no cliff on both models, all languages.

## 2.3 — Transcription noise (DONE)

`TranscriptionNoise(severity=0.05)`: i.i.d. per-character events, 80%
substitution (uniform other letter), 10% deletion, 10% insertion (uniform
letter). The 5% default is the Bruton-2026 level; severity-parameterized and
unit-tested at 0–0.3. Sweep: monotone, no cliff, both models.

## 2.5 — NULL-token exposure (DONE — acceptance met after Phase B)

`frame_with_nulls`: the 2N-slot frame of design §8 on hard tokens — each
Naibbe-style token emits two slots, `[letter, NULL]` for a unigram,
`[letter, letter]` for a bigram, at the measured unigram rate. NULL therefore
occupies 47.6% of slot-2 positions = **23.8% of all slots**, always in slot-2
position and never adjacent to another NULL (invariants unit-tested; dropping
NULLs recovers the stream exactly). This is exactly the hard-token limit of
the soft frame `diff_voyn/heads/frame.py::build_frame` emits. Phase B shows
framed text on 20% of examples (10% clean-framed, 10% noised-framed).

Acceptance ("NULL slots score as in-distribution after Phase B"), 48 tiled
held-out windows per language on the frame, per-position NELBO split
(`robustness_curve.py`, judged by `g2_check.py`):

| model | lang | NULL-slot bits: A → B | frame overhead bits/letter: A → B | letter-slot / clean (B) |
|---|---|---|---|---|
| 85M | latin | 32.1 → **0.881** | 11.4 → **0.762** | 1.20 |
| 85M | italian | 34.3 → **0.946** | 12.0 → **0.785** | 1.19 |
| 85M | german | 32.7 → **0.800** | 11.8 → **0.773** | 1.27 |
| 25M | latin | 31.7 → 0.953 | 11.0 → 0.783 | 1.19 |
| 25M | italian | 33.8 → 0.995 | 11.8 → 0.797 | 1.18 |
| 25M | german | 31.7 → 0.874 | 11.3 → 0.805 | 1.25 |

Criterion and its reasoning: NULL-slot NELBO ≤ 1.5 bits, and the **frame
overhead per plaintext letter** (total frame bits per letter, NULL slots
included, minus the clean bits/char of the same text) ≤ 1.0 bit. The
unavoidable floor is the unigram/bigram pattern's own information,
H(0.476) = 0.998 bits per token = **0.656 bits per letter** when the pattern is
unpredictable; the measured overhead of 0.76–0.81 is that floor plus ~0.1–0.15
bits of lost letter context. The pattern's cost lands partly on the NULL
slots (0.8–1.0 bits ≈ H) and partly on letter slots (a masked slot 2 pays
−log P(not NULL) before −log P(letter | not NULL)), so the letter-slot ratio
alone (1.18–1.27×) is *not* a context-loss measure — the first draft of the
gate used a 1.15× letter-slot bound, which would have failed for this
bookkeeping reason and was replaced by the overhead criterion before the
verdict (recorded here so the change is not invisible). In Phase 5 the
pattern is the head's own `w_t` and identical across language hypotheses,
so this overhead cancels in the ranking.

## 2.4 — Phase B fine-tune (DONE 2026-08-21)

Example mix (`NoiseConfig`, recorded in each run manifest and ClearML
configuration `noise`): **clean 0.50 / noised 0.30 / framed 0.10 /
framed+noised 0.10** — noised fraction 40% (design: 30–50%), NULL-framed 20%,
clean fraction 50% and guarded ≥ 0.5 in code (the calibration anchor is never
reduced). Within a noised example the families are applied independently —
substitution p=0.75 (severity ~U(0.02, 0.5)), segmentation p=0.5 (~U(0.01,
0.2)), transcription p=0.5 (~U(0.01, 0.1)) — at least one always, in
deployment order wrong key → wrong parse → transcription. Length-changing
kinds over-sample the source window ×1.5 and crop. Realized kind fractions
(0.50/0.30/0.10/0.10 at every log point), per-kind train NELBO and mean
substitution severity (0.21) are logged every 50 steps
(`train_kind_fraction/*`, `train_nelbo_by_kind/*`).

Schedule (`scripts/train.py --phase phase_b --init-from <phase_a ckpt_final>`):
init from the Phase-A **EMA** weights (raw = EMA within 0.14% at G1), fresh
AdamW, peak LR 1e-4 (≈3× the Phase-A floor), 300-step warmup, cosine to
1e-5, 6000 steps × 0.5M chars ≈ 3.1B chars (1.3B noised), dropout 0.1, EMA
**0.999** (time constant 1000 steps = 1/6 of the run — the G1 EMA lesson;
raw canary logged alongside, final raw-vs-EMA gap ≤ 0.09%). Canary every 200
steps: the clean canary (unchanged series, comparable to Phase A) plus fixed
noised (20% key / 5% parse / 5% transcription) and NULL-framed held-out
variants with the NULL-slot / letter-slot split.

Runs (ClearML project `diff-voyn`, tags `phase_b`, `task2.4`):
`phase_b-85m-seed0` (GPU 0, 63k chars/s, ~14 h, task `6f58e26ae09940a3be27b13ac66a227d`)
and `phase_b-25m-seed0` (GPU 1, 204k chars/s, ~4.3 h, task `978ff77923af42b188224d5bdb6da78b`);
final checkpoints `DATA_ROOT/runs/phase_b-<size>-seed0/ckpt_final.pt` (step
6000, EMA). Canary trajectories (EMA, steps 200 → 6000):

| model | lang | clean canary | noised canary | NULL-slot bits | train NELBO by kind at 6000 (clean / noised / framed / both) |
|---|---|---|---|---|---|
| 85M | latin | 2.148 → 2.161 | 3.900 → 3.396 | 27.4 → 0.87 | 2.021 / 3.125 / 2.124 / 2.970 |
| 85M | italian | 2.515 → 2.526 | 4.309 → 3.574 | 30.7 → 0.95 | |
| 85M | german | 1.875 → 1.884 | 3.924 → 3.220 | 29.0 → 0.75 | |
| 25M | latin | 2.381 → 2.397 | 3.958 → 3.511 | 28.0 → 0.94 | 2.273 / 3.316 / 2.352 / 3.110 |
| 25M | italian | 2.670 → 2.688 | 4.383 → 3.660 | 30.6 → 0.98 | |
| 25M | german | 2.089 → 2.105 | 3.974 → 3.416 | 28.7 → 0.82 | |

Everything had flattened by step ~4000 (noised canary and NULL-slot bits
move < 0.03 over the last 2000 steps); the run length was adequate.

**Clean anchor (the G2 hard check):** full tiled held-out NELBO of the final
EMA weights, scored with the identical estimator, tiles, seeds and batch as
the G1 calibration run (paired comparison):

| model | lang | G1 (tiled) | Phase B (tiled) ± s.e.m. | drift |
|---|---|---|---|---|
| 85M | latin | 2.3496 | 2.3554 ± 0.0133 | **+0.25%** |
| 85M | italian | 2.5538 | 2.5584 ± 0.0048 | **+0.18%** |
| 85M | german | 1.8997 | 1.9077 ± 0.0053 | **+0.42%** |
| 25M | latin | 2.5530 | 2.5573 ± 0.0119 | +0.17% |
| 25M | italian | 2.7008 | 2.7168 ± 0.0043 | +0.59% |
| 25M | german | 2.0964 | 2.1109 ± 0.0054 | +0.69% |

All within the 1% criterion; the 85M drifts are at or below the
window-to-window s.e.m. The calibration offsets of `calibration_v1.json`
therefore remain valid for the Phase-B evaluator to within 0.004–0.008
bits/char on the 85M (well inside the 0.0065/0.0030/0.0022 s.e.m. of the
offsets); the table is **not** re-versioned for Phase B. The Phase-B drift is
uniformly positive and largest for German/25M — the expected small price of
spending capacity on noised text; if a later phase needs it back, the lever
is the noised fraction (0.4 → 0.3), not the clean share.

## 2.6 — Robustness curve (DONE)

`scripts/robustness_curve.py` (reports, per-window arrays and plots under
`DATA_ROOT/analysis/phase2/robustness_<tag>.{json,_windows.npz,png}`; ClearML
tag `task2.6`: phase_a-85m `8ca025ee…`, phase_a-25m `3dc1f9ae…`, phase_b-85m
`fb18004a…`, phase_b-25m `592adef6…`). Phase-B curves are scored on the *same
noised texts* (fixed noise seed) and masking seeds as the Phase-A curves, so
the comparison is paired. 85M, own-language NELBO bits/char, before → after:

| lang | noise | 0 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 | 0.75 | 1.0 | shuffled |
|---|---|---|---|---|---|---|---|---|---|---|
| latin | substitution A | 2.374 | 2.879 | 3.216 | 3.738 | 4.000 | 4.322 | 4.534 | 4.590 | 4.656 |
| latin | substitution **B** | 2.378 | 2.615 | 2.794 | 3.013 | 3.142 | 3.337 | 3.448 | 3.332 | 4.109 |
| italian | substitution A | 2.541 | 3.057 | 3.541 | 4.077 | 4.360 | 4.775 | 5.139 | 5.220 | 5.066 |
| italian | substitution **B** | 2.547 | 2.799 | 2.911 | 3.115 | 3.243 | 3.416 | 3.517 | 3.417 | 4.157 |
| german | substitution A | 1.901 | 2.537 | 2.983 | 3.589 | 3.891 | 4.366 | 4.505 | 4.686 | 4.650 |
| german | substitution **B** | 1.909 | 2.202 | 2.364 | 2.590 | 2.825 | 3.170 | 3.401 | 3.373 | 4.217 |

| lang | noise | 0 | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|---|---|
| latin | segmentation A → **B** | 2.374 → 2.378 | 2.564 → 2.537 | 2.859 → 2.748 | 3.250 → 3.037 | 3.533 → 3.233 | 3.785 → 3.428 | 4.093 → 3.644 |
| german | segmentation A → **B** | 1.901 → 1.909 | 2.129 → 2.097 | 2.455 → 2.328 | 2.922 → 2.658 | 3.256 → 2.903 | 3.554 → 3.140 | 3.960 → 3.469 |

| lang | noise | 0 | 0.01 | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 |
|---|---|---|---|---|---|---|---|---|
| latin | transcription A → **B** | 2.374 → 2.378 | 2.528 → 2.502 | 2.681 → 2.619 | 2.998 → 2.855 | 3.475 → 3.223 | 3.801 → 3.501 | 4.065 → 3.748 |
| german | transcription A → **B** | 1.901 → 1.909 | 2.087 → 2.061 | 2.257 → 2.185 | 2.640 → 2.447 | 3.220 → 2.881 | 3.586 → 3.204 | 3.920 → 3.512 |

(Italian and the 25M follow the same pattern; full tables in the JSON
reports.) What the curriculum bought, and what it cost:

- **Smoothness.** First-step sensitivity (5% wrong key) halved: 85M
  0.505/0.516/0.636 → 0.237/0.252/0.293 bits/char (latin/italian/german); the
  total rise to a fully wrong key fell from 2.2–2.8 to 0.9–1.5 bits/char. The
  curves are concave and saturating with no accelerating increment
  (operational "no cliff": slope never > 1.5× any earlier slope, no
  increment > 1 bit, none > 50% of the rise). Segmentation and transcription
  curves moved less (−0.1 to −0.45 bits at the high end) — those families
  were applied at lower severities in training.
- **Discriminability retained.** The clean→20%-wrong-key margin is still
  0.57–0.68 bits/char on the 85M (0.57–0.80 on the 25M), 6–8× the hard
  floor of 0.1; a half-wrong key still costs ≥ 0.87 bits/char. The
  no-structure ceiling also dropped (shuffled letters 4.66 → 4.11), i.e. the
  Phase-B model is a broader density — expected, and bounded by the clean
  fraction (clean NELBO is the anchor above).
- **The severity-1.0 dip is text, not instrument.** Substitution curves are
  monotone up to 0.75 and then *drop* at 1.0 (85M latin 3.448 → 3.332). At
  severity 1.0 the generator remaps *every* letter (a fully consistent
  many-to-one key, nothing kept); the letter merging reduces the text's own
  entropy, whereas 0.75 mixes original and relabeled letters (more symbols,
  less consistency). Verified directly (`scratch: relabel_probe`, 85M, 24
  windows): Phase A dips too (latin 0.9 → 1.0: 4.762 → 4.697), a consistent
  many-to-one key scores *lower* than a bijective relabeling (B: 3.34 vs
  3.77) because it merges symbols, and the noise-trained model simply reads
  that structure better. `g2_check.py` therefore judges substitution
  monotonicity on the mixed-key range ≤ 0.75 (hard) and reports the 1.0
  point with its margin over clean (+0.87…+1.46 bits/char) — a criterion
  change made after seeing the curves, recorded here with its evidence.

## Gate G2 — verdict (2026-08-21) — **PASSED**

`scripts/g2_check.py` (ClearML task `c83d403e634149a9917c4f4288ef7625`, tag
`g2`; report `DATA_ROOT/runs/g2_report.json`), both models:

| check | result |
|---|---|
| G2.1 clean-text anchor, tiled full held-out, |drift| < 1% vs G1 | PASS all 6 cells (85M +0.25/+0.18/+0.42%, 25M +0.17/+0.59/+0.69%); raw-vs-EMA gap ≤ 0.09% |
| G2.2 noised-input degradation monotone (substitution ≤ 0.75) and cliff-free; clean→20%-key margin ≥ 0.1 bits/char | PASS all 18 curves; margins 0.57–0.80 |
| G2.3 NULL slots in-distribution: NULL-slot ≤ 1.5 bits, frame overhead ≤ 1.0 bits/letter | PASS all 6 cells (0.80–1.00 bits; 0.76–0.81 bits/letter vs floor 0.656) |
| G2.4 training-log canaries | clean canary +0.4–0.8%, noised −0.45…−0.73 bits, NULL-slot 28–31 → 0.75–1.0 |

**The Phase-B EMA weights are the evaluator candidates going forward:**
`DATA_ROOT/runs/phase_b-85m-seed0/ckpt_final.pt` (the instrument) and
`phase_b-25m-seed0/ckpt_final.pt` (the restart-heavy search sibling).
Calibration table v1 stays valid (drift ≪ offset s.e.m.); the bound-fairness
audit of Phase 3 (3.5) should be run on the Phase-B weights since the design
asks for it "after every training phase".

**→ Phase 3 (ELBO metrology, tasks 3.1–3.6) may start.** Carry-overs: the
Phase-B density is broader (shuffled text −0.55 bits), so any Phase-3
abstention threshold / negative-control margin must be measured on the
Phase-B weights, not inferred from Phase A; per-document (3.3) reporting is
still owed for the heterogeneous Latin held-out set.
