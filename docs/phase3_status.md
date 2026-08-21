# Phase 3 — ELBO metrology: status

Status record for Phase 3 of the [task breakdown](../reference_docs/Diffusion%20Model%20Training%20-%20Task%20Breakdown.md)
(design §5, §9.1–9.3; requirement R1). Started 2026-08-21 after Gate G2
(`docs/phase2_status.md`). All measurements are on the **Phase-B EMA
weights** (`DATA_ROOT/runs/phase_b-85m-seed0/ckpt_final.pt`, the evaluator
candidate; 25M sibling where stated), as the G2 carry-over requires. Code:
`diff_voyn/metrology/` (`scoring.py`, `calibration.py`); scripts named per
task below; artifacts under `DATA_ROOT/analysis/phase3/` and
`DATA_ROOT/calibration/`; ClearML project `diff-voyn`, tags `task3.x`, `g3`.

<!-- sections 3.1–3.7 and the G3 verdict are filled in below as each run completes -->

## 3.1 — Scoring harness with common random numbers (DONE, acceptance PASSED)

`diff_voyn/metrology/scoring.py::score_conditions`: per-window bits/char
under any set of *conditions* (the three conditioning languages and the
unconditional NULL language), stratified timestep sampling with
`n_strata × samples_per_stratum` draws (`per_window_nelbo_bits` gained the
`k` dimension; `k=1` reproduces the Phase-1/2 draw sequence exactly), and
CRN by construction: every condition of a chunk of windows is scored with
the identical `t` values and masked positions (chunk `i` ⇒ seed `seed+i`,
the same convention calibration v1 used, so all tables are paired
window-for-window). Two input shapes: one text shared by all conditions
(clean-text LID, calibration) or one text per condition with equal length
(trial decipherment — each language hypothesis's candidate plaintext, same
masks). `crn=False` exists only for the acceptance experiment.

Acceptance — "variance of between-language score differences under CRN ≥ 5×
smaller than under independent sampling" (`scripts/crn_check.py`, Phase-B
85M, 32 held-out windows per language, 8 replicate mask seeds, 32 strata,
ClearML `e70ef4fc…`):

| window length | pair | sd of difference, CRN | sd, independent | variance ratio |
|---|---|---|---|---|
| 1024 | latin−italian | 0.019 | 0.049 | **6.3** |
| 1024 | latin−german | 0.011 | 0.046 | **15.1** |
| 1024 | italian−german | 0.018 | 0.047 | **5.5** |
| 200 | latin−italian | 0.051 | 0.119 | **5.5** |
| 200 | latin−german | 0.031 | 0.125 | **14.5** |
| 200 | italian−german | 0.051 | 0.122 | **5.7** |

(bits/char, replicate-to-replicate sd per window averaged over windows.)
The raw per-condition score noise is unchanged by CRN (sd 0.035 at 1024
chars either way) — it is the *shared* part that cancels out of the
differences, which is what the ranking consumes. Pass, minimum ratio 5.5
(≥ 5). The residual CRN variance is the part of the masking noise that
interacts with the condition (which positions are masked matters more for
one language than another); it is what the sample budget (3.2) buys down.

## 3.4 — Calibration references: why a third tier was built

Calibration v1 (Phase A, `docs/phase1_status.md`) already recorded that its
offsets are **reference-dependent**: swapping the per-language AR reference
from v1 (6k steps) to v2 (20k steps) moved the Latin and German offsets by
+0.09 and +0.12 bits/char while Italian — whose 3.6M-char train split
starves any monolingual reference — stayed put. `NELBO − NLL_AR` then mixes
the bound gap with "how starved was this language's reference", a
language-dependent slack that tracks corpus size: exactly the bias R1
forbids. Phase 3 therefore adds **AR reference v3**: *one* multilingual
char-AR model (`scripts/train_ar_reference.py --multilingual --preset 25m
--version v3`; `CharARLM` with the backbone's own additive
`LanguageConditioning`, 10% conditioning dropout), 19.3M params = the 25M
diffusion sibling's dims made causal, trained on the backbone's τ=0.7
three-language mix (weights german 0.650 / latin 0.281 / italian 0.070),
20k steps × 64 × 1024 = 1.31B chars, 1.7 h on one 3090 (ClearML tag
`task3.4`, `multilingual`). Selection is one checkpoint for all languages
(lowest equal-weight mean held-out NLL — no per-language cherry-picking,
which would itself be a calibration bias): best @ step 19000.

| reference | latin | italian | german | note |
|---|---|---|---|---|
| AR v1 per-language (10M, 6k steps) | 2.345 | 2.637 | 1.874 | Italian selected among 3 candidates |
| AR v2 per-language (10M, 20k steps) | 2.259 | 2.636 | 1.751 | v1 table's reference |
| **AR v3 multilingual (19M, 20k steps)** | **2.218** | **2.551** | **1.702** | one model, same mix as the backbone |

(held-out bits/char, tiled split.) The multilingual reference is better on
every language — Italian by 0.085 bits, i.e. the transfer the backbone also
enjoys — so the v1/v2 Italian offset (−0.083) was an artifact of the
reference, not a tighter Italian bound. Tables produced in this phase (all
on the full tiled held-out split, 32 strata, CRN seed 0, paired with v1
window-for-window): **v2** = Phase-B 85M vs AR v2 (same reference as v1, so
the Phase-B drift is isolated), **v3** = Phase-B 85M vs AR v3, **v3-25m** =
Phase-B 25M vs AR v3 (capacity-matched), **v2-25m**, and **v3-phase_a** =
Phase-A 85M vs AR v3 (reference-swap control for the audit).

## 3.6 — Synthetic 1:1 suite: the inner search

`scripts/language_recovery.py --stage solve` (8 forked single-thread
workers, 87 min): 3 languages × 5 lengths {50, 100, 200, 400, 700} × 50
ciphers = 750 instances, plaintext from the **held-out** split, random
bijective key; each deciphered under *every* language hypothesis by the
rung-1 Sinkhorn head on the frozen pentagram n-gram evaluator (restarts 3,
ILS polish — the CH-track head unchanged). SER of the decipherment under
the true hypothesis (mean / fraction of instances with SER < 5%):

| L | latin | italian | german |
|---|---|---|---|
| 50 | 0.525 / 22% | 0.344 / 26% | 0.301 / 44% |
| 100 | 0.123 / 84% | 0.044 / 84% | 0.026 / 90% |
| 200 | 0.031 / 90% | 0.001 / 100% | 0.005 / 98% |
| 400 | 0.001 / 100% | 0.000 / 100% | 0.000 / 100% |
| 700 | 0.005 / 94% | 0.000 / 100% | 0.000 / 100% |

(= the task-5.2 "near-perfect at ≥ 200 chars" behaviour, now on 50
instances per cell.) The 50-char cell is unsolvable by construction —
those rows measure what the ranking does with a wrong decipherment, which
is the regime an early cipher-head search lives in.
