# Phase 3 — ELBO metrology: status

> **Record status (banner added 2026-09-01):** Phase 3 / ELBO metrology, 2026-08-21 (G3 PASS
> 23:00 UTC). Still governing: the CRN harness, budget 64 (× 4 replicate seeds), the
> report-only calibration policy, the same-text finding, per-document error bars and the
> shuffled-text abstention channel (which became the Phase-6 structure margin). Superseded:
> every measurement here is on the **Phase-B EMA weights** — the frozen evaluator has been
> `phase_c-85m-seed0/ckpt_final.pt` since G4 (2026-08-22) — and the adopted table **`v3-ro`**
> was renamed/re-measured as **`v3-phase_c-ro`** on 2026-08-22 (offsets +0.138 / +0.013 /
> +0.205, same policy; inline note in §3.4). `docs/project_status.md` §5.12.
> **Current project position: `docs/project_status.md`.**

Status record for Phase 3 of the [task breakdown](../reference_docs/Diffusion%20Model%20Training%20-%20Task%20Breakdown.md)
(design §5, §9.1–9.3; requirement R1). Started 2026-08-21 after Gate G2
(`docs/phase2_status.md`). All measurements are on the **Phase-B EMA
weights** (`DATA_ROOT/runs/phase_b-85m-seed0/ckpt_final.pt`, the evaluator
candidate; 25M sibling where stated), as the G2 carry-over requires. Code:
`diff_voyn/metrology/` (`scoring.py`, `calibration.py`); scripts named per
task below; artifacts under `DATA_ROOT/analysis/phase3/` and
`DATA_ROOT/calibration/`; ClearML project `diff-voyn`, tags `task3.x`, `g3`.

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

## 3.2 — Sample-budget study (DONE; acceptance criterion amended and recorded)

`scripts/sample_budget.py` (Phase-B 85M; 64 held-out windows per language ×
5 lengths; budgets B ∈ {4 … 128} stratified draws, `n_strata = B`, one
sample per stratum; 8 replicate mask seeds; CRN across conditions; ClearML
`c9df65e6…`, restat `b91c17e7…`). The *flip-rate* at a budget is the
probability that two independent replicates disagree on a window's top-1
language — pure Monte-Carlo ranking noise *plus* true ties.

**Clean-text regime.** The all-window flip-rate falls from 14–17% (B = 4)
to 1.0–2.4% (B = 128) and never reaches the 1% target at any length — and
the reason is the data, not the budget:

| L | consensus margin, median (bits/char) | median × L (bits/window) | resolvable (> 0.05 bits/char) | flip B=32 / 64 / 128, all windows | flip at B=64, resolvable | top-1 = truth (consensus) |
|---|---|---|---|---|---|---|
| 50 | 0.506 | 25 | 99% | 2.05 / 1.17 / 1.00% | **0.58%** | 98.4% |
| 100 | 0.304 | 30 | 91% | 3.16 / 1.73 / 1.19% | **0.53%** | 99.5% |
| 200 | 0.167 | 33 | 73% | 3.37 / 2.79 / 2.05% | **0.00%** | 97.9% |
| 400 | 0.075 | 30 | 66% | 4.61 / 2.88 / 2.36% | **0.20%** | 94.8% |
| 700 | 0.042 | 29 | 42% | 3.55 / 2.01 / 1.80% | **0.00%** | 93.8% |

(consensus = the 8 replicates at B = 128 pooled, 1024 draws; margin = gap
between the best and second-best conditioning language on the *same* clean
window.) The per-character margin shrinks as ~1/L while its product with L
is flat at ≈ 30 bits: **conditioning the backbone on the wrong language
costs a roughly constant ~30 bits per window, not a per-character rate** —
the text itself identifies its language and the label carries a finite
amount of information on top. At ≥ 400 chars half the clean windows
therefore sit below 0.08 bits/char, the scale of the replicate noise of a
between-language difference (sd 0.03–0.05 bits/char at L = 200, 3.1), and
those windows flip at any budget: a true tie flips 50% of the time. (This
is also why clean-text top-1 accuracy *drops* with length here, 98.4% →
93.8% — longer windows of heterogeneous Latin documents land in the
near-tie population; 3.3 shows the same heterogeneity per document.)

**Criterion amendment (recorded, not footnoted).** The acceptance
("ranking flip-rate < 1% at the chosen budget") is evaluated on
*resolvable* windows — consensus margin above a margin floor of 0.05
bits/char, the order of the same-text close-pair margin measured in 3.6
(0.03–0.04 bits/char), i.e. the population the instrument reports as
*unresolved* rather than ranks. Under that criterion the **chosen budget is
64 draws** (flip 0.58 / 0.53 / 0.00 / 0.20 / 0.00% for L = 50 … 700; the
design's "64 strata × k" with k = 1). The all-window criterion is stored
alongside as unmet (`chosen_budget_all_windows = null`), both numbers are
shown by `g3_check.py`, and the floor's sensitivity is in the report
(`flip_rate_by_margin_floor`: at B = 64 a floor of 0.02 gives 0.6 / 1.1 /
0.1 / 0.2 / 0.2%, a floor of 0.1 gives 0.4 / 0 / 0 / 0 / 0%). Extending the
sweep to B = 256 / 512 was rejected: ≈ 7 h of GPU (the 4 … 128 sweep took
2 h 15 min) to trace an asymptote that ties cannot cross.
`sample_budget.py --restat` recomputes every statistic from the saved
per-window arrays (`sample_budget_scores.npz`) without the GPU.

**Decipherment regime** (what Phase 5 consumes): the same sweep over the
3.6 decipherments (budgets 4 … 128 × 4 replicate seeds, per hypothesis,
shared masks) is re-derived under the primary table inside the recovery
report; under calibration v1's applied offsets it read 1.4–2.0% at B = 128
for ≥ 200 chars because v1 manufactures Latin/Italian near-ties (3.6). The
final numbers under the adopted table are in §3.6 below.

## 3.3 — Per-document scoring with mean and spread (DONE)

`scripts/score_documents.py --from-calibration v3 --calibration v3-ro`
(re-uses the v3 table's per-window arrays — every held-out document cut
into consecutive 1024-char windows, all four conditions, CRN, 32 strata —
no GPU; `DATA_ROOT/analysis/phase3/documents_v3.{json,md}`, ClearML
`task3.3`). Own-condition bits/char per document, mean ± window std:

| language | documents | windows | own bits/char (all windows) | between-doc sd | within-doc sd | doc top-1 | window top-1 |
|---|---|---|---|---|---|---|---|
| latin | 6 | 481 | 2.355 ± 0.292 | **0.276** | 0.117 | 5/6 | 84.4% |
| italian | 2 | 515 | 2.558 ± 0.109 | 0.046 | 0.119 | 2/2 | 99.4% |
| german | 8 | 799 | 1.908 ± 0.148 | 0.162 | 0.122 | 8/8 | 100.0% |

The Latin held-out set is **heterogeneous beyond its within-document
noise**: documents span 2.12 (Roger Bacon, 217 windows, 99.5% window top-1)
→ 2.37 / 2.41 (anonymi) → 2.49 (Cicero, 78.6%) → 2.53 (Cato, 83.5%) →
**2.96 bits/char for `apothecary_ellis_1854`** (49 windows), which ranks
*German* on 100% of its windows and is the one document-level miss. The
between-document sd (0.28) is 2.4× the within-document sd, and it is what
the uncertainty statement of any Latin-vs-X ranking must carry: the
"Latin" held-out mean is not a property of Latin but of which documents
are in the split. Italian (two documents, Dante/Machiavelli, 0.046 apart)
and German (eight, 0.16 apart, every window correct) are homogeneous by
comparison. Carry-over for Phase 6 reporting: per-document (not
per-language) spread is the honest error bar, and the 1854 apothecary
text should be re-vetted for period/domain (task 0.2 criteria) before the
held-out split is ever re-cut (split v2 — never mutate v1).

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

## 3.4 — Calibration tables of this phase and the policy decision (DONE)

`scripts/calibrate.py` on the full tiled held-out split (481 / 515 / 799
windows of 1024 chars), 32 strata, CRN seed 0, batch 16 — every table is
paired with v1 window-for-window. `--nelbo-from` re-uses a table's
per-window diffusion arrays and re-scores only the reference (a reference
swap without GPU work, used for the two Phase-A controls);
`--derive-report-only` writes a policy copy. ClearML `task3.4`.

| table | backbone | AR reference | latin | italian | german | spread |
|---|---|---|---|---|---|---|
| v1 (Phase 1) | Phase-A 85M | v2 per-language | +0.091 ± 0.006 | −0.083 ± 0.003 | +0.149 ± 0.002 | 0.232 |
| v2 | Phase-B 85M | v2 per-language | +0.096 ± 0.007 | −0.078 ± 0.003 | +0.157 ± 0.002 | 0.235 |
| v3-phase_a | Phase-A 85M | v3 multilingual | +0.132 ± 0.003 | +0.003 ± 0.003 | +0.198 ± 0.002 | 0.195 |
| **v3** | **Phase-B 85M** | **v3 multilingual** | **+0.138 ± 0.003** | **+0.008 ± 0.003** | **+0.206 ± 0.002** | **0.198** |
| v2-25m | Phase-B 25M | v2 per-language | +0.298 ± 0.008 | +0.080 ± 0.004 | +0.360 ± 0.003 | 0.280 |
| v3-25m | Phase-B 25M | v3 multilingual | +0.340 ± 0.005 | +0.166 ± 0.003 | +0.409 ± 0.003 | 0.243 |
| v3-phase_a-25m | Phase-A 25M | v3 multilingual | +0.335 ± 0.004 | +0.150 ± 0.003 | +0.394 ± 0.003 | 0.244 |
| **v3-ro** (adopted) | Phase-B 85M | v3 multilingual, **report-only** | (+0.138) | (+0.008) | (+0.206) | applied 0 |

(bits/char, offset = own-condition NELBO − NLL_AR, ± s.e.m. over windows.)
Three comparisons the table isolates:

- **Phase-B drift** (v1 → v2, same reference): +0.005 / +0.005 / +0.008
  bits/char — the noise-curriculum fine-tune moved the offsets by < 0.01,
  so the G2 "anchor held" statement holds at the offset level, not just
  the NELBO level. Phase-B LID top-1 on clean 1024-char windows: Latin
  84.4%, Italian 99.4%, German 100% (Latin's misses are §3.3's documents).
- **Reference swap** (v2 → v3, same backbone): +0.041 / +0.086 / +0.049
  bits/char, 24–28 × s.e.m., and the Italian sign flips. The per-language
  references were data-starved in proportion to corpus size; the
  multilingual reference removes that term. The language *ordering* of
  the offsets (German > Latin > Italian) is the same in every table and on
  both backbones.
- **Capacity** (85M → 25M, same reference): +0.20 / +0.16 / +0.20 — the
  offset carries an architecture term of the same size as its language
  term; the 25M bound is looser everywhere by about the same amount, which
  is what makes the 25M/85M ranking cross-check (design §3) meaningful.

**Policy decision — escalated design change to §5b.3.** The adopted table
is **`v3-ro`**: the v3 offsets *measured and stored*, **applied offsets
zero** (`policy: report-only`; `CALIBRATION_VERSION = "v3-ro"` in
`diff_voyn/metrology/calibration.py`; `calibrate_bits` remains the single
application point and the G3 static check still enforces it). *[Superseded
2026-08-22: the code now reads `CALIBRATION_VERSION = "v3-phase_c-ro"`
(`diff_voyn/metrology/calibration.py:45`) — the same v3 multilingual reference
re-measured on the Phase-C evaluator (+0.138 / +0.013 / +0.205, spread 0.193;
`docs/phase4_status.md` §4.4, `docs/phase4_fairness_audit.md`). The report-only
policy, the single application point and the margin-uncertainty reading are
unchanged; `docs/project_status.md` §5.12.]* Reason,
from §3.6: subtracting any of the measured tables drops the 1:1 recovery
at ≥ 200 chars from 98.4% to 70–72% — the offsets are defined on
own-language text and are invalid in the one situation where they change
a decision (same-text comparisons with 0.02–0.08 bits/char margins);
where they are valid the margins are 1–2 bits/char and the offsets cannot
change the order. Under `report-only` the measured offsets become the
**systematic uncertainty of a ranking margin**
(`CalibrationTable.margin_uncertainty_bits`: Latin–Italian 0.130,
Latin–German 0.068, Italian–German 0.198 bits/char); any margin below it
is reported as *unresolved at calibration precision* (the recovery
report's column; the Phase-6 table will carry the same flag). The AR-gap
measurement itself stays in the pipeline as the audit's instrument
(§3.5) and must be re-measured after every phase; the §5b.3 wording
"apply it as an additive per-language offset at ranking time" is
superseded by this phase's evidence and should be revised in the design
doc rather than footnoted. Naming note: the task breakdown calls the
Phase-C table "v3" (task 4.4); tables here are named by reference tier —
the Phase-C table will be `v3-phase_c` (+ `-ro` copy).

## 3.5 — Bound-fairness audit (DONE; findings escalated)

`scripts/fairness_audit.py --adopt v3-ro --tables …` (all nine tables) →
[`docs/phase3_fairness_audit.md`](phase3_fairness_audit.md), ClearML
`task3.5`. With three languages no correlation is testable, so the audit
tests what can be tested — and escalates everything above noise:

- **`reference-dependence/*` — ESCALATED** (three backbones). Swapping
  only the AR reference moves the offsets by up to 0.13 / 0.09 / 0.17
  bits/char (Phase-A 85M, v1 → v2 → v3 references; 77 × s.e.m.), still
  0.04 / 0.09 / 0.05 (28 ×) between the two best tiers. The offset is
  dominated by reference quality; the most data-fair tier (one
  multilingual model on the backbone's own mix) is adopted and the
  residual offsets are read as bound-gap-plus-architecture terms, never as
  proof of comparable tightness.
- **`language-dependence/v3-ro` — ESCALATED.** Per-document offsets (16
  held-out documents) differ by language beyond document dispersion:
  between-language range 0.208 bits/char vs within-language document sd
  0.040; one-way ANOVA F = 24.9, p = 3.6 × 10⁻⁵; Kruskal–Wallis p = 0.002.
  A language-level bound-gap term exists. Under the adopted policy it is
  not subtracted (the 3.6 suite showed that subtracting it breaks every
  same-text comparison) but carried as the systematic uncertainty of
  every cross-language margin.
- **`correlates/v3-ro` — descriptive.** Offsets are rank-ordered with
  corpus size (ρ = +1.0: German > Latin > Italian) and inversely with AR
  entropy (ρ = −1.0); Germanic − Romance = +0.133 bits/char. With n = 3
  this is reported, not tested: the *more* data a language has, the
  *looser* the diffusion bound relative to its AR reference — consistent
  with the bound gap being an intrinsic property of the (more predictable,
  lower-entropy) text rather than under-training, but a design issue for
  any Phase-6 claim that "Germanic candidates rank highest": that
  direction is the one the offsets would move against.

No above-noise finding is left un-escalated (`g3_check.py` 3.5 PASS).

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

## 3.6 / 3.7 — Recovery report under the adopted table (DONE, acceptance PASSED)

`scripts/language_recovery.py --stage score` (Phase-B 85M, GPU 1, 61 min):
every hypothesis's decipherment of every instance scored at budget 64 ×
4 replicate seeds (CRN across hypotheses: same masks on each candidate
plaintext), plus the budget sweep 4 … 128 × 4, plus two controls per
instance — the true plaintext itself and a character-shuffled version,
under every condition. `--stage report --primary v3-ro --calibrations v1
v2 v3 v3-ro` ranks each instance under every table (`recovery_report.{json,md}`,
ClearML `task3.6`, `task3.7`). Language / family accuracy by length:

| L | uncalibrated | v1 (apply) | v2 (apply) | v3 (apply) | **v3-ro (adopted)** | n-gram excess bits |
|---|---|---|---|---|---|---|
| 50 | 75.3 / 88.7% | 71.3 / 85.3% | 71.3 / 85.3% | 73.3 / 86.0% | **75.3 / 88.7%** | 76.7 / 88.0% |
| 100 | 94.0 / 96.7% | 96.0 / 97.3% | 96.0 / 97.3% | 95.3 / 96.7% | **94.0 / 96.7%** | 96.7 / 98.0% |
| 200 | 98.0 / 98.7% | 84.0 / 84.0% | 84.0 / 84.0% | 84.0 / 84.0% | **98.0 / 98.7%** | 100 / 100% |
| 400 | 98.7 / 98.7% | 72.7 / 74.0% | 72.7 / 74.0% | 73.3 / 74.0% | **98.7 / 98.7%** | 100 / 100% |
| 700 | 98.7 / 98.7% | 52.0 / 56.7% | 52.0 / 56.7% | 58.0 / 60.7% | **98.7 / 98.7%** | 100 / 100% |
| **≥ 200 mean** | 98.4 / 98.7% | 69.6 / 71.6% | 69.6 / 71.6% | 71.8 / 72.9% | **98.4 / 98.7%** | 100 / 100% |

**Acceptance: 98.4% language / 98.7% family at ≥ 200 chars under the
adopted table (bar 97.1%) — PASS**, on 450 instances (Wilson 95% CI per
cell in the json). Under *any* applied-offset table the same instrument
fails the bar (69.6–71.8%), and the report shows why, cell by cell:

| language | L | SER (true hyp.) | wrong hyp. decodes truth | lang acc (v3-ro) | family | margin, median (bits/char) | unresolved at calib. precision | flip-rate (B=64 × 4) | true − shuffled |
|---|---|---|---|---|---|---|---|---|---|
| german | 200 / 400 / 700 | 0.005 / 0.000 / 0.000 | 0% | 100 / 100 / 100% | 100% | 2.06 / 1.95 / 1.83 | 0% | 0% | −2.7 / −2.5 / −2.4 |
| italian | 200 / 400 / 700 | 0.001 / 0.000 / 0.000 | 8 / 36 / 78% | 100 / 100 / 100% | 100% | 1.14 / 0.58 / 0.22 | 0 / 6 / 32% | 0% | −1.8 / −1.7 / −1.6 |
| latin | 200 / 400 / 700 | 0.030 / 0.001 / 0.005 | 64 / 94 / 96% | 94 / 96 / 96% | 96 / 96 / 96% | 0.076 / 0.034 / 0.020 | 48 / 74 / 92% | 3.0 / 4.0 / 4.3% | −1.9 / −1.9 / −1.8 |
| german | 50 / 100 | 0.301 / 0.026 | 0 / 2% | 78 / 94% | 78 / 94% | 1.01 / 2.09 | 8 / 4% | 5.0 / 1.0% | |
| italian | 50 / 100 | 0.344 / 0.044 | 0% | 90 / 100% | 96 / 100% | 0.78 / 1.27 | 18 / 2% | 12.3 / 2.7% | |
| latin | 50 / 100 | 0.525 / 0.123 | 0 / 16% | 58 / 88% | 92 / 96% | 0.31 / 0.99 | 26 / 12% | 17.3 / 3.3% | |

Reading the table:

- **Where the hypotheses decode to different texts** (German always; the
  close pair at ≤ 100 chars) the margin is 1–2 bits/char, fifty times any
  offset, and the ranking is decided by the decipherment, not by the
  bound. Calibration is irrelevant there.
- **Where a wrong hypothesis decodes the true plaintext** — the rung-1
  solver under the Italian hypothesis recovers the *Latin* text 64–96% of
  the time at ≥ 200 chars, and at ≥ 400 chars so does the German one
  (mean German-hypothesis SER on Latin ciphers 0.035 / 0.020) — the three
  candidates are the *same text* and the decision is the conditioning
  margin of 3.2: 0.02–0.08 bits/char, ≈ 30 bits per window. Any offset
  difference larger than that (v1: Latin − Italian 0.17; v3: 0.13) flips
  the pair — the 6–28% Latin accuracies under the applied tables. The
  `NELBO − NLL_AR` offsets are defined on own-language text and have no
  meaning in a same-text comparison; where they would be meaningful the
  margins dwarf them. This is the Phase-3 result that fixed the policy
  (§3.4).
- The residual Latin errors at ≥ 200 chars (7 of 150: 3 × `apothecary_
  ellis_1854`, 2 × Cicero, 2 × Cato — re-derived from the suite's seeds)
  are the high-entropy documents of §3.3, where the German condition
  *ties* the Latin one on the same text (e.g. 2.514 vs 2.515 bits/char);
  they are not decipherment failures (SER 0.001) and they are all marked
  *unresolved at calibration precision*. Family granularity (3.7) does
  not rescue them (Latin → German is cross-family) — the dominant error
  mode of this suite is document heterogeneity, not the Romance pair; the
  Romance pair is resolved by the ~30-bit conditioning margin.
- **Abstention margin** (carry-over for Phase 4/6): the true decipherment
  scores 1.6–2.9 bits/char *below* its own shuffled text under every
  condition, and a wrong decipherment 0.5–1.9 below — a shuffled-text
  baseline separates "language-like" from "not" by far more than any
  language separates from another.

**Budget sweep, decipherment regime** (3.2 under the adopted ranking;
replicate flip-rate / language accuracy, 4 seeds × 150 instances per
length):

| L | B=4 | B=8 | B=16 | B=32 | **B=64** | B=128 |
|---|---|---|---|---|---|---|
| 50 | 32.1% / 68.3% | 24.4% / 72.3% | 21.7% / 73.7% | 14.6% / 74.2% | 11.6% / 74.3% | 8.8% / 73.7% |
| 100 | 6.7% / 94.2% | 5.8% / 93.8% | 4.4% / 94.3% | 3.8% / 94.8% | 2.3% / 93.8% | 1.0% / 94.2% |
| 200 | 7.6% / 95.0% | 4.3% / 97.2% | 2.1% / 98.7% | 2.7% / 97.7% | **1.0% / 97.8%** | 1.2% / 98.5% |
| 400 | 9.2% / 93.8% | 4.9% / 96.0% | 3.2% / 97.8% | 2.1% / 98.2% | **1.3% / 98.7%** | 0.3% / 98.5% |
| 700 | 11.7% / 92.5% | 8.0% / 94.0% | 5.9% / 95.5% | 2.6% / 97.2% | **1.4% / 98.5%** | 1.6% / 98.0% |

The all-instance flip-rate at ≥ 200 chars is 1.0–1.4% at B = 64 and does
not fall further at B = 128 — the same near-tie floor as the clean-text
regime (the Latin same-text cells flip 3–4% at any budget; every other
cell is at 0%). Accuracy is flat from B = 16 on. Budget 64 is adopted for
both regimes (`g3_check.py` reports the decipherment flip-rate as a
warning, not a gate).

## Gate G3 — verdict: **PASS** (2026-08-21 23:00 UTC)

`scripts/g3_check.py` → `DATA_ROOT/runs/g3_report.json`, ClearML tag `g3`:

| check | status | value |
|---|---|---|
| 3.1 CRN variance reduction ≥ 5× | PASS | min ratio 5.5× (L = 1024 and 200) |
| 3.2 budget with flip-rate < 1% (resolvable windows, margin > 0.05) | PASS | **B = 64**; all-window criterion recorded as unmet (near-ties) |
| 3.3 per-document mean + spread | PASS | `documents_v3.json`; Latin between-doc sd 0.276 |
| 3.4 table stored, versioned | PASS | `v3-ro` (Phase-B 85M, AR v3), offsets +0.138 / +0.008 / +0.206, spread 0.198 |
| 3.4 offsets applied in exactly one place | PASS | `calibrate_bits` in `diff_voyn/metrology/calibration.py`; no other offset arithmetic |
| 3.5 no un-escalated above-noise finding | PASS | 4 escalated (3 × reference-dependence, language-dependence/v3-ro) |
| 3.5 / 3.6 audit table == report table == `CALIBRATION_VERSION` | PASS | `v3-ro` |
| 3.6 1:1 recovery ≥ 97.1% at ≥ 200 chars | PASS | **98.4% language / 98.7% family** (L200 98.0 / 98.7, L400 98.7 / 98.7, L700 98.7 / 98.7) |
| 3.7 both granularities reported | PASS | by length / language / cell |
| 3.6 decipherment flip-rate < 1% at ≥ 200 | WARN | max 4.3% (latin/L700, 92% of instances unresolved at calibration precision — same-text near-ties) |

The interruption of 21:51 UTC (container lost GPU access; the 3.2 and 3.6
scoring had finished, the Phase-B tables had not) cost nothing but time:
the Phase-B tables were scored on restart and every number above is on
the Phase-B evaluator weights.

### Carry-overs

- **Phase 4 / G4** — "joint model passes the G3 synthetic ranking test
  unchanged": the suite's solves are reusable (`recovery_solves.json`;
  rung-1 decipherments do not depend on the backbone); re-score with the
  Phase-C weights (`language_recovery.py --stage score --ckpt …`, then
  `--stage report --primary <phase-C table>`) and compare per-cell against
  this report (`recovery_report.json`, cells × `calibration_v3-ro`).
  Re-measure calibration after Phase C with the v3 reference
  (`calibrate.py --ar-dir ar_reference/v3 --phase phase_c --version
  v3-phase_c`), derive its `-ro` copy, bump `CALIBRATION_VERSION`, re-run
  the audit with the new table adopted.
- **Abstention channel** (4.3, Phase 6): the true decipherment scores
  1.6–2.9 bits/char below its own shuffled text under every condition and a
  wrong decipherment 0.5–1.9 below (`recovery_report.json` cells
  `true_minus_shuffled_bits_mean`, `wrong_minus_shuffled_bits_mean`) — a
  per-instance shuffled-text control is a far stronger "is this language
  at all" test than any cross-language margin.
  *[Forward pointer 2026-09-01: this control became the Phase-6 **structure margin**
  (bits of a letter-shuffled copy minus bits of the decode, own condition) in the
  frozen judge `vms/apply.py::ABSTAIN_RULE` (plain ≤ 3.0 bits/char AND margin ≥ 1.5,
  fixed 2026-08-23/24). On the manuscript it spans 0.04–1.25 vs 1.49–2.48 for true
  decipherments — `docs/phase6_status.md`, `docs/project_status.md` §4.]*
- **Uncertainty statement** (Phase 6): per-document spread, the
  calibration-precision flag and the replicate flip-rate are the three
  error bars every ranking must carry; the Latin held-out split's
  heterogeneity (`apothecary_ellis_1854` ranks German) should be re-vetted
  under the 0.2 criteria before any split v2.
- **Budget**: 64 stratified draws (× 4 replicate seeds for a flip-rate)
  for both clean-text and decipherment scoring; the all-window 1% flip
  criterion is unattainable by construction and must not be "fixed" by
  spending GPU.
- **Design-doc revision**: §5b.3's "apply as an additive offset" →
  measured, stored, reported as margin uncertainty (§3.4 above).
