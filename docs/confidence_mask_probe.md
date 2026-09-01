# Confidence-masked judging: would the diffusion judge do better on a decode with its low-confidence letters blanked?

> **Record status (banner added 2026-09-01):** pre-registered 2026-08-24, run 2026-08-24, **not adopted** — see §9. §§1–8 are the plan as frozen before any number was computed; §9 records the result (E1 FAIL, E3 oracle ceiling +0.056 vs the required +0.15). The machinery (`heads/masked_bits.py`, `heads/confidence.py`, `scripts/confidence_probe.py`) stays in the tree as a probe. The "what survives" items were not pursued as masks; the denoiser posterior was built the next day as a *re-seeding proposal* instead (`heads/posterior.py`, `docs/alt_loop_plan.md` §7). **Current project position: `docs/project_status.md`.**

*Pre-registered test plan, 2026-08-24; run the same day and not adopted (§9).* *(Corrected 2026-09-01; originally read "Proposal / pre-registered test plan, 2026-08-24. Nothing here has been run." — the header was never updated when §9 was added.)*

## 1. The question

Today every cipher head hands the judge a **complete** plaintext: every ciphertext
symbol is mapped to a letter, and a "partial decipherment" means a fraction of
those letters are systematically wrong (`data/noise.py`, `SubstitutionNoise`;
`control6a` deranged keys). The judge was trained (Phase 2) to see through that.

The alternative: hand the judge only the letters the search is *sure* of and a
blank for the rest. The judge is an absorbing-state masked diffusion model, so a
blank is not a foreign token — `MASK` clamped for the whole trajectory is exactly
the "reconstruct from partial evidence" regime it was trained in. Confidence
would come from the inner n-gram search's own sensitivity: how many bits the
n-gram score loses if symbol *s* is reassigned to its best alternative letter.

## 2. Prior expectation — be honest about which direction this could go

**Reasons it might help**

- The Phase-6 verdict rests on the **structure margin** (bits of a shuffled copy
  minus bits of the decode). A wrong-key letter contributes near-zero margin at
  best and actively negative margin when it creates false structure. Removing
  such positions from both the decode and its shuffled control can only sharpen
  the margin *if* the confidence signal actually finds the wrong ones.
- On the VMS every hypothesis is at best a partial key (rung-4 synthetics: 45–66 %
  of symbols wrong). That is the regime where thinning to the reliable part is
  most valuable, and the regime where the Phase-2 curriculum was thinnest (it
  trained at 20–50 % wrong, not 60 %).
- Rare symbols are where keys go wrong in this project's own record (Borg
  < 20-occurrence types dropped; `elbo_polish` winner's curse). A confidence
  rule targets exactly them.

**Reasons it might not, or might bias the instrument**

- The judge is already flat in language verdict from 0–50 % wrong key (Phase 2,
  `docs/ngram_judge_robustness.md`). The *ranking* has little headroom.
- **Selection effect.** Training masks positions uniformly at random. A
  confidence mask blanks positions correlated with letter identity (the rare
  and hard symbols map to specific letters), so the observed subsequence has a
  different unigram distribution from the language. The score of a masked
  decode is not the Phase-6 bits/char and must never be compared across masks
  (`masked_bits.py` caveat).
- **Fairness (R1).** Latin is the hard language for the n-gram search at every
  rung. A per-language confidence mask would blank more under the Latin
  hypothesis, so a Latin decode would be judged on a different (easier or
  harder) subset than an Italian one. The ranking must use **one mask shared by
  all three conditions**; the per-language form is measured only to quantify
  the bias.
- **n-gram leakage.** The n-gram judge drifts toward German as a partial-key
  judge. Confidence derived from it inherits that prior; a rule that blanks
  "where the n-gram is surprised" would blank precisely the letters that
  distinguish languages. Sensitivity of the *key* (reassign a symbol, hold the
  rest) is far less exposed than positional surprisal, which is why the probe
  uses the former and never the latter.
- **MDL.** Blanking reduces what the decode explains. Unless blanks are charged,
  a head can "improve" by blanking everything hard — the same failure that made
  Phase 6 charge uncovered symbols at the stream's own n-gram cross-entropy.

Net prior: modest, possibly zero, gain on language ranking; a real but unproven
chance of a cleaner structure margin. The experiment is worth its cost only
because the margin is the load-bearing statistic.

## 3. Definitions

Existing untracked scaffolding (written 2026-08-24, no tests/results yet) *[plan-time text; the tests (11, `tests/test_masked_bits.py`, `tests/test_confidence.py`) and results exist — §9]*:

- `heads/masked_bits.py::paired_bits_masked(evaluator, rows, observed, conditions)`
  — blanks sit at `MASK` in `z_t` for every stratum (context, never target);
  loss sums over positions that are observed *and* masked by the draw;
  normalized per **observed** char; one masking realization shared by all rows
  and conditions (CRN). All-true `observed` must reproduce `two_tier.paired_bits`
  bit-for-bit — that identity is the first test to write.
- `heads/confidence.py`:
  `ngram_sensitivity(symbols, sym_to_letter, lm)` — per symbol,
  `min_{l'≠l(s)} [score(decode) − score(decode with s→l')]`; `sensitivity_mask`
  keeps top symbols until coverage target; `freq_mask` (keep most frequent
  symbols — language-neutral by construction); `random_symbol_mask`,
  `random_position_mask` (matched-coverage controls); `oracle_mask` (keep truly
  correct positions — ceiling); `shuffle_within_mask` (permute observed among
  observed, blanked among blanked — the structure-margin control that keeps the
  observed letter multiset fixed).

Quantities per (instance, rule, coverage c, language condition ℓ):

- `bits_obs(ℓ)` — `paired_bits_masked` on the decode.
- `bits_shuf(ℓ)` — same on `shuffle_within_mask(decode)`, same mask, same seed.
- `margin(ℓ) = bits_shuf(ℓ) − bits_obs(ℓ)`, structure margin per observed char.
- `mdl_total` — `bits_obs · n_obs + key_bits + choice_bits + (n_all − n_obs) · r_uncov`
  with `r_uncov` the stream's best held-out no-cipher n-gram cross-entropy,
  exactly the Phase-6 rule for uncovered symbols (`vms/apply.py` ~l.712). A blank
  is an uncovered symbol.
- ranking: argmin over ℓ of `mdl_total`, as in Phase 6 — never the raw ELBO.

## 4. Arms

| arm | mask | deployable | purpose |
|---|---|---|---|
| `full` | none (c = 1) | yes | frozen Phase-6 baseline; must equal current numbers |
| `oracle` | truly-correct positions | no | ceiling on any confidence rule |
| `sens_shared` | n-gram key sensitivity, `min` over the three language keys | yes | **the candidate** |
| `sens_perlang` | n-gram key sensitivity, one mask per hypothesis | yes but biased | measures the R1 bias; not a ranking arm |
| `freq` | most frequent symbols | yes | cheap language-neutral alternative |
| `rand_sym` | random symbols, matched coverage | yes | thinning-only control |
| `rand_pos` | random positions, matched coverage | yes | thinning + selection-effect control |

Coverage sweep c ∈ {0.9, 0.8, 0.7, 0.5} for every arm except `full`. If a
confidence arm does not beat `rand_sym` at the same c, the confidence signal
contributed nothing and only the thinning did.

## 5. Experiments

All scoring on the frozen evaluator (`phase_c-85m-seed0`, sha256 checked
against `analysis/phase5/evaluator_freeze.json`), budget 64 strata × 4 replicate
seeds, CRN across conditions and arms (same seed for every arm of an instance).

### E1 — Does key sensitivity find the wrong symbols? (no judge needed)

Synthetic rung-1/rung-2 instances (Phase-5 generators, all three languages,
window 300 and 1000 chars). For each, build keys at controlled wrongness
f ∈ {0, 0.1, 0.2, 0.3, 0.5, 0.65} via the `control6a` derangement, plus the
key the inner search actually converged to. Compute `ngram_sensitivity`.

- AUROC of sensitivity vs "symbol correct", per f and per language.
- Same for symbol frequency (the `freq` rule) — is sensitivity better than count?
- Pass bar: AUROC ≥ 0.75 at f ≤ 0.3 in every language, and the Latin AUROC
  within 0.05 of the others (otherwise the shared mask is Latin-driven).
  If E1 fails, stop: nothing downstream can work.

### E2 — Language ranking on synthetics with wrong keys

Same instances and keys. Score every arm × coverage, rank by `mdl_total`.

- Language-recovery accuracy vs f, per arm. Baseline is Phase 5 (99.4 % rung 1
  at ≥ 200 chars, flat under wrong keys).
- Ranking margin / calibration uncertainty (`CalibrationTable.margin_uncertainty_bits`)
  and replicate flip-rate.
- Per-language bias check: mean over instances of `bits_obs(ℓ_true)` shift from
  `full`, by language. A rule that lowers Italian by 0.3 bits and Latin by 0.1
  is a bias, not an improvement.
- `sens_perlang` reported separately, with the fraction of instances where its
  ranking differs from `sens_shared`.

### E3 — Structure-margin separability (the one that matters)

Positives: the nine Phase-6 true decipherments re-run through each arm, plus
the E2 instances at f ≥ 0.5. Negatives: `voynichesque` gibberish and
out-of-inventory contamination from the Phase-6 control battery, given a key by
the same inner search so the sensitivity rule sees the same kind of input.

- Distribution of `margin` for positives vs negatives per arm and c.
- Statistic: **gap** = min(positive margin) − max(negative margin). Phase 6:
  1.49 − 1.51 = −0.02 (one crossing each way). Also AUROC and the width of the
  threshold band that abstains on all negatives and no positives.
- `margin` must be computed with `shuffle_within_mask`, never a plain shuffle,
  or the unigram distribution the mask selected leaks into the margin.

### E4 — Selection-effect sanity check

Clean held-out text, correct key (f = 0). For each c, compare `bits_obs` under
`sens_shared` and `freq` against `rand_pos` at the same c. With a correct key
any gap here is pure selection effect (which letters got blanked), not signal.
Report it per language; it bounds how much of an E3 gain is artefact.

### E5 — VMS re-application (only if E1–E3 pass)

Re-score the 87 Phase-6 cells with the winning arm at the c chosen in E3.
Report beside, not instead of, the frozen Phase-6 table. The abstention rule is
**not** re-tuned on manuscript numbers: the threshold band comes from E3.

## 6. Pre-registered decision rule

Adopt confidence-masked judging for the structure margin only if all hold:

1. E1 passes (sensitivity is a real, language-balanced predictor of key errors).
2. E3 gap improves over `full` by ≥ 0.15 bits/char at some c, and that arm
   beats `rand_sym` at the same c by ≥ 0.10.
3. E2 language recovery does not fall below `full` by more than 1 point at any
   f ≤ 0.3, and the per-language shift in `bits_obs(ℓ_true)` is within
   `margin_uncertainty_bits` for every language.
4. E4 selection effect < half the E3 gain.

Otherwise record the negative result in `docs/` and leave the machinery as a
probe.

## 7. Known traps

- `logaddexp(−∞, −∞)` when blanks and NULL slots coincide on the 2N frame — keep
  the smoke test; the probe runs on the plain path (collapsed hard decodes)
  first, the NULL frame only for the Naibbe head if E3 passes.
- Masked scores are not comparable across masks. Every comparison in §5 is
  either same-mask (decode vs shuffle, language vs language) or on a
  mask-independent statistic (rank, gap, AUROC).
- Rung 3/4 heads: confidence is per key entry (Naibbe table cell / arithmetic
  token), not per cipher symbol; `confidence.py` currently covers only the
  1:1 position heads. Extend after E1–E3 on rungs 1–2, not before.
- Windows where a row becomes entirely blank contribute nothing
  (`paired_bits_masked`); at c = 0.5 check that no instance loses a window.

## 8. Implementation checklist

- `tests/test_masked_bits.py`: all-true mask ≡ `paired_bits`; blank positions
  never in the loss; `shuffle_within_mask` preserves both multisets; NaN smoke.
- `tests/test_confidence.py`: `oracle_mask` AUROC = 1; masks identical across
  language conditions for `freq`/`sens_shared`; coverage targets hit.
- `scripts/confidence_probe.py --stage e1|e2|e3|e4|e5 --restat`, CRN seeds
  logged; artifacts under `DATA_ROOT/analysis/confidence_probe/`; ClearML task
  per stage.
- Cost: `ngram_sensitivity` is `|symbols| × (A−1)` n-gram rescorings per key
  (~25 × 24 = 600 for rung 1, cheap); the judge cost is arms × coverages × 4
  seeds ≈ 25× the Phase-6 per-instance budget, so E2/E3 run at 300-char windows
  first and 1000 only for the surviving arm.

---

## 9. Results (run 2026-08-24) — **not adopted**

*Everything above was written before any number was computed; this section
records what happened. Artifacts: `DATA_ROOT/analysis/confidence_probe/`
(`instances.json`, `e1.json`, `e2_shard{0,1}of2.json`, `e3_shard{0,1}of2.json`,
`report.{json,md}`); code `heads/masked_bits.py`, `heads/confidence.py`,
`scripts/confidence_probe.py`; tests `tests/test_masked_bits.py`,
`tests/test_confidence.py` (11 tests, incl. the all-true ≡ `paired_bits`
identity, exact).*

### Deviations from the plan

- **Instance set.** Instead of re-running the inner search at 300 / 1000
  chars, the probe reuses the Phase-5 solved instances — rung 1 at L = 400
  (5 trials × 3 languages) and L = 700 (3 × 3, prepared but not scored), rung 2
  at L = 408 (6 × 3, 54 symbols). Their shortlists *are* the keys the inner
  search converged to (rung-2 solves cost 480 restarts each and were not
  worth repeating). Controlled-wrongness keys f ∈ {0, .1, .2, .3, .5, .65}
  via `confidence.derange_key` (the control6a derangement generalized to
  homophonic maps). 33 instances × 9 keys = 297 E2 records.
- **E3 battery.** The sub1to1 head only (the probe's stated scope): 9
  positives, 9 voynichesque, 12 contamination, 9 shuffled, each under its
  three per-hypothesis converged keys from the Phase-6 controls; scored on
  the Phase-6 window (0, 1024) with the Phase-6 job seeds. The `full` arm
  reproduces the frozen Phase-6 bits to **max |Δ| = 1.8 × 10⁻¹⁵**.
- **E2/E3 were run although E1 failed** (the plan says stop). They were
  cheap (≈ 3 GPU-hours) and the oracle arm answers a question E1 cannot —
  whether *any* confidence rule could have helped. They are reported as
  exploratory; the pre-registered decision was already settled at rule 1.
- No ClearML tasks (no other analysis script in this repo logs to ClearML;
  artifacts and seeds are on disk).

### E1 — key sensitivity finds wrong symbols, but not fairly enough: **FAIL**

AUROC of per-symbol confidence vs "symbol correct" (mean over instances):

| cell | sens, true-language LM | **sens shared (min over LMs)** | freq | shared la / it / de |
|---|---|---|---|---|
| sub1to1 L400 f0.1 / 0.2 / 0.3 | 0.98 / 0.91 / 0.82 | 0.87 / 0.75 / 0.71 | 0.45 / 0.49 / 0.52 | .83/.92/.86 · .78/.70/.77 · .69/.74/.69 |
| sub1to1 L400 f0.5 / 0.65 | 0.66 / 0.66 | 0.60 / 0.57 | 0.48 / 0.54 | — |
| homophonic L408 f0.1 / 0.2 / 0.3 | 1.00 / 0.95 / 0.91 | 0.89 / 0.80 / 0.74 | 0.49 / 0.53 / 0.50 | .90/.96/.80 · .82/.87/.71 · .73/.81/.68 |
| homophonic L408 f0.5 / 0.65 | 0.76 / 0.64 | 0.67 / 0.60 | 0.52 / 0.47 | — |
| sub1to1 L400 search keys (la/it/de, 38/56/16 % wrong) | 0.87 / 0.64 / 0.94 | 0.82 / 0.59 / 0.89 | 0.74 / 0.68 / 0.78 | |

The signal is real when you know the language (0.82–1.00 at f ≤ 0.3) but
the only fair, deployable form — the min over the three hypotheses' LMs —
loses 0.1–0.2 of it and is dragged below the 0.75 floor by whichever
language the wrong-language LMs disagree on (German at f = 0.2–0.3,
0.68–0.72). Pass bar met in 2 of 9 cells (f = 0.1 only). Frequency is chance
on deranged keys (by construction — the derangement is uniform over symbols)
and ≈ 0.7–0.8 on search keys, whose errors concentrate on rare symbols. At
f ≥ 0.5 — the regime the probe was motivated by — no rule exceeds 0.67.

### E2 — language recovery (MDL total, 33 instances, 3 conditions)

| arm | f0 | f0.1 | f0.2 | f0.3 | f0.5 | f0.65 | search |
|---|---|---|---|---|---|---|---|
| full | 0.97 | 0.94 | 0.94 | 0.91 | 0.82 | 0.58 | 0.97 |
| **oracle@0.5** (blank exactly the wrong letters) | 1.00 | 1.00 | 0.97 | 1.00 | **1.00** | **0.94** | 0.67 |
| oracle@0.9 / 0.8 / 0.7 | 0.97 / 0.97 / 0.97 | | | | 0.91 / 1.00 / 0.94 | 0.61 / 0.58 / 0.70 | |
| sens_shared@0.9 / 0.8 / 0.7 / 0.5 | 0.97 / 0.97 / 1.00 / 0.88 | | | 0.97 / 0.94 / 0.97 / 1.00 | 0.82 / 0.76 / 0.73 / 0.64 | 0.64 / 0.58 / 0.58 / 0.61 | 0.97 / 0.97 / 0.94 / 0.82 |
| freq@0.9 … 0.5 | 1.00 … 0.79 | | | 0.88 … 0.82 | 0.67 … 0.55 | 0.48 … 0.42 | |
| rand_pos@0.9 … 0.5 | 0.97 | | | 1.00 … 0.85 | 0.85 … 0.79 | 0.61 | |

- **The judge does read "correct letters + blanks" better than "correct
  letters + wrong letters".** With the wrong half of the positions blanked
  (oracle@0.5, purity 1.0) the true-language bits drop by 0.7–1.0 bits/char
  and language recovery at f = 0.65 goes from 0.58 to 0.94 (latin 0.36 →
  0.91, german 0.45 → 0.91). That is the ceiling.
- **No deployable rule reaches it.** `sens_shared` at f ≥ 0.5 is at or below
  `full` (0.64–0.82 vs 0.82; 0.58–0.64 vs 0.58) and no better than
  `rand_pos` at matched coverage — the E1 AUROC of ≈ 0.6 at this wrongness
  is not enough to enrich the observed set. At f ≤ 0.3 every arm is within
  noise of `full` (already ≥ 0.91), the Phase-2 flatness.
- **Bias.** The per-language shift of bits_obs(ℓ_true) under `sens_shared`
  is smaller for German than for Latin/Italian (la−de −0.04 / −0.06 / −0.08
  / −0.16 at c = 0.9 / 0.8 / 0.7 / 0.5; it−de −0.05 … −0.18) against a
  la–de calibration margin uncertainty of 0.067 — the shared mask blanks
  more Latin/Italian-hard letters, i.e. rule 3 fails from c = 0.7 down.
  `sens_perlang` ranks differently from `sens_shared` on 13 / 20 / 24 / 28 %
  of instances at c = 0.9 / 0.8 / 0.7 / 0.5: the per-language form is a
  materially different — and unfair — instrument.
- **E4 selection effect** (f = 0, rule − rand_pos at the same c): −0.06 to
  −0.09 at c = 0.9, −0.2 to −0.35 at c = 0.7, −0.55 to −0.81 at c = 0.5. The
  letters a confidence rule keeps are the *easy* letters; most of any raw
  bits improvement under a rule is this, not signal. Rule 4 fails everywhere.

### E3 — structure-margin separability: **every arm shrinks the gap**

Positive = 9 Phase-6 true decipherments (SER 0); negative = voynichesque +
contamination (shuffled reported separately). Own-condition margin at the
instance's MDL-top hypothesis, bits per observed char:

| arm | positives | voynichesque max | contamination max | **gap** | AUROC |
|---|---|---|---|---|---|
| full (= Phase 6) | 1.45 – 2.42 | 1.44 | 1.43 | **+0.005** | 1.000 |
| oracle @0.9 / 0.8 / 0.7 / 0.5 | 1.32–2.30 … 0.25–1.35 | (no truth) | 1.26 / 1.06 / 0.89 / 0.35 | +0.056 / +0.042 / +0.020 / −0.100 | 1.00 / 1.00 / 1.00 / 0.96 |
| sens_shared @0.9 / 0.8 / 0.7 / 0.5 | 1.24–2.36 … 0.21–0.82 | 1.38 / 1.33 / 1.16 / 0.64 | 1.27 / 1.16 / 0.87 / 0.52 | −0.14 / −0.33 / −0.52 / −0.43 | 0.99 / 0.96 / 0.86 / 0.64 |
| freq @0.9 … 0.5 | 1.03–2.02 … 0.12–0.47 | 1.41 … 0.81 | 1.16 … 0.38 | −0.38 / −0.61 / −0.69 / −0.69 | 0.95 … 0.62 |
| sens_perlang @0.9 … 0.5 | | | | −0.35 / −0.65 / −0.71 / −0.77 | 0.96 … 0.56 |
| rand_sym @0.9 … 0.5 | | | | −0.08 / −0.45 / −0.45 / −0.66 | 0.99 … 0.81 |
| rand_pos @0.9 … 0.5 | 1.30–2.27 … 0.52–1.33 | 1.26 … 0.51 | 1.28 … 0.52 | +0.02 / −0.05 / +0.03 / −0.01 | 1.00 / 1.00 / 1.00 / 1.00 |

- The positives here carry **no wrong letters**, so on this battery a rule
  can only remove context. Symbol-level blanking (every deployable rule,
  and the random-symbol control) removes it from true decipherments faster
  than from gibberish, because a blanked *symbol* takes every occurrence of
  a letter out of the observed sequence and the remaining letters lose the
  neighbours that made them predictable; the margin per observed char
  collapses for positives first. Position-level thinning is neutral
  (rand_pos gap ≈ 0 at every c) — it is the symbol structure of the mask,
  not the thinning, that hurts. Note the oracle row's gap is against
  contamination only (voynichesque has no plaintext to define an oracle).
- **The ceiling is +0.056** (oracle@0.9, vs the pre-registered ≥ +0.15 over
  `full`), so no confidence rule, however good its E1 AUROC, could have
  passed rule 2 on this statistic.
- **Half-right decipherments are not separable from gibberish by the
  margin under any arm.** The E2 instances at f ≥ 0.5 (66 records) have
  margins 0.03–0.87 under `full` (mean 0.34) and 0.07–1.55 under oracle@0.5
  (mean 0.46) — inside the voynichesque band (0.9–1.5 at full) and the VMS
  band (0.04–1.25). Blanking the wrong half with perfect knowledge lifts the
  raw score (§E2) but not the sequential-structure evidence: with half the
  neighbours masked, decode and shuffle converge. The margin measures
  *available context × structure*, and a mask removes the first factor.

### §6 decision

| rule | result |
|---|---|
| 1. E1 passes | **no** (2 / 9 cells) |
| 2. E3 gap ≥ full + 0.15 and ≥ rand_sym + 0.10 | **no** — best deployable gain −0.14; ceiling +0.06 |
| 3. E2 recovery within 1 pt at f ≤ 0.3; per-language shift within `margin_uncertainty_bits` | recovery yes at c ≥ 0.7; bias fails at c ≤ 0.7 |
| 4. E4 selection effect < ½ E3 gain | **no** (no gain; effect 0.09–0.8) |

**Not adopted.** `paired_bits_masked` and the mask rules stay in the tree as
a probe; the Phase-6 structure margin and abstention are unchanged, and E5
was not run.

### What survives

1. The judge's *raw score* genuinely benefits from blanks in place of wrong
   letters (oracle: +0.36 language recovery at 65 % wrong). Usable only with
   a confidence signal far better than n-gram key sensitivity at f ≥ 0.5
   (AUROC ≈ 0.6); a search that produces posterior marginals per symbol
   (e.g. the Sinkhorn head's soft assignment, or a bootstrap over inner
   restarts) is the natural candidate, and it must be language-shared.
   *[Note added 2026-09-01: a per-symbol posterior from the denoiser itself
   was built 2026-08-25 (`heads/posterior.py`) but used as a *proposal* —
   re-seeding the symbols whose posterior disagrees with the key inside the
   alternating n-gram ↔ diffusion loop (`docs/alt_loop_plan.md` §7) — not
   as a confidence mask for judging; confidence-masked judging was not
   retried. The 2026-08-29 judge-alternatives study found different
   survivors (rare-aware margin, lexical density ≥ 7, learned LR — none
   pre-registered or adopted; `docs/judge_alternatives.md`). The order-k
   surrogate shuffles and conditional − unconditional gap of item 2 were
   not pursued.]*
2. The structure margin is a *context-limited* statistic: anything that
   thins the observed sequence shrinks it for true decipherments first. That
   argues for the alternative nulls discussed in the review — order-k
   surrogate shuffles (structure beyond the word template) and the
   conditional − unconditional gap (language-specific benefit without a
   null) — rather than for masking.
3. Masked scores are not comparable across masks (the E4 numbers are the
   size of that artefact: up to 0.8 bits/char at c = 0.5).
