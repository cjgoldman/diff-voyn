# Plan — adaptive-budget racing polish (`race_polish`)

Status: **PLAN pre-registered 2026-08-25; code + proof-of-life done the same day (§7) — the full study (§3) is NOT run.** Addresses the
Phase-6 carry-over (`docs/phase6_status.md` §6.6 finding): `ladder.elbo_polish`
selects on noise at Borg scale (argmin over ~1,375 budget-8 paired estimates per
sweep, winner's curse; accepted 6/6 on Borg and doubled the page SER). Task
lineage: 5.3 / 6.6 outer tier.

## 1. The idea in one paragraph

Replace "score every neighbour at a fixed small budget, take the argmin" with a
**race**: screen every neighbour cheaply on the *paired difference* to the current
key, eliminate candidates whose confidence interval says they are not
improvements, re-score the survivors with **fresh masks** at a doubled budget,
pool, eliminate again, until either one candidate is confidently better than the
current key (commit it) or nobody clears a multiplicity-corrected threshold at
the maximum budget (stop — there is no real improvement). CRN makes the paired
differences low-variance and their variance is estimable from the per-stratum
draws already computed.

## 2. Code changes

### 2.1 `heads/two_tier.py` — expose per-stratum values
- Add `paired_bits_strata(evaluator, rows, conditions, *, n_strata, seed, batch)
  -> np.ndarray [n, C, K]` returning the per-stratum bits/char contributions
  (`out.reshape(K, n) / (t·L) / ln2`, length-weighted across windows exactly as
  `paired_bits` does). `paired_bits` becomes `paired_bits_strata(...).mean(-1)`
  and must stay **bit-identical** to today (the metrology-equality test
  `tests/test_heads.py::…paired_bits…` guards this). Note K ≤ n_strata (strata
  whose mask is empty are dropped) — return the actual K and keep the
  `/ n_strata` convention so means agree.
- Seed offsets: a *stage* seed `seed + 1009·stage` so every stage draws fresh
  masks; stage 0 may reuse the caller's seed for reproducibility.

### 2.2 `heads/ladder.py` — `race_polish`
Signature mirrors `elbo_polish` (drop-in): `(evaluator, cipher_ids, sym_map, *,
language, choice_fn, sweeps, budgets=(4, 16, 64, 128), batch, seed, pair_swaps,
set_moves, alpha=0.05, se_floor_frac=0.5, max_survivors=(None, 64, 12, 4),
per_move_confirm=True) -> (map, info)`.

Per sweep:
1. Enumerate the neighbourhood exactly as `elbo_polish` does (row 0 = current key).
2. **Stage 0 (screen)** at `budgets[0]`: `paired_bits_strata` on all rows; per
   candidate `d_k = bits_k − bits_0` per stratum (plus the choice-bits delta from
   `choice_fn`, which is deterministic). Mean `Δ̂`, standard error from the K
   per-stratum differences, **shrunk**: `se = max(se_k, se_floor_frac ·
   median_k se_k)` (an 8-draw SE is itself noisy; the floor stops a lucky
   low-variance candidate slipping through).
3. **Elimination rule** with N candidates alive: drop k if `Δ̂_k − z_N·se_k > 0`
   (confidently not an improvement) where `z_N = Φ⁻¹(1 − alpha / N)` (Bonferroni
   over the live set; ≈ 3.9 at N = 1375, ≈ 2.9 at 64, ≈ 2.4 at 12). Also cap
   survivors at `max_survivors[stage]` by `Δ̂` (log the cap when it binds — "no
   silent caps").
4. **Stages 1..S**: re-score survivors + row 0 at `budgets[s]` with fresh masks,
   pool draws across stages (all fresh, so pooling is unbiased), recompute
   `Δ̂`/`se`, eliminate again.
5. **Commit rule**: after the last stage (or earlier if it already holds) commit
   the best survivor iff `Δ̂ + z_N·se < 0` on the **pooled non-screen draws**
   (stage-0 draws are excluded from the commit decision — they selected the
   candidate). If nothing qualifies the sweep ends with "no move" and the polish
   stops.
6. `per_move_confirm=True`: the commit test above *is* the per-move confirmation
   the carry-over asks for; keep the final start-vs-end comparison at
   `budgets[-1]` as well (so `info["accepted"]` keeps its meaning for the
   `vms/apply.py` record) but it should never be the only gate.

`info` records per sweep: neighbourhood size, survivors per stage, budget spent
(candidate-draws), the committed move, `Δ̂ ± se` at commit, and the z used.
Total cost per sweep must be logged so the report can compare against
`elbo_polish`'s fixed 1375 × 8 + 2 × 64.

Keep `elbo_polish` untouched (it is the recorded Phase-5/6 behaviour). Add a
`method` switch (`"greedy"|"race"`) where it is called: `vms/apply.py::score_*`
(`polish_sweeps` neighbour), `scripts/rung2_diffusion.py`,
`scripts/control6b_pooled_search.py`; default stays `"greedy"` until §4 passes.

### 2.3 Unit tests (`tests/test_race_polish.py`)
CPU only, toy evaluators already in the tree:
- `paired_bits_strata(...).mean(-1) == paired_bits(...)` bit-for-bit, single and
  multi-window rows.
- On `_ToyEvaluator` (`tests/test_control6.py`, exact target logits, zero noise)
  from a one-swap-away start: `race_polish` recovers the true key in 1 sweep,
  `set_moves=False` keeps a bijection.
- **Winner's-curse test** — a `_NoisyToyEvaluator` whose logits are the target
  logits plus per-forward Gaussian noise: start *at the true key*, neighbourhood
  ~300 candidates. Assert `elbo_polish` (budget 8, 4 sweeps) moves the key on
  ≥ 1 of 5 seeds (reproduces the failure) and `race_polish` moves it on 0 of 5.
  Then start one swap away: `race_polish` still recovers it (power, not just
  conservatism).
- Cost accounting: `info["draws_total"]` ≤ the equivalent fixed-budget cost for
  the same neighbourhood at `budgets[-1]`.

## 3. Validation on ground truth (GPU, frozen evaluator `phase_c-85m-seed0`)

Script `scripts/race_polish_study.py --stage {run,report}`, artifacts
`DATA_ROOT/analysis/race_polish/`. Every arm uses the *same* start keys as the
recorded runs so the comparison is polish-only.

| testbed | start keys | ground truth | metric |
|---|---|---|---|
| **Borg** (the failure case) | MDL-pick keys from `analysis/phase6/anchors/borg_scores_shard*.json` (6 cells: 3 hyps × 2 windows) | Örneholm plaintext via `anchors._borg_ser` | page-median SER before/after; accepted?; symbols changed; draws spent |
| **Rung-2 synthetics** (the case where the old polish helped) | MDL-pick keys from `analysis/phase5/rung2_scores.json` (18 true-hyp cells + wrong-hyp cells) | true maps in `rung2_solves.json` | SER before/after; must not lose the 8/18 improvements `elbo_polish` gave |
| **Null control** | rung-2 cells started **at the true key** | trivially the start | move rate (should be ~0; `elbo_polish`'s is the winner's-curse rate) |
| **Power control** | rung-2 true keys with 1, 2, 4 random symbols corrupted | true key | fraction repaired vs corruption count, per method |
| **VMS regression** (verdict must not change) | the 2 best homophonic VMS cells from `analysis/phase6/` (pre-polish margins 0.83/0.84) | none | structure margin post-race vs post-greedy (1.05–1.11); abstention holds |

Arms: `greedy` (= `elbo_polish`, re-run so both get identical starts and the same
evaluator), `race` with default budgets, `race` with `budgets=(8,32,128)` as a
cost-sensitivity point. 4 replicate seeds per cell (the project's flip-rate
convention). Run on one GPU; Borg cells are the expensive ones (~1,375 × 4000
chars per screen ≈ the old cost per sweep; expect ≤ 1 h per cell).

## 4. Acceptance (fixed before running)

- **A1 (fixes the failure)**: Borg, all 6 cells: race never worsens page-median
  SER by more than 0.005 over the start key, and the Latin cells improve or hold
  (greedy: 0.110 → 0.217).
- **A2 (keeps the win)**: rung-2 true-hypothesis cells: mean SER after race ≤
  mean SER after greedy + 0.002; oracle-reaching count ≥ greedy's 8/18 − 1.
- **A3 (no false moves)**: null control move rate ≤ 1/18 per seed (greedy's
  rate is the baseline to report).
- **A4 (power)**: 1-symbol corruptions repaired ≥ 90 %, 2-symbol ≥ 75 %.
- **A5 (verdict stable)**: both VMS cells still abstain under `ABSTAIN_RULE`;
  report the margins either way.
- **A6 (cost)**: mean candidate-draws per sweep ≤ greedy's 11k on Borg.
- Unit tests green; `ruff`/`black` clean.

If A1–A5 pass, flip the default `method` to `"race"` in `vms/apply.py` and
record it as a versioned outer-tier change (Phase-6 numbers are *not* re-run
retroactively; the VMS regression arm is the bridge). If A2 or A4 fail the race
is too conservative — the knobs are `alpha`, `se_floor_frac`, and adding a
budget stage; retune on the rung-2 synthetics only, never on Borg (Borg is the
held-out failure case).

## 5. Out of scope / follow-ons

- The **alternating loop** (race polish → re-seed the n-gram search → race
  again) is a separate study that depends on this one passing; it gets its own
  pre-registration.
- Rung-3/4 heads (block maps, arithmetic keys) have different neighbourhoods;
  `race_polish` is written against the symbol→letter map interface only.
- Borg's < 20-occurrence glyphs and the transcription/edition mismatch stay as
  they are — A1 is about the polish not making things worse, not about the
  4.10 % literature target.

## 6. Order of work

1. `paired_bits_strata` + equality test (½ day, CPU).
2. `race_polish` + toy/noisy-toy tests (1 day, CPU).
3. `race_polish_study.py` run stage on rung-2 synthetics + controls (GPU, ~3 h),
   tune on those only.
4. Borg + VMS arms (GPU, ~8 h wall).
5. Report stage → `analysis/race_polish/report.md`, acceptance table, update
   `docs/phase6_status.md` carry-over list and `CLAUDE.md` state line.

## 7. Proof-of-life results (2026-08-25, before the full study)

Implemented: `two_tier.paired_bits_strata` (per-stratum values; `paired_bits`
unchanged bit-for-bit), `ladder.race_polish`, `tests/test_race_polish.py`
(3 pass), `scripts/race_polish_pol.py` → `analysis/race_polish/pol.{json,log}`.
Frozen evaluator, **2 sweeps** each, greedy = `elbo_polish` budget 8 /
confirm 64; race = budgets (4, 16, 64, 128), caps (—, 64, 12, 4), α = 0.05.

| cell | start SER | greedy | race | race draws |
|---|---|---|---|---|
| rung-2 latin t2, **null** (true key) | 0 | 0 (no move) | 0 (no move; 2655 → 2598 → 64 → 1 → 0) | 57k |
| rung-2 german t0, **null** | 0 | 0 | 0 (no move) | 57k |
| rung-2 latin t1, MDL pick | 0.0049 | 0.0025 | **0.0000** (2 moves) | 115k |
| rung-2 italian t3, MDL pick | 0.0172 | 0.0172 (moves rejected) | **0.0074** (1 move) | 112k |
| rung-2 latin t2, 2 symbols corrupted | 0.0196 | 0 | 0 | 109k |
| rung-2 german t0, 2 symbols corrupted | 0.0368 | 0 | 0 | 109k |
| **Borg latin w0** [0, 4000), MDL pick | 0.1195 | **0.3100** | **0.2251** (2 moves) | 45k |

Toy (noisy exact judge, 200 chars): race never moves a true key (0/8 at two
noise levels), repairs 6/8 one-symbol corruptions; greedy's own confirmation
also catches its bad moves at that size, so the toy does not reproduce the
curse — reported, not required (§2.3).

**Reading.** As a *selection rule* the race behaves exactly as designed:
null controls unmoved, both corruption controls repaired, and on the two
rung-2 picks it beats greedy (one to the true key). But it does **not**
rescue Borg, and the trace says why — the first committed move, symbol 0
(415/4000 occurrences, truly `e`) → `z` (an unused letter), had
Δ̂ = −0.134 ± 0.019 bits/char on *fresh* 16-draw masks: a 7σ effect, not a
lucky draw. Decomposed at budget 64:

| | ELBO bits/char | choice bits/char | MDL total | SER |
|---|---|---|---|---|
| start | 2.961 | 1.559 | 4.520 | 0.120 |
| + symbol 0 → `z` | 3.048 (+0.087) | 1.316 (−0.243) | 4.364 | 0.225 |
| + symbol 14 (`i`) → `w` | 3.121 (+0.160) | 1.176 (−0.383) | 4.297 | 0.225 |

The diffusion judge *dislikes* both moves; the **homophonic choice-bits
term** rewards them three times as strongly, because `e` has 5 homophones
in the start key (log₂5 bits per occurrence) and making its most frequent
symbol the sole homophone of a spare letter costs nothing in that
accounting. This is the "`e` → `z`, 104 occurrences" of
`docs/phase6_status.md` §6.6. **Borg's degradation is therefore mostly an
objective problem, not a winner's-curse problem**: the polish maximises
ELBO + choice bits, that total genuinely prefers a wrong key at Borg scale,
and a better selection rule finds that preference more reliably. The
judge's tolerance of a wrong symbol in 10 % of positions (+0.087 bits/char
for `z` in place of `e`) is the §2.7 contradiction of
`docs/project_goals_and_progress.md` — the curriculum made the judge
insensitive to exactly this kind of key error, and the choice term exploits
it. The winner's-curse diagnosis in Phase 6 was incomplete: greedy's
2-sweep result here (0.310, worse than the race's 0.225) shows selection
noise adds damage on top, but the direction is set by the objective.

**Consequence for the plan.** A1 cannot pass by changing the selection
rule alone. Candidate fixes to the *objective* (to be chosen, not
pre-registered here): (a) polish on the ELBO alone with the choice term
held at the start key's value (the choice bits are a cell-ranking device,
not evidence about which of two same-size keys is right); (b) charge the
key description for enlarging the letter inventory / moving mass between
homophone classes; (c) forbid moves that create a new homophone class for a
letter unused by the start key. The pure-ELBO check (a) is being run now on
the same Borg window (`analysis/race_polish/borg_pure_elbo.json`).

**Pure-ELBO check (option a), same Borg window, 3 sweeps, race defaults
(`analysis/race_polish/borg_pure_elbo.json`):** 3 moves committed (symbols
14, 39, 53; gains 0.016 / 0.007 / 0.009 bits/char), SER 0.1195 → **0.1194**,
median page 0.1092 → 0.1088, 26 min. With the choice term frozen the race
**does no harm** on the failure case (A1's "never worsens by > 0.005"
holds on this cell) while the MDL-objective race and greedy degrade it to
0.225 / 0.310. It does not materially *improve* Borg either — consistent
with the judge's built-in tolerance of wrong symbols — so on a 55-symbol
key the honest expectation for any ELBO-driven polish is "hold", not
"repair"; the repairs the race does deliver are the rung-2-scale ones.

**Where this leaves the plan.** Selection rule: fixed (race). Objective:
the homophonic choice term must not be part of the *polish* objective
(keep it in the cell ranking, where it does its job of penalising verbose
decodes; a polish never changes the decode length, so the only thing the
term does inside a polish is reward moving mass onto spare letters). The
full study (§3) should therefore run the race with `choice_fn=None` as the
primary arm, keep the MDL-objective race as a documented arm, and add the
rung-2 check that dropping the choice term from the polish does not
re-admit the degenerate maps Phase 5 saw in the *pure-ELBO pick* (a
different step — the shortlist selection — which keeps its MDL ranking).
Not run: the other five Borg cells, the 18-cell rung-2 sweep with
replicate seeds, the VMS regression arm, cost tuning (the budget-4 screen
eliminates only 2–8 % at N ≈ 2,600, so the cap does the pruning; a
budget-8 screen or a two-sided cap rule would cut the ~110k draws per
rung-2 cell — greedy spends ~21k there and ~11k on Borg, against the
race's 45k).

**Greedy with the choice term off, same cell, 3 sweeps, budget 8 / confirm 64
(`analysis/race_polish/borg_pure_elbo_greedy.json`):** accepted, 3 moves
(symbols 26 → `h`, 53 → `h`, 25 → `r`; gains 0.006–0.008 bits/char), SER
0.1195 → **0.1198**, median page 0.1092 → 0.1093, 11 min. Side by side on
Borg latin w0 from the same MDL-pick key:

| polish objective | greedy (`elbo_polish`) | race (`race_polish`) |
|---|---|---|
| ELBO + choice bits (as run in Phase 6) | 0.310 (2 sweeps) | 0.225 (2 sweeps) |
| ELBO only | **0.1198** (3 sweeps, 11 min) | **0.1194** (3 sweeps, 26 min) |

Reading: on this cell the objective is the whole story — removing the
choice term turns both polishes from "destroys the key" into "holds the
key", and the greedy-vs-race difference collapses to 0.0004 SER (different
moves, five symbols apart, all within the judge's tolerance band). The
selection-rule fix still matters where the noise regime bites (greedy's
2-sweep MDL result was 0.085 SER worse than the race's on the same wrong
objective, and greedy's accepted pure-ELBO moves have gains of 0.006–0.008
bits/char at budget 8, which is the winner's-curse range), but the race's
extra cost — 2.4× the wall time here — buys nothing measurable once the
objective is right on a 55-symbol key. Neither polish repairs Borg.
