# Plan — alternating n-gram ↔ diffusion key search (proof of life)

Status: **PLAN 2026-08-25, not yet run.** Follow-on to `docs/race_polish_plan.md`
§5 ("the alternating loop is a separate study"). Task lineage: 5.3 / 6.6 outer
tier. Scope is deliberately *proof of life*: does the frozen diffusion judge
move an n-gram search out of a local optimum it provably cannot leave on its
own, and does it do so **because of what the judge knows** rather than because
it adds randomness?

## 1. The question, and why it is not already answered

The n-gram inner tier (SA over uniform-random symbol reassignments + greedy
polish) has two documented failure modes with ground truth on disk:

| testbed | evidence the n-gram search is stuck | evidence the judge knows better |
|---|---|---|
| **rung-2 latin t5** (54 symbols, ~408 chars) | 120 → 480 restarts never reach the basin; SER 0.797, shortlist oracle 0.684 (`phase5_status.md` rung 2) | true map is 0.35–0.44 bits/char *below every found map* under the ELBO |
| **wordhom synthetics at 5.6–6.8 tokens/type** (German/Italian, 8000 letters, 1 154–1 383 types) | SER 0.30–0.65, 65–103 rule violations, found key 3.5–4.9k nats below the truth; 3M vs 0.4M SA steps barely moves it (`wordhom_study.md` §3.2) | truth is a strict optimum of the *n-gram* objective (oracle polish holds it) — the ELBO side is unmeasured, §3 measures it first |

The race-polish study showed the ELBO-alone polish **holds** a key at Borg scale
but does not repair it: single-symbol moves under the ELBO have gains of
0.006–0.016 bits/char, inside the judge's tolerance band. So "ELBO polish → n-gram
polish → repeat" (the loop as first imagined) is *expected* to do little, and
running only that would be an uninformative proof of life. The mechanism that
can plausibly do more is different: the denoiser's **per-position posterior**
(`backbone.forward_soft`) says, for every masked position, which letter the
context wants; aggregated over a symbol's occurrences that is a proposal
distribution over reassignments that the n-gram objective's landscape cannot
see (it is a *local* objective; the denoiser conditions on 1024 characters of
context). Nothing in the tree uses it as a proposal yet.

## 2. Mechanisms under test

Both operate on the symbol→unit map interface shared by `HomophonicHead`
(rung 2) and `WordHomophonicHead` (units = letters + top-5 doubled-letter
bigrams). Both alternate with the *existing* n-gram polish/SA, unchanged.

**M1 — posterior-guided re-seeding (the bet).** New `heads/posterior.py`:
`symbol_posterior(evaluator, cipher_ids, sym_map, language, *, n_draws, mask_rate,
seed) -> (n_sym, A)`: decode under the current key, mask a random `mask_rate`
(default 0.3) fraction of positions per draw, one forward pass, read the SUBS
posterior at masked positions, accumulate log-probabilities per cipher symbol
(for wordhom: per type; bigram units sum the two slots; only same-length unit
reassignments are scored from the posterior — length-changing moves stay with
the n-gram search). Cost: `n_draws` (default 16) forward passes on the window.
Then one **alternation round** =
1. `P = symbol_posterior(current key)`;
2. **disagreement set** `D = {s : argmax P[s] ≠ key[s]}`, ranked by
   `P[s, argmax] − P[s, key[s]]` weighted by occurrence count;
3. re-seed: apply the top-`k` disagreements (`k ∈ {4, 8, all}`) to the key —
   *without* consulting the n-gram objective;
4. n-gram **short SA** from the re-seeded key (low temperature 2→0.3, 50k steps
   rung 2 / 200k wordhom, then greedy polish) — the n-gram side gets to undo
   what it disagrees with;
5. accept the round iff the n-gram penalized objective improved (the n-gram
   optimum is the thing we claim to escape; the ELBO is not the acceptance
   criterion here, which also keeps the judge out of the loop's own
   selection noise); record both objectives, SER, violations either way.
Repeat for ≤ 6 rounds or until `D` is empty / no acceptance in 2 rounds.

**M2 — race-polish handoff (the documented loop, run as a comparison arm).**
`race_polish(choice_fn=None)` for 2 sweeps → n-gram short SA → repeat. Same
round accounting. Expected to hold, not repair; included so the write-up can
say so on the same cells.

**Control arm — random re-seeding.** M1 with step 2–3 replaced by `k` *uniformly
random* symbols reassigned to *uniformly random* units, same `k`, same n-gram
SA. This is the comparison the user's question is about: if M1 ≈ control the
judge is only contributing randomness; if M1 ≫ control the posterior is the
lever. (Occurrence-weighted random symbol choice as a second control if M1
wins, to rule out "it just picks the frequent symbols".)

**Null arm.** Start *at the true key*: M1 must produce `|D|` ≈ 0 and accept no
round. (`race_polish` already has this property; the posterior needs to show
it too — a denoiser that "disagrees" with a correct key on frequent symbols
would also poison the loop on real ciphertext.)

## 3. Testbeds (all synthetic, all with truth; nothing on the manuscript)

Start keys are the *recorded* stuck keys wherever they exist — the comparison
is loop-vs-no-loop from the same start, not a re-solve.

| id | instance | start key | tokens or chars / symbol | what "escape" means |
|---|---|---|---|---|
| R2-t5 | rung-2 latin t5 (`analysis/phase5/rung2_solves.json`, own-hypothesis) | recorded MDL pick (SER 0.797) and the 3 next shortlist entries | 7.5 chars/symbol | SER < 0.1, n-gram objective above the recorded best |
| R2-bank | 6 rung-2 cells (2 per language) whose *shortlist* holds a non-basin local optimum with SER ≥ 0.2 alongside the true-basin winner | that non-basin optimum | 7.5 | reach the recorded winner's basin (SER ≤ 0.02) |
| WH-6.7 | wordhom German 1 154 types & Italian 1 144 types, 8000 letters (`analysis/wordhom/controls_solves.json`; regenerate with the study's seeds if maps are absent) | recorded solve (SER 0.30 / 0.54) | 6.7–6.8 | SER halves; violations drop; n-gram objective up |
| WH-5.6 | same at 1 375 / 1 383 types | recorded solve (SER 0.57 / 0.64) | 5.6–5.7 | any monotone improvement; this is the hard edge |
| WH-4 (stretch, one cell) | German A-like (3 259 types, 14k letters, SER 0.60) | recorded solve | 4.1 | not expected — reported as the identifiability wall |

Arms per cell: `M1-k4`, `M1-k8`, `M1-all`, `M2`, `control-k8`, `null`. Two
replicate seeds (the posterior draws and the SA are both seeded). Frozen
evaluator `phase_c-85m-seed0` on one GPU; wordhom decodes at 8000 letters are
8 windows of 1024 → 16 draws ≈ 128 forward passes per posterior, seconds.
The n-gram SA is the CPU cost: keep `steps` small (above) and `workers ≤ 4`
while `wordhom_study --set controls` (12 workers) is still running.

## 4. Pre-registered readings (fixed before running)

- **PoL-1 (the judge can see the exit)**: on R2-t5 and WH-6.7 the
  posterior-disagreement set at the *start key* has precision ≥ 0.6 against
  the truth (a symbol in `D` really is wrong) and the top-8 by weighted margin
  ≥ 0.75. If this fails nothing downstream can work and the study stops with
  that number — it is the cheapest, most decisive measurement (minutes).
- **PoL-2 (escape)**: `M1-*` beats `control-k8` on both SER and the n-gram
  objective on ≥ 5 of the 6 R2-bank cells and on both WH-6.7 cells, with
  non-overlapping replicate ranges. R2-t5: any arm reaching SER < 0.1 is the
  headline; report if none does.
- **PoL-3 (not just noise)**: `null` arm moves nothing (`|D|` ≤ 2 at the true
  key on rung 2, ≤ 2 % of types on wordhom; zero accepted rounds).
- **PoL-4 (M2 is what we think it is)**: M2 holds every start (SER within
  ±0.005) — confirms the ELBO-polish loop alone is not the lever.
- WH-5.6 and WH-4 are reported, not gated.

Interpretation guard: an accepted round is judged on the *n-gram* objective, so
the loop cannot import the judge's own tolerance of wrong symbols (the Borg
`e→z` lesson) — the judge proposes, the n-gram disposes, and truth adjudicates
after the fact.

## 5. Code

- `diff_voyn/heads/posterior.py` — `symbol_posterior` (rung-2 and wordhom
  variants share the aggregation; wordhom passes `expand_units` positions →
  type index), `disagreements(P, key, occ) -> ranked list`.
- `diff_voyn/heads/altloop.py` — `alternate(head, evaluator, cipher, key, *,
  mechanism="posterior"|"race"|"random", k, rounds, seed) -> (key, info)`;
  `info["rounds"]` holds per-round n-gram objective, ELBO bits/char (budget 64,
  paired to the start key), SER, violations, `|D|`, symbols changed by the
  re-seed and by the SA.
- `HomophonicHead.short_sa(cipher, key, language, steps, t_start, t_end)` and
  the same on `WordHomophonicHead` — thin wrappers over the existing
  `_sa_phase`/`sa_phase` with an explicit init (they already take one).
- `scripts/altloop_pol.py --stage {pol1,run,report}` → `analysis/altloop/`.
- `tests/test_altloop.py` on the toy evaluators (`_ToyEvaluator` from
  `tests/test_control6.py`): posterior at the true key ⇒ empty `D`; one-swap-
  away start ⇒ the swapped pair heads `D`; `alternate` recovers it in 1 round.
- Nothing frozen changes; `race_polish`/`elbo_polish` untouched; no new deps.

## 6. Order and time

1. `symbol_posterior` + PoL-1 measurement on R2-t5 / WH-6.7 (½ day; go/no-go).
2. `alternate` + toy tests + R2 arms (½ day; each rung-2 round is seconds of
   GPU + ~1 min of SA).
3. WH arms (½ day wall, mostly CPU SA); WH-4 last and only if PoL-2 passed.
4. Report: `analysis/altloop/report.md` with the PoL table, one paragraph in
   `docs/alt_loop_plan.md` §7, `CLAUDE.md` state line.

## 7. Out of scope

VMS cells (a proof of life on synthetics must precede any manuscript run; the
VMS regression arm belongs to the full study), rung-3/4 heads, tuning `k` /
`mask_rate` beyond the three values above, and the manuscript-shaped wordhom
instances beyond the single stretch cell.

## 7. Proof-of-life results (2026-08-25)

Implemented: `heads/posterior.py` (`position_posterior`, `symbol_scores`,
`unit_scores`, `disagreements`), `heads/altloop.py` (`alternate`, mechanisms
`posterior` | `posterior_sample` | `random` | `race` | `none`),
`scripts/altloop_pol.py` (`--stage pol1|run|report`), `tests/test_altloop.py`
(4 pass). Artifacts `analysis/altloop/{pol1,runs_r2,runs_cold,runs_wh,runs_wh2}.json`.
Frozen evaluator `phase_c-85m-seed0`; posterior = 16 draws at mask rate 0.3;
round acceptance on the **n-gram** penalized objective only; ≤ 6 rounds,
patience 2. Wordhom SER below is the letter-level edit-distance SER of the
expanded decode (`unit_ser`), which is why the start values (0.42) differ
from the occurrence-weighted map error the wordhom study tabulated.

### 7.1 PoL-1 — can the judge see the exit?

| cell (start SER) | wrong symbols | \|D\| | top-8 precision: is wrong | top-8: exact letter | \|D\| at true key |
|---|---|---|---|---|---|
| rung-2 latin t5 pick (0.80) | 44/54 | 25 | 0.75 | 0.00 | 7 |
| rung-2 bank ×6 (0.15–0.41) | 8–24/54 | 5–15 | 0.62–0.83 | 0.25–0.60 | 0–2 |
| wordhom German / Italian, 6.6 tok/type (0.42) | 644/1171, 705/1165 | 588 / 476 | 0.75 / 0.88 | 0.00 | 30 / 95 |

**PASS** (≥ 0.6 everywhere; ≥ 0.75 on 6/10 top-8 sets). Two readings that
shaped the rest: on the bank cells the disagreement set is far above the base
rate of wrong symbols (0.83 vs 0.20 on italian t4) *and* the proposed letter
is exactly right 25–60 % of the time; on the badly-wrong keys (t5, wordhom)
the judge knows *which* symbols are wrong at about the base rate but its
argmax letter is never the right one — a decode that is mostly garbage gives
the denoiser no context to name the replacement.

### 7.2 Rung 2 — warm SA (T 2.0→0.3, 50k steps), 6 bank cells × 2 seeds

Escapes (final SER ≤ 0.005): posterior k=8 **9/12**, posterior all 9/12,
posterior k=4 7/12, random k=8 7/12, SA alone 4/12. Separation exists but at
this temperature the SA itself leaves half the traps, so the arms are
confounded; latin t5 (pick and oracle starts) is untouched by every arm, and
its **null** arm drifts away from the true key (posterior to 0.17–0.29,
random to 0.77) because the true key scores 300 nats *below* the found maps
under the pentagram objective — the acceptance rule, not the proposer, moves
it. The other five nulls stay within 0–1 symbol of truth (= the n-gram
optimum's own offset).

### 7.3 Rung 2 — cold SA (T 0.5→0.2, 20k steps), 6 bank + 6 deep cells × 2 seeds

Final SER per seed (start SER in parentheses):

| cell | post-k8 | post-all | psamp-k8 | psamp-all | rand-k8 | SA alone |
|---|---|---|---|---|---|---|
| latin t0 bank (0.41) | **0.002 / 0.002** | 0.002 / 0.002 | 0.002 / 0.027 | 0.002 / 0.002 | 0.32 / 0.41 | 0.41 / 0.41 |
| latin t3 bank (0.25) | 0.002 / 0.002 | 0.002 / 0.002 | 0.002 / 0.002 | 0.002 / 0.002 | 0.002 / 0.002 | 0.25 / 0.25 |
| italian t0 bank (0.22) | **0.000 / 0.000** | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.025 | 0.22 / 0.22 |
| italian t4 bank (0.15) | 0.105 / 0.125 | 0.105 / 0.125 | 0.15 / 0.042 | 0.15 / 0.042 | 0.15 / 0.005 | 0.15 / 0.15 |
| german t0 bank (0.24) | 0.24 / 0.24 | 0.24 / 0.24 | 0.24 / 0.24 | 0.24 / 0.24 | 0.24 / 0.24 | 0.24 / 0.24 |
| german t1 bank (0.15) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.15 / 0.15 |
| german t1 deep (0.61) | **0.000 / 0.000** | 0.59 / 0.77 | 0.22 / 0.56 | **0.000 / 0.078** | 0.58 / 0.30 | 0.65 / 0.59 |
| german t0 deep (0.59) | 0.41 / 0.58 | 0.52 / 0.63 | 0.30 / 0.24 | 0.51 / 0.28 | 0.51 / **0.064** | 0.55 / 0.63 |
| italian t4 deep (0.43) | 0.43 / 0.43 | 0.43 / 0.43 | 0.43 / **0.005** | 0.43 / **0.005** | 0.145 / 0.074 | 0.43 / 0.43 |
| latin t0 / t3, italian t0 deep (0.55–0.59) | stuck | stuck | stuck | stuck | stuck | stuck |
| **escapes ≤ 0.005 / 24** | **10** | 8 | 8 | **10** | 6 | 0 |
| **improved by > 0.05 / 24** | 11 | 9 | 13 | 14 | 13 | 0 |

Reading. With the SA too cold to leave any trap on its own (0/24), the
posterior kick clears four of the six bank traps on both seeds and one
0.61-SER deep trap; a same-size random kick clears two. That is the
proof of life the plan asked for (PoL-2 on the bank: posterior beats random on
SER and objective on latin t0, italian t0 and — only via sampling — italian
t4 / german-t1-deep; ties on latin t3, german t1; loses on german t0 deep).
Two qualifications the plan did not anticipate:

1. **The argmax proposer is deterministic.** From a mostly-wrong decode its
   proposed letters are wrong (§7.1), the SA rejects the round, and the next
   round proposes the same set — so on the deep starts it stalls while random
   makes progress by trying six different things. Sampling the letter from the
   per-occurrence posterior (`posterior_sample`) fixes the stall where the
   judge's ranking of *symbols* is right (italian t4 deep 0.43 → 0.005,
   german t1 deep → 0.000/0.078) at the cost of one bank seed (latin t0
   0.027). Judge-picked symbols + sampled letters is the form to carry
   forward; random letters on judge-picked symbols is the untested middle.
2. **Some traps are objective traps, not search traps.** German t0's 0.240
   optimum resists every arm (and the deep German t0 start lands *in* it), and
   latin t5's true key is 300 nats below the pentagram optimum. No proposer
   fixes those under "accept on the n-gram objective"; they need the judge in
   the acceptance rule (§5 of `race_polish_plan.md` — with the choice term
   off), which this study deliberately kept out.

Null (start at truth) under cold SA: 0–1 symbol moved on every bank/deep
instance for every arm (PoL-3 PASS outside t5).

### 7.4 Word-homophonic cells (6.6 tokens/type, 8000 letters, 1 seed)

The wordhom study recorded these as unsolvable by the n-gram search (SER
0.30–0.65 after 2M SA steps, hundreds of rule violations). Same recorded start
keys, SA 200k steps at T 2.0→0.3 per round:

| cell | arm | SER | n-gram objective | violations | accepted rounds |
|---|---|---|---|---|---|
| German (start 0.425, obj −18486) | posterior **all** (~590 types) | **0.026** | −13419 | see runs_wh.json | 4/6 |
| | posterior k=8 | 0.410 | −18357 | | 6/6 |
| | random k=8 | 0.416 | −18382 | | 5/6 |
| | SA alone | 0.425 | −18486 | | 0/2 |
| | null (truth, obj −13648) → | 0.026–0.029 | −13419…−13425 | | |
| Italian (start 0.421, obj −19183) | posterior **all** (~480 types) | **0.047** | −15528 | | 5/6 |
| | posterior k=8 | 0.413 | −19095 | | 5/6 |
| | null (truth, obj −15973) → | 0.043 | −15503 | | |

The posterior-all loop takes both cells from the recorded stuck key to the
n-gram objective's own optimum near the truth (the null rows show that
optimum is 0.026–0.043 SER from the true key: the objective refits rare
types, as the study's oracle row said). k = 8 is a drip against 640–700
wrong types. The size-matched random control (`rand-k512`) is recorded in
§7.5.

### 7.5 Size-matched random control on the word-homophonic cells

`rand-k512`: 512 uniformly random types re-seeded to random same-length
units per round, same SA (200k steps, T 2.0→0.3), same acceptance rule.

| cell | posterior-all (~590 / ~480 types) | random 512 types | random 8 | SA alone |
|---|---|---|---|---|
| German (start 0.425) | **0.026**, obj −18486 → −13419, 4/6 rounds accepted | 0.425, **0 rounds accepted** | 0.416 | 0.425 |
| Italian (start 0.421) | **0.047**, obj −19183 → −15528, 5/6 accepted | 0.421, **0 rounds accepted** | 0.415 | 0.421 |

(German null under random-512: 0 → 0.041, obj −13648 → −13589 — the
objective's own refit of rare types, as with every other arm.)

This is the decisive comparison for the user's question. A random kick of the
same size never produces a key the n-gram SA can improve on; the judge's kick
of the same size hands the SA a key it polishes to the objective's optimum
near the truth. On a 1 100–1 200-type word-homophonic key at 6.6 tokens per
type, the information in the denoiser posterior — not the perturbation — is
what moves the search.

### 7.6 Pre-registered readings

| | result |
|---|---|
| PoL-1 (judge sees the exit) | PASS — top-8 precision 0.62–0.88 (≥ 0.6); ≥ 0.75 on 6/10; exact-letter precision 0 on badly-wrong keys |
| PoL-2 (escape, posterior beats random) | PASS on wordhom (2/2 cells, control 0 accepted rounds) and on the rung-2 bank under cold SA (posterior-k8 10/24 vs random 6/24, SA alone 0/24); NOT the pre-registered "≥ 5 of 6 cells with non-overlapping seeds" — 3 of 6 bank cells are clean posterior-only escapes, 2 tie, german t0 resists all arms. latin t5: no arm reaches SER < 0.1 |
| PoL-3 (null unmoved) | PASS outside latin t5 (0–1 symbol; wordhom 0.026–0.046 = the objective's own refit, identical for every arm); t5's null drifts because its true key is 300 nats below the pentagram optimum |
| PoL-4 (race-polish handoff holds) | NOT RUN — the race arm was dropped for cost once §7.3 showed the objective, not the polish, is the lever on rung 2 |

### 7.7 What to carry forward

1. **Judge-picked symbols + sampled letters** (`posterior_sample`) is the
   proposer; `posterior_all` for large keys (wordhom), k ≈ 8 for 54-symbol
   keys.
2. **The judge must enter the acceptance rule** for the objective traps
   (german t0, latin t5, and the wordhom refit offset): accept on the ELBO
   with the choice term off and the race's confirmation, per
   `race_polish_plan.md` §7 — untested here by design.
3. Before any manuscript use: a VMS regression arm (the abstention must not
   move), the wordhom 5.6-tokens-per-type and manuscript-shaped cells
   (§3's WH-5.6 / WH-4, not run), and two seeds on the wordhom cells.
