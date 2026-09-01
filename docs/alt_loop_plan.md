# Plan — alternating n-gram ↔ diffusion key search (proof of life)

> **Record status (banner added 2026-09-01):** running log of the post-Phase-6 word-homophonic solver development, 2026-08-25 → 2026-08-30 (results §7–§10 appended in date order below a plan, §1–§6a, written 2026-08-25).
> Still current: §8.6 (wildcard → anneal, the **solver of record**), §8.7 (judge vs SER), §10 (manuscript-shaped control battery); the §10.5 carry-forward items are all still OPEN. Superseded: the plan header's "not yet run"; every statement that the judge in the acceptance rule is "the lever" / "the only change these traps respond to" (§7.3 (2), §7.7 (2), §8, §8.2, end of §8.4, §8.5 last sentence, §8.7 consequence) — tested 2026-08-28 in §8.5 with no gain; the traps were broken by the wildcard objective (§8.4) and the anneal (§8.6); §7.3's "german t0 is an objective trap" (§9.1: a search trap). §8.7 judged the *pre-anneal* keys — the anneal finals are called for German and Latin (§10.1 A-like table). **Current project position: `docs/project_status.md`.**

## 0. Current state (2026-09-01) and table of contents

- **Solver of record** (word-homophonic cells): hapax-wildcard n-gram objective → anneal re-admission of the wildcards → post-all diffusion-guided loop, patience ≥ 10 (§8.4–8.6): `scripts/altloop_vms.py --wild`, then `--wild-anneal 0,40 --patience 10 --rounds 80 --start-from <wild tag>`. Plain multi-restart SA (`WordHomophonicHead.solve`) is the baseline arm only.
- **Findability wall by solver stage** (tokens per type at which a 30-unit synthetic key is recovered): plain SA ≥ 8 (`docs/wordhom_study.md` §3.1) → §7 posterior re-seeding 6.6 → §8.4 wildcard objective A-like 4.1–4.4 to SER 0.13–0.24 → §8.6 anneal A-like to 0.05 / 0.07 / 0.12 (German and Latin **called**, Italian not) — **≈ 4 overall**; ~3.5 fails at every stage. Manuscript: Currier A 3.0 (below the wall), Currier B 4.6 (inside it, still NOISE). The synthetic "A-like/B-like" cells are 4.1–4.4 / 5.5–5.7 tokens per type with ~45 % hapax types, not the manuscript's 3.0 / 4.6 with 69–75 %.
- **Battery (§10, 2026-08-29/30)**: 12 negatives + 6 cross-language cells all NOISE ≤ 0.48; B-like positives German 0.026 (margin 2.22, called), Latin 0.036 (1.83, called), Italian 0.070 (1.46, not called; truth ceiling 1.61); `nodouble` costs nothing; `revdouble` German still called; dirty-10 % truth ceilings 1.54 / 1.06 / 0.91.
- **Manuscript runs**: `docs/altloop_vms_plan.md` §12 (posterior loop, 72/72 NOISE, 2026-08-26) and §13 (wildcard → anneal, 24/24 NOISE, 2026-08-29); `d5b20` variant 8/8 NOISE (`docs/wordhom_bigram_variant.md`, 2026-08-31).
- **Contents**: §1–§6 the plan (2026-08-25); §6a out of scope as planned; **§7** proof-of-life results (2026-08-25); **§8**–8.3 hapax masking, not adopted (2026-08-26/27); §8.4 hapax-wildcard objective (2026-08-27); §8.5 judge in the acceptance rule, no gain (2026-08-28); §8.6 wildcard → anneal, adopted (2026-08-28); §8.7 judge vs SER (2026-08-28); **§9** small-commitment loop, refuted (2026-08-26/27); **§10** manuscript-shaped control battery (2026-08-29/30).

*[Header as written 2026-08-25 — superseded, see the banner and §0:]* Status: **PLAN 2026-08-25, not yet run.** Follow-on to `docs/race_polish_plan.md`
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
| WH-4 (stretch, one cell) | German A-like (3 259 types, 14k letters, SER 0.60) | recorded solve | 4.1 | not expected — reported as the identifiability wall *[Superseded 2026-08-27/28: the A-like cells were solved by the wildcard objective + anneal, §8.4–8.6 (SER 0.05 / 0.07 / 0.12); `docs/project_status.md` §3.]* |

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

## 6a. Out of scope (as planned 2026-08-25)

*(renumbered 2026-09-01 from a duplicate "## 7." so that the results section below keeps the §7 number other documents cite)*

VMS cells (a proof of life on synthetics must precede any manuscript run; the
VMS regression arm belongs to the full study), rung-3/4 heads, tuning `k` /
`mask_rate` beyond the three values above, and the manuscript-shaped wordhom
instances beyond the single stretch cell.

*[Superseded: VMS cells were run 2026-08-26 and 2026-08-29 (`docs/altloop_vms_plan.md` §12, §13 — all NOISE) and manuscript-shaped wordhom instances throughout §8–§10 below; `docs/project_status.md` §1.]*

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

   *[Superseded: german t0's 0.240 optimum is a *search* trap — the `rand-k2` arm opens it at 32 rounds (§9.1 side result, 2026-08-27); latin t5 remains the objective trap. The judge in the acceptance rule was tested 2026-08-28 (§8.5) with no gain. `docs/project_status.md` §5.2, §5.10.]*

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
   *[Superseded 2026-08-28: tested in §8.5 — no gain. The traps were broken instead by the hapax-wildcard objective (§8.4) and the anneal (§8.6). `docs/project_status.md` §5.2.]*
3. Before any manuscript use: a VMS regression arm (the abstention must not
   move), the wordhom 5.6-tokens-per-type and manuscript-shaped cells
   (§3's WH-5.6 / WH-4, not run), and two seeds on the wordhom cells.
   *[Done: manuscript runs `docs/altloop_vms_plan.md` §12 (2026-08-26) and §13 (2026-08-29), abstention unmoved; the A-like (WH-4) cells §8–§8.7 and B-like cells §10.1, three seeds in §8.6.]*

## 8. Hapax-masked proposer on the high-hapax word-homophonic cells (2026-08-26)

**Question.** The proposer reads the denoiser posterior from the decode under
the current key, so at 3–5 tokens per type the context the judge conditions
on is mostly the key's guesses for types it has essentially no evidence
about. Variant: before the posterior read, replace the letter positions
emitted by *hapax* types (≤ 1 occurrence, `--hapax-max`) with `MASK` in every
draw (`position_posterior(force_mask=…)`, arms `post-all-hm` /
`psamp-all-hm` in `scripts/altloop_pol.py`), on top of the usual 30 % random
masking. The rest of the loop (SA 200k steps T 2.0→0.3, accept on the n-gram
objective, 6 rounds, patience 2) is unchanged. One seed.

Cells: the two t0 cells (6.6 tok/type, 21 % hapax types = 3 % of tokens) and
the three A-like positives (4.15 tok/type, 44 % hapax types = 11 % of
tokens, the manuscript's Currier-A shape). Artifacts
`analysis/altloop/runs_hm.json`, `run_hm.log`.

| cell (stuck start) | arm | SER start → final | n-gram obj final | accepted | hm − plain (nats) |
|---|---|---|---|---|---|
| german/t0 | post-all / **-hm** | 0.425 → 0.027 / 0.028 | −13418 / −13421 | 5/6, 4/6 | −3 |
| german/t0 | psamp-all / **-hm** | 0.425 → 0.031 / 0.029 | −13433 / −13431 | 3/5, 4/6 | +2 |
| italian/t0 | post-all / **-hm** | 0.421 → 0.056 / 0.054 | −15616 / −15537 | 6/6, 5/6 | +79 |
| italian/t0 | psamp-all / **-hm** | 0.421 → 0.073 / 0.053 | −15855 / −15532 | 6/6, 6/6 | +323 |
| german/Alike | post-all / **-hm** | 0.642 → 0.549 / 0.617 | −31828 / −32538 | 6/6, 4/6 | **−710** |
| german/Alike | psamp-all / **-hm** | 0.642 → 0.569 / 0.650 | −32132 / −33438 | 4/6, 1/3 | **−1306** |
| italian/Alike | post-all / **-hm** | 0.759 → 0.756 / 0.697 | −34327 / −33141 | 5/6, 6/6 | **+1187** |
| italian/Alike | psamp-all / **-hm** | 0.759 → 0.750 / 0.718 | −34683 / −33906 | 5/6, 6/6 | **+777** |
| latin/Alike | post-all / **-hm** | 0.764 → 0.729 / 0.690 | −34224 / −33604 | 6/6, 5/6 | **+620** |
| latin/Alike | psamp-all / **-hm** | 0.764 → 0.757 / 0.761 | −35167 / −34858 | 2/4, 3/5 | +309 |

Null starts (truth) under the hm arms drift exactly as under the plain arms
(german/t0 0.026–0.027, italian/t0 0.044–0.046, german/Alike 0.040–0.044,
italian/Alike 0.097–0.102, latin/Alike 0.055–0.057): masking a correct key's
hapax letters does not damage it — the SER a null reaches is the objective's
own refit of rare types (0.10 on Italian A-like: at 4 tokens per type the
n-gram optimum is already 2 000 nats and 10 % of letters from the truth).

**Reading.** No effect where hapaxes are 3 % of tokens (t0, as expected).
On the 11 %-hapax A-like cells the sign is language-dependent: masking
helps Italian on both proposers and Latin on the argmax proposer (+0.6–1.2k
nats, −0.03–0.06 SER) and hurts German on both (−0.7–1.3k nats, +0.07–0.08
SER, fewer accepted rounds). The magnitude is the same as the plain
argmax-vs-sample proposer gap on the same cell (0.3–1.3k nats), i.e. one
seed cannot separate it from proposer noise, and no variant moves an A-like
cell out of its trap: every final key sits at SER 0.55–0.76 with 120–220
rule violations and 9–11k nats above the truth-basin optimum the null rows
reach. Withholding the hapax guesses removes bad context but also removes
11 % of the letters the denoiser needs for the *non*-hapax positions, and
the two effects roughly cancel. **Not adopted**; the code stays as an
option (`force_mask`, `--hapax-max`, `--hapax-mask-rate`) for a partial-mask
sweep if the loop is ever run with the judge in the acceptance rule, which
remains the only change these traps are expected to respond to (§7.7).

*[Superseded the next day (2026-08-27) by §8.4: the hapax-*wildcard objective*, not the acceptance rule, broke these traps; the judge in the acceptance rule was then tested 2026-08-28 (§8.5) with no gain. `docs/project_status.md` §5.2.]*

### 8.1 Why: the judge's proposals are hapax-heavy but not hapax-discriminating

Per-occurrence-class split of the disagreement set at the A-like stuck
starts (plain proposer): hapax types are ~48 % of every proposal set (44 %
of types, 11 % of tokens) and the per-type flag rate falls with occurrence
(≈ .68 hapax → .64 occ 2–3 → .50 occ 4+). But on hapaxes the rate on
*wrong* types (.64–.72) barely exceeds that on *right* ones (.38–.55) — a
one-occurrence posterior row is noise — while frequent types are separated
(.42–.59 vs .32–.38). At the true key 81–91 % of the false alarms are
hapaxes. Hapax masking raises the flag rate ~.10 in every class including
correct frequent types and enlarges the false-alarm set at the truth
(86 → 158, 171 → 232): it is a larger, noisier kick, not a sharper judge.
German's judge separates hapaxes best (.72 vs .38; 5 % false alarms at the
truth) and Italian's worst (.64 vs .53; 29 %), which is the one language
asymmetry in the data.

### 8.2 Dropping hapax types from the proposal set (`-nh` arms)

Follow-up: types with ≤ 1 occurrence are excluded from the disagreement
set (`post-all-nh` / `psamp-all-nh`; score rows set to −inf), posterior
unchanged. Same cells, seed, SA.

| cell (stuck) | proposer | plain | hapax-masked (hm) | no-hapax (nh) |
|---|---|---|---|---|
| german/t0 | post / psamp | 0.027 −13418 / 0.031 −13433 | 0.028 −13421 / 0.029 −13431 | 0.027 −13419 / 0.030 −13429 |
| italian/t0 | post / psamp | 0.056 −15616 / 0.073 −15855 | 0.054 −15537 / 0.053 −15532 | 0.054 −15548 / 0.054 −15585 |
| german/Alike | post / psamp | **0.549 −31828** / **0.569 −32132** | 0.617 −32538 / 0.650 −33438 | 0.599 −32193 / 0.608 −32718 |
| italian/Alike | post / psamp | 0.756 −34327 / 0.750 −34683 | **0.697 −33141** / **0.718 −33906** | 0.753 −34230 / 0.752 −34973 |
| latin/Alike | post / psamp | 0.729 −34224 / 0.757 −35167 | **0.690 −33604** / 0.761 −34858 | 0.752 −34864 / 0.739 −35153 |

Null starts under `-nh`: 0.025–0.026 (german/t0), 0.044–0.045, 0.040–0.041,
0.100–0.102, 0.054 — unchanged.

Dropping the hapax proposals changes nothing on t0 and never beats the
plain proposer on the A-like cells (German: worse than plain, better than
hm; Italian/Latin: equal to plain, i.e. the hm gain there did *not* come
from suppressing hapax noise). Taken with §8.1: the hapax entries are
noise the SA repairs at no cost (removing them buys nothing), and what
moved Italian/Latin under masking was the larger perturbation of the
frequent types, which the German start (shallower, better-judged) cannot
absorb. Neither variant is adopted; the recommendation stands — the lever
is the acceptance rule (judge in the loop), not the proposal set.

*[Superseded 2026-08-27/28: the lever turned out to be the *objective* (hapax wildcards, §8.4, then re-admitted, §8.6); the judge in the acceptance rule gave no gain (§8.5).]*

### 8.3 Hapax-only masking (2026-08-27, `runs_hm0.json`)

Same `-hm` arms with `--mask-rate 0`: only hapax positions are blanked, so
the proposal set is the hapax types read against full non-hapax context.
A-like stuck starts, one seed, 6 rounds (post / psamp, nats):

| cell | plain | hm (hapax + 30 %) | hm0 (hapax only) |
|---|---|---|---|
| German | −31 828 / −32 132 | −32 538 / −33 438 | −31 847 / −31 779 |
| Italian | −34 327 / −34 683 | −33 141 / −33 906 | −34 379 / −34 421 |
| Latin | −34 224 / −35 167 | −33 604 / −34 858 | −34 461 / −34 391 |

The German penalty under `-hm` was the 30 % general mask, not the hapax
mask (hm0 matches the best plain arm, 6/6 rounds accepted); the
Italian/Latin gain under `-hm` disappears (it was the larger kick). SER
0.61–0.76 everywhere: harmless, not a trap-breaker.

### 8.4 Hapax types as wildcards in the *objective* (2026-08-27)

At the A-like stuck starts the wrong-rate of a type is monotone in its
occurrence count (hapax 0.87–0.94, ≥ 30 occurrences 0.49–0.75; t0: 0.73–0.76
vs 0.30–0.37). A hapax type is a free parameter the SA sets after the fact
to patch its neighbours' n-grams — 44 % of the A-like key. Test:
`WordHomophonicHead.wild_types` (`wordhom_state.py`: a wildcard letter is
charged a constant and resets the n-gram context for the letters after it;
wildcard types are frozen out of SA/polish proposals), driven by
`scripts/hapax_wildcard.py` (`--no-wild` = same SA under the standard
objective), 1M SA steps, 3 seeds, artifacts `hapax_wildcard{,_nowild}.json`.

Gap truth − stuck, standard / wildcard objective (nats): t0 German +4 838 /
+4 605, Italian +3 210 / +3 189; A-like German +11 990 / +8 823, Italian
+11 535 / +8 999, Latin +13 356 / +9 825. The hapax "fudge" is ~25 % of the
A-like gap; the frequent types alone still prefer the trap by 9–10k.

SA from the stuck start (SER per seed): **German t0 escapes under the
wildcard objective — 0.055, 0.052, 0.394 vs 0.417, 0.422, 0.409 standard; from
a cold `frequency_init` 0.050/0.050/0.055 vs 0.456/0.520/0.507** (non-hapax
map error 0.02–0.03; the reachable truth with hapaxes frozen has SER 0.025).
This is the cell the posterior loop took to 0.027 with the judge; the
wildcard objective gets there with n-grams alone. Italian t0 (0.41 both) and
all three A-like cells (0.61–0.76 both; cold 0.52–0.75) do not move. Truth
starts stay in their basin under both (wildcard 0.13–0.19 on A-like vs
standard 0.07–0.15, the frozen-hapax cost). Reading: a sound, free objective
change that reshapes the landscape enough to unstick one t0 trap, not a
route across the A-like regime, where the frequent types are themselves
mostly wrong.

*[Revised below the same day: with 32–96 rounds and patience 6–10 the wildcard objective does cross the A-like regime (German 0.13, Latin 0.17, Italian 0.21–0.24), and the anneal (§8.6) takes it to 0.05 / 0.07 / 0.12.]*

**In the alternating loop** (`altloop_pol.py --wild`, `runs_wild.json`; stuck
start, 6 rounds, one seed; SER, post-all / psamp-all): Italian t0 0.066 / 0.067
(plain 0.047–0.056 / 0.073, same basin); German A-like **0.482** / 0.577 (plain
0.549 / 0.569); Italian A-like **0.714 / 0.709** (0.756 / 0.750); Latin A-like
**0.661** / 0.765 (0.729 / 0.757). First movement on the A-like cells — the
argmax proposer gains 0.04–0.07 SER on all three and keeps accepting (5–6/6
rounds) — but 0.48–0.71 is far from the reachable truth (~0.10); one seed, six
rounds: a lead for a longer run, not a result.

**32 rounds, 3 seeds, patience 6** (`runs_wild32.json`, `post-all --wild`, A-like
stuck starts; SER per seed): German **0.132 / 0.134 / 0.134** (17/29, 19/32,
14/20 rounds accepted; objective −23 253…−23 268 vs the truth's −23 464);
Italian 0.268 / 0.237 / 0.246 (26–28/32 accepted, still descending at the
cap); Latin **0.173** / 0.661 (stalled, 4/12) / **0.183**. Five of nine runs
reach the frequent-type truth basin (reachable-truth SER ~0.10) from the
n-gram trap that every previous arm left untouched; the descent is gradual
(German s0: 0.64 → 0.48 over rounds 0–4, then 0.40 → 0.27 → 0.17 at rounds
8–10 as the disagreement set halves, flat at 0.13 from round 13). The
six-round runs above stopped exactly where the descent accelerates. A
96-round / patience-10 continuation for Italian and Latin is `runs_wild96.json`.

**96 rounds, patience 10** (`runs_wild96.json`, Italian and Latin, 3 seeds; SER,
rounds run / accepted): Italian **0.206 / 0.227 / 0.235** (78/45, 51/33, 51/31;
objectives −25 981…−26 421 vs the truth's ≈ −26 390); Latin **0.173 / 0.171 /
0.174** (41/21, 52/23, 56/29; −25 161…−25 178 vs ≈ −25 400). Every one of the
nine A-like (cell × seed) runs now converges from the trap: German 0.13,
Latin 0.17, Italian 0.21–0.24, all at or above the truth's own wildcard
objective. The Latin seed that stalled at patience 6 (0.661) converges at
patience 10 — the slow early phase needs ~10 rejected rounds tolerated. The
residual above the reachable truth (~0.10) is an objective-level trap the
wildcard n-gram score prefers to the truth by 0–400 nats; resolving it needs
the judge in the acceptance rule.

*[Superseded 2026-08-28: §8.5 found no gain from the judge in the acceptance rule; §8.6's re-admission of the wildcards resolved most of the residual instead (German 0.13 → 0.05, Latin 0.17 → 0.07, Italian 0.21–0.24 → 0.12).]*

### 8.5 The judge in the acceptance rule, on the residual (2026-08-28)

Prerequisite holds: the frozen judge prefers the reachable truth (hapaxes
frozen at the stuck assignment) to every converged wildcard key — Italian
3.24 vs 3.34–3.39 bits/char, Latin 3.03 vs 3.15–3.17 (CRN seed noise ≈ 0.01;
one 13.7k-char `score_stream` call = 15 s). Aside: frozen wrong hapaxes (11 %
of tokens) cost the judge 0.86 bits/char (German reach-truth 2.75 vs truth
1.89).

`altloop_pol.py --judge-accept 0.005 --start-from _wild96` (accept iff the
judge's bits/char drop by the margin, CRN-paired within a round, fresh masks
across rounds; `alternate(accept_fn=…)`), from the converged keys, seed 0,
`runs_judge{,_shortsa}.json`: Italian 0.206 → 0.204 (5/18 accepted) / 0.205
with a 10k-step inner SA; Latin 0.173 → 0.172 (1/9) / 0.173 (0/8). **No
gain.** Every candidate returns from the SA (+ greedy polish) with
essentially all re-seeded symbols rewritten (`sa_changed ≈ reseed`), i.e.
another key of the same n-gram basin within ±0.01 bits/char, so the judge
only arbitrates noise. The raw proposals themselves lower the judge's bits
(Italian 3.35 → 3.31–3.33, Latin 3.15 → 3.11) with SER flat or worse
(0.208–0.219 / 0.168–0.173): the judge's per-symbol argmax on the residual
~150–450 disagreeing types is not truth-directed either, so a judge-only
loop would collect a 0.03-bit self-consistent optimum, not the 0.10–0.15
gap to the truth. The residual 0.13–0.24 SER is not reachable by either
objective's single-symbol moves from here.

*[Superseded the same day by §8.6: by re-admitting the wildcards it is — anneal finals 0.05 / 0.07 / 0.12.]*

### 8.6 Annealing the wildcard set (2026-08-28)

§8.4 treats the hapax types as a binary switch: charged a constant, frozen
out of every proposal, for the whole run. That is what breaks the A-like
traps, but it also pins 44 % of the key at the *stuck* assignment for
ever — the reachable-truth ceiling (SER ≈ 0.10, judge cost 0.86 bits/char
on German, §8.5) is the price of never re-admitting them. The obvious
continuation is a schedule: once the frequent types have converged, hand
the hapaxes back to the objective a batch at a time so the now-correct
context can set them, ending on the standard objective.

Implementation: `alternate(schedule=…)` (`heads/altloop.py`) calls
`schedule(r)` at the start of each round; when the wildcard set changes
the incumbent is re-scored under the new objective and the patience
counter restarts. `WHCell.wild_schedule(start, end, seed)`
(`scripts/altloop_pol.py --wild-anneal START,END`) re-admits the hapax
types in equal batches over rounds `start..end` in a seeded random order;
re-admitted types are scored normally and re-enter the SA, polish and
posterior proposals. Runs, from the converged `_wild96` keys (Italian /
Latin A-like, `post-all`, 3 seeds, patience 10, ≤ 80 rounds):
`--wild-anneal 0,40` (tag `_anneal`) against the instant-switch control
`--wild-anneal 0,0` (tag `_anneal0`, standard objective from round 0 —
"does the standard objective just fall back into the trap?").

Pre-registered reading: the anneal is worth adopting iff its final SER
beats both the `_wild96` endpoint (0.206/0.227/0.235 Italian, 0.173/0.171/
0.174 Latin) and the instant switch on ≥ 5 of 6 (cell × seed) runs; a tie
with the instant switch means the schedule is irrelevant and the gain (if
any) is the re-admission itself; a loss on both means the standard
objective's basin at these keys is the §8.4 trap and the hapaxes must stay
wild.

**Results** (`runs_anneal{,0}.json`, SER per seed; standard objective in
nats, higher is better):

| cell | `_wild96` start | instant switch `_anneal0` | anneal 0–40 `_anneal` |
|---|---|---|---|
| Italian A-like | 0.206 / 0.227 / 0.235 | 0.129 / 0.135 / 0.150 (obj −26 045 / −26 045 / −26 332; 22–49 rounds) | **0.119 / 0.122 / 0.121** (−25 944 / −25 924 / −25 894; 64–80 rounds, 45–50 accepted) |
| Latin A-like | 0.173 / 0.171 / 0.174 | 0.068 / 0.074 / 0.072 (−24 563 / −24 581 / −24 598; 28–59 rounds) | **0.069 / 0.066 / 0.073** (−24 536 / −24 520 / −24 591; 52–78 rounds, 42–47 accepted) |
| German A-like (from `_wild32map`, 0.132 / 0.134 / 0.134) | 0.132 / 0.134 / 0.134 | 0.050 / 0.048 / 0.052 (−22 422 / −22 436 / −22 448; 23–40 rounds) | 0.050 / 0.049 / 0.048 (−22 430 / −22 428 / −22 449; 59–68 rounds, 43–47 accepted) |

Three things, in order of size:

1. **Re-admitting the hapaxes is the gain.** Handing the frozen types back
   to the standard objective from the converged wildcard key does *not*
   fall back into the §8.4 trap: the instant switch alone takes Italian
   0.21–0.24 → 0.13–0.15 and Latin 0.17 → 0.07, i.e. through the
   "reachable truth with hapaxes frozen" ceiling (~0.10) that bounded §8.4.
   Most of it lands in the first five rounds (Italian s0 0.206 → 0.161 on
   the first re-scored round, 0.13 by round 5). The frequent types the
   wildcard objective fixed carry enough context to set the hapaxes right
   — the reverse of the trap, where wrong hapaxes patched wrong neighbours.
   The judge agrees: the final keys score 2.73–2.83 (Italian) / 2.38–2.41
   (Latin) bits/char against 3.34–3.39 / 3.15–3.17 for the wildcard keys
   (§8.5), a drop of ~0.6–0.8 bits/char that is the §8.5 "frozen wrong
   hapaxes" cost being paid back.
2. **The schedule beats the switch, narrowly, 5/6** (pre-registered
   criterion met): Italian 0.119–0.122 vs 0.129–0.150 on all three seeds,
   by 100–440 nats of the standard objective as well; Latin 0.066/0.073 vs
   0.074/0.072 on two seeds and 0.069 vs 0.068 on the third — the Latin
   difference is inside seed noise (≈ 0.005), the Italian one is not. The
   anneal's advantage is variance: the instant switch lands on a different
   basin per seed (Italian 0.129–0.150, obj spread 290 nats) where the
   anneal converges to the same place (0.119–0.122, spread 50 nats). The
   anneal is also slower — it keeps accepting through the whole 40-round
   window (42–50 accepted) and needs 52–80 rounds vs 22–59.
3. **Where it stops is again the objective, not the search.** The final
   standard objectives (Italian −25 894…−26 045, Latin −24 520…−24 598) sit
   1 000–1 900 nats *above* the truth's own standard objective (−27 813 /
   −25 633, `hapax_wildcard_nowild.json`): the standard n-gram objective
   prefers these SER-0.07–0.12 keys to the truth by a wide margin, so no
   proposer acting under it can get closer. This is the same statement as
   §8.4–8.5 one level down: the wildcard objective's trap was 0–400 nats
   below the truth; the standard objective's is 1–2k nats above it. The
   residual (map error 0.07–0.12 by occurrence) is the n-gram fudge the
   wildcard objective was built to remove, now re-admitted; the judge in
   the acceptance rule (§8.5) is the only instrument that could arbitrate
   it, and §8.5 found its single-symbol moves are not truth-directed at
   this SER either.

**German (added 2026-08-28, `runs_anneal{,0}_de.json`; the `_wild32`
keys had no `final_map`, regenerated as `_wild32map` seeds 1–2: 0.134 /
0.134).** Same picture, with the two arms now indistinguishable: instant
switch 0.048–0.052, anneal 0.048–0.050, seed noise ≈ 0.003; both go through
the German reachable-truth ceiling (0.097 with hapaxes frozen), judge
2.06–2.07 bits/char against the truth's 1.89 (wildcard keys were 2.75),
and the standard objective at the found keys (−22 422…−22 449) is again
~600 nats above the truth's (−23 052). Across the three languages the
schedule-vs-switch tally is 5/9 with the German and Latin differences
inside noise; the schedule's only demonstrated edge is Italian, where the
instant switch scatters across basins. Re-admission is the result, the
schedule is a variance reducer.

Adopted as the default continuation of a wildcard run: `--wild
--wild-anneal 0,40 --patience 10 --rounds 80` from the converged wildcard
key (or, from a stuck start, a wildcard phase to convergence followed by
the anneal). Final A-like state of the wildcard-then-re-admit pipeline from the
stuck starts: German 0.05, Latin 0.07, Italian 0.12 — each the standard
n-gram objective's own optimum, above the truth by 0.6–1.9k nats.

### 8.7 What the judge says at the loop's residual SER (2026-08-28, `scripts/judge_at_ser.py`, `analysis/altloop/judge_at_ser.{json,md}`)

Question: the wildcard loop leaves the A-like cells at letter SER 0.13 (German) / 0.17 (Latin) / 0.21–0.24 (Italian) *[clarified 2026-09-01: these are the **pre-anneal** `_wild96` / `_wild32map` keys; the §8.6 anneal finals (0.05 / 0.07 / 0.12) were judged afterwards with the same script — German called at margin 2.13–2.14 and Latin called at 1.70–1.72 on all 3 seeds, Italian not called at 1.39–1.40; `analysis/altloop/judge_at_ser.md`, tabulated in the §10.1 A-like table]*. Does the frozen Phase-6 judge call the language there? Every key below was pushed through the exact Phase-6 full-stream scoring (13 windows × 4 replicate seeds × 3 language conditions, budget 64, paired letter-shuffled copies, `cell_from_score`, `ABSTAIN_RULE`). Keys per cell: truth, the solve's stuck start, the loop's recorded final maps (`runs_wild96.json`; German re-run once as `runs_wild32map.json`, SER 0.132 as before), and truth corrupted at controlled SER — uniformly over types (`uni@`) and rarest-types-first (`rare@`, the search's own error profile).

| cell | truth: plain / margin / lang margin ± unc | last **called** key | loop's final key (SER) → plain / margin / called |
|---|---|---|---|
| German A-like | 1.89 / 2.33 / 0.138 ± 0.067 | rare@ SER 0.084 (margin 1.59); uni@ SER 0.102 sits at exactly 1.50 | 0.132 → 2.83 / **1.40** / no (rank ge>la>it, lang margin 0.086 ± 0.067) |
| Latin A-like | 2.17 / 1.94 / 0.023 ± 0.067 | uni@ SER 0.039 (margin 1.66); SER 0.056 → 1.47 | 0.17 ×3 → 3.14 / **0.99** / no (rank la>ge>it, lang margin 0.02 ± 0.067) |
| Italian A-like | 2.60 / **1.56** / 0.022 ± 0.193 | truth only; SER 0.030 → 1.33 | 0.21–0.24 ×3 → 3.34–3.39 / **0.73–0.82** / no (rank it>ge>la, lang margin 0.01 ± 0.193) |

Readings:

1. **The structure margin is a near-linear function of SER and crosses the 1.5 threshold at SER ≈ 0.10 (German), ≈ 0.045 (Latin), and below 0.03 (Italian — the true key itself is at 1.56).** ~0.08–0.09 bits of margin per point of SER on all three cells, uniform and rare-first alike. The loop's residual (0.13–0.24) is therefore 1.3×–8× beyond what the frozen rule will call. *[Pre-anneal keys. The anneal finals (§8.6) are called for German (SER 0.049–0.050 → margin 2.13–2.14) and for Latin (0.066–0.073 → 1.70–1.72, 3 seeds — above the ≈ 0.045 forecast from this corruption curve: search-shaped residuals are n-gram-plausible and cost the judge less than uniform corruption), not for Italian (0.119–0.122 → 1.39–1.40); `analysis/altloop/judge_at_ser.md`; `docs/project_status.md` §5.8.]* Plain bits cross 3.0 at about the same SER (German 0.13, Latin 0.10, Italian 0.05), so both halves of the rule fail together.
2. **The language *ranking* is right at every SER, but only German's is significant.** The truth language tops all 3 conditions on all 48 non-stuck keys (flip-rate 0 everywhere) — even at SER 0.5. German's language margin stays 0.08–0.15 (uncertainty 0.067) down to SER ≈ 0.3; Latin's is 0.02–0.04 against 0.067 and Italian's 0.01–0.05 against 0.193 at *every* SER including truth, i.e. within calibration uncertainty. On the stuck starts (SER 0.64–0.76) the ranking is German-first for German, Latin-first for both Latin and Italian — the wrong-key drift to a default language, not a signal.
3. **Error profile barely matters.** At matched SER the rare-first corruption (hapaxes wrong first) scores marginally better than uniform (German SER 0.084 rare → 1.59 vs SER 0.102 uni → 1.50), but the loop's actual final keys sit *on* the synthetic curve (German 0.132 → 1.40 between rare 0.127 → 1.34 and uni 0.152 → 1.20; Latin 0.17 → 0.99 vs rare 0.176 → 0.82 / uni 0.160 → 0.89), slightly above it because the wildcard objective's errors concentrate on rare types. No hidden benignness in the search's errors.
4. **The Italian A-like cell is not callable even at the true key by a margin of 0.06 bits** — the same shape as the 1.49/1.51 borderline instances in the Phase-6 acceptance. The A-like Italian source text is simply harder for the judge (2.60 bits/char at truth vs 1.89 German, 2.17 Latin), so anything short of a perfect key abstains there.

Consequence: to turn the wildcard loop's A-like results into *calls* the loop must reach SER ≲ 0.08 (German) / ≲ 0.04 (Latin) — the 0–400-nat objective-vs-truth gap noted in §8.4 has to close — or the judge must enter the acceptance rule (§8.5). The ranking signal that survives to SER 0.3 is a weak argument for the latter: the judge sees the right language long before it is willing to say so, but on Latin/Italian that preference is one calibration-uncertainty wide and cannot carry a decision on its own.

*[Superseded 2026-08-28: the anneal (§8.6) reached it — German 0.05 called (2.13), and Latin was called at 0.07 (1.70–1.72) despite the ≲ 0.04 estimate above; Italian 0.12 not called (1.39). The judge in the acceptance rule was tested in §8.5 with no gain. `analysis/altloop/judge_at_ser.md`; `docs/project_status.md` §5.2, §5.8.]*

## 9. Small-commitment loop (2026-08-26)

**Question.** Image and masked diffusion samplers commit a little per step
and re-read the denoiser between steps because the x₀ prediction is a
per-position *marginal*: with many plausible completions it is an average
over modes, and committing everything at once lands on an inconsistent
one. §7.1 measured exactly that regime on the badly-wrong keys (the judge
ranks *which* symbols are wrong above base rate but its argmax letter is
never right). The §7 arms committed either 8 symbols or the whole
disagreement set per round and stopped after ≤ 6 rounds (patience 2), so the
diffusion-faithful regime — commit the 1–2 most-supported symbols, re-read
the posterior under the new key, repeat for many rounds — was never run.
This section runs it. No training, no change to the proposer or the
acceptance rule (still the n-gram objective).

**Arms** (`scripts/altloop_pol.py`, tag `_smallk`): `psamp-k1`, `psamp-k2`,
`post-k1` (argmax letter, k = 1), against `psamp-k8`, `psamp-all` and the
size-matched control `rand-k2`, all at **32 rounds, patience 32** (no early
stop) so every arm gets the same number of posterior reads and SA calls;
the §7.3 6-round numbers are the second reference. Rung 2: the 8 bank/t5
cells + 6 deep cells, cold SA (T 0.5→0.2, 20k steps), 2 seeds; null start
(truth) for `psamp-k1`. Wordhom t0 cells (6.6 tokens/type): `psamp-k8`,
`psamp-k64`, `psamp-all` at 32 rounds, 1 seed, SA 200k at T 2.0→0.3.

**Pre-registered readings** (fixed before any number was read):

- R9.1 *Deep traps.* The three rung-2 deep cells stuck for every §7.3 arm
  (latin t0 / t3, italian t0 deep, SER 0.55–0.59) and german t0 deep: any
  seed reaching SER ≤ 0.005 under `psamp-k1`/`k2` where the 32-round
  `psamp-k8`/`psamp-all`/`rand-k2` do not is evidence for gradual
  commitment. Reaching it under all arms means the extra rounds, not the
  granularity, did it.
- R9.2 *Equal-round comparison.* Escapes (≤ 0.005 / 28 seed-cells) and
  "improved > 0.05" for k1/k2 vs k8/all vs rand-k2 at 32 rounds. Small-k
  is supported only if it beats k8 *and* all at the same round budget
  (it commits fewer symbols in total, so a tie is not support).
- R9.3 *Null.* `psamp-k1` from the true key moves ≤ 1 symbol on every cell
  except latin t5 (whose true key sits 300 nats below the pentagram
  optimum); more drift than the 6-round arms would mean 32 rounds of small
  commits erode a correct key, which would disqualify the sampler.
- R9.4 *Wordhom.* §7.4 recorded k = 8 as "a drip" at 6 rounds. With 32
  rounds: `psamp-k8` / `psamp-k64` final SER vs `psamp-all` (0.026 / 0.047
  at 6 rounds). If the small arms stay ≥ 0.3 the hypothesis is refuted
  at 6.6 tokens/type — large simultaneous commits are what the SA needs
  there.
- Objective traps (german t0 bank 0.240, latin t5) are expected to resist
  every arm; they are not evidence either way.

### 9.1 Results (2026-08-27; artifacts `analysis/altloop/runs_smallk_{r2_1..4,wh}.json`)

Rung 2, 32 rounds, final SER per seed (start SER in parentheses):

| cell | psamp-k1 | psamp-k2 | post-k1 | psamp-k8 | psamp-all | rand-k2 |
|---|---|---|---|---|---|---|
| german t0 bank (0.24) | 0.240 / 0.240 | 0.240 / 0.240 | 0.240 / 0.240 | 0.240 / 0.027 | 0.240 / 0.027 | 0.240 / **0.000** |
| german t1 bank (0.15) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.078 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| italian t0 bank (0.22) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| italian t4 bank (0.15) | 0.113 / 0.113 | 0.113 / 0.113 | 0.113 / 0.113 | 0.005 / 0.005 | 0.005 / 0.005 | 0.113 / 0.113 |
| latin t0 bank (0.41) | 0.397 / 0.027 | 0.002 / 0.002 | 0.002 / 0.027 | 0.002 / 0.002 | 0.002 / 0.002 | 0.002 / 0.002 |
| latin t3 bank (0.25) | 0.002 / 0.002 | 0.002 / 0.002 | 0.002 / 0.002 | 0.002 / 0.002 | 0.002 / 0.002 | 0.002 / 0.002 |
| german t0 deep (0.59) | 0.282 / 0.525 | 0.515 / 0.282 | 0.480 / 0.240 | 0.240 / 0.240 | 0.240 / 0.240 | **0.000** / 0.311 |
| german t1 deep (0.61) | 0.586 / 0.583 | 0.000 / 0.596 | 0.608 / 0.419 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| italian t0 deep (0.59) | 0.591 / 0.591 | 0.591 / 0.637 | 0.591 / 0.591 | 0.637 / 0.637 | **0.000 / 0.000** | 0.637 / 0.637 |
| italian t4 deep (0.43) | 0.426 / 0.426 | 0.426 / 0.426 | 0.426 / 0.426 | 0.005 / 0.005 | 0.005 / 0.005 | 0.113 / 0.150 |
| latin t0 deep (0.56) | stuck | stuck | stuck | stuck | stuck | stuck |
| latin t3 deep (0.55) | 0.554 / 0.554 | 0.554 / 0.554 | 0.554 / 0.554 | 0.554 / 0.002 | 0.002 / 0.002 | 0.554 / 0.002 |
| latin t5 pick / oracle (0.80 / 0.68) | ≥ 0.66 | ≥ 0.66 | ≥ 0.68 | ≥ 0.66 | ≥ 0.68 | ≥ 0.66 |
| **escapes ≤ 0.005 / 28** | **6** | **9** | **6** | 15 | **18** | 13 |
| **improved > 0.05 / 28** | 9 | 11 | 11 | 18 | 21 | 16 |

Wordhom t0 (32 rounds, 1 seed, ~14 min per arm): German k8 **0.030** /
k64 0.028 / all 0.027 (obj −13439 / −13422 / −13412); Italian k8 **0.406** /
k64 0.049 / all 0.049.

Null (`psamp-k1` from truth, 32 rounds): 0–1 symbol moved on every cell
except latin t5 (12 symbols, SER 0.174 — the pentagram optimum, as before).

**Readings.**

- R9.1 **FAIL.** No deep trap is opened by k1/k2 that the large-commit arms
  do not open; the reverse holds — italian t0 deep is opened only by
  `psamp-all` (both seeds), latin t3 deep by k8/all/rand-k2, german t1 deep by
  k8/all/rand-k2 on both seeds but by k2 on one and k1 on none.
- R9.2 **FAIL, decisively.** At equal rounds the ordering is
  all 18 > k8 15 > rand-k2 13 > k2 9 > k1 = post-k1 6. Judge-picked 1–2
  symbol commits lose to a *random* 2-symbol kick.
- R9.3 PASS (≤ 1 symbol outside t5).
- R9.4 Mixed: German k8 catches `all` once given 32 rounds (the §7.4
  "drip" was the round budget), Italian k8 does not (0.406); k64 = all on
  both. Large simultaneous commits are still what the SA needs at 6.6
  tokens/type.

**Why the analogy fails here.** The stuck keys are, by construction,
local optima of the n-gram objective under single-symbol reassignment —
that is what the SA converged to. A 1-symbol commit therefore always lowers
the objective, and the cold SA that follows walks straight back to the
optimum it came from (32 rounds, 0–4 accepted, SER unchanged on 20/28 k1
seed-cells). Escape needs several symbols moved *together* so that the
SA lands in a different basin, which is why kick size dominates and why the
random 2-symbol kick beats the judge's 1–2. The diffusion sampler's
gradual commitment works because each partial commit is *kept* while the
rest is re-denoised; under "accept on the n-gram objective" a partial commit
is never kept. Gradual commitment would need the commit to be protected
from the SA (frozen symbols) or the judge in the acceptance rule — the same
conclusion as §7.7 (2).

**Side result worth keeping.** 32 rounds instead of 6 (patience 32) is a
free gain for every arm: `psamp-all` 18/28 escapes vs 10/24 at 6 rounds,
opening italian t0 deep, latin t3 deep and one german t0 bank seed — the
latter is the "objective trap" of §7.3, which `rand-k2` also opens
(0.000), so german t0's 0.240 optimum is a *search* trap after all; latin
t5 remains the objective trap. Not adopted as a change to any recorded
number; carry the round budget forward in any future loop.

## 10. Manuscript-shaped control battery for the wildcard → anneal pipeline (2026-08-29/30)

**Question.** §8.6 validated the wildcard → anneal loop on clean A-like
positives only. Before an exhaustive treatment, the pipeline needs the
controls the manuscript comparison actually rests on: negatives of the
manuscript's shape (does the loop *invent* structure?), positives at the
Currier-B shape, the wrong-language hypothesis, plaintext with transcription
errors, a foreign block inside the text, and — added mid-run — a cipher that
lacks the hypothesis' doubled-letter units. Nothing here touches the
manuscript; the readings were fixed in `docs/wordhom_battery_restart.md`
before any cell finished.

**Instances** (`scripts/wordhom_battery.py --stage prepare|solve|report`;
`analysis/wordhom/battery/wordtypesall/`, 24 instances + the reused
`positive/<lang>/{Alike,Blike}` controls). Shapes: A-like 14 000 letters /
5 200 key types (4.2–4.4 tokens per type — Currier A's letter and type *counts*;
the manuscript itself is 3.0 tokens per type with 74 % hapax types), B-like 30 000 / 7 200
(5.5–5.7 — Currier B's counts; the manuscript is 4.6) *(clarified 2026-09-01; originally read "(4.2–4.4 tokens per type, Currier A)" and "(5.5–5.7, Currier B)"; `docs/project_status.md` §3)*. Per language (German, Latin, Italian):
`shuffled/{Alike,Blike}` (letters permuted, then enciphered — the language's
unigram statistics with no structure), `voynichesque/{Alike,Blike}` (pinned
gibberish generator at the shape's token count, draw selected by
tokens/type), `dirty/Alike_s05|_s10` (Phase-2 `TranscriptionNoise` at 5 % /
10 % per character, enciphered under the clean key; SER measured against
the noisy plaintext), `mixed/<lang>+<other>/Alike` (80 % host language with
a 20 % block of the other in the middle, one key), `nodouble/Alike` (see
below), and the cross-language cells `positive/<lang>/Alike` run under each
*other* language's hypothesis. Every cell: n-gram MDL start → wild 96 rounds
/ patience 10 → anneal `0,40` 80 rounds / patience 10 → Phase-6 judge
(`scripts/judge_at_ser.py --battery`, keys `stuck`, `truth`, finals). One
seed; seed 1 added on the three borderline positives. Chains
`analysis/altloop/battery/chain_*.sh`, tags `_bat_{wild,anneal}_{g0,g1,g0b,l0,l1,l1b,i0,i1,x1}`,
judge `_battery_*`, report `analysis/wordhom/battery/report.md`.

**A data defect found on the way.** Two Latin instances drew the same
held-out document — a pharmacopoeia ("pulveris ipecacuanhae … gr xii divide
in pulveres …": drug names, abbreviations, Roman-numeral doses) at 4.66
bits/char under the Latin LM, 10 % of the Latin held-out sampling weight
(the other five Latin docs score 2.6–3.0, German/Italian docs 2.2–2.9). On it
even the *true* key is judged non-language-like (margin 1.23, ranked German).
The sampler now redraws any window above 3.6 bits/char (`MAX_OWN_BPC`,
commit 7d4991f); the two instances were rebuilt, re-solved and rerun, and
the stale rows purged. Any earlier study that sampled Latin held-out windows
may have hit this document — check a Latin cell's plaintext bits/char before
attributing a failure to "Latin is hard".

### 10.1 Results (seed 0 unless stated; judge margin = structure margin of the anneal final; ceiling = margin of the true key)

Negatives and wrong hypotheses — every cell NOISE:

| cell | tok/type | wild → anneal margin | rank | called |
|---|---|---|---|---|
| shuffled A: de / la / it | 4.2–4.4 | 0.14→0.26 / 0.11→0.26 / 0.13→0.27 | noise | no |
| shuffled B: de / la / it | 5.6–5.7 | 0.12→0.17 / 0.10→0.18 / 0.11→0.21 | noise | no |
| voynichesque A: de / la / it | 4.4–5.0 | 0.17→0.35 / 0.18→0.37 / 0.14→0.25 | noise | no |
| voynichesque B: de / la / it | 6.0–6.4 | 0.14→0.23 / 0.15→0.25 / 0.14→0.21 | noise | no |
| German text under :latin / :italian | 4.1 | 0.21→0.43 / 0.17→0.39 | noise | no |
| Latin text under :german / :italian | 4.2 | 0.30→0.44 / 0.17→0.43 | noise | no |
| Italian text under :german / :latin | 4.2 | 0.31→0.44 / 0.29→0.48 | noise | no |

The n-gram objective climbs +8–40 k nats on every one of these keys while
the judge margin stays at 0.17–0.37 (negatives) / 0.39–0.48 (wrong
hypothesis); the language rank is 0.000–0.012 ± 0.067 and flips between
rows. The loop does not invent structure and does not invent its
hypothesis language; the top of the negative band is ≈ 0.5.

Positives:

| cell | tok/type | SER stuck → wild → anneal | margin | ceiling | called |
|---|---|---|---|---|---|
| positive/german/Blike | 5.5 | 0.69 → 0.085 → **0.026** | 2.22 | 2.36 | YES |
| positive/latin/Blike | 5.6 | 0.78 → 0.095 → **0.036** | 1.83 | 1.97 | YES |
| positive/italian/Blike | 5.6 | 0.78 → 0.125 → **0.070** | 1.46 | 1.61 | no |
| nodouble/german | 4.3 | 0.76 → 0.120 → **0.027** | 2.40 | 2.52 | YES |
| nodouble/latin | 4.3 | 0.73 → 0.131 → **0.040** | 1.88 | 2.06 | YES |
| nodouble/italian | 4.4 | 0.78 → 0.165 → **0.094** | 1.43 | 1.58 | no |
| mixed de+la (:german) | 4.2 | 0.75 → 0.141 → **0.050** (de 0.045 / la block 0.071) | 2.13 | 2.37 | YES |
| mixed la+de (:latin) | 4.2 | 0.76 → 0.76 → 0.64; seed 1: 0.70 → 0.27 → **0.17** (+80 rounds; la 0.13 / de block 0.33) | 0.52; 1.24 | 1.99 (ranked ge) | no |
| mixed it+la (:italian) | 4.2 | 0.77 → 0.58 → 0.52 | 0.59 | 1.59 (ranked ge) | no |
| dirty de 5 % | 4.2 | 0.75 → 0.60 → **0.148**; seed 1: 0.64 → 0.251 | 1.51; 1.13 | 1.95 | YES; no |
| dirty la 5 % | 4.2 | 0.76 → 0.77 → 0.68; seed 1: 0.76 → 0.66 | 0.43; 0.41 | 1.56 | no |
| dirty it 5 % | 4.2 | 0.76 → 0.59 → 0.45 | 0.62 | 1.22 | no |
| dirty de / la / it 10 % | 4.2 | 0.63 / 0.74 / 0.69 | 0.44 / 0.37 / 0.46 | 1.54 / 1.06 / 0.91 | no |

A-like positives (anneal finals of §8.6, seeds 0–2; from `analysis/altloop/judge_at_ser.md`; table added 2026-09-01 — until then these judge rows were recorded only in the `d5` reference column of `docs/wordhom_bigram_variant.md`):

| cell | tok/type | SER (anneal final) | margin | ceiling | called |
|---|---|---|---|---|---|
| positive/german/Alike | 4.1 | 0.049–0.050 | 2.13–2.14 | 2.33 | YES |
| positive/latin/Alike | 4.2 | 0.066–0.073 | 1.70–1.72 | 1.94 | YES |
| positive/italian/Alike | 4.2 | 0.119–0.122 | 1.39–1.40 | 1.56 | no |

### 10.2 Readings

- **B-like positives solve better than A-like** (0.026 / 0.036 / 0.070 vs
  0.05 / 0.07 / 0.12 at A-like): more tokens per type outweighs the larger
  key. The manuscript's Currier-B shape is the easier one for this pipeline.
- **`nodouble` — a hypothesis with doubled-letter units the cipher never
  used costs nothing.** The decoder drains the unused units by itself
  (n-gram start 60–80 types in them, wild 40–60, anneal 6 / 10 / 27 types =
  0.04–0.2 % of occurrences) and ends at or below the matched positive's
  residual in every language. The reverse mismatch (cipher uses doubled
  units, hypothesis lacks them) was not built.
- **The structure-margin ceiling is a property of the language**: true keys
  on clean text score 2.4–2.5 (German), 2.0 (Latin), 1.6 (Italian). Under
  the frozen ≥ 1.5 rule that is 0.9 / 0.5 / 0.1 bits of headroom, which is
  why solved Italian cells (SER 0.07–0.09) are never called and Latin calls
  are narrow. Transcription noise eats it fast: truth-key margins at 5 % /
  10 % are 1.95 / 1.54 (de), 1.56 / 1.06 (la), 1.22 / 0.91 (it) — a perfect
  decipherment survives the rule at 10 % noise only in German and at 5 %
  only in German and (barely) Latin.
- **Perturbed Romance A-like positives stay in the n-gram trap.** Dirty
  5 % Latin fails identically at two seeds (wild stage never leaves the
  start); mixed Latin escapes at seed 1 but not seed 0; Italian dirty/mixed
  reach 0.45–0.52. The 80-round anneal budget — set on clean A-like text,
  where patience triggers first — is binding on these cells (62/80 and
  53/80 rounds accepted at the end; the mixed-Latin continuation converged
  at 0.17 with 56 more rounds). The single dirty-German seed-0 call (0.148,
  margin 1.51) did not replicate (seed 1: 0.251, 1.13).
- **A 20 % foreign block does not lower the ceiling but flips the
  ranking**: the German host is called with the Latin block decoded at
  0.071; the Latin and Italian hosts' true keys are language-like
  (1.99 / 1.59) but ranked German at 0.006–0.019 ± 0.067. "Language-like"
  and "which language" separate for the Romance hosts.
- **Unsolved dirty-10 % positives judge like negatives** (0.37–0.46): a
  stuck key on noisy real text is worth ~0.1 bits more than a stuck key on
  gibberish.

### 10.3 Where the manuscript sits (charts `analysis/plots/structure_margin_stage{1,2,3,4a,4b}*.png`)

Like-for-like on the word-homophonic head (n-gram keys through this
pipeline), the manuscript's 12 wordhom cells score 0.36–0.51 against
negatives at 0.17–0.37 — just above, with one shared bin. Across all 72
altloop psamp cells the manuscript spans 0.36–1.05 (median 0.74; Currier A
0.48–0.95, Currier B 0.36–1.05), above every n-gram-key negative here
(≤ 0.48) and every unsolved dirty-10 % positive (≤ 0.46), and ~0.45 bits
below the call line. The Phase-6 negatives on *ELBO-polished* keys —
shuffled ≈ 0, voynichesque + contamination 0.3–1.51 — cover the
manuscript's whole range and reach the threshold, so the gap between the
manuscript and "no language" depends on how the key was obtained; the
honest control for any manuscript number is a same-shape negative through
the same key search.

### 10.4 Reverse mismatch — `revdouble` (2026-08-30, German only)

The cipher uses the language's top-5 doubled units, the hypothesis has
letters only (`truth.hyp_bigrams = []`, honoured by the solver, the loop and
the judge; the judge's `truth` row is the *projected* truth — doubled-unit
types mapped to their base letter — since the exact key is unrepresentable).
German A-like, 13 515 tokens / 3 258 types (4.1 per type); 148 types
(≈ 2.4 % of occurrences) sit on doubled units, so the floor is SER **0.035**
(485 unrecoverable second letters) and the true text violates the
letter-only repeat rule 19 times.

| key | SER | margin | rank | called |
|---|---|---|---|---|
| n-gram start | 0.673 | 0.47 | ge | no |
| wild | 0.170 | 1.23 | ge | no |
| anneal (80/80 rounds, 48 accepted) | **0.098** | **1.88** | ge>la>it | **YES** |
| projected truth | 0.035 | 2.11 | ge | YES |

Reading: omitting units the cipher does use is *not* free, unlike the
forward mismatch — the residual is 0.098 against 0.027 (nodouble) and 0.05
(matched positive), the wild stage starts from a smaller unit space (start
SER 0.67 rather than 0.75) but the anneal is again budget-bound. The damage
concentrates where expected: 73 of the 148 doubled-unit types are still
wrong after the anneal (their tokens decode to one letter and break the
repeat-rule pattern the objective relies on) against 5.5 % of occurrences
among ordinary letter types. The cell is nevertheless **called** (margin
1.88 with German's headroom); for Latin/Italian, whose ceilings are 2.0 /
1.6, the same 0.06–0.07 residual penalty would likely cost the call.
Latin/Italian instances are built (`revdouble/{latin,italian}/Alike`) and
not run.

### 10.5 Carry forward

1. Anneal budget: use patience-terminated runs (e.g. 200 rounds / patience
   10) on perturbed positives; the 80-round cap cost at least two solves.
2. Report the structure margin with the per-language truth ceiling, not
   only the binary abstention: a Latin/Italian plaintext with a few percent
   transcription noise is a decipherment the frozen rule cannot call.
3. Seeds: negatives are one seed each; the positive/negative gap is large
   enough that this is fine for the negatives, not for borderline positives.
4. Run `revdouble` for Latin/Italian (built, unsolved) if the doubled-unit
   hypothesis is to be argued for the Romance languages.
5. Latin held-out set: the pharmacopoeia document should be excluded (or
   down-weighted) at the next corpus version.
