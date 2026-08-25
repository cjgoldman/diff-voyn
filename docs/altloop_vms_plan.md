# Plan — diffusion-guided n-gram search on the manuscript, with controls in every cell

Status: **RUNNING 2026-08-25** — `scripts/altloop_vms.py` implemented (tests in `tests/test_altloop.py`); wordhom and homophonic launched 23:10 UTC (GPU 0/1), sub1to1 queued. Implementation note: start keys are the `vms_solves.json` MDL-pick n-gram candidates (§2), not the Phase-6 ELBO-polished keys — the polished key sits ~6500 nats below the n-gram optimum, so an n-gram-accepting loop leaves it in round 1 on every arm. Follows `docs/alt_loop_plan.md` §7
(proof of life) under time pressure: the three §7.7 carry-forward items are
*deferred*, and in their place every manuscript cell runs its own
same-size random control (`rand-k512` / `rand-k8`) and SA-alone control
(`none`) so that any improvement can be read on the spot. Metrics stream to
ClearML per round; a "promising" flag is raised by the script, surfaced in
ClearML tags/scalars, and watched by the agent session, so the user can decide
early whether to add compute — for speed, or to fund option B.

## 1. What this run can and cannot say

- **Can say:** whether the diffusion-guided loop finds, in any cipher × language
  × dialect cell, a key that (a) the n-gram objective prefers, (b) the frozen
  judge scores as more language-like than Phase 6's best (structure margin
  above the manuscript's recorded 1.25 ceiling), and (c) the random and
  SA-alone arms on the *same cell* do **not** reach. Only (a)+(b)+(c) together
  is "promising"; any one alone is noise or the objective refitting rare types
  (§7.4/§7.5 of the PoL doc).
- **Cannot say:** that a null result means "nothing to find". The proposer is
  blind on mostly-wrong keys (PoL-1 exact precision 0) and the synthetic
  battery never tested the manuscript's 3.0–4.6 tokens-per-type regime. A null
  here is "the method did not find it", not evidence of absence — and the
  Phase-6 abstention stands regardless.
- **Does not change the Phase-6 record.** `ABSTAIN_RULE`, controls and the 87
  cells stay as recorded; this is a new search on top of them, reported
  separately.

## 2. Cells

The loop needs a symbol → unit map interface, so three of the five heads run:

| head | presentations | start keys | key size | cells | proposer |
|---|---|---|---|---|---|
| `wordhom` | wordtypesall, IT2a/RF1b × A/B | `analysis/wordhom/vms_solves.json` best-`inner` candidate | 3.6k–5k types | 12 (4 instances × 3 languages) | `posterior_sample`, k = all |
| `homophonic` | eva (20/23 symbols) + boxer top-20 | `analysis/phase6/vms_solves.json` MDL-pick candidate per window | 20–23 | 30 (18 eva + 12 boxer) | `posterior_sample`, k = 8 |
| `sub1to1` | eva + boxer | same | 20–23 | 30 | `posterior_sample` restricted to **pair swaps** (keeps the bijection); k = 4 swaps |

Not run: `naibbe` (block-map key, different neighbourhood) and `arithmetic`
(no admissible canonical order on the Boxer stream, Phase 6) — state this in
the report as scope, not evidence. Currier A and B are never pooled (each cell
is one dialect already). Windows are the recorded Phase-6 windows.

Ordering (so the most informative cells report first): wordhom → homophonic
eva → homophonic boxer → sub1to1. Within a head, Currier A before B (A is
where the Naibbe grammar compressed).

## 3. Arms — every cell runs all three, same start key, same seeds

| arm | proposer | k |
|---|---|---|
| `psamp` | judge-picked symbols, letter sampled from the per-occurrence posterior | all (wordhom) / 8 / 4 swaps |
| `rand` | uniformly random symbols → random same-length units | size-matched: 512 (wordhom) / 8 / 4 swaps |
| `none` | no kick; the short SA alone | — |

Loop settings as in the PoL: ≤ 6 rounds, patience 2, accept on the n-gram
penalized objective, SA 200k steps (wordhom) / 50k (symbol heads) at T 2.0→0.3,
posterior 16 draws at mask rate 0.3. Two seeds per arm on wordhom and
homophonic (the flip-rate convention), one on sub1to1. **No polish, no choice
term inside the loop** (`CHOICE_TERM_WARNING` stands).

## 4. Metrics recorded per round (and streamed to ClearML)

Computed on the round's *accepted* key (or the proposed one if rejected —
both are logged, flagged), own-language condition, budget 64, same masks
across arms (CRN):

| scalar | meaning | source |
|---|---|---|
| `ngram_obj` | penalized n-gram objective (nats) | head `objective` |
| `plain_bits` | ELBO bits/char of the decode, own condition, calibrated `v3-phase_c-ro` | `DiffusionEvaluator.score_stream` |
| `structure_margin` | bits of a letter-shuffled copy of the decode − `plain_bits` (the Phase-6 statistic) | same, on the shuffled copy |
| `mdl_total_per_symbol` | plaintext bits + key + choice bits per **all** ciphertext symbols (cross-head ranking scale) | `scale.cell_score` |
| `n_changed`, `n_accepted`, `seconds`, `draws` | bookkeeping | loop info |
| `delta_vs_rand`, `delta_vs_none` | `structure_margin(psamp) − structure_margin(control)` at the same round (control gate) | computed when both arms have reported |

Plus once per cell at the end: `language_like` under `ABSTAIN_RULE`
(plain ≤ 3.0 and margin ≥ 1.5), top-language margin and its calibration
uncertainty, replicate flip-rate.

## 5. "Promising" — fixed now, before any manuscript number is read

Three tiers, evaluated after every round of every cell, on the `psamp` arm:

| tier | condition | meaning |
|---|---|---|
| **NOTABLE** | `structure_margin ≥ 1.26` (above the manuscript's Phase-6 ceiling 1.25) **and** ≥ 0.15 above both controls' best margin on the cell | first time a manuscript decode leaves the manuscript band; worth a look |
| **PROMISING** | `structure_margin ≥ 1.49` (the lowest true decipherment in the Phase-6 battery) **and** both controls < 1.26 | inside the true-decipherment band, and the judge's kick — not the perturbation — got it there |
| **LANGUAGE-LIKE** | `ABSTAIN_RULE` satisfied (plain ≤ 3.0 **and** margin ≥ 1.5) **and** replicate flip-rate 0 across seeds **and** controls fail the rule | the Phase-6 verdict would flip for this cell |

Anything that the random or `none` arm *also* reaches is downgraded to
**noise** and logged as such, whatever its margin (that is the §7.5 lesson: a
lower number is not evidence unless the control cannot get it). A rising
`ngram_obj` with a flat or falling `structure_margin` is the objective
refitting rare types — logged, never flagged.

## 6. ClearML layout

Project `diff-voyn`, one Task `altloop-vms-<UTC date>` per launch (resumable
launches reuse the task id via `Task.init(continue_last_task=True)`), tags
`phase6-followup`, `altloop`, plus the head being run.

- Scalars: one **title per cell** (`wordhom/IT2a-A/latin`), **series per arm**
  (`psamp`, `rand`, `none`), **iteration = round**, for each metric of §4 —
  so the ClearML scalar view shows the three arms of a cell on one plot.
- Summary titles updated after every round: `best_structure_margin`
  (series per head; the running maximum over cells of the `psamp` arm),
  `best_delta_vs_controls`, `cells_done`, `promising_count` per tier.
- Task tags flipped by the script: `NOTABLE`, `PROMISING`, `LANGUAGE-LIKE`
  (added the first time a tier fires; never removed) — the ClearML
  experiment list shows them without opening the task.
- Configuration object: the cell list, arm settings, §5 thresholds, evaluator
  sha256, calibration version (so the pre-registration is inside the task).
- Artifacts: `analysis/altloop_vms/runs_<head>.json` uploaded after every
  cell; `promising.json` (see §7) uploaded whenever it changes.
- Debug samples: for any NOTABLE-or-better round, the first 400 characters of
  the decode as a text sample on the task (the user can read it without
  the repo).

The scalar names are fixed here so the dashboard and the report agree; the
report stage reads the same JSON the task uploads.

## 7. Awareness — how the agent and the user find out

1. **Script side.** After every round the script appends one line to
   `analysis/altloop_vms/events.log`: `EVENT <tier> <cell> round=<r>
   margin=<m> rand=<m_r> none=<m_n> plain=<p>` for tier ∈ {NOISE, NOTABLE,
   PROMISING, LANGUAGE-LIKE}; NOTABLE-or-better also rewrites
   `promising.json` (cell, round, arm metrics, decode sample, key) and flips
   the ClearML tag. Nothing waits on a human — the run continues.
2. **Agent side.** The launching session arms a `Monitor` on `events.log`
   filtered to `NOTABLE|PROMISING|LANGUAGE-LIKE|Traceback`, plus a heartbeat
   line per finished cell. On any tier ≥ NOTABLE the agent re-reads
   `promising.json`, checks the control gate by hand, and posts a short
   read-out in the session (what fired, what the controls did, whether the
   window/seeds agree). If the session has push notifications available
   (`PushNotification`) it sends one for PROMISING or better; otherwise the
   read-out is the notification.
3. **User side.** The ClearML experiment list (project `diff-voyn`, filter
   tag `altloop`) shows the tags; the task's scalar page shows
   `best_structure_margin` per head against the two reference lines the
   plan fixes (1.25 manuscript ceiling, 1.49 lowest true decipherment —
   logged as constant series `ref_vms_ceiling`, `ref_true_min` so they draw
   on the same plot). No email or external service is wired: ClearML's own
   alerting can be attached by the user if wanted.

## 8. Compute, order, and the decision points

Estimates from the PoL timings on this box (12 cores shared with any other
SA job; two 24 GB GPUs; the SA is CPU-bound, the posterior/ELBO GPU-bound):

| head | cells × arms × seeds | per run | wall on 2 GPUs |
|---|---|---|---|
| wordhom | 12 × 3 × 2 = 72 | 10–40 min (SA 200k over 10–23k tokens) | ~10–14 h |
| homophonic | 30 × 3 × 2 = 180 | ~5–10 min | ~8–12 h |
| sub1to1 | 30 × 3 × 1 = 90 | ~5 min | ~4 h |

≈ 1.5 days serial on two GPUs, wordhom results in the first half-day. Each
head is a separate launch (`--head`) with its own runs JSON, resumable
(cell × arm × seed keyed; the crash we hit in the PoL is fixed).

Decision points for the user — each is a place where the ClearML numbers
answer "add compute?":

| when | if | then |
|---|---|---|
| after wordhom (½ day) | any NOTABLE with the control gate passing | add GPUs: run the remaining heads in parallel *and* start option-B item 3 (manuscript-shaped synthetic battery) to calibrate what that margin means |
| after wordhom | nothing above NOISE, `psamp` ≈ `rand` ≈ `none` | the method is blind in this regime as §1 warned; the remaining heads are cheap to finish but option B is the only path to a stronger statement — decide whether to fund it |
| any time | PROMISING or LANGUAGE-LIKE | stop adding cells; spend compute on that cell: 4 seeds, both windows, both transcriptions, race-polish confirmation with the choice term off, and the full §5 control battery of the PoL on a synthetic of that cell's shape |
| any time | `rand` or `none` reaches the same margin as `psamp` on a cell | that cell is noise regardless of the number; do not escalate on it |

## 9. Code (small; everything reuses PoL machinery)

- `scripts/altloop_vms.py --head {wordhom,homophonic,sub1to1} --stage {run,report}`:
  cell builder from the recorded solves (start keys, window spans, stream),
  the three arms, per-round metrics of §4 (shuffled-copy scoring reuses the
  Phase-6 `controls` helper), §5 tiers, `events.log`/`promising.json`,
  ClearML reporting via a thin `infra/clearml_task.py` addition
  (`init_analysis_task(name, tags, config)` + `report_cell_round(...)`).
- `heads/altloop.py`: a `pair_swap` proposer for sub1to1 (posterior-ranked
  symbol pairs whose swap both symbols' posteriors favour); ELBO metrics
  callback already exists.
- `heads/posterior.py`: unchanged (wordhom `unit_scores`, symbol
  `symbol_scores` already cover the three heads).
- Tests: tier logic on synthetic numbers (a NOISE case, a NOTABLE case, a
  downgraded case), ClearML reporter mocked.
- Report stage: `analysis/altloop_vms/report.md` — per cell the three arms'
  final margins, the best round, tier, and the control-gate verdict; a
  one-table summary per head; the §1 caveat verbatim.

## 10. Run commands

```bash
uv run python scripts/altloop_vms.py --head wordhom     --stage run --device cuda --seeds 2   # GPU 0
uv run python scripts/altloop_vms.py --head homophonic  --stage run --device cuda --seeds 2   # GPU 1
uv run python scripts/altloop_vms.py --head sub1to1     --stage run --device cuda --seeds 1   # whichever frees first
uv run python scripts/altloop_vms.py --stage report
```

Launch via pid-based queue scripts (not `pgrep -f` waiters); verify with
`ps -eo args | grep altloop_vms` and a Traceback-free log 30 s after launch;
arm the `Monitor` of §7.2 before leaving the session.

## 11. Out of scope (deferred option-B items, unchanged)

Judge-in-acceptance (`race_polish` confirmation inside the loop), the
manuscript-shaped synthetic battery, and the VMS regression arm of the race
study. If §8's first decision point fires, item 3 is the first to fund.
