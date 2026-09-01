# Word-homophonic decoder with doubles + bigrams as units (2026-08-30/31)

> **Record status (banner added 2026-09-01):** `d5b20` decoder-variant study, 2026-08-30/31; complete.
> Still current in full (newest study of the word-homophonic track): letters-only ciphers unharmed under `d5b20` 3/3, Italian `d5` cipher hurt (0.314), a matched 48-unit key unfindable 3/3 at ~4 tokens/type, VMS 8/8 NOISE; `d5` remains the default unit spec. The `d5` reference column here was, until 2026-09-01, the only record of the A-like Latin call (0.066–0.073 → 1.70–1.72, YES, 3 seeds); it is now also tabulated in `docs/alt_loop_plan.md` §10.1. **Current project position: `docs/project_status.md`.**

**Question (user, 2026-08-30).** The `nodouble` control of the battery
(`docs/alt_loop_plan.md` §10) showed that a decoder whose unit set carries the
language's top-5 doubled letters loses nothing on a cipher that never used
them — the search drains the spare units by itself. Does that still hold when
the decoder's unit set is much wider: letters + top-5 doubled letters + the
top-20 non-doubled bigrams (48 units, spec `d5b20`, vs 30 for `d5`)? And, for
fun, what does the manuscript look like under that decoder?

Unlike the doubles, frequent bigrams (`en er ch nd …`) are *good* fits in many
places, so a wrong bigram assignment can be a stable local optimum rather than
something the anneal drains — the mechanism that made `nodouble` free is
weaker here. Nothing in this study is a claim about the manuscript's cipher:
the bigram-unit reading was rejected as a *hypothesis* by the doubling-rate
control in `docs/wordhom_study.md` (with bigram units the plaintext's repeated-
unit rate is 35–44 per 1000 against the manuscript's 7–9). This is a decoder
over-specification study.

## Design

* **Unit sets** (`heads/wordhom.py::language_targets(units=…)`,
  `parse_units`): `d5` = the Phase-6 hypothesis (letters + top-5 doubled
  letters by p1(a)p2(a|a)); `d5b20` = the same five doubles first, then the
  20 most probable non-doubled bigrams (a≠b) under the frozen n-gram LM. The
  doubles come first so any `d5` key is a `d5b20` key unchanged. Per language:
  German `ss nn ff ll tt | en er ch nd de ge ei in te he es ie se vn ne st ic
  an be re`; Latin `ss ll ii ee mm | er ti in um qu it te is es tu us nt re
  ri en et ta at si on`; Italian `ll ss tt ee aa | er on an al en co ra el to
  es ar la ch or ia re di no in te`.
* **Truth projection** (`project_key`): a cipher key is expressed in the
  decoder's unit space — shared bigram units keep their identity, units the
  decoder lacks fall back to the first letter (the closest representable key;
  the letter-level SER then charges one wrong letter per occurrence). Replaces
  the ad-hoc `hyp_bigrams` projections of the `revdouble` cell.
* **Plumbing**: `--units d5b20` on `wordhom_study.py --stage solve --set vms`
  (→ `vms_solves_d5b20.json`), `wordhom_battery.py --stage solve`
  (→ `battery_solves_d5b20.json`, positives from the Phase-6 controls included
  via `--only`), `altloop_pol.py --battery`, `judge_at_ser.py --battery`,
  `altloop_vms.py --head wordhom` (+ `--langs`). The head, the incremental
  numba objective, the posterior proposer (`unit_scores`: a type moves only
  within its length class) and the choice-bits term were already generic over
  (a, b) units; tests `tests/test_wordhom.py` (unit-set parsing, ordering,
  projection, incremental deltas with general bigrams, cipher inversion).
* **New battery instance** `bigram/<lang>/Alike`: the cipher itself uses the
  `d5b20` units (`truth.cipher_units`). Bigram units absorb ~25 % of the
  letters into fewer tokens, so the key is scaled to keep tokens per type equal
  to the `d5` cipher of the same plaintext (2 639–2 700 observed types, 10.3–
  10.9 k tokens, 3.9–4.1 tokens/type; the other A-like cells are 4.2–4.4).
  Everything else as in §10: n-gram MDL start (2 restarts × 2 M SA steps) →
  wild 96 rounds / patience 10 → anneal `0,40` 80 rounds / patience 10 →
  Phase-6 judge on `stuck`, `truth` (projected), finals; one seed.

## Cells

Manuscript (GPU 1, `analysis/altloop_vms/d5b20/`, tags `_d5b20_{wild,anneal}_{de,la}`):
the four `wordtypesall` streams IT2a/A, RF1b/A, IT2a/B, RF1b/B under German
first, then Latin; arms `none` / `rand` / `post`, 1 seed, starts from the
`d5b20` n-gram MDL picks (`analysis/wordhom/vms_solves_d5b20.json`); tiers read
by `heads.altloop.classify_tier` exactly as in `docs/altloop_vms_plan.md` §5.

Battery (GPU 0 German → Latin after the `revdouble` chain; GPU 1 Italian after
the manuscript; tags `_big_{wild,anneal}_{ge,la,it}`, reverse cell
`_bigrev_*`, judge `_battery_big_*` / `_battery_bigrev_*`), per language:

| cell | cipher units | decoder units | role |
|---|---|---|---|
| `positive/<l>/Alike` | d5 | d5b20 | over-specified decoder on the standard positive |
| `nodouble/<l>/Alike` | letters only | d5b20 | the user's question proper: 25 spare units, none used |
| `bigram/<l>/Alike` | d5b20 | d5b20 | matched positive: is a 48-unit key findable at all? |
| `bigram/<l>/Alike` (rev) | d5b20 | d5 | reverse mismatch: decoder lacks 20 units the cipher uses |
| `shuffled/<l>/Alike` | d5 (no structure) | d5b20 | negative: does the wider decoder invent structure? |
| `voynichesque/<l>/Alike` | — | d5b20 | negative, manuscript-like |

## Readings (fixed 2026-08-30 22:45 UTC, before any cell finished)

Reference values from §10 under `d5` (anneal finals; judge margin; called):
positives 0.05 / 0.07 / 0.12 (de / la / it), nodouble 0.027 (2.40, YES) /
0.040 (1.88, YES) / 0.094 (1.43, no); negatives ≤ 0.48 margin; truth-key
ceilings 2.4–2.5 / 2.0 / 1.6.

- **R1 — over-specification cost.** For `positive` and `nodouble` under
  `d5b20`: "no large negative effect" means the anneal final SER is within
  +0.03 absolute of the `d5` final on the same instance AND the judge verdict
  is unchanged (called where `d5` called, margin within 0.15). Leakage is the
  occurrence fraction assigned to general-bigram units at the final (the
  cipher uses none): ≤ 1 % is "drained" like the doubles (0.04–0.2 % there);
  ≥ 5 % means bigram units are a trap the anneal does not leave.
- **R2 — matched positive.** `bigram` under `d5b20`: a final SER ≤ 0.10 and a
  judge call in German/Latin (Italian's ceiling is 1.6) says a 48-unit
  unstructured key is findable at 4 tokens/type; a final ≥ 0.3 says the wider
  unit space is beyond the pipeline at the manuscript's shape, which would
  cap what the manuscript run below can show.
- **R3 — reverse mismatch.** `bigram` under `d5`: the projected truth has a
  letter SER floor > 0 (the second letter of every bigram unit is missing);
  record that floor from the judge's `truth` row and whether the loop reaches
  it. Expected: no call (compare `revdouble`).
- **R4 — negatives.** `shuffled` / `voynichesque` under `d5b20` must stay
  NOISE (anneal margin ≤ 0.5, the top of the `d5` negative band). A margin
  ≥ 0.75 on any negative is a false-structure alarm for the wider decoder and
  disqualifies its manuscript numbers from comparison with the `d5` band.
- **Manuscript.** Tiers per §5 (`NOISE` unless margin ≥ 1.26 and ≥ 0.15 above
  both controls). The `d5` wild→anneal run ended 24/24 NOISE with anneal
  margins A 0.40–0.54, B 0.30–0.43. A `d5b20` reading is *notable for this
  study* only if the `post` arm ends ≥ 0.15 above the `d5` anneal margin of
  the same cell AND above both controls of the same cell; a rise that the
  `none`/`rand` arms show equally is the wider unit space fitting noise, and
  R4 says how much of that to expect.

## Results

### Manuscript, German (2026-08-31 03:00 UTC; `runs_wordhom_d5b20_{wild,anneal}_de.json`)

Structure margin (own condition) at the anneal final, one seed; plain bits of
the `post` arm; the `d5` wild→anneal anneal band of the same streams
(`docs/altloop_vms_plan.md` §13) for reference. Start keys (n-gram MDL pick
in the 48-unit space) scored 0.69 / 0.68 / 0.55 / 0.59.

| stream | none | rand | post | plain (post) | tier | d5 anneal band |
|---|---|---|---|---|---|---|
| IT2a/A | 0.714 | 0.707 | 0.676 | 3.15 | NOISE | 0.40–0.54 |
| RF1b/A | 0.717 | 0.711 | 0.724 | 3.13 | NOISE | 0.40–0.54 |
| IT2a/B | 0.528 | 0.562 | 0.540 | 3.32 | NOISE | 0.30–0.43 |
| RF1b/B | 0.612 | 0.608 | 0.581 | 3.24 | NOISE | 0.30–0.43 |

**4/4 NOISE.** The judge-guided arm ends within −0.04…+0.01 of the controls
on every stream; the wild stage behaves as under `d5` (one accepted round
then a stall for `post`/`none`, margins 0.22–0.49), and the anneal returns all
three arms to the start basin. The whole level sits ~0.2 bits above the `d5`
band — but so do the 48-unit decoder's *wrong* keys on synthetic text (the
German battery's stuck starts score 0.59–0.60 at SER 0.70–0.76 against ~0.4
under `d5`, and the unsolved `bigram` cipher's 98 %-wrong final 0.56): the
extra 20 units buy a fixed ~0.15–0.2 bits of margin on any key. Top language
German on 11/12 runs with a language margin of 0.001–0.015 against 0.067
uncertainty — noise, as before.

### Manuscript, Latin (2026-08-31 06:50 UTC; `runs_wordhom_d5b20_{wild,anneal}_la.json`)

Start keys scored 0.60 / 0.61 / 0.42 / 0.49.

| stream | none | rand | post | plain (post) | tier | d5 anneal band |
|---|---|---|---|---|---|---|
| IT2a/A | 0.618 | 0.595 | 0.632 | 3.17 | NOISE | 0.40–0.54 |
| RF1b/A | 0.611 | 0.604 | 0.633 | 3.13 | NOISE | 0.40–0.54 |
| IT2a/B | 0.433 | 0.429 | 0.476 | 3.25 | NOISE | 0.30–0.43 |
| RF1b/B | 0.482 | 0.495 | 0.496 | 3.23 | NOISE | 0.30–0.43 |

**4/4 NOISE** (8/8 with German). `post` ends +0.00…+0.05 above the better
control — inside the round-to-round noise of the anneal and an order of
magnitude below the +0.15 / ≥ 1.26 bar. Latin runs ~0.1 below German on every
stream, as under `d5`; top language Latin on 12/12 runs at 0.001–0.008
against 0.067 uncertainty.

### Battery, German (2026-08-31 05:20 UTC; tags `_big_*_ge`, `_bigrev_*_ge`, judge `_battery_big(rev)_ge`)

Anneal finals, one seed; the `d5` value of the same instance (§10) in the last
column. Leakage = occurrence fraction on general-bigram units at the final.

| cell | decoder | SER stuck → wild → anneal | judge plain / margin / called | leakage | d5 reference |
|---|---|---|---|---|---|
| positive/german | d5b20 | 0.762 → 0.527 → **0.064** | 2.07 / 2.12 / YES | 5.0 % → 1.3 % | 0.05, 2.07, YES |
| nodouble/german | d5b20 | 0.700 → 0.134 → **0.035** | 1.84 / 2.40 / YES | 2.3 % → 0.7 % | 0.027, 2.40, YES |
| bigram/german | d5b20 (matched) | 0.701 → 0.704 → **0.700** | 3.28 / 0.56 / no (truth 1.94 / 2.30 / YES) | — | — |
| bigram/german | d5 (reverse) | 0.700 → 0.704 → **0.702** | 3.47 / 0.40 / no (projected truth SER 0.237: 3.52 / 0.92 / no) | — | revdouble: 0.098, 1.88, YES |
| shuffled/german | d5b20 | 0.818 → 0.817 → 0.819 | 3.42 / **0.45** / no | — | 0.26 |
| voynichesque/german | d5b20 | — | 3.30 / **0.52** / no | — | 0.35 |

Stuck-start margins under d5b20: 0.59 / 0.60 / 0.59 / 0.48 / 0.52 (positive,
nodouble, bigram, shuffled, voynichesque) against ~0.4 / 0.14 / 0.17 under
`d5`; the *same* `bigram` stuck key scores 0.59 under d5b20 and 0.40 under d5.

- **R1 PASS (both cells).** The over-specified decoder ends within +0.014 /
  +0.008 SER of the `d5` finals with identical verdicts and margins (2.12 vs
  2.07; 2.40 vs 2.40). Cost: the wild stage is slower on the d5 cipher
  (0.53 at its stall vs 0.13 under `d5`; the anneal was still improving at
  the 80-round cap, 61/80 accepted) and a residual leakage of 1.3 % / 0.7 % of
  occurrences into never-used bigram units — above the doubles' 0.04–0.2 %,
  far below the 5 % trap line, and still draining when the budget ran out.
- **R2 FAIL.** A cipher that really uses all 48 units is not recovered
  (0.70 throughout; wild stalls in 11 rounds, anneal climbs +6 k nats without
  approaching the truth), although its true key is perfectly callable (SER 0,
  margin 2.30). At ~4 tokens per type the pipeline finds 30-unit keys and not
  48-unit ones — this caps the manuscript reading: a true 48-unit cipher on
  the manuscript would look exactly like this cell.
- **R3.** Reverse mismatch unrecovered as well (0.70), and its projected truth
  is itself not language-like (margin 0.92, plain 3.52 — 24 % of the letters
  are missing), while the language *ranking* at that key is still significant
  (German +0.167 ± 0.067). Contrast `revdouble` (five doubles missing):
  recovered to 0.098 and called (margin 1.88).
- **R4.** `shuffled` 0.45 (PASS), `voynichesque` **0.52 — 0.02 over the
  pre-stated 0.5 line**, far under the 0.75 alarm. The wider decoder lifts
  every wrong or structureless key by a near-constant +0.17…+0.19 bits
  (stuck keys +0.19 like-for-like; negatives +0.19 / +0.17; manuscript A
  +0.2). It is an offset, not invented structure, but the `d5b20` numbers must
  be read against their own negative band (≈ 0.45–0.52), not the `d5` band.

### Battery, Latin and Italian (2026-08-31 13:00 UTC; tags `_big_*_{la,it}`, `_bigrev_*_{la,it}`)

| cell | decoder | SER stuck → wild → anneal | judge plain / margin / called | d5 reference (SER, margin, called) |
|---|---|---|---|---|
| positive/latin | d5b20 | 0.783 → 0.194 → **0.092** | 2.43 / 1.67 / YES | 0.066–0.073, 1.70–1.72, YES (3 seeds) |
| nodouble/latin | d5b20 | 0.797 → 0.152 → **0.056** | 2.22 / 1.87 / YES | 0.040, 1.88, YES |
| bigram/latin | d5b20 (matched) | 0.724 → 0.722 → **0.720** | 3.30 / 0.50 / no (truth 2.40 / 1.72 / YES) | — |
| bigram/latin | d5 (reverse) | 0.713 → 0.718 → **0.712** | 3.41 / 0.35 / no (projected truth SER 0.210: 3.57 / 0.60 / no) | — |
| shuffled/latin | d5b20 | 0.796 → 0.799 → 0.804 | 3.35 / **0.41** / no | 0.26 |
| voynichesque/latin | d5b20 | — | 3.30 / **0.48** / no | 0.35 |
| positive/italian | d5b20 | 0.779 → 0.720 → **0.314** | 3.01 / 0.99 / no | 0.12, not called |
| nodouble/italian | d5b20 | 0.799 → 0.188 → **0.120** | 2.69 / 1.44 / no | 0.094, 1.43, no |
| bigram/italian | d5b20 (matched) | 0.719 → 0.717 → **0.719** | 3.25 / 0.61 / no (truth 2.56 / 1.61 / YES) | — |
| bigram/italian | d5 (reverse) | 0.712 → 0.713 → **0.714** | 3.40 / 0.42 / no (projected truth SER 0.214: 3.62 / 0.52 / no) | — |
| shuffled/italian | d5b20 | 0.823 → 0.825 → 0.839 | 3.31 / **0.49** / no | 0.27 |
| voynichesque/italian | d5b20 | — | 3.30 / **0.51** / no | 0.35 |

Stuck-start margins under d5b20 vs d5 on the *same* key (`bigram` cell):
Latin 0.48 vs 0.36, Italian 0.55 vs 0.43 (German 0.59 vs 0.40).

## Conclusions (all three languages, one seed each)

1. **R1, letters-only cipher (the question asked) — PASS 3/3.** A decoder
   carrying 5 doubled-letter units and 20 bigram units that the cipher never
   used ends within +0.008 / +0.016 / +0.026 SER (de / la / it) of the
   30-unit decoder, with the judge's margin and verdict unchanged (2.40 / 1.87
   / 1.44 vs 2.40 / 1.88 / 1.43). Residual leakage into the unused bigram
   units is 0.7 % of occurrences (German) — above the doubles' 0.04–0.2 %,
   an order of magnitude below the 5 % trap line, still draining at the round
   cap. Over-specifying the unit set is cheap when the cipher is letters-only.
2. **R1, d5 cipher — PASS 2/3, FAIL Italian.** German and Latin end at
   +0.014 / +0.02 SER with unchanged calls; Italian ends at SER 0.314 against
   0.12 (margin 0.99, neither decoder calls Italian A-like). The Italian
   wild stage stalls at 0.72 after 22 rounds — its top-20 bigrams (`er on an
   al en co ra el to es …`) cover the most text of the three inventories and
   compete hardest with the letter units. One seed; not separable from
   variance without a second.
3. **R2 — FAIL 3/3.** A cipher that genuinely uses all 48 units is never
   recovered (SER 0.70–0.72 from start to finish; wild stalls in 11–13
   rounds; anneal climbs 6–8 k nats without approaching the truth) although
   its true key is callable in every language (margins 2.30 / 1.72 / 1.61).
   At ~4 tokens per type the pipeline finds 30-unit keys and not 48-unit
   ones. This caps the manuscript reading below: a genuine 48-unit cipher on
   the manuscript would produce exactly these cells — margin 0.5–0.6, top
   language noise, not called — which is what the manuscript produces under
   this and every other hypothesis tried.
4. **R3.** The reverse mismatch is doubly hopeless: the search does not
   move (0.71), and the closest representable key is itself not language-like
   (projected truth SER 0.21–0.24, margins 0.92 / 0.60 / 0.52) because a
   quarter of the letters are gone. Contrast `revdouble/german` (only the five
   doubles missing): recovered to SER 0.098 and called (margin 1.88). Missing
   doubles are survivable; missing bigrams are not.
5. **R4 and the offset.** `shuffled` 0.41–0.49 (PASS 3/3); `voynichesque`
   0.48 / 0.51 / 0.52 — two of three a hair over the pre-stated 0.5 line, all
   far below the 0.75 alarm. The 48-unit decoder adds a near-constant
   +0.12…+0.20 bits of margin to any wrong or structureless key (same stuck
   key like-for-like: +0.19 / +0.12 / +0.12; negatives +0.13…+0.19;
   manuscript +0.2). It is an offset from the extra free parameters, not
   invented structure — but `d5b20` margins are only comparable to their own
   negative band (0.41–0.52).
6. **Manuscript — 8/8 NOISE.** German A 0.68–0.72 / B 0.54–0.58, Latin A
   0.63 / B 0.48–0.50 on the judge-guided arm, never more than +0.05 above
   the controls; against the `d5b20` negative band the separation
   (+0.15…+0.25 for Currier A, ~0 to +0.1 for B) is the same as under `d5`.
   The abstention stands; the doubles+bigrams decoder neither helps the
   manuscript nor, per R2, could it have.

Caveats: one seed everywhere (the R1 Italian miss and the two 0.51–0.52
negatives are the readings a second seed would firm up); the `bigram` cipher
is at 3.9–4.1 tokens/type against 4.2–4.4 for the other A-like cells; no
B-like cells were run; the doubling-rate argument against bigram units as a
*hypothesis* (`docs/wordhom_study.md`) is untouched by any of this.

