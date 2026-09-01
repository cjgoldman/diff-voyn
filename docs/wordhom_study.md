# Boxer's hypothesis without the arithmetic — word-level homophonic head

> **Record status (banner added 2026-09-01):** post-Phase-6 word-homophonic study, COMPLETE 2026-08-25; **plain-SA baseline** (`WordHomophonicHead.solve`).
> Still current: §2.1 (units = top-5 doubled letters), §2.2 (every type must be mapped), §4.1 (the pipeline calls German at 8.3 tokens/type), the §5 manuscript numbers *for plain SA* and the MDL-above-baseline point. Superseded: **every findability figure here is plain SA** — the "≥ 8 tokens per type" wall, "not findable at the manuscript's ratio", "more text per type is the only lever"; the current solver (hapax-wildcard → anneal, `docs/alt_loop_plan.md` §8.4–8.6, battery §10) recovers keys at ≈ 4 tokens per type — Currier B (4.6) is inside that regime, Currier A (3.0) below it (`docs/project_status.md` §3, §5.1). Sections were reordered into numeric order 2026-09-01 (text unchanged). **Current project position: `docs/project_status.md`.**

Post-Phase-6 study, started 2026-08-24. Code: `diff_voyn/heads/wordhom.py`
(head, synthetic cipher), `diff_voyn/vms/presentations.py::wordtypes_presentation`,
`diff_voyn/vms/controls.py::build_wordhom_controls`, the `wordhom` branch of
`diff_voyn/vms/apply.py`; script `scripts/wordhom_study.py`
(prepare / solve / score / report); tests `tests/test_wordhom.py`; artifacts
`DATA_ROOT/analysis/wordhom/`. Evaluator, calibration, budget, outer tier,
MDL scale and abstention rule are the Phase-6 ones (`docs/phase6_status.md`).

## 1. What is being tested

Phase 6 ran Boxer's `pseudo_vms` generator (rung 4) as the "arithmetic"
cipher hypothesis. After discussion with Boxer (via the user, 2026-08-24)
that generator is better read as a *key-compression device* — the integer
sum-to-target rule is one way to name thousands of homophones with a
16-value key — and the hypothesis he actually advances is the cipher
behind it:

1. the unit of ciphertext is the **word token**;
2. the plaintext alphabet is the language's letters plus its **4–5 most
   frequent doubled letters as single characters** (`ss`, `ll`, `tt` … —
   the ß analogy);
3. many tokens stand for one plaintext unit (homophonic, many-to-one), with
   no key structure assumed;
4. the **repeat rule**: a unit written twice in a row is enciphered with the
   same token twice.

With the arithmetic dropped this is a homophonic substitution over word
types with an (A + 5)-symbol target alphabet — rung 2's problem, with two
additions: targets that emit one *or two* letters, and the converse of the
repeat rule as a constraint the decoder can use (two *different* adjacent
tokens may never decode to the same unit; `repeat_weight` = 4 nats per
violation in the discrete objective, the true key has none).

## 2. Two definitional findings before any manuscript number

### 2.1 The extra characters must be doubled letters, not frequent bigrams

The first implementation took the five most frequent bigrams of the
language overall (Italian er/on/an/al/en, German en/er/ch/nd/de). The
user's independent tests had found that the hypothesis' doubled-unit rate
matches the manuscript; under that definition it does not. Doubled units
per 1000 units in held-out text (200k letters per language, greedy
segmentation) by unit set:

| extra units | Latin | Italian | German | VMS adjacent identical tokens |
|---|---|---|---|---|
| none (letters only) | 35.5 | 39.0 | 43.8 | 7.1–8.9 per 1000 tokens |
| top-5 bigrams overall | 35.2 | 38.0 | 41.3 | |
| **top-5 doubled letters** | **10.7** | **15.6** | **9.7** | |
| top-8 doubled letters | 5.6 | 9.4 | 3.5 | |

With general bigrams as units the text's `ll`/`ss`/`tt` stay doubled units
and the rate is 4–5× the manuscript's; with the doubled letters as units
German and Latin land in the manuscript's band and Italian is close. The
study uses the **top-5 doubled letters per language** (frozen in
`language_targets`: Latin ss ll ii ee mm; Italian ll ss tt ee aa; German ss
nn ff ll tt; the stored control is `analysis/wordhom/doubling_control.json`:
10.3 / 16.5 / 9.5 per 1000 on 20 × 20k-letter held-out samples, VMS 8.9 /
8.1 (IT2a A/B), 7.7 / 7.1 (RF1b)). This is a genuine consistency check the
hypothesis passes — a rule-following cipher over Latin or German text with
these five extra characters would show repeated tokens at about the rate
the manuscript does.

### 2.2 The rare-type tail cannot be dropped

The plan was to map only the top-K word types (K = 100–400 covers 43–68 %
of tokens) and charge the rest at the stream's own surface cost, as Phase 6
does for uncovered symbols. That fails on its own positive control: on a
synthetic Italian instance (8000 letters, 2500 homophones, Zipf usage, 1732
types realised) the **true** decode of the top-400 stream costs 5.7 bits/char
under the pentagram (clean text: 2.86) because the deleted 31 % of tokens
destroy the n-gram context, and greedy polish from the true key then
moves 56 % of the symbols away from it with 60 rule violations. Word-level
homophony has no "covered subsequence" that is still language.

So the head maps **every type** (`wordtypesall`: 3.6–3.7k types for
Currier A, 5.0–5.3k for B, 74 % / 69 % of them hapaxes). Rare types are
free parameters paid for in key bits (n_types · log2 30 ≈ 4.9 bits each —
~18 kbits for Currier A against 55.7k ciphertext characters, i.e. 0.32
bits per character before any plaintext is encoded) and re-fitted at will:
on the synthetic positive, polish from the true key moves 22 % of types
(0.6 % of the top-400, occurrence-weighted), letter SER 0.077, decode at
2.69 bits/char. The objective's optimum is the true key for every type
that occurs often enough to matter; that is what the search has to find.

## 3. Search: the objective is right, the key is not findable by plain SA at the manuscript's type/token ratio

*(heading qualified 2026-09-01; originally read "…the key is not findable at the manuscript's type/token ratio". The wildcard → anneal solver of `docs/alt_loop_plan.md` §8.4–8.6 later moved the wall to ≈ 4 tokens per type; `docs/project_status.md` §3.)*

All calibration below is on the synthetic Italian positive of §2.2 (8000
letters, 1732 types, 6100 tokens → 3.5 tokens per type; the manuscript has
3.0 (A) and 4.6 (B)). "found − truth" is the penalized pentagram objective
of the search result minus that of the true key (nats; positive would mean
the objective itself prefers a wrong key).

| initializer / search | steps | wall | found − truth | rule violations | occurrence-weighted map error | letter SER |
|---|---|---|---|---|---|---|
| frequency init + SA 15→0.5 + polish | 0.4M | 8 min | −3 543 | 100 | 0.815 | 0.746 |
| frequency init + SA 15→0.5 + polish | 3M | 19 min | −2 612 | 74 | 0.754 | 0.726 |
| frequency init + SA 3→0.3 + polish | 3M | 19 min | −6 098 | 166 | 0.895 | 0.764 |
| random init + SA 15→0.5 + polish | 1.5M | 10 min | −4 593 | 133 | 0.844 | 0.764 |
| EM (trigram segmental DP, annealed, 40 it) + polish + SA 2→0.2 | 40 + 0.2M | 24 min | −4 485 | 170 | 0.766 | 0.732 |
| EM, German instance (13 866-nat truth) | 40 + 0.2M | 24 min | −8 389 | 221 | 0.887 | 0.768 |
| *oracle*: polish from the true key | — | 1 min | +1 005 | 3 | 0.075 (top-400: 0.006) | 0.077 |

Every search ends thousands of nats *below* the true key with dozens of
rule violations: the true key is a strict local optimum the search never
reaches (rung 2 recovers Zodiac-408-class keys — 54 symbols, 7.5 chars per
symbol — with the same SA in ~1/8 of restarts; here there are 1732
symbols at 3.5 tokens each). The EM initializer that solves rung 3 does no
better than SA: the per-type posterior has almost nothing to condition on
when most types occur once or twice. §3.1 measures where this breaks.

### 3.1 Feasibility curve: tokens per type

Same generator and search (frequency init + one random restart, SA
15→0.5, 1.5M steps, polish), 8000 held-out letters, homophone budget
varied. **Plain multi-restart SA** (this study's solver) is essentially perfect down to ~8 tokens per type and
fails outright at 3.5 *(qualified 2026-09-01: the wildcard → anneal solver later recovers 4.1–4.4 tokens per type, `docs/alt_loop_plan.md` §8.6, §10.1; originally read "The search is essentially perfect…")*:

| language | types | tokens / type | hapax types | found − truth (nats) | violations | occurrence-weighted map error | letter SER |
|---|---|---|---|---|---|---|---|
| German | 30 | 257 | 2 | +34 | 1 | 0.010 | 0.010 |
| German | 120 | 64 | 2 | +34 | 0 | 0.004 | 0.004 |
| German | 250 | 31 | 2 | +34 | 1 | 0.012 | 0.012 |
| German | 498 | 15.5 | 9 | +66 | 0 | 0.010 | 0.010 |
| German | 943 | 8.2 | 132 | +123 | 2 | 0.028 | 0.028 |
| German | 1 7xx | ~3.5 | ~700 | −8 389 (EM) | 221 | 0.887 | 0.768 |
| Italian | 30 | 261 | 0 | +61 | 0 | 0.005 | 0.005 |
| Italian | 120 | 65 | 0 | +61 | 0 | 0.005 | 0.005 |
| Italian | 250 | 31 | 0 | +61 | 0 | 0.005 | 0.005 |
| Italian | 495 | 15.8 | 11 | +103 | 0 | 0.012 | 0.012 |
| Italian | 935 | 8.4 | 127 | +272 | 3 | 0.031 | 0.032 |
| Italian | 1 732 | 3.5 | 674 | −2 612 (best of §3) | 74 | 0.754 | 0.726 |

(A positive "found − truth" at small vocabularies is the polish re-fitting
the handful of rare types, as in the oracle row; the decode bits/char equal
clean text, 2.50 German / 2.86 Italian.) The manuscript has 3.0 tokens per
type in Currier A and 4.6 in Currier B (74 % / 69 % hapax types). Runs at
the threshold (5–6 tokens per type) and at the manuscript's own shape
(A-like 10.8k tokens / 3.6k types, B-like 23k / 5k) follow in §3.2.

### 3.2 Threshold and manuscript-shaped positives

Same search, 2 restarts (frequency init + random), 1.5M steps at 8000
letters and 2M steps for the manuscript-shaped instances:

| language | letters | tokens | types | tokens / type | hapax types | found − truth | violations | occurrence-weighted map error | letter SER | decode bits/char (pentagram) |
|---|---|---|---|---|---|---|---|---|---|---|
| German | 8 000 | 7 707 | 1 154 | 6.7 | 213 | −3 683 | 65 | 0.294 | 0.302 | 3.12 |
| German | 8 000 | 7 707 | 1 375 | 5.6 | 346 | −4 904 | 79 | 0.572 | 0.573 | 3.30 |
| German, **A-like** | 14 000 | 13 502 | 3 259 | 4.1 | 1 466 | −7 820 | 166 | 0.596 | 0.598 | 3.18 |
| German, **B-like** | 30 000 | 28 932 | 5 238 | 5.5 | 1 927 | −21 369 | 478 | 0.567 | 0.571 | 3.34 |
| Italian | 8 000 | 7 831 | 1 144 | 6.8 | 233 | −3 532 | 87 | 0.542 | 0.542 | 3.43 |
| Italian | 8 000 | 7 831 | 1 383 | 5.7 | 378 | −3 494 | 103 | 0.649 | 0.643 | 3.40 |
| Italian, **A-like** | 14 000 | 13 707 | 3 266 | 4.2 | 1 455 | −8 062 | 199 | 0.887 | 0.763 | 3.55 |
| Italian, **B-like** | 30 000 | 29 378 | 5 237 | 5.6 | 1 900 | −20 910 | 590 | 0.870 | 0.761 | 3.69 |

The break is sharp under this study's SA: essentially perfect recovery at ≥ 8 tokens per type,
a third to a half of the map wrong at 6.7–6.8, more than half wrong at
≤ 5.7 — and a true cipher with Currier A's or B's letter and type *counts*
(A-like 4.1 and B-like 5.5 tokens per type in the table above — above the
manuscript's own A: 10.8k tokens / 3.6k types = 3.0 and B: 23k / 5.0k = 4.6) comes back with
letter SER 0.57–0.76, hundreds of rule violations and a decode at 3.2–3.7
bits/char — the same numbers the manuscript produces (§5) *(corrected 2026-09-01; originally read "a true cipher with Currier A's or B's own type/token shape (A: 10.8k tokens / 3.6k types = 3.0; B: 23k / 5.0k = 4.6)" — the table gives 4.1 / 5.5; `docs/project_status.md` §5.9)*. The search
budget is not the lever: 3M steps versus 0.4M moved the 3.5-tokens-per-type
case from SER 0.746 to 0.726 (§3), and the EM initializer did no better.
More text per type was the only lever found for plain SA, and the manuscript's is fixed *(corrected 2026-09-01; originally read "More text per type is the only lever" — the objective turned out to be another: hapax types as wildcards, then annealed back in, `docs/alt_loop_plan.md` §8.4–8.6)*.

This is the identifiability statement behind the user's question. An
unstructured word-level homophonic cipher with the manuscript's vocabulary
statistics — three-quarters of the types seen once — cannot be brute-forced
through the tokens even when the plaintext language, its extra characters
and the repeat rule are all *given*: the true key is a strict optimum of
the objective (the oracle row of §3) but sits in a basin that no
frequency-driven search reaches. Boxer's arithmetic is not an incidental
detail of his demonstration; some compact key rule is what would make the
hypothesis testable at all, and its specific form is the thing a
decipherment would have to supply.

*[Superseded 2026-08-28/30 for ≥ 4 tokens per type: the wildcard → anneal loop does brute-force the A-like and B-like synthetics with language, extra characters and repeat rule given (`docs/alt_loop_plan.md` §8.6, §10.1). The statement stands at 3.0–3.5 tokens per type, i.e. for Currier A. `docs/project_status.md` §3.]*

## 4. Controls through the diffusion pipeline

Every control is a `wordtypesall` instance through the identical pipeline
(2 restarts × 2M steps, paired ELBO, MDL selection, 16 × 1024-char
windows × 4 seeds with shuffled copies). Positives are the synthetic
cipher of §2.2 at three shapes: **findable** (8000 letters, ~940 types,
8.3 tokens per type — inside the recoverable regime of §3.1), **A-like**
(14 000 letters, ~3 270 types, 4.1–4.2 tokens per type) and **B-like**
(30 000 letters, ~5 240 types, 5.5–5.6); a fourth set at 6.7 tokens per
type (`t0`, 1 300 homophones requested) landed in the degraded regime by
mis-sizing and is kept as the intermediate point. Negatives: letter-shuffled
text under the same cipher, `voynichesque` word tokens presented directly
as word types, and out-of-inventory contamination (Dutch, English, French,
Spanish under the cipher with the document's own top-5 doubled letters).
`analysis/wordhom/report.{json,md}`:

| instance | tokens / type | own-language plaintext bits/char | own structure margin | letter SER (own hypothesis) | abstain | MDL-top language |
|---|---|---|---|---|---|---|
| positive German findable | 8.3 | **1.941** | **2.27** | **0.008** | **no — called German** | German |
| positive Italian findable | 8.3 | 2.702 | 1.44 | 0.040 | yes (rule: margin < 1.5) | Italian |
| positive Latin findable | 8.3 | 3.303 | 0.38 | 0.606 | yes | German |
| positive German t0 | 6.6 | 3.175 | 0.61 | 0.425 | yes | German |
| positive Italian t0 | 6.7 | 3.366 | 0.51 | 0.421 | yes | German |
| positive Latin t0 | 6.8 | 3.397 | 0.35 | 0.674 | yes | German |
| positive German A-like | 4.1 | 3.311 | 0.49 | 0.642 | yes | German |
| positive Italian A-like | 4.2 | 3.418 | 0.38 | 0.768 | yes | German |
| positive Latin A-like | 4.2 | 3.429 | 0.34 | 0.769 | yes | German |
| positive German B-like | 5.5 | 3.653 | 0.27 | 0.690 | yes | Latin |
| positive Italian B-like | 5.6 | 3.723 | 0.23 | 0.775 | yes | Latin |
| positive Latin B-like | 5.6 | 3.674 | 0.23 | 0.780 | yes | Italian |
| shuffled (3) | 6.6–6.8 | 3.45–3.59 | 0.20–0.23 | 0.77–0.78 | yes | — |
| voynichesque (3) | 2.6–3.2 | 3.28–3.34 (best hyp) | 0.48–0.64 | — | yes | — |
| contamination (4) | 6–7 | — | 0.31–0.45 | — | yes | German ×3, Italian ×1 |

Reading:

* **The manuscript's cells are indistinguishable from true ciphers of its
  shape.** The A-like positives (4.1–4.2 tokens per type) give plaintext
  3.31–3.43 bits/char and margins 0.34–0.49; Currier A (3.0) gives
  3.31–3.37 and 0.47–0.50. The B-like positives (5.5–5.6) give 3.65–3.72
  and 0.23–0.27; Currier B (4.6) gives 3.66–3.71 and 0.19–0.24. Same
  numbers, same abstention, same failed map (letter SER 0.64–0.78 where the
  truth is known).
* **Below the threshold the structure margin carries no cipher signal.**
  Shuffled text under the cipher (0.20–0.23) and the B-like positives
  (0.23–0.27) overlap; `voynichesque` gibberish (0.48–0.64) is *above* every
  A-like positive. The margin that separated true from false decipherments
  by ≥ 1.26 bits in Phase 6 is a property of *found* keys; an unfound
  word-level key produces a decode whose residual structure is whatever
  the frequent types happen to carry.
* **The language call is meaningless at this ratio**: 0/9 positives are
  called (all abstain, as they should) and the MDL-top language is right
  for 2/9 — German is the MDL favourite for 6 of the 9 positives and 3 of
  the 4 contaminations, the "drift to German" of the n-gram judges
  (`docs/ngram_judge_robustness.md`) reappearing because the inner search
  is the n-gram objective.

### 4.1 The findable positives: the pipeline works when the key is findable

At 8.3 tokens per type the identical pipeline recovers the German key
(letter SER 0.008), decodes at 1.94 bits/char with a structure margin of
2.27 — inside the Phase-6 true-decipherment band (1.49–2.48) — and *calls*
German; it recovers the Italian key (SER 0.040, margin 1.44) and abstains
only because the frozen rule's margin threshold is 1.5 (Phase 6 had the
same borderline case at 1.49); it misses the Latin key with 2 restarts
(SER 0.61, margin 0.38) — Latin is the hard language for the n-gram inner
search at every rung (`docs/phase5_status.md`), and here as there the fix
is restarts, not the objective. So the instrument is sound: a word-level
homophonic cipher whose vocabulary is rich enough per type is found,
scored as language and named. What it cannot do is find one at the
manuscript's 3–5 tokens per type, and no positive at that ratio (6/6) got
above a margin of 0.49.

*[Superseded 2026-08-28/30: with the wildcard → anneal solver the same A-like (4.1–4.2) and B-like (5.5–5.6) positives are recovered (SER 0.05 / 0.07 / 0.12 and 0.026 / 0.036 / 0.070) and German and Latin are called — `docs/alt_loop_plan.md` §8.6, §10.1; `docs/project_status.md` §3.]*

## 5. The manuscript

All types mapped (100 % coverage), one solve window of ≤ 12 000 tokens
(the whole of Currier A; the first 12 000 tokens of B), 2 restarts × 2M
SA steps + polish per (instance × hypothesis), outer tier = paired ELBO
rescoring of the shortlist + MDL selection (no `elbo_polish`, §1), full
stream scored on 16 × 1024-char windows × 4 seeds with letter-shuffled
copies. `analysis/wordhom/report.{json,md}`:

| stream | tokens / type | hypothesis | MDL total / ciphertext char | no-cipher baseline | plaintext bits/char ± sem | key bits | choice bits/char | structure margin | rule violations of the map | flip-rate | language-like |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IT2a A | 3.0 | Italian | 2.485 | 2.373 | 3.307 ± 0.013 | 17 895 | 7.67 | 0.50 | 336 | 0 | no |
| | | Latin | 2.496 | | 3.346 ± 0.007 | | 7.75 | 0.47 | | 0 | no |
| | | German | 2.500 | | 3.360 ± 0.016 | | 7.73 | 0.49 | | 0 | no |
| IT2a B | 4.6 | Latin | 2.448 | 2.085 | 3.661 ± 0.085 | 24 755 | 7.73 | 0.19 | 1 252 | 0 | no |
| | | Italian | 2.451 | | 3.680 ± 0.085 | | 7.70 | 0.22 | | 0 | no |
| | | German | 2.458 | | 3.701 ± 0.088 | | 7.73 | 0.22 | | 0 | no |
| RF1b A | 2.9 | Latin | 2.524 | 2.433 | 3.322 ± 0.009 | 18 327 | 7.79 | 0.50 | 324 | 0 | no |
| | | Italian | 2.525 | | 3.335 ± 0.011 | | 7.72 | 0.49 | | 0 | no |
| | | German | 2.537 | | 3.365 ± 0.017 | | 7.77 | 0.49 | | 0 | no |
| RF1b B | 4.4 | Italian | 2.552 | 2.121 | 3.663 ± 0.085 | 26 218 | 7.76 | 0.24 | 1 373 | 0 | no |
| | | Latin | 2.554 | | 3.687 ± 0.083 | | 7.77 | 0.22 | | 0.25 | no |
| | | German | 2.564 | | 3.712 ± 0.083 | | 7.76 | 0.24 | | 0 | no |

(Rule violations and key bits are per stream — the same order of magnitude
across hypotheses.)

Reading, in the order the numbers allow:

1. **Every cell abstains** — plaintext 3.31–3.71 bits/char against the
   rule's 3.0, structure margin 0.19–0.50 against the 1.5 threshold and the
   Phase-6 true-decipherment band of 1.49–2.48. Currier A's margin (0.47–0.50)
   is inside the Phase-6 manuscript range (0.04–1.25); B's (0.19–0.24) is
   at its low end.
2. **The cipher hypothesis explains the manuscript worse than its own
   surface statistics**: the MDL total per ciphertext character (2.45–2.56)
   is above the stream's held-out n-gram baseline in all four streams
   (2.09–2.43), by 0.09–0.43 bits. The choice bits alone — 7.7 bits per
   plaintext character to name which of a unit's homophones was used (median 106–175 per
   unit under the found maps, up to 430) — are what make an unstructured word-level homophonic cipher an
   expensive description: the manuscript's ~3 900-fold vocabulary buys the
   hypothesis nothing it can pay for.
3. **The language ranking is noise**: within-stream margins 0.001–0.011
   bits/symbol (calibration uncertainty 0.067–0.193); the top language is
   Italian (IT2a A, RF1b B) or Latin (IT2a B, RF1b A), i.e. it flips between
   transcriptions of the same dialect.
4. **None of this distinguishes the manuscript from a true cipher of its
   shape**: the manuscript-shaped positives of §3.2 come back with the same
   decode bits (3.2–3.7), the same hundreds of rule violations and — see
   §4 — the same structure margins through the identical pipeline. The
   abstention is therefore *correct but uninformative*: it says the pipeline
   cannot find a word-level homophonic key at 3–5 tokens per type, not that
   there is none. What the study does establish about the hypothesis is
   §2.1 (the doubling-rate consistency check it passes) and point 2 (as an
   unstructured cipher it is not a compressive description of the text).

   *[Superseded 2026-08-29: the current solver does find word-level keys at 4.1–5.7 tokens per type on synthetics, and Currier B (4.6) still returned NOISE 24/24 under it (`docs/altloop_vms_plan.md` §13) — B's abstention is no longer uninformative in the way this sentence says; Currier A (3.0) remains below anything ever recovered. `docs/project_status.md` §1, §3.]*

## 6. Conclusion

Answer to the question that started this (can the arithmetic be bypassed
by brute-forcing the tokens, given letters + the top-5 doubled letters and
the repeat rule?): **not at the manuscript's vocabulary statistics.** The
hypothesis passes a real consistency check — with the doubled letters as
extra characters, Latin and German text would show repeated tokens at
about the manuscript's rate (§2.1) — and the machinery to test it exists
and works (§4.1). But an unstructured word-level homophonic key is
identifiable from the ciphertext alone only above ~8 tokens per type
**under plain SA (this study's solver)** — ≈ 4 under the later wildcard → anneal loop (`docs/alt_loop_plan.md` §8.6, §10.1), and still unfindable at 3.0–3.5 *(qualified 2026-09-01; originally read "…only above ~8 tokens per type;")*;
Currier A has 3.0 and B has 4.6, three-quarters of the types are hapaxes,
and a true cipher of that shape comes through the pipeline with exactly
the manuscript's numbers (plaintext 3.3–3.7 bits/char, margin 0.2–0.5,
hundreds of rule violations, letter SER 0.6–0.8 where truth is known).
The manuscript's 12 cells all abstain and its MDL total sits 0.09–0.43
bits/char above the stream's own n-gram description, but that abstention
is uninformative: it is what the pipeline says about any cipher of this
shape. Boxer's arithmetic — or some other compact key rule that makes the
rare types cost nothing — is not a detail of his demonstration; it is the
part that would make the hypothesis testable, and it is the part a
decipherment claim has to supply. Two ways forward if one wanted to push:
(a) a structured-key head (targets as a function of a token's glyph
composition, rung-3 style), which turns 3.6k free parameters into a few
hundred and moves the instance above the threshold; (b) a much larger
restart budget on the manuscript only makes sense once (a) exists.

*[Postscript 2026-09-01: neither was pursued. The wall was moved instead by posterior re-seeding (`docs/alt_loop_plan.md` §7, to 6.6 tokens per type) and by changing the objective — hapax types as wildcards, then annealed back in (§8.4–8.6, to ≈ 4). (a) is still not built and remains the only proposed route below ≈ 4 tokens per type (`docs/project_status.md` §6).]*
