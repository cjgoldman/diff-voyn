# Phase 6 — VMS application and reporting: status

> **Record status (banner added 2026-09-01):** Phase 6, 2026-08-23 → 2026-08-24 (the manuscript abstains, 0/87 cells; acceptance FAIL on one P0 sub-criterion), record maintained through 2026-08-25. Still current: the frozen abstention rule (`apply.ABSTAIN_RULE`), the 87-cell result, the controls, anchors and acceptance roll-up as the frozen verdict. Superseded or annotated in place below: the first "selection bias" account of the Borg polish in §6.6 (corrected 2026-08-25, same section); the safe threshold corridor (1.26–1.48 → ≈ 1.41–1.48 against a strict negative, 2026-08-31); the voynichesque control's status (a wrong-hypothesis control, not a strict negative — the P0 FAIL was a mis-specified test, verdict left as recorded); and the carry-over list. Later manuscript work — alternating loop 72/72 NOISE (2026-08-26, `docs/altloop_vms_plan.md` §12), word-homophonic wildcard → anneal 24/24 NOISE (2026-08-29, §13), `d5b20` 8/8 NOISE (2026-08-31, `docs/wordhom_bigram_variant.md`) — confirms the abstention. **Current project position: `docs/project_status.md`.**

Status record for Phase 6 of the [task breakdown](../reference_docs/Diffusion%20Model%20Training%20-%20Task%20Breakdown.md)
(design §8 R5, §9; the Phase-5 carry-overs in `docs/phase5_status.md`).
Started 2026-08-23 after Gate G5. Every number below is produced by the
Gate-G4 frozen evaluator (`runs/phase_c-85m-seed0/ckpt_final.pt`, step 2000,
sha256 `e2cfb3c6…`) under calibration `v3-phase_c-ro` (report-only), scoring
budget 64 stratified draws × 4 replicate seeds, paired masks within every
shortlist. Code: `diff_voyn/vms/{presentations,apply,controls}.py`,
`ArithmeticHead.solve_segmented`; scripts `vms_apply.py`, `vms_controls.py`,
`anchors.py`, `phase6_analysis.py`, `fairness_audit.py --phase-tag phase6`,
`phase6_samples.py`, `phase6_check.py`; artifacts `DATA_ROOT/analysis/phase6/`;
write-up `docs/phase6_writeup.md`.

## 6.1 — Per-dialect scoring: how the manuscript is presented to the heads

Design §9 fixes only that Currier A and B are scored separately; it says
nothing about how a *transcription* becomes the ciphertext form each head
was validated on. Those mappings are Phase-6 decisions, frozen in
`diff_voyn/vms/presentations.py` with their coverage:

| presentation | source | what the head sees | heads | coverage |
|---|---|---|---|---|
| `eva` | Takahashi IT2a (primary) / Reference RF1b (replicate) IVTFF, per `$L` dialect | the EVA character stream, whitespace stripped (the Phase-0 model-facing stream; RF1b's `{ } '` residue removed here) | rung 1 (20 ≤ 25 symbols → injective map, key bits log2 25!/5!), rung 2 (`n_symbols` = 20 / 23) | 100% |
| `words` | same | the word tokens; only tokens the published Naibbe tables can produce | rung 3 | A: 73% of words / 65% of characters (IT2a), B: 82% / 78% — the unparseable residue Greshko reports |
| `boxer` | Boxer's independent glyph transcription (voynich-attack @ e324bee), folios mapped to dialect via the IVTFF headers (`102ra` ↔ `f102r1`) | glyph-unit stream; top-16 types (98.2% of glyph occurrences) for rung 4, tokens with any other glyph (incl. `?`) dropped; top-20 types as a second symbol presentation for rungs 1–2 | rung 4 (16 symbols, tokens of length 2–6); rungs 1–2 (`boxer20`) | 90–92% of glyphs (boxer16), 97–98% (boxer20) |

Stream sizes: A 55,705 EVA characters (10,846 words; 7,888 Naibbe-parseable),
B 119,481 (22,984 words; 18,907 parseable); Boxer A 46.7k glyphs / 11.1k
tokens, B 97.0k / 21.7k.

**Rung 4 on the manuscript — a structural finding before any solve.** The
Phase-5 arithmetic head infers the cipher's canonical within-token glyph
order from the unsegmented stream and prunes the segment lattice with it.
On the Boxer glyph stream *no candidate order is admissible*: every exact
LOP optimum (and every Gumbel-perturbed one) forces adjacent descents,
i.e. length-1 tokens, which the pinned generator's 2–6 length range
forbids. The sum-to-target cipher's strongest structural signature
(globally sorted tokens) is absent from the manuscript. The manuscript
does, however, have *visible* word boundaries, so the hypothesis is tested
in its honest remaining form: `ArithmeticHead.solve_segmented` builds the
lattice from the observed token boundaries (tokens of length 2–6, 90% of
kept tokens), seeds the 16 glyph values from within-token positions, and
runs the same Sinkhorn-assignment + integer-polish machinery.

**Solve-window protocol.** A key is global per dialect, so each head's
inner search runs on windows of the validated Phase-5 scale (rung 1: 4,000
symbols; rung 2: 2,000; rung 3: 4,000 tokens; rung 4: 500 tokens) — two
evenly spaced windows per dialect on IT2a/Boxer, one on RF1b — and the
chosen key decodes the whole dialect stream. The outer tier per (window ×
hypothesis): paired ELBO of every shortlist decode under every condition
(budget 64), MDL selection (calibrated plaintext bits + choice bits per
plaintext char), `elbo_polish` for the homophonic head (single moves,
budget 8, confirmed at 64), then the full-stream decode scored on ≤ 16
evenly spaced 1024-char windows × 4 replicate seeds under all three
conditions, each window paired with a letter-shuffled copy of itself (the
Phase-3 per-instance structure control). Inner-search budgets: rung 1 — 4
Sinkhorn restarts + ILS, 8-deep shortlist; rung 2 — 64 SA restarts × 100k
steps, pair polish of the top 4, 12-deep shortlist; rung 3 — 3 block-Sinkhorn
restarts × 250 steps + fixed-parse swap polish; rung 4 — 3 restarts × 400
steps + integer polish. CPU: 87 solve jobs on 12 workers (rung-3 jobs ≈ 20
min each, rung 4 ≈ 58 min, rung 2 ≈ 20 min, rung 1 ≈ 4 min; 2.5 h wall
clock). GPU: 87 score jobs sharded over two 3090s, 3.6 h.

## 6.2 — The ranking rule, its uncertainty columns, and the abstention rule

Per cell (head × presentation × window × hypothesis): calibrated plaintext
bits/char of the full-stream decode (mean over windows × seeds, s.e.m. over
windows), key bits, choice bits, **total per covered ciphertext symbol**
(the §5.6 cross-head comparator: plaintext bits × n_plain + key + choice,
divided by the symbols the head actually consumed) and per *all* symbols
(uncovered symbols charged at the stream's own best held-out n-gram
cross-entropy — a head that explains part of the manuscript pays the
surface price for the rest, or partial coverage would read as compression;
**the cross-head ranking is on this all-symbols total**), coverage, the
**structure margin** (shuffled − decode bits under the own condition), the
language rank of the decode under the three conditions with its margin and
the calibration margin uncertainty (`margin_uncertainty_bits`, 0.01–0.19),
the **replicate flip-rate** (fraction of the 4 seeds whose full-stream
top language differs from the pooled one) and the **window vote**
(fraction of scored windows agreeing with the pooled top language).
Per head the language order is by the best cell's total; *head agreement*
is the fraction of (presentation × window) votes agreeing with that
order's top, and the fraction of heads agreeing with the table's top
language; *transcription agreement* compares IT2a and RF1b per head and
dialect; *dialect agreement* compares A and B.

**No-cipher baseline.** Next to every table: the held-out cross-entropy of
order-1…5 n-gram models of the *ciphertext itself* (fit on the first half
of the stream, evaluated on the second), in bits per ciphertext symbol. A
(cipher × language) cell whose total is above this number describes the
manuscript worse than its own surface statistics do. EVA A: 2.37
bits/symbol at order 2 (order-0 entropy 3.83); Boxer-20 A: 2.65; Boxer-16 A: 2.58.

**Abstention rule (fixed before any manuscript number was read;
`apply.ABSTAIN_RULE`).** A cell is *language-like* only if (i) its
full-stream calibrated plaintext bits are at the clean-text level, ≤ 3.0
bits/char (Phase-5 solved cells sit at 1.9–2.7; the wrong-key plateau at
3.4–4.4), and (ii) its structure margin is in the true-decipherment band,
≥ 1.5 bits/char (Phase 3: 1.6–2.9 for true decipherments, 0.5–1.9 for
wrong-hypothesis ones). A dialect none of whose cells is language-like
**abstains**; its language ranking is still printed, flagged as a ranking
among non-decipherments. The rule's false-abstention rate on true
decipherments and its abstention rate on the negative controls are
measured by the 6.3 battery on the same pipeline.

## 6.2 — Results (87 cells; `analysis/phase6/vms_report.{json,md}`)

**Every dialect of every transcription abstains: 0 of 87 cells is
language-like.** The best plaintext bits any cell reaches are 2.468
bits/char (homophonic head, German hypothesis, Currier B) — inside the
clean-text band — but the largest structure margin over the whole table is
**1.249 bits/char**, below the 1.5 the rule requires and inside the
wrong-hypothesis band Phase 3 measured (0.5–1.9). No decode is more
structured, relative to its own shuffle, than a wrong-key decipherment of a
synthetic cipher is.

| table | MDL top cell (total bits / ciphertext symbol) | beats the stream's own n-gram baseline? | per-head language order (margin, bits/symbol) | head agreement | verdict |
|---|---|---|---|---|---|
| IT2a + Boxer / **A** | naibbe / latin, 2.091 | **yes** (baseline 2.179) | naibbe la>de>it (0.018); arithmetic it>la>de (0.023); homophonic la>de>it (0.001); sub1to1 de>la>it (0.037) | 0.50 | ABSTAIN |
| IT2a + Boxer / **B** | naibbe / latin, 1.921 | no (baseline 1.909) | naibbe la>de>it (0.005); arithmetic it>la>de (0.034); homophonic de>la>it (0.029); sub1to1 de>la>it (0.071) | 0.25 | ABSTAIN |
| RF1b / **A** | naibbe / latin, 2.123 | **yes** (baseline 2.231) | naibbe la>de>it (0.029); homophonic la>de>it (0.012); sub1to1 de>la>it (0.033) | 0.67 | ABSTAIN |
| RF1b / **B** | naibbe / german, 1.974 | no (baseline 1.943) | naibbe de>la>it (0.002); homophonic la>it>de (0.077); sub1to1 de>la>it (0.083) | 0.67 | ABSTAIN |

Reading of the table, in order of what the numbers support:

1. **No decipherment.** Nothing in the (cipher × language) grid produces a
   letter stream that behaves like language under the frozen evaluator. The
   two rungs that reach clean-text-level bits (sub1to1 2.81–3.05, homophonic
   2.47–2.76 bits/char) do so the way Phase 5 showed a verbose head *can* do
   on any stream — by finding a low-entropy mapping — and their structure
   margins (0.87–1.25) say so: shuffling the decode costs about as much as
   it costs to shuffle a wrong-key decipherment. The two rungs with real
   cipher grammars produce decodes far above clean text (naibbe 3.54–3.99,
   arithmetic 4.01–4.26 bits/char), i.e. no key was found.
2. **The Naibbe hypothesis wins the description-length comparison, and on
   Currier A only it beats the no-cipher baseline.** Its total is lowest in
   all four tables (1.92–2.16 bits per ciphertext symbol) because a Naibbe
   parse compresses the glyph stream roughly 2:1; on Currier A that total
   (2.09 / 2.12) is below the manuscript's own held-out bigram cross-entropy
   (2.18 / 2.23), on Currier B it is not (1.92 / 1.97 vs 1.91 / 1.94). The
   honest statement is about *description length of the word structure*, not
   decipherment: a Naibbe-shaped unigram/bigram token grammar describes
   Currier A slightly better than a character n-gram of Currier A does,
   while its decode is not language. Currier A and B differ on this, which
   is consistent with the dialects differing in token structure (the
   Phase-0/CH-track doubling and token-length findings) — and it is the one
   place in the table where a cipher hypothesis buys anything at all.
3. **The language ranking is noise.** Within-head margins are 0.001–0.083
   bits per ciphertext symbol against a calibration margin uncertainty of
   0.067–0.193 (`margin_uncertainty_bits` for the relevant language pair) —
   every margin in the table is at or below the systematic uncertainty.
   Heads disagree (head agreement 0.25–0.67), the two transcriptions
   disagree on Currier B for two of three heads, and A and B disagree under
   the homophonic head. Replicate flip-rates are 0 for 80 of 87 cells,
   i.e. the *Monte-Carlo* noise is small — the disagreement is systematic,
   not sampling: different heads and transcriptions genuinely prefer
   different languages, which is what a ranking among non-decipherments
   looks like.
**Representative decodes** behind the abstention — per head the cell with
the lowest full-stream plaintext bits, decoded from the stored final key
at fixed excerpt positions, with a positive-control (true-decipherment)
and letter-shuffled contrast — are generated by `scripts/phase6_samples.py`
→ `analysis/phase6/samples.{json,md}` and embedded in the write-up §2.1.
The best homophonic cell reaches its 2.468 bits/char by re-expressing the
manuscript's word-level repetitiveness as a small set of recurring
pseudo-words (`ersti`, `wunam`, `accp`, …), not by producing German.

4. **The outer tier behaved as in Phase 5.** `elbo_polish` was accepted on
   30 of 30 homophonic cells (it always improves the MDL of a
   non-decipherment, which is why acceptance is not evidence), and the pure
   ELBO pick differed from the MDL pick in 26 of 87 cells — the
   degeneracy the choice-bits term exists to charge for.
   *[Note added 2026-09-01: see the 2026-08-25 correction in §6.6 — these
   polished keys ran with the choice-bits term in the polish objective,
   which is now disallowed by default; their verdict is unaffected
   (pre-polish margins 0.83–0.84, post-polish 1.05–1.11, both far below
   1.5).]*

## 6.3 — Negative-control battery (39 instances, 309 solves, 309 scored cells)

`scripts/vms_controls.py`; every instance through the identical pipeline
(same heads, windows, outer tier, MDL scale, abstention rule).
`analysis/phase6/controls/{manifest,solves,scores,report}.*`.

| control | n | abstain rate (95% CI) | structure-margin band | reading |
|---|---|---|---|---|
| **positive** (in-inventory 1:1 + Naibbe, 2000 / 1000 chars) | 9 | 0.11 (0.02–0.44) | **1.49 – 2.48** | 8/9 not abstained and language correct (8/9, family 8/9); all nine 1:1 solves have **SER 0.000** under the true hypothesis |
| **voynichesque** | 9 | 0.89 (0.56–0.98) | 0.92 – 1.51 | one miss (italian t1, margin 1.51, called latin) |
| **shuffled** | 9 | **1.00** (0.70–1.00) | 0.03 – 0.06 | no head finds structure in a permuted stream |
| **contamination** (Dutch, English, French, Spanish under in-inventory-fit 1:1 / Naibbe) | 12 | **1.00** (0.76–1.00) | 0.60 – 1.43 | no out-of-inventory language is *called*; the MDL top-1 (ignoring abstention) is family-correct 7/12 |

**Acceptance: shuffled and contamination PASS; voynichesque 0.89 misses the
> 0.95 bar — FAIL as written, and reported as such.** The single miss and
the single false abstention are the same fact: the frozen threshold of 1.5
bits/char sits inside the overlap Phase 3 measured. The margin is very
nearly a separating statistic — the worst true decipherment scores 1.49 and
the best structured-gibberish instance 1.51 — and the whole battery has one
error on each side out of 39. Post-hoc thresholds (1.3–1.4 gives 0
false-abstentions and 2 false-accepts; 1.6 gives 1 and 0) are recorded here
only to show the sensitivity: **the rule is not re-tuned after seeing the
data**, which is the entire point of freezing it, and the VMS verdict does
not depend on the choice — see below.

*[Re-classification 2026-08-31 (`docs/voynichesque_nocontent_restart.md`,
`analysis/phase6/controls_nocontent/report.json`; `docs/project_status.md`
§5 item 6): `voynichesque` is generated from real held-out text. Its
strict-negative twins — same generator, seeds and alphabets on a
letter-shuffled source — through this identical pipeline abstain 9/9
(0/66 cells language-like). The homophonic twin scores lower than its
real-text partner on all 27 pairs (Δ −0.27 mean, range −0.39…−0.16; real
0.85–1.51 → twin 0.55–1.24); sub1to1 Δ −0.02 and naibbe Δ +0.02 are pure
glyph grammar. The single miss above (italian t1: 1.43 / 1.51 / 1.50 across
its three cells) deflates to 1.15 / 1.24 / 1.14. Maximum twin margin 1.40.
Reading: real-text voynichesque is a wrong-hypothesis control, not a strict
negative, and the > 0.95 bar was tested on a non-negative. The FAIL above
stands as the frozen-rule outcome; this is a re-classification of the
control, not a pass.]*

**The number that matters for the manuscript.** The whole VMS table's
structure margins span **0.04 – 1.25**, i.e. *every one of the 87 VMS cells
is below every one of the nine true decipherments* (1.49–2.48) and inside
the band of the structured-gibberish and out-of-inventory controls. Any
threshold between 1.26 and 1.48 abstains on the whole manuscript and on no
positive control. The manuscript's best decode behaves like `voynichesque.py`
output, not like a decipherment.
*[Note added 2026-09-01: the 1.26–1.48 corridor is stated against this
battery, whose gibberish row is content-inflated (previous note). Strict
gibberish — the letter-shuffled-source twins — reaches 1.40, so the corridor
that is also safe against a strict negative is ≈ 1.41–1.48. The manuscript
(≤ 1.25) is below both. `docs/project_status.md` §5 item 7.]*

**Contamination detail** (MDL top-1 language, abstention ignored): Dutch →
italian 2 / german 1; English → latin 2 / italian 1; French → italian 3;
Spanish → italian 3. Untrained Romance languages are pulled to Italian
(family-correct), untrained Germanic ones scatter (Dutch → Italian twice,
English → Latin twice) — family-correct 7/12, barely above the 5/12
expected by chance. The operative result is that all 12 are *abstained*, so
an out-of-inventory language never reaches a language call; but had the
abstention been skipped, an untrained language would have been assigned an
in-inventory label with family accuracy no better than chance. This is the
cross-contamination risk the design named, quantified.

## 6.4 — Bound-fairness audit re-run

`scripts/fairness_audit.py --phase-tag phase6` → `docs/phase6_fairness_audit.md`,
`analysis/phase6/fairness_audit.json`. The evaluator is the frozen Gate-G4
checkpoint, so the numbers are those of Phase 4, re-attached here: adopted
table `v3-phase_c-ro`, measured offsets latin +0.138 ± 0.003, italian
+0.013 ± 0.003, german +0.205 ± 0.002 bits/char (spread 0.193), **not
subtracted** (report-only: subtracting them breaks same-text comparisons,
the Phase-3 finding) but carried as the systematic uncertainty of every
cross-language margin; four ESCALATED findings (three reference-dependence,
one language-dependence: per-document offsets differ by language, range
0.203 vs within-language sd 0.040, ANOVA p = 4.5e-5) and one within-noise
correlate note, all carried into the write-up as stated residual risk.

## 6.5 — Length sensitivity and family confusion

`scripts/phase6_analysis.py` → `analysis/phase6/length_family.{json,md,png}`.
Assembled from the Phase-4 recovery suite (1:1, n = 50 per language × length),
the Phase-5 rung-1 two-tier suite, the LID head's clean-decipherment curve,
the rung-2/3/4 reports and the 6.3 contamination set.

| L | language acc (95% CI) | family | per-language la / it / de | flip-rate | margin unresolved | structure margin true / wrong | two-tier language / solved |
|---|---|---|---|---|---|---|---|
| 50 | 0.740 (0.66–0.80) | 0.880 | 0.56 / 0.88 / 0.78 | 0.129 | 0.16 | 2.26 / 1.59 | 0.70 / 0.30 |
| 100 | 0.940 (0.89–0.97) | 0.967 | 0.88 / 1.00 / 0.94 | 0.023 | 0.03 | 2.40 / 1.02 | 0.90 / 0.83 |
| 200 | 0.980 (0.94–0.99) | 0.987 | 0.94 / 1.00 / 1.00 | 0.012 | 0.16 | 2.16 / 1.01 | 1.00 / 0.98 |
| 400 | 0.993 (0.96–1.00) | 0.993 | 0.98 / 1.00 / 1.00 | 0.016 | 0.27 | 2.03 / 1.16 | 1.00 / 0.98 |
| 700 | 0.993 (0.96–1.00) | 0.993 | 0.98 / 1.00 / 1.00 | 0.011 | 0.41 | 1.95 / 1.24 | 0.98 / 1.00 |

Confusions: at L = 50 the within-Romance pair dominates (latin → italian
0.34, italian → latin 0.08); from L = 100 on the residual errors are
Latin → German (0.04 at L100/L200, 0.02 at L700) — cross-family, the
document-heterogeneity mode of Phase 3 — and rungs 2–4 (n = 18 / 12 / 9)
show 0 within-Romance and 2 cross-family confusions (both rung 4). The
"margin unresolved" column grows with length (0.16 → 0.41): longer texts
make the margin *smaller than the calibration uncertainty* more often,
because the same-text margin shrinks toward the calibration spread while
the ranking stays correct — the reason the offsets are not subtracted.
Claims at the granularity the data support: language-level ranking at
≥ 200 plaintext characters (0.989 / 0.991 family); 50 characters is the
LID-head anchor on clean text (0.88 family at L50 for the ELBO ranking),
not an ELBO-ranking regime; within-Germanic resolution is not realizable
inside the inventory and is measured only through the Dutch / English
contamination set — where, with the abstention rule off, Dutch is called
Italian twice and German once and English Latin twice, i.e. family accuracy
7/12 against 5/12 by chance. **Within-Germanic resolution is therefore not
supported at all**, and the design's "Germanic candidates" headline is the
finest granularity the instrument could in principle deliver, not one it
delivers here.

## 6.6 — Known-benchmark anchors (`analysis/phase6/anchors/`)

| anchor | language | instrument | result | target | verdict |
|---|---|---|---|---|---|
| **Zodiac-408** (54 symbols, 408 chars) | English — **outside the inventory** | n-gram tier only (240 SA restarts, Gutenberg English pentagram at 2.55 bits/char held-out) | **SER 0.0098** | ≤ 0.019 | **PASS**, as a pre-diffusion baseline only |
| **Borg** (Borg.lat.898; 55 symbol types with ≥ 20 occurrences = 99.9% of 120,191 symbols) | Latin | full pipeline (rung-2 + MDL selection + `elbo_polish`, frozen evaluator) | n-gram winner **0.129**, final **0.226** (weighted over 274 alignable pages / 86,375 letters; median page 0.110 → 0.217, best page 0.035) | ≤ 0.041 | **FAIL** |
| **BnF fr2988** | French — outside the inventory | — | — | ≤ 0.0113 | **NOT RUN** (no transcription available: DECODE requires a login; the benchmark repo lists its BnF/Gallica material as transcription-pending) |

**Zodiac-408** validates the inner search at the literature's own difficulty
(0.98% vs the 1.9% target) but says nothing about the diffusion evaluator:
English is not in the frozen inventory, so only the n-gram tier can run.

**Borg — the head largely solves it; our protocol and our outer tier do
not reach the literature number.** The n-gram winner's decode is readable
Latin — e.g. `sarumestremediumsiniigulareetuitellumouicridicumlanaimposiiitum
estutilissimum` against the published `sarumestremediumsingulareetuitellum
ouicrudicumlanaimpositumestutilissimum` — with the residual errors
concentrated in the `n/i/u/t` group and the rare symbols. 111 of 274 pages
decode at < 10% and the best at 3.5%. Two things separate this from the
4.10% literature figure, and both are ours, not the head's: (a) the
published plaintext is Örneholm's **corrected and expanded** Latin edition,
not a symbol-aligned transcription, so a Levenshtein SER against it charges
every editorial expansion as an error (a floor of several percent), and
(b) we reduce 77 glyph types to the 55 occurring ≥ 20 times, deleting the
rare types from the stream. The honest statement is that this is **not a
like-for-like comparison with the literature number and it fails the target
as measured**; a comparable run needs the DECODE aligned transcription.

**Language recovery on Borg is correct and is the strongest margin in
Phase 6**: latin > german > italian by **0.250 bits per ciphertext symbol**
(against a calibration margin uncertainty of 0.13 for latin–german), and
the Latin cells' structure margins (0.91, 0.97) are three times the wrong
hypotheses' (0.30–0.32). A real Latin homophonic cipher, run through
exactly the pipeline the manuscript was run through, is ranked Latin.

**Finding — `elbo_polish` does not transfer to a 55-symbol real cipher.**
It was accepted on all six Borg cells and made the decipherment
*worse* every time: median page SER 0.110 → 0.217, best page 0.035 →
0.124, 25 of 55 symbols reassigned (systematically `e` → `z`, 104
occurrences on one page alone). The mechanism is selection bias, and it is
a scale effect the Phase-5 rung-2 suite could not show: each sweep takes
the argmin over ~1,375 noisy paired-ELBO estimates at budget 8, so the
winner's-curse bias grows with the neighbourhood (rung 2: 54 symbols on
408 characters, few sweeps; here 55 symbols on 4,000 characters, six
sweeps), and the single confirmation at budget 64 compares only the final
map against the start map — it cannot undo a chain of six individually
biased moves. **Carry-over: the discrete outer tier needs a per-move
confirmation at full budget (or a bias-corrected selection rule) before it
is used on anything larger than the rung-2 synthetics.** *[Superseded
2026-08-25 — resolved in the next paragraph: the dominant cause is the
choice-bits term in the polish objective, not selection bias;
`docs/race_polish_plan.md` §7; `docs/project_status.md` §5 item 4.]*

**Post-Phase-6 correction (2026-08-25, `docs/race_polish_plan.md` §7).**
The winner's-curse account above is incomplete. Re-running the polish on
the same Borg window from the same MDL-pick key shows the *objective* is
the main cause: the first move any polish takes is symbol 0 (415/4000
occurrences, truly `e`) → `z`, which the frozen judge dislikes (+0.087
bits/char) but the homophonic **choice-bits term** rewards by −0.243
bits/char (it makes a frequent symbol the sole homophone of a spare
letter). With the term in the objective greedy reaches SER 0.310 in two
sweeps and a noise-robust racing polish still reaches 0.225; with the term
**off** both hold the key (0.1198 greedy / 0.1194 race). Selection noise
adds damage on the wrong objective but does not set the direction. Code
defaults changed accordingly: `elbo_polish` / `race_polish` refuse a
`choice_fn` unless `choice_term_in_polish=True`, and `vms/apply.py`,
`rung2_diffusion.py`, `control6b_pooled_search.py` polish on the ELBO alone
unless `--polish-choice-term` / `polish_choice_term=True` is passed to
reproduce the recorded runs. The VMS homophonic cells' *polished* keys are
therefore partly an artefact of this term; their verdict is not (next
paragraph).

**This does not affect the VMS verdict.** Re-scoring the two best VMS
homophonic cells with their *pre-polish* (MDL-pick) keys gives structure
margins of 0.828 and 0.840 against the post-polish 1.106 and 1.050 — the
polish moved the manuscript's margins *up*, and they still fall far short
of the 1.5 the rule requires. The abstention is not an artifact of the
outer tier.

Sources: Borg (MSS Borg.lat.898; transcription + Örneholm's corrected Latin
plaintext) and the Zodiac-408 record from `matthewdgreen/cipher_benchmark`
(sparse clone pinned at `729aad62`, `fetch_external.py`); modern-English
public-domain texts (Project Gutenberg ids 98, 1342, 2701, 1661, 84, 11,
345, 76; sha256 in `anchors/english_lm.json`) for the Zodiac pentagram.
BnF fr2988: no transcription is available to us (DECODE requires a login;
the benchmark repo lists the BnF/Gallica material as transcription pending)
— NOT RUN, a standing WARN.

## 6.7 — Write-up

`docs/phase6_writeup.md` — exploratory framing, the assumption list
(inventory, cipher inventory, presentations, bound comparability, search
fairness, length regime), the resolution statement (family level is the
ceiling; within-Germanic is not supported at all), the residual-risk list,
and §2.1's representative decodes (`scripts/phase6_samples.py` →
`analysis/phase6/samples.{json,md}`) so the abstention verdict can be read
against the actual letter streams rather than a threshold.

## Assessment — what Phase 6 established

1. **The manuscript abstains under every cipher hypothesis in the
   inventory, on both dialects and both transcriptions.** 0 of 87 cells is
   language-like; the whole table's structure margins (0.04–1.25) sit below
   every one of nine true decipherments run through the identical pipeline
   (1.49–2.48) and inside the band of structured gibberish (0.92–1.51) and
   out-of-inventory contamination (0.60–1.43). The result is robust to the
   frozen threshold: any cut between 1.26 and 1.48 abstains on the entire
   manuscript and on no positive control.
2. **The language ranking among those non-decipherments is noise, and the
   report says so.** Within-head margins of 0.001–0.083 bits per ciphertext
   symbol against a calibration uncertainty of 0.067–0.193; heads,
   transcriptions and dialects disagree with each other while the
   Monte-Carlo flip-rate is 0 for 80 of 87 cells — the disagreement is
   systematic, not sampling.
3. **The one positive structural result is a description-length statement,
   not a decipherment.** A Naibbe-shaped unigram/bigram token grammar
   describes **Currier A** (both transcriptions) below the manuscript's own
   held-out bigram cross-entropy (2.09 vs 2.18; 2.12 vs 2.23 bits/symbol),
   while its decode sits at 3.5 bits/char with a 0.07–0.09 structure margin.
   On **Currier B** no cipher hypothesis beats the surface baseline. The A/B
   difference is consistent with the dialects' known token-structure
   difference and is the only place a cipher hypothesis buys anything.
4. **The instrument's own limits were measured on real ciphers, not just
   synthetics.** Zodiac-408 confirms the inner search at literature
   difficulty (0.98% SER); Borg confirms the language ranking on a genuine
   Latin homophonic cipher (0.25 bits/symbol, the largest margin in the
   phase) while failing its SER target under our alignment protocol; and
   `elbo_polish`, the Phase-5 outer tier, actively degrades a 55-symbol
   real cipher through selection bias. *[Superseded 2026-08-25: mostly
   through the choice-bits term in the polish objective — §6.6 correction
   paragraph; an ELBO-only polish holds Borg.]*
5. **The abstention rule is very nearly a separating statistic but not
   quite**: one voynichesque instance at 1.51 and one true decipherment at
   1.49 straddle the frozen 1.5, so the battery has one error on each side
   out of 39 and voynichesque abstention (0.89) misses its > 0.95 bar.

### Carry-overs

- **Outer tier**: per-move full-budget confirmation or a bias-corrected
  selection rule for `elbo_polish` before any further use (6.6 finding) —
  **resolved 2026-08-25**: the dominant cause was the choice term in the
  polish objective, now off by default (`docs/race_polish_plan.md` §7);
  `ladder.race_polish` exists as the noise-robust selection rule.
- **Anchors**: obtain the DECODE aligned Borg transcription and the BnF
  fr2988 transcription for a like-for-like SER comparison; both remain
  open.
- **Abstention**: the margin's overlap band is now measured on 39 real
  instances — a larger control battery (or a length-matched margin
  normalization) would place the threshold on firmer ground.
  *[Partly done as of 2026-09-01: a manuscript-shaped battery for the
  word-homophonic solver of record (12 negatives + 6 cross-language, all
  NOISE ≤ 0.48; `docs/alt_loop_plan.md` §10, 2026-08-29/30) and the
  strict-negative voynichesque twins through this pipeline (ceiling 1.40,
  9/9 abstain; `docs/voynichesque_nocontent_restart.md`, 2026-08-31).
  Length-matched margin normalisation: not started.]*
- **Presentations**: EVA character tokenization is one choice among
  several (glyph units, benched gallows as single symbols); a
  tokenization sweep is the obvious sensitivity study the abstention
  verdict deserves. *[Still open as of 2026-09-01; never started.]*
- **4.7** (25M seed replication) is still paused. *[Still paused as of
  2026-09-01, resumable per `docs/phase4_status.md` §4.7.]*

## Acceptance roll-up

`scripts/phase6_check.py` → `runs/phase6_report.json`, ClearML tag
`phase6`.

| check | status | value |
|---|---|---|
| 6.1 both dialects scored independently, both transcriptions, every head | PASS | dialects ['A', 'B'], sources ['IT2a', 'RF1b', 'boxer16', 'boxer20'], heads ['arithmetic', 'homophonic', 'naibbe', 'sub1to1'], 87 cells, tables ['IT2a+boxer/A', 'IT2a+boxer/B', 'RF1b/A', '… |
| 6.1 A and B never pooled | PASS | IT2a+boxer/A; IT2a+boxer/B; RF1b/A; RF1b/B |
| 6.2 every cell carries uncertainty columns (per-window spread, flip-rate, calibration margin uncertainty, structure margin) | PASS | 87 cells |
| 6.2 head agreement + transcription / dialect agreement reported | PASS | IT2a+boxer/A: top naibbe/latin, head agreement 0.5, abstain True; IT2a+boxer/B: top naibbe/latin, head agreement 0.25, abstain True; RF1b/A: top naibbe/latin, head agreement 0.66666666666… |
| 6.2 no-cipher baseline attached to every presentation | PASS | 12 presentations |
| 6.2 verdict IT2a+boxer/A | PASS | ABSTAIN (no language-like cell); ranking among heads: naibbe:latin, arithmetic:italian, homophonic:latin, sub1to1:german |
| 6.2 verdict IT2a+boxer/B | PASS | ABSTAIN (no language-like cell); ranking among heads: naibbe:latin, arithmetic:italian, homophonic:german, sub1to1:german |
| 6.2 verdict RF1b/A | PASS | ABSTAIN (no language-like cell); ranking among heads: naibbe:latin, homophonic:latin, sub1to1:german |
| 6.2 verdict RF1b/B | PASS | ABSTAIN (no language-like cell); ranking among heads: naibbe:german, homophonic:latin, sub1to1:german |
| 6.3 voynichesque abstention > 95% | **FAIL** | 0.889 (n=9) |
| 6.3 shuffled-text abstention > 95% | PASS | 1.000 (n=9) |
| 6.3 positives not abstained, language recovered | **WARN** | false-abstain 0.111, language correct 0.889 (n=9) |
| 6.3 contamination confusions documented | PASS | n=12, abstain 1.00, family-correct (MDL top) 0.58; MDL-top confusion {"dutch": {"latin": 0, "italian": 2, "german": 1}, "english": {"latin": 2, "italian": 1, "german": 0}, "french": {"lat… |
| 6.4 bound-fairness audit re-run attached | PASS | adopted v3-phase_c-ro; phase6_fairness_audit.md; findings: 5 |
| 6.5 length curves + confusion matrices published, claims restated | PASS | ≥200 language 0.989 / family 0.991; dominant error mode: cross-family (document heterogeneity: high-entropy Latin documents tie with the German condition), not the Romance pair; contamina… |
| 6.6 Borg ≤ 4.10% SER (full pipeline) and Latin recovered | **WARN** | SER 0.2258 (n-gram 0.1291) on 274 aligned pages; language ['latin', 'german', 'italian'] |
| 6.6 Zodiac-408 ≤ 1.9% SER (n-gram tier, English outside inventory) | PASS | SER 0.0098 (oracle 0.0098); pre-diffusion baseline only |
| 6.6 BnF fr2988 | **WARN** | no transcription available (DECODE login; benchmark repo lists BnF material as transcription pending) |
| 6.7 write-up states assumptions and residual risks explicitly | PASS | phase6_writeup.md (16990 chars) |
| freeze discipline: every Phase-6 artifact scored by the Gate-G4 frozen evaluator under the adopted calibration | PASS | frozen step 2000, sha e2cfb3c6998c; artifacts [('vms_report.json', 2000, 'v3-phase_c-ro'), ('report.json', 2000, 'v3-phase_c-ro')] |

**Verdict: FAIL on one P0 sub-criterion** — 6.3's voynichesque abstention
(0.89 against > 0.95). Everything else passes or is a stated WARN: one true
decipherment falsely abstained (the same 1.5 threshold, from the other
side), Borg's SER target missed under a non-like-for-like alignment
protocol, and BnF fr2988 unavailable. The manuscript result itself (6.1,
6.2) passes every check and does not depend on the failing bar: any
threshold in 1.26–1.48 abstains on the whole manuscript while abstaining on
no positive control.

*[Note added 2026-09-01: the failing P0 sub-criterion was re-classified on
2026-08-31 — the 1.51 voynichesque instance is content-inflated (its
strict-negative twin scores 1.24; strict negatives abstain 9/9 with a
ceiling of 1.40), so the bar was tested on a non-negative. The verdict above
is not re-tuned and stands as the frozen-rule outcome; the corridor safe
against a strict negative is ≈ 1.41–1.48. See §6.3 notes and
`docs/project_status.md` §1 "Phase-6 acceptance".]*
