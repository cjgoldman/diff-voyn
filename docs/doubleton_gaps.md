# Doubleton gaps — where the manuscript's twice-occurring words recur, and what page order says about it

> **Record status (banner added 2026-09-02, extended 2026-09-03):** side study, 2026-09-02/03. §1–5 doubletons; §6 rare types 2–5 and pooled leaf test; §7 glyph n-gram confirmation with hand control (supersedes §4's "cannot tell" and §6's "suggestive": the stacked tilt replicates on an independent unit at z +3 to +5 and survives the hand control); §8 stacked-order windows (tokens/type unchanged); §9 corpus sweep and literature; §10 sheet-order optimisation (noise-ceiling gain only); §11 containment (rare types not held by sheet); §12 empirical tail of the §7 excess (0/5 000 even-spread draws at n ≥ 7); §13–15 one-quire seriation of the stacked order (plain mean → burst-scaled metric → replication on T, A, B, C: **quire T** 1-6-5-4-2-3 on 31/32 cells, C 1-4-3-2, outermost-next-to-innermost pattern in T/C/M; adjacency only); §16 **correction** — an order and its sheet-reversal are *not* equivalent (§13–15's "≡ reversal" wording is superseded; reading direction is not established in any quire); §17 null-shape check (all p-values empirical); §18 burst front-loading (prose bursts symmetric → no time arrow; direction route closed; Currier-A sheet n-gram back-loading is a leaf-level layout question). Complete as of 2026-09-03. Motivated by the "bound out of order" literature (Davis: the bifolia may have been meant to be stacked, not nested into quires). Not on the Phase 0–6 critical path; no solver, judge or verdict depends on it. Current-status arbiter: `docs/project_status.md`.

**Question.** For every word type that occurs exactly twice in the manuscript (a *doubleton*), how far apart are its two occurrences, and does that match known texts? Two sub-questions: (1) how *bursty* are the manuscript's rare words compared with real prose and verse; (2) does the bound page order carry information about where rare words recur, and can the doubletons distinguish the nested (as-bound) reading order from Davis's stacked-bifolia order?

**Artifacts.** `scripts/doubleton_gaps.py` (gap distributions, page-order nulls, stacked order; `summary.json`, per-doubleton CSVs, page-affinity CSVs), `scripts/doubleton_leaf_affinity.py` (leaf-pair test and its `--control` power run), all under `DATA_ROOT/analysis/doubleton_gaps/`. Later sections add `rare_type_clustering.py` (§6), `glyph_ngram_leaf_test.py` (§7), `order_optimize.py` (§10), `leaf_test_pvalue.py` (§12), `quire_order_poc.py` (§13), `quire_order_burst.py` (§14–15), `quire_order_direction.py` (§16), `quire_order_nullshape.py` (§17), `burst_frontloading.py` and `burst_frontloading_control.py` (§18); each section names its JSON and log. Both EVA transcriptions (Takahashi IT2a, Reference RF1b); tokens with an uncertain glyph are dropped whole; extended-EVA codes and ligature braces are kept as part of the word type. 1000 page-order shuffles (200 per known-text window), 2000 for the leaf test.

## 1. Set-up

- Manuscript stream in folio order: 37 759 (IT2a) / 38 463 (RF1b) tokens on 225 / 227 pages, mean 168 tokens per page; 896 / 997 doubleton types (hapax 5 527 / 6 685). Page headers give quire `$Q`, bifolium `$B`, Currier language `$L`, section `$I`.
- Known texts: Latin (Macer, Gordon, Hyginus, Seneca NQ, Isidore, Pliny, Bacon), German (Küchenmeisterei, Promptuarium, Bullinger, Staden), Italian (Principe, Decameron, Commedia, Orlando Furioso), in up to three non-overlapping windows of 37 759 tokens each, cut into pseudo-pages of 168 tokens and 8-page pseudo-quires.
- Reference null 1 (uniform pairs): two positions drawn uniformly; P(gap ≤ g) and the median are exact.
- Reference null 2 (page order): pages permuted globally, within Currier language, within section, within language × section, within quire; and quires permuted as blocks. Statistics on cross-page doubletons only: fraction on adjacent pages, within 3 pages, within 1000 tokens, mean log10 gap.
- Davis order: quires in current order; within a quire, bifolia from outermost (`$B=1`) inward, each bifolium's pages in file order (leaf a r/v, leaf b r/v). Evaluated as a fixed alternative order under the same statistics.

## 2. Burstiness: the manuscript's doubletons are nearly uniformly placed

| text | N | doubletons | median gap / N (uniform ≈ 0.29) | P(gap ≤ 100) / uniform | P(gap ≤ 1000) / uniform | same page |
|---|---|---|---|---|---|---|
| VMS IT2a | 37 759 | 896 | 0.221 | **3.8** | 2.5 | **0.029** |
| VMS RF1b | 38 463 | 997 | 0.206 | **3.9** | 2.2 | **0.023** |
| Latin prose (Isidore, Bacon, Gordon, Pliny, Seneca) | 37 759 | 955–2049 | 0.06–0.23 | 12–50 | 3.1–8.0 | 0.05–0.23 |
| German prose (Promptuarium, Bullinger, Staden) | 37 759 | 655–879 | 0.13–0.22 | 18–30 | 4.1–6.3 | 0.09–0.13 |
| Italian prose (Decameron) | 37 759 | 756–820 | 0.19–0.23 | 13–16 | 3.8–4.1 | 0.06–0.07 |
| Italian verse (Commedia, Orlando Furioso) | 37 759 | 1001–1059 | 0.24–0.26 | 6.8–10 | 2.2–2.9 | 0.038–0.045 |

Currier A alone (11.3k tokens, 376–391 doubletons): P(≤ 100)/uniform 2.4–2.8, same page 0.048; Currier B alone (23–24k, 570–629): 2.7–3.6, same page 0.046–0.049. Within-10-token excess exists (ratio 4–8) and is the known line-level near-repeat phenomenon.

Reading: in prose a word that occurs twice in 38k tokens is a topic word and its two uses sit in the same passage — 12–50× the uniform rate within 100 tokens, 5–23 % on the same page. The manuscript's doubletons show a tenth of that locality (3.8×, 2–3 % same page), below even Italian narrative verse (7–10×, 4 %), whose rare forms are scattered by rhyme and metre. Whatever a manuscript "word" is, its rare recurrences carry very little topical locality. This is the same conclusion the wordhom twins reached below the findability wall (`docs/project_status.md` §1: zero content at 3.4 tokens/type) from a different direction and without any solver.

## 3. Page order: real but weak, and mostly section grouping

Fraction of cross-page doubletons on adjacent pages (IT2a / RF1b), observed vs the null mean, with z:

| pages permuted … | null mean | z (IT2a) | z (RF1b) |
|---|---|---|---|
| globally | 0.009 | 6.6 | 6.1 |
| within Currier language | 0.013 | 4.4 | 3.9 |
| within section | 0.022 | 2.0 | 1.4 |
| within language × section | 0.023 | 1.6 | 0.8 |
| within quire | 0.022 | 1.9 | 1.2 |
| quires as blocks (fraction ≤ 1000 tokens) | 0.086 / 0.083 | 3.8 | 2.2 |

Observed adjacent-page fraction 0.031 / 0.028. Known prose in its true order on the same pseudo-page grid: global z 10–37, within-quire z 4–9; Italian verse: global z 4–8, within-quire z 1.6–3.5.

Reading: the bound order is far from random (z 6–7), but two thirds of that is language and section grouping (which the null already reproduces once pages are only permuted within section). The residual within-section / within-quire locality is z ≈ 1–2.7 — the same range as verse in its correct order, and consistent with a text whose rare words carry little locality to begin with. Quire order carries some information at the 1000-token scale (z 2–4; sections span consecutive quires). None of this is evidence of disorder; it says the doubletons are too weakly local for order to leave a strong mark.

## 4. Nested vs stacked bifolia: the doubletons cannot tell

Page-level: under the stacked order the adjacent-page fraction rises from 0.031 to 0.039 (IT2a) and 0.028 to 0.038 (RF1b), i.e. ≈ +1.7 within-quire null sd. Leaf-level test (`doubleton_leaf_affinity.py`): doubletons with one occurrence on each of two leaves of the same quire, by category, against a null that permutes page contents among the page slots of the same quire and language:

| category (adjacent only if …) | IT2a obs / null | z | RF1b obs / null | z |
|---|---|---|---|---|
| conjugate leaves (stacked) | 18 / 14.3 ± 3.6 | +1.0 | 18 / 16.4 ± 3.8 | +0.4 |
| nested-adjacent leaves (nested) | 27 / 27.6 ± 4.7 | −0.1 | 32 / 29.7 ± 4.9 | +0.5 |
| innermost bifolium (both) | 5 / 6.8 | −0.7 | 10 / 9.0 | +0.4 |
| other same-quire leaf pairs | 101 / 112.2 ± 5.8 | −1.9 | 112 / 125.3 ± 6.1 | −2.2 |
| conjugate − nested-adjacent | −9 / −13.3 ± 6.3 | **+0.7** | −14 / −13.4 ± 6.6 | **−0.1** |

Power control (500 shuffles): five known texts laid onto the manuscript's own page slots (same tokens per page, same quire and bifolium structure), written in the nested order or in the stacked order, then bound nested and analysed identically:

| control text | conj − nested z, written nested | written stacked |
|---|---|---|
| Isidore (Latin prose) | −4.0 | +1.0 |
| Seneca NQ (Latin prose) | −1.8 | +0.2 |
| Bullinger (German prose) | −2.0 | +1.9 |
| Decameron (Italian prose) | −3.6 | +3.9 |
| Orlando Furioso (Italian verse) | −2.6 | +1.1 |

Reading: even for real texts with 3–13× the manuscript's burstiness, the leaf test separates the two hypotheses by only 2–7 sd, and for the least bursty control (verse) the stacked-written case is indistinguishable from the null. The manuscript's +0.7 / −0.1 lies above every nested-written control and inside the stacked-written band, but its expected effect under *either* hypothesis is smaller than any control's because its doubletons are so much less local (§2). The honest statement for doubletons alone: they do not contradict Davis's stacked reading and cannot confirm it; the test is underpowered on this text by roughly an order of magnitude in locality. *(§6 pools types with 2–10 occurrences and finds a weak but consistent tilt toward stacked.)*

## 5. What would give power

- Use all rare types, not only doubletons (types with 2–5 occurrences, consecutive-occurrence gaps), and score page-pair affinity by a likelihood ratio rather than counts — perhaps 3–5× the pairs.
- Test the specific reorderings the codicology proposes (Davis's per-quire proposals, the herbal/pharma foldouts, the Currier-A herbal bifolia) as fixed alternative orders against the within-quire null, rather than the blanket stacked rule used here.
- Sub-word units: repeat the analysis on rare glyph n-grams, which have more instances and, if the "word" is a cipher unit, may carry the locality the words lack.
- Nothing here needs the GPU; all runs are seconds on CPU.

## 6. Words used three, four, five times — and the pooled order test

`scripts/rare_type_clustering.py` (`rare_types.json`, `rare_types_controls.json`, logs `rare_types.log`, `rare_controls.log`). Consecutive-occurrence gaps per frequency class k against a Monte-Carlo uniform-placement null with the same N and k (200 000 draws); known texts in one 37 759-token window with 168-token pseudo-pages.

### 6.1 Clustering by frequency class

P(consecutive gap ≤ 100) ÷ uniform, and (in brackets) the fraction of types with all occurrences on one page:

| text | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|
| VMS IT2a (types 896 / 399 / 216 / 145) | 3.8 (2.9 %) | 3.3 (0.3 %) | 3.5 (0 %) | 4.5 (0 %) |
| VMS RF1b (997 / 413 / 250 / 157) | 3.9 (2.3 %) | 3.7 (0.2 %) | 5.0 (0.8 %) | 4.4 (0.6 %) |
| Currier A alone (IT2a / RF1b) | 2.4 / 3.1 | 2.7 / 2.9 | 2.9 / 3.2 | 3.2 / 2.6 |
| Currier B alone | 2.6 / 3.2 | 3.3 / 3.0 | 2.9 / 3.0 | 3.0 / 1.9 |
| Isidore (Latin) | 51 (23 %) | 32 (7.0 %) | 23 (2.2 %) | 18 (0.5 %) |
| Pliny (Latin) | 21 (10 %) | 19 (3.5 %) | 12 (0.3 %) | 12 (1.3 %) |
| Seneca NQ (Latin) | 13 (5.4 %) | 9.5 (1.9 %) | 8.6 (0.3 %) | 6.0 (0 %) |
| Bullinger, Staden (German) | 23–26 (11 %) | 16–17 (3 %) | 14–15 (0.5–1.3 %) | 9–11 (0.6–0.8 %) |
| Decameron (Italian prose) | 13 (6.0 %) | 13 (1.3 %) | 7.1 (0.4 %) | 7.9 (0 %) |
| Commedia, Orlando Furioso (verse) | 6.6–8.4 (3.8 %) | 4.4–4.7 (0–0.6 %) | 4.4–5.4 (0 %) | 2.4–3.9 (0 %) |

Median span (first to last occurrence) over N for k = 3: VMS 0.41 (uniform 0.50); prose 0.25–0.42; verse 0.45–0.48. Mean distinct pages per type for k = 3: VMS 2.92 (uniform 2.98); Isidore 2.57; verse 2.94.

Reading: the answer for three- and four-times words is the same as for doubletons, at every frequency class and in both Currier languages. Consecutive uses of a rare manuscript word fall within 100 tokens 3–5× more often than chance; in prose it is 9–32× at k = 3 and 7–23× at k = 4, in Italian verse 4–5×. The manuscript's rare vocabulary clusters like the vocabulary of rhymed verse, not like topic words in prose, and it does not become more prose-like as k grows.

### 6.2 Pooled leaf test (all occurrence pairs of types with 2–5 or 2–10 occurrences)

Same categories and within-quire-and-language null as §4, 2000 shuffles:

| pairs pooled | IT2a: conj z / nested z / **conj − nested z** | RF1b |
|---|---|---|
| k = 2 (896 / 997 pairs) | +1.0 / −0.1 / **+0.7** | +0.4 / +0.5 / **−0.1** |
| k = 2–5 (4 839 / 5 306) | +2.6 / −1.4 / **+2.5** | +2.0 / −0.7 / **+1.8** |
| k = 2–10 (13 665 / 14 625) | +2.2 / −1.8 / **+2.4** | +2.3 / −1.8 / **+2.5** |
| k = 2–10, Currier A pages only | +0.7 / +0.2 / +0.3 | +1.4 / +0.2 / +0.7 |
| k = 2–10, Currier B pages only | +1.1 / −1.4 / +1.8 | +1.0 / −1.7 / +1.9 |

Power controls (six known texts on the IT2a page slots, 1000 shuffles), conj − nested z:

| control | written nested, k 2–5 / 2–10 | written stacked, k 2–5 / 2–10 |
|---|---|---|
| Isidore | −7.1 / −7.3 | +3.1 / +3.5 |
| Seneca NQ | −4.2 / −5.5 | +1.3 / +2.4 |
| Bullinger | −5.2 / −7.2 | +3.3 / +2.8 |
| Decameron | −4.4 / −5.4 | +3.6 / +4.7 |
| Commedia | −0.8 / −2.0 | +2.0 / +1.5 |
| Orlando Furioso | −3.1 / −3.3 | +0.9 / +1.7 |

Per-quire contributions (k 2–10, conj − nested, observed vs null): quire A (herbal, Currier A, f1–f8) +4 vs −5.3 ± 4.6 and +4 vs −5.0 ± 4.4 (≈ +2.0 sd on both transcriptions); quire T (stars, B, 23 pages) −76 vs −117 ± 22 and −85 vs −120 ± 24 (+1.9 / +1.5); quire H (8 pages, two bifolia, mixed languages) +48 vs −18 ± 36 and +62 vs −14 ± 47 (+1.8 / +1.6, null nearly degenerate); quire M (bio, B) +0.5 / +1.4; herbal quires B–G between −2.0 and +1.0 with mixed signs; quires with one bifolium contribute nothing by construction.

Reading: pooling rare types puts the manuscript at conj − nested z ≈ +2.4 on both transcriptions, inside the band the controls give when written in the stacked order (+0.9 to +4.7) and outside the band they give when written nested (−0.8 to −7.3, all six texts, both pool sizes). Conjugate leaves share rare words *more* than random same-quire leaves, and consecutive leaves of different bifolia share *less*. Because the manuscript's locality is 3–13× weaker than any control's, its expected magnitude under either hypothesis is smaller than the controls', so a +2.4 is if anything a larger relative departure from "written nested" than the raw number suggests. Caveats that keep this at "suggestive": (i) the pooled statistic was chosen after the k = 2 test came back null, so the nominal p ≈ 0.01 (one-sided, per transcription, the two not independent) is not pre-registered; (ii) the signal is spread over quires A, T, H and M and absent from herbal quires B–G; (iii) conjugate leaves are one physical sheet, so any sheet-level effect (one writing session per sheet, sheet-level choice of subject) would produce the same excess without a stacked *reading* order — which is close to, but not identical with, Davis's claim. Confirmation would need an independent statistic (glyph n-grams, or a whole-vocabulary leaf similarity with language and section controlled) and, ideally, the specific reorderings proposed in the codicological literature as fixed alternatives.

## 7. Glyph n-gram check (independent unit) and the scribal-hand control

`scripts/glyph_ngram_leaf_test.py` (`glyph_ngrams.json`, log `glyph_ngrams.log`); hand control in `hand_control.json` / `hand_control.log`. Units: character n-grams over the space-stripped text of each page (extended-EVA codes are one symbol), n = 4…8, rare = 2–10 occurrences (k = 2 alone also shown); variant **cross** keeps only n-grams that straddle a word boundary, so they are fragments of word *pairs*, not of single rare words. Same leaf categories and within-quire-and-language null (500 shuffles; 1000 for the hand control).

### 7.1 Manuscript

conj − nested z (IT2a / RF1b), types with 2–10 occurrences:

| n | all n-grams | boundary-straddling only | pairs (all) |
|---|---|---|---|
| 4 | +2.6 / +1.2 | +2.6 / +1.0 | 33k / 37k |
| 5 | +3.4 / +2.8 | +3.1 / +2.3 | 74k / 81k |
| 6 | +3.3 / +3.5 | +3.2 / +3.5 | 113k / 119k |
| 7 | +4.1 / +4.4 | +3.7 / +4.2 | 119k / 118k |
| 8 | +4.7 / +4.8 | +4.4 / +5.2 | 98k / 92k |

Doubleton n-grams alone (k = 2): n = 7 gives +4.5 / +3.4, n = 8 +4.0 / +4.2 (all), +3.8 / +2.8 and +4.2 / +4.0 (cross). Both components move: conjugate leaves share more than the null (z +2.2 to +4.1 at n ≥ 5) and nested-adjacent leaves share less (z −1.4 to −3.4).

### 7.2 Controls (six known texts on the IT2a page slots by symbol count, k 2–10)

| control | written nested: n5 / n6 / n7 (all; cross) | written stacked |
|---|---|---|
| Isidore | −6.0 / −5.6 / −5.2 (−4.4 / −4.3 / −3.9) | +2.3 / +2.5 / +2.6 (+2.1 / +2.4 / +2.4) |
| Seneca NQ | −5.5 / −6.2 / −6.4 (−4.9 / −6.5 / −6.0) | +2.4 / +2.9 / +2.4 (+2.0 / +2.6 / +2.3) |
| Bullinger | −6.3 / −7.6 / −7.8 (−7.0 / −7.2 / −7.3) | +3.2 / +3.4 / +3.7 (+2.3 / +2.6 / +2.9) |
| Decameron | −5.3 / −5.7 / −5.4 (same) | +4.7 / +4.5 / +4.5 (+4.7 / +4.9 / +4.3) |
| Commedia | −3.3 / −3.9 / −3.7 (−2.3 / −2.4 / −2.4) | +3.7 / +2.7 / +2.4 (+2.8 / +1.6 / +1.4) |
| Orlando Furioso | −2.2 / −3.4 / −4.0 (−1.7 / −2.8 / −3.1) | +1.1 / +1.7 / +1.9 (+1.3 / +1.6 / +1.8) |

The manuscript's +3 to +5 sits in the upper part of the stacked-written band (+1.1 to +4.9) and nowhere near the nested-written band (−1.7 to −7.8, thirty-six of thirty-six control cells negative).

### 7.3 Scribal hand

Conjugate leaves are in the same hand (and language) for 97 % of page pairs (138/142 IT2a, 146/152 RF1b); nested-adjacent leaves only 70 % (203/290, 203/292); other same-quire pairs 73–75 %. The hands (`$H`, Davis's five scribes) therefore run by *sheet*, which is itself a codicological observation consistent with sheets being the working unit. A hand-specific spelling habit would make conjugate leaves share rare n-grams for reasons unrelated to reading order, so the null was restricted to pages of the same quire, language **and hand** (28 / 30 groups):

| unit | null within quire + lang (IT2a / RF1b) | null within quire + lang + hand |
|---|---|---|
| words, k 2–10 | +2.4 / +2.5 | +1.7 / +2.0 |
| n = 6 all / cross | +3.4 / +3.3 ; +2.9 / +3.2 | +3.5 / +3.7 ; +3.0 / +3.6 |
| n = 7 all / cross | +3.9 / +4.1 ; +3.6 / +4.1 | +4.0 / +4.2 ; +3.6 / +4.2 |

The n-gram result is unchanged under the hand control; the word result drops by about a third but keeps its sign.

### 7.4 Reading

Three units (whole words, character n-grams inside words, character n-grams across word boundaries), two transcriptions, and a null that holds quire, Currier language and scribal hand fixed all agree: text on the two leaves of one physical sheet shares its rare material more than text on leaves that are neighbours only in the nested binding, and the nested neighbours share *less* than random same-quire leaves. Real texts written in the nested order never do this (36/36 control cells negative); real texts written sheet-by-sheet and then nested do (36/36 positive), at the magnitude the manuscript shows. That is the pattern Davis's stacked-bifolia proposal predicts. What it does not settle: whether the sheet was the unit of *composition* (text flows a-r → a-v → b-r → b-v) or only the unit of *work* (one scribe, one session, one topic per sheet, with the intended reading order still nested) — the two are indistinguishable by any rare-material statistic. Nor does it say which sheets or quires are misplaced; the per-quire word-level picture (§6.2) is uneven, and a per-quire n-gram breakdown plus the literature's specific reorderings remain the natural next step. Caveat on sequential testing: the doubleton test (§4) was null, the pooled word test (§6) was chosen after seeing it, and the n-gram test was run as its check; the n-gram result is the pre-stated confirmation of the §6 hypothesis and is the number to quote (z +3 to +5, both transcriptions, hand-controlled).

## 8. Does the stacked order change tokens/type in solver windows? No.

Asked 2026-09-02 because the wordhom findability wall is ≈ 4 tokens/type and windowing was rejected on 2026-09-01 at 1.8–2.0 tokens/type per 1024-token window. Non-overlapping windows over the Currier A and B streams in the nested vs the stacked order, against the within-quire page-shuffle null (200 shuffles); log `window_tokens_per_type.log`.

| stream, W | tokens/type nested → stacked (IT2a) | (RF1b) | hapax fraction |
|---|---|---|---|
| A, 512 | 1.607 → 1.620 (+3.2 sd) | 1.541 → 1.544 | 0.78–0.80 |
| A, 1024 | 1.867 → 1.875 | 1.767 → 1.769 | 0.75–0.77 |
| A, 2048 | 2.192 → 2.206 | 2.035 → 2.046 | 0.73–0.76 |
| A, 4096 | 2.638 → 2.635 | 2.408 → 2.407 | 0.72–0.74 |
| B, 512 | 1.721 → 1.724 | 1.610 → 1.613 | 0.75–0.78 |
| B, 1024 | 2.028 → 2.032 | 1.870 → 1.882 (+3.2 sd) | 0.73–0.75 |
| B, 2048 | 2.422 → 2.430 | 2.219 → 2.205 | 0.71–0.73 |
| B, 4096 | 2.972 → 2.911 | 2.661 → 2.606 | 0.70–0.72 |

The sheet-level sharing is real but tiny in mass: it moves tokens/type by ≤ 0.01 (under 1 %) at W ≤ 1024, sometimes detectably against the null, and by nothing or slightly negative at W ≥ 2048. Every window stays at 1.5–3.0 tokens/type with 70–80 % hapax types, far below the ≈ 4 wall. The 2026-09-01 rejection of windowing stands under either reading order; the order result of §7 does not open a solver path.

## 9. Distance from the corpus and from the published literature (2026-09-02)

Corpus sweep (`corpus_sweep.json`, log `corpus_sweep.log`): every corpus document with ≥ 37 759 words, up to two windows each — 148 unique known windows (German 113, Latin 25, Italian 10; the Italian manifest documents are the raw texts, listed twice in the log). Statistic: consecutive gaps of all types with 2–5 occurrences pooled, P(gap ≤ 100) ÷ uniform (and ≤ 1000).

| | P(≤ 100)/uniform | P(≤ 1000)/uniform |
|---|---|---|
| known: min / 5th pct / median / max | 3.39 / 4.66 / 12.9 / 34.0 | 1.51 / 1.65 / 3.35 / 5.53 |
| Latin (25): min / median | 8.6 / 20.1 | — |
| German (113): min / median | 4.8 / 13.2 | — |
| Italian (10, verse epics + Decameron + Principe): min / median | 3.4 / 5.0 | — |
| VMS IT2a (whole, 37 759 tokens) | 3.89 — 2.5th percentile, log-z −2.3; below it only Tasso's *Gerusalemme liberata* (both windows) | 2.06 — 10th percentile, log-z −1.5 |
| VMS RF1b (whole) | 4.38 — 3.8th percentile, log-z −2.1; below it Tasso ×2 and Dante | 2.11 — 10th percentile |

The whole-manuscript stream mixes Currier A and B and the sections, which *adds* clustering; Currier A or B alone score 2.8–2.9 (≤ 100) and 1.4–1.6 (≤ 1000), below every known window, though at a shorter N the ratio is not strictly comparable. So the manuscript's rare vocabulary is at the extreme low edge of the corpus — matched only by Italian ottava-rima / terza-rima epics, where rhyme forces rare word forms to scatter — and two to three log-sd below the typical prose window.

Published literature. Montemurro & Zanette (2013, PLoS ONE 8:e66344) measure the information that *all* words carry about the section they fall in, at a scale of ≈ 800 words, and find the manuscript comparable to English and below Chinese; that is section-/topic-scale clustering, and this study sees the same thing (§3: adjacent-page sharing z 6–7 under a global page shuffle, most of it removed by permuting only within section). Reddy & Knight (2011, ACL LaTeCH) report that manuscript words "do not show significant long-distance correlations", cite Schinner (2007) for the finding that the distance between repeats of similar words follows a geometric distribution (i.e. is memoryless, as under uniform placement), and note the near-absence of repeated word bigrams; those statements are the passage-scale side, and §2/§6 quantify it for the rare vocabulary: 3–5× uniform within 100 tokens against 7–34× in prose. The two scales are not in conflict: the manuscript has topic-sized sections whose vocabularies differ, but within a section a rare word's second use is almost as likely anywhere as next to its first.

## 10. Can a better stacking order push the rare-word locality toward the corpus? No — the optimizer gains the same on noise.

`scripts/order_optimize.py <steps> <none|inverted>` (`order_optimize_{none,inverted}.json`, logs `order_optimize_{none,inverted}.log`; a first run of 2026-09-02 that allowed a physically impossible leaf swap, `order_optimize.log`, gave the same numbers and is superseded). IT2a, 52 sheets (bifolia; single surviving leaves count as sheets), every sheet read stacked (leaf a r/v, leaf b r/v). Orientation: **none** (the sheet's page order is fixed — the run of record) or **inverted** (a sheet may be folded the other way: b-v, b-r, a-r, a-v; considered unlikely, run as a sensitivity check). Simulated annealing, 20 000 steps, objective Σ exp(−gap/1000) over consecutive occurrences of types with 2–5 occurrences. Constraint levels: **strict** (sheets swap only within the same Currier language, section and hand; 12 groups), **topic** (language + section; 10), **language** (3), **free**. Reported: P(gap ≤ 100)/uniform (r100) and P(gap ≤ 1000)/uniform (r1000), the corpus-sweep statistics of §9 (corpus r1000: 5th pct 1.65, 25th 2.74, median 3.35; r100: 5th pct 4.66, median 12.9).

Orientation fixed (run of record), r100 / r1000:

| start | init | strict | topic | language | free |
|---|---|---|---|---|---|
| manuscript, stacked order | 4.04 / 2.09 | 4.24 / 2.40 | 4.28 / 2.45 | 4.24 / 2.50 | 4.48 / 2.62 |
| manuscript, random sheet orders (3 starts) | 4.04 / 2.09 | 4.2 / 2.42–2.45 | 4.1–4.2 / 2.43–2.47 | 4.2–4.4 / 2.42–2.51 | 4.3–4.4 / 2.58–2.68 |
| manuscript, **page contents shuffled** within quire + language first (no order information; 3 draws) | 4.0–4.5 / 1.89–1.97 | 5.3–5.7 / 2.32–2.54 | 5.1–5.4 / 2.37–2.59 | 5.3–5.9 / 2.51–2.83 | 6.1–6.6 / 2.67–3.00 |
| Isidore written stacked, sheets shuffled (true order 29.3 / 5.09) | — | 29.2 / 5.09 | 29.3 / 5.11 | 29.2 / 5.09 | 29.2 / 5.16 |
| Seneca NQ (true 8.69 / 2.51) | — | 8.64 / 2.52 | 8.69 / 2.56 | 8.59 / 2.61 | 8.77 / 2.64 |
| Bullinger (true 14.8 / 3.36) | — | 14.6 / 3.46 | 14.7 / 3.46 | 14.5 / 3.50 | 14.8 / 3.55 |
| Decameron (true 9.27 / 2.81) | — | 9.30 / 2.95 | 9.20 / 2.94 | 9.30 / 3.07 | 9.30 / 3.16 |
| Orlando Furioso (true 5.36 / 1.96) | — | 5.56 / 2.07 | 5.43 / 2.10 | 5.52 / 2.21 | 5.49 / 2.21 |

Inverted folding allowed: manuscript from stacked 4.36 / 2.42 (strict), 4.36 / 2.44, 4.60 / 2.53, 4.24 / 2.70 (free); page-content-shuffled manuscripts 5.7–6.8 / 2.53–2.56 (strict) … 6.7–7.4 / 2.84–3.03 (free). Same picture, with the extra degree of freedom adding ≈ +0.05 to both the manuscript and the noise runs.

Reading. (i) Reordering the sheets under any constraint moves r1000 from 2.09 to 2.4–2.6 — but a manuscript whose page contents were first shuffled, so that no order information exists, is moved by the same optimizer to 2.3–3.0. The whole gain is what an optimizer extracts from noise on ~5 000 rare pairs; nothing beyond the noise ceiling appears at any constraint level. (ii) Random sheet orders are optimized to the same values as the actual stacked order: the current order is not a special optimum and the optimizer cannot tell it from a shuffle. (iii) Known texts written sheet-by-sheet and shuffled are put back to their true-order value (plus 0.0–0.35 of overfitting, growing with freedom), so the method would recover a real order if one carried locality. (iv) Even the overfitted manuscript values stay at the bottom of the corpus: r1000 2.4–2.6 is below the 25th percentile (2.74), and r100 barely moves (4.0 → 4.2–4.5, ≈ 5th percentile) because gaps under 100 tokens fall inside a sheet (≈ 670 tokens) and no sheet order can change them. Sheet order is not what separates the manuscript's rare vocabulary from prose; the separation is within the sheet.

## 11. Where the occurrences of a rare type sit (2026-09-03)

Asked whether the §7 sheet effect could mean that rare types are *contained* in one bifolium. They are not. Percentage of types whose occurrences all fall in the given unit (null: page contents shuffled within quire + language, 200 draws; the same-page and 2+-quire columns are order-independent under that null). Log `containment.log`.

| k | types (IT2a / RF1b) | same page | same leaf (r+v) | same bifolium, both leaves | 2+ bifolia, same quire | 2+ quires |
|---|---|---|---|---|---|---|
| 2 | 896 / 997 | 2.9 / 2.3 | 2.7 / 2.6 (null 1.6 / 1.8) | 2.6 / 2.8 (null 2.3 / 2.6) | 14.3 / 14.4 (null 15.6 / 15.5) | **77.6 / 77.8** |
| 3 | 399 / 413 | 0.3 / 0.2 | 0.3 / 0.5 (null 0.2 / 0.1) | 1.0 / 0.2 (null 0.4 / 0.3) | 7.3 / 8.0 (null 8.0 / 8.4) | **91.2 / 91.0** |
| 4 | 216 / 250 | 0.0 / 0.8 | 0.0 / 0.0 | 0.0 / 0.4 (null 0.1 / 0.1) | 7.4 / 6.8 (null 7.4 / 7.0) | **92.6 / 92.0** |
| 5 | 145 / 157 | 0.0 / 0.6 | 0.0 / 0.0 | 0.0 / 0.6 (null 0.0 / 0.1) | 3.4 / 3.8 (null 3.4 / 4.3) | **96.6 / 94.9** |

Three quarters of doubletons and over 90 % of the 3–5-times types span more than one quire; only 2–3 % of doubletons and ≈ 0–1 % of the rest are held by a single sheet. The §7 sheet effect is a small excess in the roughly 20 % of pairs that fall within a quire (same-leaf 2.7 % vs 1.6 % expected; same-bifolium 2.6 % vs 2.3 %), not a containment of vocabulary by sheet, and it is why re-stacking (§8) and re-ordering the sheets (§10) move the locality statistics so little: the material the order can act on is a fifth of the pairs, and the rest lies across quires.

## 12. How often would even spreading produce the stacked-over-nested excess? (2026-09-03)

`scripts/leaf_test_pvalue.py` (`leaf_test_pvalue.json`, log `leaf_test_pvalue.log`): the §6/§7 statistic (conjugate-leaf pairs minus nested-adjacent-leaf pairs, types with 2–10 occurrences) with 5 000 draws per cell and the tail counted directly, p = (#null ≥ observed + 1)/(n + 1). Two readings of "evenly spread": **perm** — page contents permuted within quire + language (+ hand), each page keeping its own rare-word load (the §6/§7 null; the conservative one, because page-level clumping stays in the variance); **uniform** — every occurrence of a rare type lands independently on a page of its language with probability proportional to page length (the literal even spread; smaller variance, so larger z).

Null draws reaching the observed excess, out of 5 000 (perm; IT2a / RF1b), and the normal-tail p for the recorded z:

| unit | null quire + lang | null quire + lang + hand |
|---|---|---|
| words k 2–10 | 54 / 26 (p ≈ 1/90 · 1/190; z +2.3 / +2.5) | 157 / 57 (p ≈ 1/30 · 1/90; z +1.8 / +2.0) |
| n = 5 | 1 / 14 (z +3.5 / +2.9) | 2 / 2 (z +3.5 / +3.3) |
| n = 6 | 3 / 3 (z +3.5 / +3.5, p ≈ 1/1 700) | 5 / 1 (z +3.6 / +3.7) |
| n = 7 | 0 / 0 (z +4.2 / +4.3, normal tail ≈ 1/70 000) | 0 / 0 (z +4.1 / +4.3) |
| n = 8 | 0 / 0 (z +4.6 / +5.0, normal tail ≈ 1/500 000) | 0 / 0 (z +4.5 / +4.9) |

Under the uniform null every n-gram cell is 0/5 000 (z +3.6 to +8.2) and the word cell 10 / 0 (IT2a / RF1b).

Reading. The word-level excess alone is a 1-in-30 to 1-in-200 event under even spreading — real but not decisive, and it was chosen after the doubleton test came back null. The glyph n-gram excess, the pre-stated confirmation, never occurs in 5 000 even-spread draws at n ≥ 7 on either transcription or either null; its normal-tail chance is of order 10⁻⁵–10⁻⁶ per cell. The cells are not independent (n-grams of neighbouring n overlap; the two transcriptions are the same text), so the family is best counted as one test per transcription at n = 7 or 8: even a twenty-fold multiplicity correction leaves it below 1/1 000. So: if the rare material were evenly spread, seeing this increase would be a chance of roughly one in ten thousand or less; what it does not exclude is the sheet being the unit of *work* rather than of reading order (§7.4).

## 13. Proof of concept: can a clustering metric point to an intended stacking order inside one quire? (2026-09-03)

`scripts/quire_order_poc.py` (`quire_order_poc_M.json`, log `quire_order_poc_M.log`; the first run's log, `quire_order_poc_M_maskbug.log`, had a wrong null mask for the between-sheet metric and is superseded). Quire **M** (f75–f84: 20 pages, 5 bifolia, Currier B, hand 2, section B, 6 911 / 6 969 tokens IT2a / RF1b) — the only quire with more than one sheet that is one language, one hand and one section throughout.

**Metric.** Units: word types and glyph n-grams (n 5–8) with 2–10 occurrences in the quire. For a reading order, the mean distance in tokens (symbols) between the two members of every occurrence pair of a rare type — the "total distance of all members" per pair (L1); a short-range kernel mean exp(−d/300) as a check (K). Two questions use two pair sets: *which stacking* uses only pairs whose members sit on different sheets, because the four pages of a sheet are contiguous in every stacked order and within-sheet pairs would only add the §7 sheet effect as a constant; *stacked at all vs as bound* uses every cross-page pair. Candidates: all 5! = 120 sheet orders read a-r, a-v, b-r, b-v *(§13–15 reported orders up to reversal, treating an order and its sheet-reversal as equivalent — **corrected 2026-09-03, §16**: reversing the sheet order does not reverse the page sequence, so the two costs differ and the difference is the direction signal)*, the nested binding as the 121st, and the inverted folding per sheet (b-v, b-r, a-r, a-v; 3 840 candidates) as an extension. Nulls: page contents permuted among the quire's slots (200 draws), the best-of-120 recomputed each time; consistency across units = correlation of the 120-vector between units against the same correlation on shuffled contents (which keeps the overlap of n-grams with their words).

**Power (known prose on quire M's page slots, 6 911 tokens, three texts × three windows).** Written stacked in the current or a random sheet order, the true order ranks 1/120 in 5 of 18 cases, in the top 10 in 13, and 44–52 in 3 (Isidore w0, Bullinger w0); its within-candidate z is −0.0 to −2.9. Written nested, the nested order ranks 1/121 in 8 of 9. So at this quire size the metric recovers a prose writing order about a third of the time outright and usually to within the top tenth; the manuscript's rare-material locality is about a tenth of prose (§2, §9), so low power is expected there.

**Manuscript, which stacking (between-sheet L1; IT2a / RF1b):**

| unit | best order (reported up to reversal — see §16) | best z within the 120 | p (best vs best on shuffled contents) | current 1-2-3-4-5: rank / 120 |
|---|---|---|---|---|
| words | 2-3-5-1-4 / 2-3-5-1-4 | −2.36 / −2.33 | 0.025 / 0.070 | 98 / 92 |
| n = 5 | 1-4-5-2-3 / 3-2-5-1-4 | −2.55 / −1.94 | 0.050 / 0.119 | 86 / 95 |
| n = 6 | 1-4-5-2-3 / 1-4-5-2-3 | −2.52 / −1.75 | 0.035 / 0.383 | 40 / 65 |
| n = 7 | 1-4-2-3-5 / 1-4-2-3-5 | −2.53 / −2.41 | 0.060 / 0.060 | 31 / 33 |
| n = 8 | 1-4-2-3-5 / 2-3-5-1-4 | −2.34 / −2.42 | 0.184 / 0.065 | 46 / 48 |

The minimum of 120 roughly-normal values, of which reversal pairs are nearly (not exactly, §16) equal, sits at ≈ −2.3 to −2.5 sd, which is exactly where every best order falls: no candidate stands out from the candidate set. Against shuffled contents the best order is mildly better (p 0.03–0.4 per unit, units not independent), and the 120-vectors of different units agree slightly more than shuffled contents do (words vs n-grams z +1.0 to +2.1 IT2a, +1.4 to +1.5 RF1b; n-gram vs n-gram z +0.4 to +1.2). The between-sheet affinity matrix (shared rare pairs, observed/expected; sheets 1..5 outer→inner) says what that whisper is: sheets 1 and 5 share most (1.35–1.42 words, 1.04–1.08 n7), 2–3, 3–5 and 2–5 share 1.1–1.2, and sheet 4 (f78/f81) shares least with everything (0.63–0.87, except 1.0 with sheet 1). The recurring best orders are the chains that put 2–3 together, 5 next to 1, and 4 at an end (2-3-5-1-4; the n-gram optima 1-4-5-2-3 and 1-4-2-3-5 are one transposition away). The inverted-folding extension changes nothing (best of 3 840 at z −2.4 to −2.9 within, vs ≈ −3.5 for the minimum of that many; the fixed-orientation optimum or a one-flip variant of it is the top candidate). K agrees on the words optimum (2-3-5-1-4 both transcriptions) and scatters on n-grams.

**Manuscript, stacked at all vs as bound (all cross-page pairs).** The nested binding ranks 100–121 of 121 on every unit and transcription (121/121 for n 5, 8 IT2a and n 7, 8 RF1b) and is no better than random page contents (z −0.6 to +0.9), while the stacked candidates as a set beat shuffled contents at z +2.3 to +6.4 — §7 again in seriation form: any reading that keeps a sheet's four pages together beats the binding, and the binding is as bad as chance.

**Reading.** As a proof of concept the metric works: on prose of quire M's size it recovers the writing order outright a third of the time and to the top tenth usually, and on the manuscript it reproduces the sheet effect with the nested binding at the bottom. What it does not find is an intended stacking: the best of the 120 orders is exactly as good as the best of 60 random values, the optimum differs between units, and the only structure is a faint between-sheet affinity pattern (sheet f78/f81 apart, f75/f84 with f79/f80) at p ≈ 0.03–0.4 per unit and cross-unit agreement of z +1 to +2 — a hint at best, not a result, and one that the ~10× lower locality of the manuscript's rare material would predict to be about this weak even if a true order existed. Next steps if pursued: replicate on quire T (23 pages, 6 sheets, Currier B, hand 3, 10 673 tokens) and on the Currier-A herbal quires A–C (4 sheets, hand 1, ~1 400 tokens each); pool the between-sheet affinities of several quires with the stacked hypothesis as a prior; and test Davis's specific proposals as fixed candidates rather than searching the full 120.

## 14. Burst-scaled seriation metric (2026-09-03)

Asked after §13: the plain mean pair distance gives every cross-sheet pair the same weight, although a type that clusters tightly where it lives carries more positional information than one spread over a sheet, and gives a k-occurrence type ~k² pairs. `scripts/quire_order_burst.py` (`quire_order_burst_M.json`, log `quire_order_burst_M.log`) adds three weighted metrics, same quire M, same 120 candidates, nulls and controls as §13:

- **burst**: for each rare type, home sheet = the sheet with most of its occurrences (types tied, e.g. doubletons split 1–1, are skipped as carrying no intra-sheet information); burst scale λ_t = mean gap between consecutive home-sheet occurrences, shrunk toward the median over types (one pseudo-gap) and floored at 30; cost = Σ over occurrences outside the home sheet of d/λ_t, d = distance under the candidate order to the nearest home occurrence. This is the negative log-likelihood of an exponential-tail burst model; λ_t depends only on within-sheet gaps, so it is identical across all stacked orders and no candidate can tune its own weights. One term per outlier, not per pair.
- **blog**: Σ log(1 + d/λ_t) — caps a single far outlier.
- **modal**: Σ w_t·d with w_t = fraction of home occurrences on the type's modal page — the discrete version of the same intuition.

**Power (prose on quire M's slots, 18 stacked-written cases).** Rank of the true order among 120: L1 rank-1 4, top-10 13, median rank 6; burst 3 / 13 / 4; blog 6 / 13 / 2; modal 5 / 13 / 3. (A run with a different random writing order gave rank-1 5 / 6 / 8 / 8.) The weighting halves to thirds the median rank on prose; the rank-1 count is sensitive to the particular order and window.

**Manuscript (IT2a / RF1b): best order, z of the best within the 120, and p of the best against the best on shuffled contents (200 draws):**

| unit | burst | blog | modal | L1 (§13) |
|---|---|---|---|---|
| words | 2-3-5-1-4, p 0.40 / 0.26 | 2-3-5-1-4, 0.38 / 0.11 | 2-3-5-1-4, 0.19 / 0.09 | 2-3-5-1-4, 0.03 / 0.06 |
| n = 5 | 3-2-5-1-4, **0.015 / 0.005** | 3-2-5-1-4, 0.045 / 0.020 | 3-2-5-1-4, **0.005 / 0.005** | 1-4-5-2-3 · 3-2-5-1-4, 0.09 / 0.14 |
| n = 6 | 3-2-5-1-4, 0.060 / 0.095 | 3-2-5-1-4, 0.045 / 0.12 | 3-2-5-1-4, **0.005 / 0.005** | 1-4-5-2-3, 0.03 / 0.33 |
| n = 7 | 4-1-2-3-5, 0.030 / 0.13 | 2-3-5-1-4 · 4-1-2-3-5, 0.010 / 0.08 | 3-2-5-1-4 · 4-1-2-3-5, 0.025 / 0.015 | 1-4-2-3-5, 0.07 / 0.06 |
| n = 8 | 2-3-5-1-4, 0.020 / 0.045 | 2-3-5-1-4, 0.010 / 0.030 | 4-1-2-3-5 · 2-3-5-1-4, 0.020 / 0.035 | 1-4-2-3-5 · 2-3-5-1-4, 0.18 / 0.08 |

z of the best within the candidate set: burst −2.2 to −3.2, blog −2.5 to −3.2, modal −2.4 to −3.0 (the minimum of 60 distinct random values sits at ≈ −2.3 to −2.5). Current outer→inner order 1-2-3-4-5: rank 30–114 of 120 under every weighted metric. Cross-unit consistency of the 120-vectors is unchanged by the weighting (words vs n-grams z +0.5 to +2.0, n-gram vs n-gram mean z +0.7 to +1.1).

**Reading.** The weighting does what it was meant to: on prose it lowers the median rank of the true order from 6 to 2–4, and on the manuscript the n-gram units now agree on one family of orders — sheets 2 and 3 (f76/f83, f77/f82) together, then 5 (f79/f80), then 1 (f75/f84), then 4 (f78/f81) at the end (or the reverse; 4-1-2-3-5 is that family with 1 moved next to 4) — with the best-of-120 beating shuffled contents at p 0.005–0.05 in 14 of the 16 n-gram × metric × transcription cells (modal 8/8 at ≤ 0.035). Three cautions keep this at "a hint that has become a candidate": (i) the cells are not independent (n-grams of neighbouring n overlap, the two transcriptions are one text, the three metrics share the outliers), so the family-wise chance is well above 0.005; (ii) the word unit stays weak (p 0.1–0.4), and the cross-unit agreement did not sharpen; (iii) a seriation finds the best chain through sheet *affinities*, and affinities can come from subject matter shared by two sheets as well as from reading order — sheet 4 being the odd one out (§13 affinity matrix) is equally a topic statement. What would make it a result: the same optimum from an independent quire-internal signal (e.g. the words alone at p < 0.05), replication of the method's behaviour on quires T and A–C, and a check against the codicological literature on quire M's sheet order.

## 15. Replication on quires T, A, B, C; geometry null; nested-written control (2026-09-03)

Same script (`scripts/quire_order_burst.py`, v2), all five quires rerun in parallel with 200 content shuffles (`quire_order_burst_{M,T,A,B,C}.{json,log}`; the first M/A/B/C runs without the additions below are archived in `burst_v1/`). Two additions: a **geometry null** — the real page lengths stay in their slots and every occurrence of a rare type is re-placed uniformly over the tokens (200 draws; best-of-candidates recomputed) — which asks whether the winning order is favoured by sheet sizes alone; it is valid for L1 and modal (both are means of distances) but **not** for burst/blog, whose scale λ_t changes under uniform placement (their "p_uni" is ≈ 1 everywhere and is discarded). And a **nested-written control**: which stacked order does prose written as bound prefer? Answer: the current outer→inner order or a near variant (M 7/9, B 8/9, C 6/9, T 7/9 texts×windows under L1; 1-adjacent-to-innermost appears in 1/9 on T), so "outermost next to innermost" is not what nested writing produces.

Sheets (outer→inner): **T** 1 f103/f116 (3 pages), 2 f104/f115, 3 f105/f114, 4 f106/f113, 5 f107/f112, 6 f108/f111 — 23 pages, Currier B, hand 3, section S, 10 673 tokens, 720 orders (reversal pairs nearly equal, random minimum ≈ −2.9 sd). **A** 1 f1/f8, 2 f2/f7, 3 f3/f6, 4 f4/f5 (1 495 tokens); **B** 1 f9/f16, 2 f10/f15, 3 f11/f14, 4 f13 alone (1 019); **C** 1 f17/f24, 2 f18/f23, 3 f19/f22, 4 f20/f21 (1 401) — all Currier A, hand 1, herbal; 24 orders (reversal pairs nearly equal, random minimum ≈ −1.6 sd).

| quire | n-gram best order (n 5–8, IT2a / RF1b) | content-shuffle p (L1 · blog · modal) | z of best within candidates | geometry p (L1 · modal) | current order rank | words |
|---|---|---|---|---|---|---|
| **T** | **1-6-5-4-2-3** in 31/32 unit×metric cells (IT2a n5 L1/burst/modal: 6-1-5-4-2-3) | 0.005–0.02 on every unit and metric | −3.1 to −4.05 | 0.005 · 0.005–0.02 | 150–626 / 720 | 1-6-5-4-3-2 / 1-6-5-4-2-3, L1 p 0.005, modal 0.01–0.015, burst/blog 0.36–0.74 |
| M (§14) | 2-3-5-1-4 · 3-2-5-1-4 · 4-1-2-3-5 | 0.005–0.07 · 0.01–0.12 · 0.005–0.03 | −2.3 to −3.2 | 0.005–0.01 · 0.005–0.03 | 30–114 / 120 | 2-3-5-1-4, p 0.03–0.4 |
| C | **1-4-3-2** in 30/32 cells | 0.005–0.15 · 0.01–0.07 · 0.005–0.27 | −1.2 to −2.1 | 0.005 · 0.005–0.07 | 4–10 / 24 | 2-3-1-4 / 1-2-3-4 / 2-1-3-4, p 0.03–0.7 |
| A | split: 1-2-4-3 (L1, burst n5–7), 1-4-2-3 (n7–8), 3-1-2-4 (blog/modal n5–6) | 0.01–0.1 · 0.035–0.26 · 0.02–0.4 | −1.4 to −2.2 | 0.005 · 0.01–0.9 | 10–20 / 24 | 2-4-1-3, p 0.1–0.5 |
| B | 1-2-3-4 = current (n 6–8) | 0.18–0.65 · 0.06–0.56 · 0.11–0.75 | −1.3 to −2.4 | 0.005–0.17 · 0.005–0.68 | 1–4 / 24 | 1-4-3-2 etc., p 0.1–0.8 |

Prose power on each quire's slots (18 stacked-written cases; L1 / blog): rank-1 6/6 (M), 1/2 (T; median rank 10/6 among 720), 6/5 (A), 4/4 (B), 5/3 (C); top-10 in 13–15 of 18 on the four-to-five-sheet quires and 9–13 on T.

**Reading.** (i) **Quire T is unambiguous by this test**: every n-gram length, both transcriptions and all four metrics choose 1-6-5-4-2-3 (words: 1-6-5-4-3-2), it beats shuffled contents in ≤ 1 of 200 draws on every cell, beats sheet geometry likewise, sits 3–4 sd below the candidate mean where a random minimum of 360 would sit at 2.9, and the current outer→inner order is in the bottom half to bottom tenth. (ii) **Quire C** replicates the shape with a smaller quire: 1-4-3-2 on 30 of 32 cells, p 0.005–0.05 on most, geometry excluded. (iii) **Quire A** shows sharing above shuffled contents (L1 p 0.01–0.05, geometry excluded) but no single order; the current order is disfavoured. (iv) **Quire B** (three sheets and a single leaf, 1 000 tokens) prefers its current order but not above shuffled contents — no information. (v) The pattern across T, C and M: **the outermost sheet sits next to the innermost sheet, and the remaining sheets then run outward in descending order** — T 1-6-5-4-(3-2 words / 2-3 n-grams), C 1-4-3-2, M 4-1-5-3-2 (words, n8) with sheet 4 displaced to the front. Nested-written prose does not produce this (its best stacked chain is the bound order), and neither sheet geometry nor page-level clustering (the content shuffle keeps each page's own bursts) does. What could: composition sheet by sheet in that order; or a chain of sheet-level subject affinities that happens to run outermost → innermost → outward — the test cannot tell the two apart (§7.4), and direction is tested in §16 (an order and its sheet-reversal are *not* equivalent, as first stated here). Caveats: cells within a quire are correlated (overlapping n-grams, one text in two transcriptions, metrics sharing outliers); the herbal quires are 5–7× smaller than M/T and their candidate sets tiny (12 distinct values), so their p-values are coarse; the "descending" reading of M requires ignoring one displaced sheet. Quire T's result is the one to quote; C is a replication in a different language, hand and section; M is consistent; A and B are uninformative. Not run: the remaining multi-sheet quires (D–G, mixed language/hand; O, Q, S), the literature's specific reorderings as fixed candidates, and a check of T's sheet order against the codicology (f103/f116 is the sheet with the missing f116v text).

## 16. Direction: an order and its sheet-reversal are not equivalent (2026-09-03)

Correction to §13–15, raised by the user: reversing the sheet order does not reverse the page sequence, because every sheet is still read a-r, a-v, b-r, b-v. For a type at within-sheet position x in sheet A and y in sheet B, order AB costs (L_A − x) + y and BA costs (L_B − y) + x; the difference is the direction signal (shared types falling *late* in one sheet and *early* in the next favour that reading sense). §13–15 reported each best order as the smaller label of the pair and so discarded it. `scripts/quire_order_direction.py` (`quire_order_direction_{T,M,C,A,B}.{json,log}`, 200 content shuffles): the un-folded argmin, its reversal's cost and rank, the gap Δ = cost(reversal) − cost(best) in candidate sd, and a null for Δ taken from each shuffle's own best order and *its* reversal (same selection, so the selection bias that makes any best beat its reversal is in the null).

| quire | un-folded best (n 5–8, both transcriptions; L1 · blog · modal) | reversal's rank | Δ (sd) | null Δ (sd) | p |
|---|---|---|---|---|---|
| T | **3-2-4-5-6-1** in 23 of 24 cells (IT2a n5 L1/modal 3-2-4-5-1-6); words 2-3-4-5-6-1 / 3-2-4-5-6-1 | 5–27 / 720 | 0.6–1.7 | 0.8–1.4 ± 0.5–0.7 | 0.20–0.80 |
| M | inconsistent: 2-3-5-1-4, 4-1-5-2-3, 1-4-5-2-3, 1-4-2-3-5, 4-1-2-3-5 | 2–35 / 120 | 0.1–2.1 | 0.8–1.5 | 0.07–0.98 |
| C | 1-4-3-2 (n 6–8), 2-3-4-1 (n 5) | 2–6 / 24 | 0.05–1.1 | 1.1–1.6 | 0.66–1.00 |
| A | 3-1-2-4 / 1-2-4-3 / 1-4-2-3, split | 2–8 / 24 | 0.0–0.9 | 1.0–1.5 | 0.55–1.00 |
| B | 1-2-3-4 and 4-3-2-1 both appear | 2–9 / 24 | 0.0–1.5 | 1.0–1.3 | 0.38–1.00 |

**Reading.** No quire's direction is established: every gap between the best order and its reversal is the size that selecting the best of the candidate set produces on shuffled contents (p ≥ 0.07, mostly ≥ 0.3). Quire T's direction is at least *consistent* — 3-2-4-5-6-1 on 23 of 24 cells, i.e. f105/f114, f104/f115, f106/f113, f107/f112, f108/f111 and the outermost f103/f116 **last** — but the cells are correlated and the gap is ordinary; note that "outermost sheet holds the end" is also where a text written as bound ends (f116). What §13–15 established is adjacency (which sheets neighbour which); the sense of reading along the chain lives only in whether shared types fall early or late within a sheet, a small part of the between-sheet distance, and is below this test's resolution at one quire. §13–15's "≡ reversal" and "60 / 360 / 12 distinct values" statements are corrected in place; their adjacency results are unaffected (reversal pairs differ by ≤ 2 sd, and the two members were always ranked together).

## 17. Shape of the nulls behind §13–16 (2026-09-03)

Asked whether the "± sd" summaries and the "random minimum of N candidates ≈ −2.5 sd" heuristic hide a Gaussian assumption. `scripts/quire_order_nullshape.py` (`quire_order_nullshape_{T,M,C}.{json,log}`, IT2a, 200 content shuffles, L1 and modal):

- **Candidate cost vector (all stacked orders).** On shuffled contents it is near-normal but mildly flat: skew ≈ 0, excess kurtosis −0.3 to −0.8; Shapiro–Wilk rejects at 5 % in 6–34 % of shuffles for M (120 orders) and C (24), and in 80–92 % for T (720 orders, where the test has power against the flatness). On the **real** contents T's vector is clearly non-normal (skew −0.4 to −1.0, kurtosis up to +0.8 on n 6–7): a left tail of low-cost orders — the winning family itself — which is the signal, not a null defect. M's real vectors are mildly left-skewed (−0.1 to −0.6), C's mildly right-skewed.
- **Winner's within-candidate z, empirically.** Median z of the shuffled winner: M −2.26 to −2.42 (Gaussian heuristic −2.40), C −1.80 to −1.91 (heuristic −1.75), T −2.65 to −2.98 (heuristic −2.99). The heuristic was adequate. Empirical p of the real winner's z against the shuffled winners' z: **T n 6–8 p 0.005–0.015** (both metrics), n 5 0.045 / 0.16, words 0.13–0.15; M modal n 5 0.005, n 6 0.07, the rest 0.16–0.51; C 0.50–0.97. These match the §14–15 p_best values, which were already empirical.
- **Direction gap Δ.** Bounded at zero and right-skewed (skew +0.2 to +1.3; median 0.7–1.4 sd, upper quartile 1.0–2.0, maxima 2.2–4.0), so "mean ± sd" was a poor summary; the p-values in §16 were empirical and stand. The real Δ falls inside the null interquartile range on every T cell (p 0.20–0.83) and on C (p 0.48–0.99); M's only low cell is n 7 L1 at p 0.085.

Net: nothing in §13–16 rests on normality. The candidate distribution's departure from normal in T is the finding; the gap distribution's departure is why only the empirical p was ever quoted.

## 18. Burst front-loading: is there a time arrow inside a burst? (2026-09-03)

Proposed as the only route to reading direction that does not depend on the sliver of within-sheet position used in §16: if a rare type's burst is front-loaded (introduced, reused quickly, tails off), a burst straddling a sheet boundary tells which sheet came first from its *shape*. `scripts/burst_frontloading.py` (`burst_frontloading.{json,log}`): for every type with 3–10 occurrences inside one segment, the fraction of types whose first inter-occurrence gap is shorter than the last (front-loaded → > 0.5; sign test) and the skew (mean position − midpoint of first/last)/span (front-loaded → negative; t-test). Segments: 1 500-token windows of six known texts (first 60 000 tokens each; words and character 5–8-grams, spaces stripped); the manuscript's 50 sheets of ≥ 3 pages read a-r, a-v, b-r, b-v (direction known, no order search), by Currier language, and its pages. Control `burst_frontloading_control.py` (`burst_frontloading_control.log`): the four pages of every four-page sheet put in a random order (200 draws), keeping every page-level effect and destroying only the reading order.

**Prose has no usable arrow.** Pooled over six texts: words first<last 0.503 (n 19 551), n 5–8 0.502–0.508; skew −0.0004 to −0.0011. By text the fraction ranges 0.48–0.53 with signs in both directions (Seneca and Isidore n-grams faintly front-loaded, Bullinger and Decameron faintly the other way). The only trace is long n-grams with ≥ 5 occurrences (0.52–0.55, skew −0.004 to −0.010), a few per cent. A burst is, to within a per cent, symmetric in time; the premise fails, so the shape of a straddling burst cannot give direction, in prose or anywhere.

**Manuscript.** All sheets pooled: words 0.49–0.51, n-grams 0.49–0.50 (skew +0.002 to +0.003, p 0.004–0.24 on RF1b); pages alone 0.48–0.51: symmetric. Currier B sheets: symmetric on every unit (0.49–0.50, permutation p 0.17–0.61). **Currier A sheets, glyph n-grams only**: back-loaded — first<last 0.46 / 0.47 / 0.46 (n 6/7/8, IT2a) and 0.46 / 0.46 / 0.44 (RF1b), skew +0.004 to +0.014, driven by k = 3 types (0.44–0.49); against the within-sheet page-permutation null the real a-r, a-v, b-r, b-v order sits at p 0.005 / 0.025 / 0.10 (IT2a n 6/7/8) and 0.025 / 0.045 / 0.085 (RF1b); words show nothing (0.52–0.55, p 0.5–0.9). Page length by position in A sheets falls 104 → 95 → 87 → 82 tokens, but the permutation null carries page lengths with the pages and still gives 0.50, so length is not the cause.

**Reading.** The direction-by-burst-shape route is closed: prose bursts are symmetric to within a per cent, so there is no arrow to calibrate on, and the manuscript's bursts are symmetric too in Currier B and at page level. The one asymmetry — Currier A herbal sheets, n-grams, k = 3, p 0.005–0.1 on six correlated cells, both transcriptions — is not the mirror of any prose property (prose k = 3 is 0.50 exactly) and so is not evidence that A sheets are read backwards; its shape (middle occurrence nearer the last) is what a *leaf-level* difference would produce — the two pages of leaf b (the later folio) sharing rare n-grams with each other more than the two pages of leaf a do — which is a layout or production question (herbal a-recto pages carry the large drawings), not a reading-direction one. Worth one follow-up check (within-leaf sharing, leaf a vs leaf b, herbal quires) before any interpretation; not run. Direction along the sheet chain (§16) therefore stays undetermined at this sample size, and adjacency (§13–15) remains the deliverable.
