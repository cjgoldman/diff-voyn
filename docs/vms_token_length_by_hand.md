# VMS token length by scribal hand — is length a scribal choice?

**Date:** 2026-08-21 (length distributions; extended the same day with the
transition FSM §5b and the positive control §5c) · **Status:** side-quest,
complete · companion to
`docs/vms_doubling_rate.md`. **Not on the Phase 0–6 critical path**, but it
constrains rung 4 (CH.8) length calibration and Phase 6 per-dialect scoring.

**Artifacts**

| what | where |
|---|---|
| measurement script | `scripts/token_length_by_hand.py` (reuses the IVTFF `$H/$C/$L` parser in `scripts/doubling_rate.py`) |
| positive control (non-length statistics by hand) | `scripts/hand_positive_control.py` → `data/analysis/token_length/hand_positive_control.md/.json` |
| length-transition FSM | `scripts/length_transitions.py` (`--measure eva_collapsed|eva_glyphs|eva_chars|boxer_glyphs`) → `data/analysis/token_length/length_transitions_*.md/.json` |
| full tables + JSON + figure | `data/analysis/token_length/token_length_report.md`, `token_length_results.json`, `token_length_by_hand.png` |

**TL;DR.** Token length varies between Davis hands, but the variation is the
Currier A/B split: H1 (A) is 0.2–0.4 glyphs shorter than every B hand; the
two large B hands agree to −0.02 glyphs once `iin`/`ee` are counted as
single units (§4). A first-order length FSM is weak (≈0.015 nats/token),
first-order only, twice as strong in B as in A, and shared across hands
within a dialect (§5b). The positive control is decisive: across Currier B,
H2 and H3 differ at z ≈ 7–13 on every other glyph-level statistic (glyph
inventory, word-initial/final glyphs, bigrams, line layout, vocabulary) and
not at all on unit length (z = 0.8) (§5c). Length behaves like a property
of a shared system or of the underlying text, not a choice made by the
scribe at the time of writing. Rung 4 should use a per-dialect length mix
(§7).

## 1. The question

Boxer's arithmetic cipher makes the length of a Voynich word arbitrary: every
plaintext letter has homophones of 2–6 glyphs (his encoder draws from a
global length mix 10/22/26/26/16 % for L2..L6,
`voynpy/pseudo_vms/encoder.py::DEFAULT_LENGTH_DISTRIBUTION`, pinned @
`e324bee`) and the scribe picks one per letter. If that choice is really free,
it is natural to expect it to vary with the scribe. Does the token-length
distribution vary with Lisa Fagin Davis' hands (IVTFF `$H`)?

## 2. What "length in glyphs" means — four segmentations

The answer turns out to depend on this, so all four are reported:

| measure | source | unit |
|---|---|---|
| `boxer_glyphs` | Boxer's `transcription/vms.csv` | his own glyph alphabet, already comma-segmented (`cc`=ch, `c^c`=sh, `c`=e, `m`/`n` ≈ the iin/in ligatures) |
| `eva_glyphs` | Takahashi IT2a, Reference RF1b | EVA with `ch`, `sh`, `cth/ckh/cph/cfh` as one glyph each |
| `eva_chars` | IT2a | raw EVA characters |
| `eva_collapsed` | IT2a | `eva_glyphs` with every `i`-run (+ closing stroke) and every `e`-run counted once — the coarsest defensible segmentation |

Policy (frozen in the script): paragraph text only, uncertain tokens dropped,
hands from IT2a page headers (Boxer's pages mapped as in the doubling
analysis). **Pages are the clustering unit** throughout: CIs are page
bootstraps and p-values are page-label permutation tests (4000 permutations).
Token-level χ² p-values are in the generated report for reference only — with
~10k tokens per hand they are all < 1e-10 and meaningless.

Hand is confounded with Currier language and section: H1 is entirely
Currier A (herbal + pharma); H2, H3, H5 are Currier B (H2: biological +
herbal-B + cosmological; H3: stars/recipes + 7 herbal-B pages; H5: 6 herbal-B
pages + f66r); H4 is the 9 astronomical pages without a Currier label.
The clean comparisons are therefore *within Currier B* (H2 / H3 / H5) and
*within the Herbal section* (H1/A vs H2, H3, H5 /B), which the script reports
separately.

## 3. Per-hand distributions

Takahashi IT2a, `eva_glyphs` (mean with page-bootstrap 95 %; "long/short" =
P(len ≥ 6) / P(len ≤ 3)):

| hand | pages | tokens | mean | P(≤3) | P(≥6) | long/short | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8+ |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (A) | 112 | 9802 | 4.39 (4.32–4.46) | 0.34 | 0.22 | 0.66 | 2.8 | 9.2 | 21.9 | 21.0 | 22.8 | 11.5 | 5.6 | 5.1 |
| 2 (B) | 46 | 10546 | 4.58 (4.51–4.65) | 0.25 | 0.28 | 1.09 | 1.0 | 7.9 | 16.4 | 24.5 | 22.8 | 16.6 | 8.1 | 2.8 |
| 3 (B) | 32 | 11970 | 4.83 (4.75–4.91) | 0.20 | 0.34 | 1.68 | 0.9 | 6.4 | 12.8 | 22.3 | 23.9 | 19.6 | 10.2 | 4.0 |
| 4 (–) | 9 | 485 | 4.48 (4.14–4.83) | 0.31 | 0.26 | 0.84 | 2.3 | 8.0 | 20.6 | 21.0 | 22.1 | 15.5 | 6.2 | 4.3 |
| 5 (B) | 7 | 838 | 4.82 (4.70–5.07) | 0.22 | 0.31 | 1.43 | 1.9 | 5.6 | 14.1 | 22.2 | 25.4 | 16.7 | 7.8 | 6.4 |

Means under the other segmentations (same pages):

| hand | boxer_glyphs | eva_chars | eva_collapsed | RF1b eva_glyphs |
|---|---:|---:|---:|---:|
| 1 | 4.17 | 5.08 | 3.80 | 4.37 |
| 2 | 4.42 | 5.06 | 4.02 | 4.46 |
| 3 | 4.51 | 5.30 | 4.05 | 4.71 |
| 4 | – | 5.00 | 3.91 | 4.45 |
| 5 | 4.58 | 5.38 | 4.43 | 4.84 |

Boxer's L2..L6 mix, renormalised per hand from his own transcription:

| | L2 | L3 | L4 | L5 | L6 |
|---|---:|---:|---:|---:|---:|
| encoder default | 10 | 22 | 26 | 26 | 16 |
| H1 (A) | 10 | **29** | 25 | 23 | 13 |
| H2 (B) | 10 | 20 | 27 | 27 | 16 |
| H3 (B) | 10 | 17 | 26 | 28 | 18 |
| H5 (B) | 6 | 17 | 27 | 30 | 20 |

His hard-coded mix is a Currier-B (essentially H2) calibration. Currier A has
a pronounced 3-glyph mode (26 % vs 15–18 % — `daiin`, `chol`, `chor`, `shol`
are 3 Boxer glyphs) that the default mix does not reproduce.

## 4. Tests

### 4.1 Across all hands — significant, but it is the A/B split

Page-permutation p < 0.001 for every segmentation and both transliterations.
Every pairwise contrast involving H1 against a B hand is significant
(Δmean −0.19 to −0.63 glyphs); H4 (astronomical, 485 tokens) is
indistinguishable from H1.

### 4.2 Within Currier B — H2 vs H3 depends entirely on how `iin`/`ee` are counted

| segmentation | Δmean(H2 − H3) | page-perm p |
|---|---:|---:|
| eva_chars (IT2a) | −0.25 | 0.001 |
| eva_glyphs (IT2a) | −0.25 | < 0.001 |
| eva_glyphs (RF1b) | −0.25 | < 0.001 |
| boxer_glyphs | −0.09 | 0.12 |
| **eva_collapsed** | **−0.02** | **0.75** |

Decomposing the EVA difference by word type: words in `-aiin/-ain` contribute
−0.27 and `-ey/-eey` −0.12 (H3-favoured: `aiin`, `okeey`, `qokeey`, `otaiin`,
`qokaiin`), words in `-edy/-eedy` +0.36 (H2-favoured: `qokedy`, `shedy`,
`chedy`, `qokain`). Collapsing i-runs alone takes Δ from −0.23 to −0.08;
collapsing i- and e-runs takes it to −0.01. In other words the two large B
hands differ only in how many strokes their words contain, not in how many
glyph *units* — and under Boxer's arithmetic the unit inventory is exactly the
thing that is undetermined. Under either of the two segmentations closest to
his (his own, and the collapsed EVA) H2 and H3 have the same length
distribution. H1–H3 survives every collapse (−0.41 → −0.23 → −0.25 under
`eva_collapsed`, p ≤ 0.001).

### 4.3 Within the Herbal section (same subject matter, both languages)

H1/A vs H2/B: Δ −0.18 (p = 0.04, `eva_glyphs`); H1 vs H3: −0.26 (p = 0.10, 7
pages); H2 vs H3 within Herbal-B: Δ −0.08, p = 0.50 (`eva_collapsed` −0.00,
p = 0.98). Same picture: language, not hand.

### 4.4 Hand 5 — longer tokens, but also a different glyph inventory (§5c)

H5 (f41r/v, f48r/v, f57r, f66r/v — 6 herbal-B pages + one text page) writes
longer tokens than H2 and H3 *in the same section and language*:

| comparison (Herbal-B only) | eva_glyphs IT2a | eva_glyphs RF1b | eva_collapsed |
|---|---|---|---|
| H2 vs H5 | −0.38 (p = 0.006) | −0.48 (p = 0.001) | −0.58 (p = 0.002) |
| H3 vs H5 | −0.31 (p = 0.04) | – | −0.61 (p = 0.003) |

All six H5 herbal pages have page-mean collapsed length 4.31–5.27 against an
H2 herbal-B median of 3.95, so it is not one outlier page. Caveats: six
pages; in Boxer's own segmentation the H5 contrast is *not* significant
(Δ −0.16 / −0.07, p = 0.14 / 0.45, all of Currier B); and the H5 vocabulary
(`qokeody`, `sheody`, `cheky`, `okedy`) is itself atypical, so "content" and
"hand" are not separable on seven pages. The positive control (§5c) settles
how to read it: H5 differs from H2/H3 just as strongly in glyph unigrams,
word-final glyphs and bigrams as in length — it is different text, not a
length-only scribal preference.

### 4.5 How much of the page-to-page spread is hand at all?

Token-weighted between-page variance of page mean length, R² by grouping
(IT2a `eva_glyphs` / `eva_collapsed` / Boxer):

| explained by | R² |
|---|---|
| Currier language | 0.16 / 0.10 / 0.18 |
| Davis hand | 0.26 / 0.17 / 0.22 |
| hand beyond language | 0.11 / 0.08 / 0.04 |

Between-page SD of page means is 0.31–0.35 glyphs. 75–85 % of the
page-to-page variation in mean token length is *within* a hand. The
hand-beyond-language share (4–11 %) is mostly H5.

### 4.6 Transliteration noise floor

Same pages, IT2a − RF1b mean `eva_glyphs`: H1 +0.02, H2 +0.12, H3 +0.12,
H4 +0.03, H5 −0.02. Absolute levels move by ≈0.1 glyph between
transliterations (RF1b reads fewer `e`/`i` strokes in B text), so contrasts
must be made within one transliteration — which they are above; the H2–H3 gap
is −0.25 in both.

## 5. A second constraint: adjacent token lengths are not independent

Boxer's encoder draws each token's homophone (hence its length) independently
of the previous one, except for doublings (9/1000 pairs, worth ≤ 0.01 in r).
Lag-1 correlation of interior token lengths within lines, against a
within-line shuffle null (which also absorbs the between-line mean
differences that make the null positive), and after subtracting the hand's
mean length at each relative line position (10 bins):

| hand | r | shuffle null | z | r detrended | z detrended |
|---|---:|---|---:|---:|---:|
| 1 | +0.072 | +0.031 ± 0.011 | +3.9 | +0.066 | +2.9 |
| 2 | +0.117 | +0.055 ± 0.010 | +6.3 | +0.112 | +6.5 |
| 3 | +0.145 | +0.063 ± 0.009 | +9.3 | +0.141 | +9.4 |
| 5 | +0.072 | +0.034 ± 0.033 | +1.2 | +0.055 | +0.9 |

(IT2a `eva_glyphs`; the same pattern, z = 3–13, holds for all four
segmentations and RF1b.) Adjacent words are ~0.04–0.09 more length-correlated
than an i.i.d. choice allows, in every well-sampled hand, more strongly in B
than in A, and it is not a line-position gradient. Natural-language word
length has this property; an independent per-letter homophone draw does not.
The model as written predicts r = null; it would need a "sticky" length
preference (cf. the sticky-reuse open item in the doubling note) or
length-dependent table structure to produce this. Synthetic rung-4 text from
the pinned encoder will have r ≈ null — an easy tell against the real
manuscript.

## 5b. A first-order length "finite-state machine" per hand / language

`scripts/length_transitions.py` (reports `data/analysis/token_length/length_transitions_<src>_<measure>.md`)
fits P(next length | this length) over 7 states (1…6, 7+) from adjacent
within-line pairs. To separate "the matrices differ" (guaranteed, because
the marginals differ A/B) from "the *dependence structure* differs", four
nested models are compared per group set: **M0** per-group marginal only;
**M1pooled** one shared matrix; **M1s** per-group marginal × shared lift
(log-linear model without the group×this×next interaction, fitted by IPF);
**M1g** per-group full matrix; **M2g** per-group second-order. Scored by
page-grouped 5-fold held-out log-likelihood per token (nats, Δ vs M0) and a
page-permutation LRT of M1g vs M1s (1000 permutations).

**The FSM itself (Currier B, `eva_collapsed`, lift = P(next|this)/P(next)):**

| this \ next | 1 | 2 | 3 | 4 | 5 | 6 | 7+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.60 | 1.30 | 1.22 | 0.97 | 0.73 | 0.47 | 0.58 |
| 2 | 2.19 | 1.36 | 1.24 | 0.99 | 0.73 | 0.49 | 0.49 |
| 3 | 1.33 | 1.28 | 1.11 | 1.01 | 0.84 | 0.76 | 0.74 |
| 4 | 0.75 | 0.95 | 1.01 | 1.03 | 1.04 | 0.99 | 0.89 |
| 5 | 0.59 | 0.80 | 0.85 | 1.03 | 1.13 | 1.21 | 1.29 |
| 6 | 0.34 | 0.67 | 0.82 | 0.93 | 1.21 | 1.54 | 1.42 |
| 7+ | 0.38 | 0.66 | 0.81 | 0.88 | 1.19 | 1.47 | 2.06 |

A smooth, monotone "persistence" structure: short follows short (1→1 lift
2.6, and 2.2 for 2→1), long follows long (7+→7+ lift 2.1), with the middle
lengths near 1. Currier A has the same shape but flatter (1→1 lift 3.6 from
only 311 pairs, 6→6 1.3, 7+→7+ 1.4). Mutual information I(L_t; L_t+1):

| | Currier A | Currier B | H1 | H2 | H3 | H5 |
|---|---:|---:|---:|---:|---:|---:|
| `eva_collapsed` (bits) | 0.015 | 0.035 | 0.015 | 0.041 | 0.032 | 0.037 |
| `boxer_glyphs` (bits) | 0.022 | 0.032 | 0.022 | 0.035 | 0.033 | 0.044 |
| `eva_glyphs` (bits) | 0.009 | 0.017 | 0.009 | 0.018 | 0.017 | 0.050 |

**Model comparison** (`eva_collapsed`; other segmentations in the generated
reports agree on every conclusion):

| comparison | pairs | M1pooled | M1s | M1g | M2g | LRT M1g vs M1s page-perm p (collapsed / eva / Boxer) |
|---|---:|---:|---:|---:|---:|---|
| Currier A vs B | 25092 | +0.007 | +0.015 | **+0.016** | +0.007 | **0.001 / 0.002 / 0.001** |
| All hands | 25431 | +0.005 | **+0.014** | +0.011 | −0.008 | 0.001 / 0.011 / 0.002 |
| Within B: H2 / H3 / H5 | 17664 | +0.020 | **+0.020** | +0.018 | +0.001 | 0.34 / 0.23 / 0.043 |
| Within B: H2 / H3 | 17014 | +0.021 | **+0.021** | +0.020 | +0.004 | 0.10 / 0.44 / 0.049 |

Reading it:

- **Fitting the FSM is easy and a first-order chain is enough.** Order 2
  never improves held-out likelihood (M2g ≤ M1g everywhere), so there is no
  deeper length grammar to find. The dependence is real but weak: the
  first-order model gains 0.01–0.02 nats/token (≈1 % of the ≈1.77-nat
  length entropy), consistent with the lag-1 autocorrelation in §5.
- **Currier B has roughly twice the length-dependence of A** (MI 0.035 vs
  0.015 bits collapsed; 0.032 vs 0.022 in Boxer's units; H1 = A matches A
  exactly, H2 ≈ H3 ≈ B). The A/B transition structure differs beyond the
  marginals (page-permutation p ≤ 0.008 in all three segmentations), but the
  practical gain of separate A/B matrices over "own marginal × shared lift"
  is ≤ 0.001 nats — almost all of the A/B difference is the marginal (§3–4),
  with a flatter persistence pattern in A on top.
- **Within a language, hands do not have their own transition structure.**
  H2 vs H3 (≈17k pairs): p = 0.10 collapsed, 0.44 EVA glyphs, 0.049 in
  Boxer's units (not significant after the three-way multiplicity), and
  held-out likelihood *prefers the shared-lift model to per-hand matrices*
  in every segmentation — per-hand matrices overfit. The per-row
  page-permutation tests for H2 vs H3 give p = 0.36/0.17/0.013/0.039/0.13/
  0.50/0.67 for this-length 1…7+: nothing that survives seven rows. H5's
  higher MI rests on 744 pairs.

So the length FSM is a property of the dialect, not the scribe: one lift
matrix (shared) × a per-dialect marginal describes every hand as well as
anything fitted per hand. That is the same verdict as §4 and as the
doubling rate. For rung 4, the encoder could reproduce it with a
length-persistent homophone choice (e.g. a two-state short/long mood with
per-dialect stickiness) — but note that natural-language word length has
exactly this persistence already, which is the more parsimonious source.

## 5c. Positive control: is hand visible in anything *other* than length?

`scripts/hand_positive_control.py` (report `data/analysis/token_length/hand_positive_control.md`)
runs the same page-permutation test on a battery of non-length statistics —
glyph unigrams, gallows rate, word-initial and word-final glyph, glyph
bigrams (top 60), line-initial glyph, tokens per line, word types (top 60) —
with unit length (`len_units`, collapsed) in the same table. Effect size is
the token-weighted Jensen–Shannon divergence between hands (bits); z is
against the page-permutation null, so category count is accounted for.

| comparison | len_units | glyph unigram | word-initial | word-final | bigrams | line-initial | tokens/line | word types |
|---|---|---|---|---|---|---|---|---|
| **All-B, H2 vs H3** (46 / 29 pages) | z +0.8, **p = 0.17** | z +8.8, p < .001 | z +10.0, p < .001 | z +9.2, p < .001 | z +10.7, p < .001 | z +10.3, p < .001 | z +7.4, p = .001 | z +12.9, p < .001 |
| Herbal-B, H2 vs H3 (20 / 7) | z +1.1, p = 0.13 | z +0.8, p = 0.17 | z +0.6, p = 0.23 | z +0.5, p = 0.25 | z +1.6, p = 0.08 | z −0.5, p = 0.65 | z +1.0, p = 0.15 | z +1.6, p = 0.07 |
| Herbal-B, H2 vs H5 (20 / 6) | z +5.3, p < .001 | z +4.1, p = .006 | z 0.0, p = 0.42 | z +3.8, p = .004 | z +4.2, p = .004 | z +0.8, p = 0.21 | z 0.0, p = 0.42 | z +1.7, p = 0.06 |
| Herbal-B, H3 vs H5 (7 / 6) | z +5.4, p = .002 | z +5.8, p = .002 | z +1.4, p = 0.10 | z +5.2, p = .002 | z +5.7, p = .002 | z +2.2, p = .02 | z +1.1, p = 0.13 | z +2.6, p = .01 |
| reference: Herbal H1/A vs H2/B | z +9.2, p < .001 | z +26.8 | z +28.4 | z +16.9 | z +44.6 | z +12.4 | z +24.0 | z +41.6 |

Four things fall out:

1. **The decisive row is All-B H2 vs H3.** Across all of Currier B the two
   scribes' text differs on *every* non-length statistic at z ≈ 7–13 —
   different glyph frequencies (`y` 12.1 vs 9.3 %, `d` 9.0 vs 7.1 %, `i`/`a`
   the other way), different word-initial (`d`, `a`, `ch`, `l`) and
   word-final (`y` 49 vs 41 %, `n` 15 vs 20 %) glyphs, different bigrams,
   different line-initial glyphs (`q` 19 vs 6 %), different line lengths,
   different vocabulary — yet **unit token length does not differ (z = 0.8,
   p = 0.17)**. Whether the H2/H3 difference is scribe or section (it is
   both: biological vs stars), whatever varies between these two bodies of
   text changes everything about the glyph stream except how many units a
   word has.
2. **The length statistic has power** — it separates Herbal A from Herbal B
   at z = 9 on 2201 B tokens — so the All-B null is not a weak test.
3. **Herbal-B H2 vs H3 (the "cleanest" comparison) is uninformative**: with
   7 H3 pages nothing differs, length included. Their length agreement in
   §4.2–4.3 carries no weight on its own; the All-B row does the work.
4. **H5 is different text, not a length preference.** Its length difference
   from H2 and H3 (§4.4) comes with equally large differences in glyph
   unigrams (`e` 14.9 vs 7–10 %, `i` 3.3 vs 7–9 %), word-final glyphs and
   bigrams (z ≈ 4–6). Wherever unit length differs in the manuscript, the
   glyph inventory differs with it; there is no case of a length-only
   signature anywhere.

So the positive control sharpens the verdict: hand (or hand-plus-section) is
plainly visible in the text — in which glyphs are written, where words start
and end, how lines are laid out — and invisible in unit length. If length
were a free per-token choice in the scribe's hands, it is hard to see why it
would be the *one* statistic two otherwise different scribes agree on.

## 6. Verdict

1. **Token length varies between hands, but the variation is the Currier A/B
   split, not the scribe.** H1 (A) is shorter than every B hand under every
   segmentation (Δ ≈ 0.2–0.4 glyphs, p ≤ 0.001, also within the Herbal
   section). The two large B hands, H2 and H3, differ in EVA stroke counts
   only because H3 favours `-aiin`/`-eey` forms and H2 `-edy` forms; counted
   in glyph *units* (Boxer's own, or collapsed EVA) they are identical
   (Δ = −0.02, p = 0.75) — and so is their length-transition structure
   (§5b) — while the same two hands differ at z ≈ 7–13 on every other
   glyph-level statistic (§5c). That is the same verdict the doubling rate
   gave: shared, not personal.
2. Against Boxer's framing: if length were a free scribal choice, the two
   best-sampled scribes writing the same dialect should be the place to see
   it, and they don't — while the same two scribes differ at z ≈ 10 on
   every other glyph-level statistic (§5c). The data are consistent with
   his model only if all scribes share one length preference (his global
   mix) — which is indistinguishable from length being fixed by the
   table/text rather than chosen. Hand 5's length difference (§4.4) comes
   with equally large glyph-inventory differences, so it is different text,
   not a length preference.
3. The A/B difference is large in his units (3-glyph share 26 % vs 15–18 %).
   Under his model that requires either different tables, different plaintext
   (language/register), or an A-scribe length preference — all of which are
   things the project's Phase-6 A/B-unpooled scoring is designed to
   distinguish, and none of which his single global length mix captures.
4. Adjacent-length autocorrelation (§5) is a property of the real text that
   the encoder lacks. Fitted as a first-order FSM (§5b) it is weak (≈0.015
   nats/token), first-order only, about twice as strong in B as in A, and —
   again — shared across hands within a dialect.

## 7. Consequences for the project

- **Rung 4 (CH.8):** Boxer's `DEFAULT_LENGTH_DISTRIBUTION` is a Currier-B
  calibration. If the arithmetic head's synthetic training/validation text is
  meant to mimic A as well as B, use a per-dialect mix (A ≈ 10/29/25/23/13,
  B ≈ 10/19/27/27/17 in his units; table above). Lengths are measured on his
  own transcription so they drop straight into `length_distribution=`.
- **Phase 6:** token length, like doubling, is stable across hands within a
  dialect, so pooling pages within Currier A or within Currier B for scoring
  is safe; A and B stay unpooled (design §9). If per-hand scoring is ever
  wanted, H5's pages (f41, f48, f57r, f66) differ from the rest of B in glyph
  inventory as well as length (§5c) and are the one B sub-block worth
  scoring separately as a check.
- **Hand is visible in the glyph stream** (§5c: word-initial/final glyphs,
  line-initial glyphs, tokens per line differ between H2 and H3 at z ≈ 10).
  Any Phase-6 statistic that is *not* length-like should therefore not
  assume H2 and H3 are exchangeable even within Currier B; the doubling
  rate and unit length are the exceptions, not the rule.
- **Glyph segmentation is load-bearing** for any length-based statistic on
  the VMS: the same pages give a "significant hand effect" in EVA and none
  in glyph units. Anything in Phase 5/6 that counts glyphs (the 2N-slot frame,
  NULL blending rates, the arithmetic head's slot count) should state its
  segmentation explicitly; the four variants here are implemented in
  `token_length_by_hand.py` (`eva_glyph_len`, `eva_collapsed_len`).
- **No Phase-0 artifacts touched.** Reads the pinned Boxer repo and the
  voynich.nu transliterations; writes only under `data/analysis/token_length/`.

## 8. Open items

- Repeat §5 on the rung-4 synthetic corpus once generated, to confirm r ≈ null
  there (a cheap real-vs-synthetic discriminator). Done for Naibbe (stock
  and sticky-on-repeat, Latin/Italian, 130k tokens, EVA chars): lag-1
  r = −0.03 to −0.05, MI 0.0015–0.0027 bits, against VMS r = +0.10 (A) /
  +0.18 (B), MI 0.010 / 0.027 bits. Naibbe's i.i.d. 1-or-2-letter respacing
  (unit-size lag-1 r = +0.002) produces no length persistence at all.
- The `eva_collapsed` rule is one choice of ligature inventory; a proper
  Currier/v101-style glyph transliteration would settle the H2/H3 question
  without regex heuristics. RF1b's `@nnn;` rare-glyph codes were dropped as
  uncertain, which removes ~10 % of RF1b tokens, mostly in B.
- H5's distinctness is now known to extend to the glyph inventory (§5c),
  which removes it as a length-preference candidate; whether those pages are
  a different scribe, a different source text, or both is a palaeographic
  question this data cannot answer.
- The H2-vs-H3 positive control is strongest across all of Currier B, where
  section (biological vs stars) is confounded with hand. The within-section
  version (Herbal-B) has only 7 H3 pages and differs on nothing. A
  same-section, same-dialect pair with ≥ 20 pages each does not exist in the
  manuscript, so "scribe vs section" for the non-length statistics stays
  open; the length conclusion does not depend on resolving it.
