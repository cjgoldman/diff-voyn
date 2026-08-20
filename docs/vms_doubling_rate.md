# VMS token doubling rate — verification, scribal-hand breakdown, and what it says about Boxer's cipher

**Date:** 2026-08-20 · **Status:** side-quest, complete · **Not on the Phase 0–6 critical path**, but it constrains rung 4 (CH.8) and Phase 6 VMS scoring.

**Artifacts**

| what | where |
|---|---|
| measurement script (rates, hand/dialect/section breakdowns) | `scripts/doubling_rate.py` |
| geminate-collapse script | `scripts/doubling_collapse.py` |
| full tables | `data/analysis/doubling/doubling_report.md`, `collapse_report.md` (+ `.json`) |

## 1. The question

Boxer's pseudo-VMS encoder hard-codes a Voynich "doubling rate" of 0.0092
(`voynpy/pseudo_vms/encoder.py::tune_to_vms`, pinned @ `e324bee`), i.e. ~9.2 per
1000 — the fraction of adjacent token pairs that are the *same* token. The
pinned repo contains no measurement code, only this consumer, so the figure was
taken on trust. Two questions:

1. Does 9.1–9.2/1000 reproduce on the actual transliterations?
2. Does the rate vary with the scribal hand?

## 2. Definition and policy sensitivity

Boxer's definition: `#{i : tok[i] == tok[i+1]} / (N−1)` over whitespace-split
tokens, counted straight through line breaks, uncertain glyphs left in place.
Measured on three transliterations (paragraph text `$P`; `scripts/doubling_rate.py`
reports the full policy cross-product):

| source | Boxer-style (across lines, uncertain kept) | within-line, uncertain dropped |
|---|---:|---:|
| Boxer's own `transcription/vms.csv` | **9.29** (8.3–10.4) | 10.36 |
| Takahashi IT2a (EVA) | 8.42 (7.5–9.5) | 9.31 (8.3–10.5) |
| Reference RF1b (EVA) | 6.59 (5.8–7.5) | 8.64 (7.6–9.8) |

**Verdict: confirmed.** His counting convention on his own transcription gives
9.29/1000 ≈ 0.0092. The honest range across transliterations and policies is
**6.6–10.4/1000**; RF1b sits lowest because its `@nnn;` rare-glyph codes break
token equality. Quote the figure with its transcription, not as a bare constant.

Parsing notes (frozen in the script): Davis' hands come from IVTFF page variable
`$H`, Currier's old hands from `$C`, Currier language from `$L`; f115r switches
hand mid-page via a `<@H=n>` text tag and is handled; Boxer's folio ids map to
IVTFF sub-pages (`89ra` → `f89r1`, and his `f85r/f85v/f86r` are the rosettes
foldout = IVTFF `f85r1/f85r2/f86v3-6`, all `$H=2 $L=B`).

## 3. Variation by scribal hand

Takahashi IT2a, paragraph text, within-line pairs, uncertain tokens dropped:

| Davis hand | pages | pairs | doubles | per 1000 | page-bootstrap 95% | top doubled types |
|---|---:|---:|---:|---:|---|---|
| 1 | 112 | 8325 | 89 | 10.69 | 8.6–13.0 | chol×19, daiin×13, chor×5 |
| 2 | 47 | 9426 | 94 | 9.97 | 7.9–12.1 | qokedy×9, shedy×7, ol×7 |
| 3 | 32 | 10632 | 86 | 8.09 | 6.0–10.4 | qokeedy×11, qokeey×10, ar×7 |
| 4 | 9 | 397 | 3 | 7.56 | 0.0–14.1 | — (3 events) |
| 5 | 7 | 744 | 3 | 4.03 | 0.0–6.7 | — (3 events) |

χ² homogeneity **p = 0.18** (RF1b p = 0.37, Boxer's csv p = 0.08). Currier
language: A 10.25 vs B 8.95, p = 0.29. Currier hands `$C` p = 0.07. Sections
`$I` p = 0.34.

**Verdict: no significant variation.** The raw rates order H1 > H2 > H3 > H5 in
all three transliterations, but the three well-sampled hands (~90 doublings
each) agree within ±20–25%, and hands 4/5 have three doublings each — no power.

## 4. Reinterpretation under Boxer's cipher model

Boxer's conjecture: each Voynich word is a homophone for **one plaintext
letter**, selected so its glyph values sum to that letter's value. Two
consequences reshape the statistic:

- Words for *different* letters can never coincide (distinct sums ⇒ distinct
  words). So **every doubled word implies a doubled plaintext letter.**
- A doubled plaintext letter yields a doubled word only if the scribe reuses the
  same homophone — probability **s**, the `doubling_strength` parameter.

Hence `VMS_doubling ≈ plaintext_letter_doubling × s`.

**A shuffled-words baseline is therefore the wrong null.** An earlier pass of
this analysis compared the observed rate against Σp<sub>w</sub>² (≈4.5/1000,
"2× chance") and read the per-hand trend as a vocabulary-concentration
artefact. Under the model that baseline mostly counts pairs the cipher forbids;
chance homophone collisions contribute only ~1/1000 of the ~9/1000 observed, so
**the raw per-hand rates are the right comparison** and they are (weakly)
proportional to s. The `chance /1000` and `obs/chance` columns in the generated
report are retained as model-free description only.

Plaintext letter-doubling measured on the Phase-0 corpora, whitespace stripped
(matching the encoder, which preserves cross-word doublings):

| plaintext | letter doubling /1000 | implied s at 9.2/1000 | repeated-bigram /1000 | Phase-0 tuned `doubling_strength` |
|---|---:|---:|---:|---:|
| Latin | 26.2 | 0.35 | 2.8 | 0.328 |
| Italian | 44.0 | 0.21 | 3.3 | 0.172 |
| German | 39.6 | 0.23 | 2.8 | 0.422 |

(The last column is `data/ciphers/acceptance_stats.json` from task 0.7,
independently bisected to hit 0.0092 — it agrees with the closed-form inversion
for Latin and Italian; German's tuned value is higher than the naive inversion,
worth a look if rung 4 leans on it.)

Three findings:

1. **A strict always-reuse rule (s = 1) is excluded for single-letter units.**
   It would force the VMS to double at 26–44/1000; it doubles at ~9. On these
   corpora the scribe reuses roughly **one time in three to five**.
2. **Pure bigram units are excluded.** Adjacent repeated bigrams occur at only
   ~3/1000 — *below* the VMS rate — so even s = 1 cannot produce 9/1000 under an
   injective bigram→value map.
3. **s looks shared, not personal.** s₁ ≈ s₂ ≈ s₃ within ±25%. Individual
   preference would be expected to scatter across scribes; it doesn't.

## 5. The geminate-collapse variant (rescuing s = 1)

If the source orthography wrote common geminates as a *single* symbol (German
`ss` → `ß`, nasal bar for `nn`/`mm`), those pairs never reach the cipher as
doubled letters. Collapsing the top-n geminates (`scripts/doubling_collapse.py`)
lowers the residual until s = 1 becomes possible:

| n collapsed | Latin | Italian | German |
|---:|---|---|---|
| 0 | 26.2 | 44.0 | 39.6 |
| 3 | ss ll ii → 13.9 | ll ss tt → 24.6 | nn ss ff → 21.1 |
| 5 | + mm ee → **9.6** | + ee aa → 16.2 | + tt ll → **11.5** |
| 6 | + rr → 8.0 ✗ | + cc → 13.1 | + dd → 8.1 ✗ |
| 8 | | + nn rr → **8.7** ✗ (7 → 10.7) | |

**n ≈ 5 (Latin), ≈ 8 (Italian), 5–6 (German)** — 65–80% of geminate occurrences
absorbed. The constraint is two-sided and therefore falsifiable: collapse one
pair too many and the plaintext doubles *less* than the ciphertext, which the
model forbids (s ≤ 1). Each language admits essentially one n, and all land in a
plausible range.

Attested medieval devices alone reach only **s ≈ 0.45**:

| | devices | residual /1000 | implied s |
|---|---|---:|---:|
| Latin | titulus for mm/nn, ij ligature | 20.8 | 0.44 |
| German | nasal bar, ß, ff ligature | 20.1 | 0.46 |
| Italian | nasal bar only | 41.4 | 0.22 |

So s = 1 needs a systematic convention over the top 5–8 geminates — beyond
ordinary orthography, but natural as part of the *cipher table*. That makes it a
mixed unigram–digraph alphabet: structurally the same move as Naibbe (rung 3).
Italian needs the longest list (8), a mild point against it. Attractively, s = 1
*predicts* the scribe-invariance of §3 — with no scribal freedom, the residual
hand-to-hand differences are differences in text (language, register, subject),
and the hands do correlate with Currier A/B.

**The doubling statistic alone cannot separate** "s = 1 over collapsed text"
from "s ≈ 0.25 over uncollapsed text": both reproduce 9.2/1000 exactly. They
differ in the plaintext alphabet (≈25 vs ≈31–33 symbols).

## 6. Consequences for the project

- **Rung 4 (CH.8):** a cheap, well-motivated variant — extend the arithmetic
  head's alphabet with the top-n geminate digraphs and generate synthetic
  ciphertext from collapsed plaintext at s = 1 (`synth.gen_arithmetic` +
  alphabet extension in `rung4_arithmetic.py`). If the extended alphabet scores
  better on the real VMS at Phase 6, that is evidence for the collapse reading.
  Note the alphabet size is itself a learnable/selectable hypothesis.
- **Direct measurement of s, language-free:** with any glyph→value key,
  s = P(same word | adjacent words with equal glyph-sum). No plaintext needed,
  and per-hand s becomes a plain proportion with real power. The pinned repo
  publishes no value table for real Voynich glyphs (the hex values exist only
  inside the synthetic encoder), and our rung-4 head *learns* values — so this
  is available once rung 4 runs on the manuscript, and doubles as a free
  consistency check on the inferred key.
- **Phase 6:** doubling rate is a per-hand-stable quantity, so it is safe to
  pool pages within a Currier dialect for scoring; A and B stay unpooled as
  design §9 requires. `scripts/doubling_rate.py` is the reference implementation
  for `$H`/`$C`/`$L` parsing (including the f115r mid-page hand switch) if
  per-hand VMS scoring is ever wanted.
- **No Phase-0 artifacts were touched.** This analysis reads the frozen corpora,
  cipher tables and VMS ingest; it writes only under `data/analysis/doubling/`.

## 7. Open items

- German's task-0.7 tuned `doubling_strength` (0.422) does not match the
  closed-form inversion (~0.23) the way Latin's and Italian's do — probably the
  tuning sample's letter-doubling rate differs from the corpus-wide rate; worth
  a check before rung 4 relies on the German table.
- The collapse analysis uses normalized modern editions. Medieval spelling wrote
  fewer geminates than these corpora do, which lowers the plaintext rate and
  therefore lowers the required n — the numbers here are upper bounds on n.
- Not yet measured: the repeat-at-distance profile (lags 2–10) per hand, which
  would show whether the excess is strictly adjacent (a doubling mechanism) or
  whether scribes also favour recently-used homophones ("sticky" reuse) — the
  latter is outside Boxer's model as written.
