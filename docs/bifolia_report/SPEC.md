# Bifolia-ordering report — working spec (shared by all agents)

Source of every number: `docs/doubleton_gaps.md` (sections cited as §N below) and the
JSON/log artifacts copied into `docs/bifolia_report/data/` (from
`DATA_ROOT/analysis/doubleton_gaps/`). Analysis scripts remain in `scripts/`
(`doubleton_leaf_affinity.py`, `rare_type_clustering.py`, `glyph_ngram_leaf_test.py`,
`leaf_test_pvalue.py`, `order_optimize.py`, `quire_order_poc.py`, `quire_order_burst.py`,
`quire_order_direction.py`, `quire_order_nullshape.py`, `burst_frontloading*.py`).

## Layout

```
docs/bifolia_report/
  bifolia_order.tex      main LaTeX source (pdflatex, one column, 11pt article)
  Makefile               `make` → latexmk -pdf; `make figures` → python figures/make_all.py
  README.md              what is here, how to rebuild, provenance
  SPEC.md                this file
  data/                  JSON + log + CSV artifacts (copied 2026-09-03)
  figures/style.py       shared matplotlib style + palette (READ IT FIRST)
  figures/fig_*.py       one script per figure, writes fig_<name>.pdf and .png
  figures/make_all.py    runs every fig_*.py
  tables/*.tex           optional generated table bodies (\input from the .tex)
```

Run figure scripts with `cd docs/bifolia_report/figures && uv run python fig_<name>.py`.
Each script must be idempotent, read only from `../data/`, and use `from style import *`.
Numbers that exist only in the markdown tables / logs (no JSON) may be hard-coded in the
script with a comment `# source: docs/doubleton_gaps.md §N` — verify them against the
.log file in data/ when one exists.

## Terminology (use consistently in text and figures)

- **sheet = bifolium**: one physical sheet folded once → two **leaves** (a, b) → four
  **pages** a-r, a-v, b-r, b-v. In a quire of *S* sheets, sheet 1 is outermost.
- **nested (as-bound) order**: 1a-r,1a-v, 2a-r,2a-v, …, Sa, Sb, …, 2b-r,2b-v, 1b-r,1b-v.
- **stacked order**: sheet after sheet, each read a-r, a-v, b-r, b-v.
- **conjugate leaves**: the two leaves of one sheet (adjacent only under stacked reading).
- **nested-adjacent leaves**: leaves consecutive in the binding but on different sheets
  (adjacent only under nested reading).
- **rare type**: word type or glyph n-gram with k ∈ [2,10] occurrences (k=2 doubleton).
- **statistic D = (conjugate-leaf pairs) − (nested-adjacent-leaf pairs)**, z against the
  null that permutes page contents among page slots of the same quire × Currier
  language (× hand for the hand control).
- Transcriptions: **IT2a** (Takahashi), **RF1b** (Reference).
- Quire sheets are labelled outer→inner 1..S; the current (as-bound) stacked order is
  1-2-…-S. T: 6 sheets (f103/f116 … f108/f111); M: 5 (f75/f84 … f79/f80); A: 4 (f1/f8 …
  f4/f5); B: 4 (f9/f16, f10/f15, f11/f14, f13 alone); C: 4 (f17/f24 … f20/f21).

## Figure list (filenames are fixed; the .tex includes them by these names)

Made in matplotlib by the figure agents:

| file | content | source |
|---|---|---|
| `fig_locality_baseline` | (a) P(consecutive gap ≤ 100)/uniform vs k=2..5, log y: VMS IT2a/RF1b, Currier A, Currier B; known Latin/German/Italian prose and Italian verse as grey lines w/ language markers. (b) corpus sweep: ECDF or strip of 148 known windows' r100 (log x) coloured by language, VMS IT2a/RF1b as vertical lines with percentile annotation. | rare_types.json (§6.1), corpus_sweep.json (§9) |
| `fig_leaf_test` | (a) D-statistic z per unit (words k2, k2–5, k2–10; n4…n8 all; n5…n8 cross) for IT2a and RF1b, dots; control bands: nested-written range (red wash) and stacked-written range (aqua wash) from the six known texts; hand-controlled values as hollow markers. (b) the two components: conjugate z and nested-adjacent z per unit. | leaf_affinity.json, rare_types.json leaf_test_*, rare_types_controls.json, glyph_ngrams.json, hand_control.json (§4, §6.2, §7) |
| `fig_leaf_test_controls` | per-control-text panel: D z for nested-written vs stacked-written, words k2–10 and n5/n6/n7 all, six texts, with manuscript values as reference lines — shows 36/36 separation | rare_types_controls.json, glyph_ngrams.json (§6.2, §7.2) |
| `fig_leaf_test_tail` | empirical null: number of 5 000 even-spread draws reaching the observed D per unit (words, n5..n8) × null (quire+lang, +hand) × transcription, log scale with "0 / 5000" shown explicitly; include normal-tail p for reference | leaf_test_pvalue.json (§12) |
| `fig_containment` | (a) where a rare type's occurrences sit: stacked horizontal bars per k (2..5): same page / same leaf / same sheet / 2+ sheets same quire / 2+ quires, both transcriptions, null ticks. (b) tokens/type per window (W=512..4096) nested vs stacked, Currier A and B. | containment.log, window_tokens_per_type.log (§11, §8) |
| `fig_order_optimize` | r1000 (and r100 small) after simulated-annealing sheet reordering under strict/topic/language/free: manuscript from stacked, from random sheet orders, page-content-shuffled manuscript (noise ceiling band), five known texts vs their true-order value; corpus percentiles (5th, 25th, median) as reference lines | order_optimize_none.json (§10) |
| `fig_seriation_power` | prose power on each quire's page slots: rank of the true writing order among the candidates (18 stacked-written cases per quire) per metric (L1, burst, blog, modal), strip/dot plot, with rank-1 and top-10 marked; nested-written control: rank of nested order among 121 | quire_order_burst_{M,T,A,B,C}.json controls (§13–15) |
| `fig_quire_T` | quire T: (a) distribution of within-candidate z for the 720 orders on real contents vs on shuffled contents, winner 1-6-5-4-2-3 marked, shuffled winners' z distribution; (b) grid unit (words,n5..n8) × metric (L1,burst,blog,modal) × transcription showing the best order per cell (colour = is it 1-6-5-4-2-3 / one transposition away / other) with content-shuffle p in the cell; (c) between-sheet affinity matrix (shared rare pairs observed/expected, 6×6) if derivable from the JSON | quire_order_burst_T.json, quire_order_nullshape_T.json (§15, §17) |
| `fig_quires_all` | all five quires: heatmap of content-shuffle p of the best order per unit×metric (log colour), IT2a/RF1b side by side; plus current-order rank/N and best-order labels; T, M, C, A, B rows | quire_order_burst_*.json (§14–15) |
| `fig_nesting` | every nesting pattern of a quire's sheets (L1 only): (a) best z per number of nested blocks, n7, binding marked; (b) nested−stacked margin vs shuffled-content 5–95 % band per quire×unit; (c) rank/N of best fully nested, binding, best stacked (log) | quire_order_nesting_{A,B,C,M,T}.json (§19) |
| `fig_es_raretext` | executive summary (S2), two rows in Voynich glyphs (fonts/VoynichUnicode.ttf, laid out letter by letter from the font's advance widths), each row the two leaves of one sheet in manuscript order joined by a blue elbow connector: words — f103r.11 then f116r.47 words 2–6 with manuscript counts, the rare word *qorain* (4 uses) washed blue on both; glyph strings — f19r.6 then f22v.6 (quire C, sheet 3) with spaces kept for legibility, the 7-letter window *shorshy* (4 uses) washed blue on both: whole words *shor shy* on f19r, starting inside *fshor* on f22v, a maximal match (neighbouring letters differ) so the box is exactly what recurs; teaches only what the two units are | counts hard-coded in `raretext_data.py` from the IT2a stream as built by `scripts/glyph_ngram_leaf_test.py`; `--recount` verifies both examples (§7) |
| `fig_raretext` | main text §2, after the definition of rare types: the same line with every one of its 44 windows classified (rare / once / > 10), five windows called out (across a gap vs inside one word), and the recurrences of three rare strings and the rare word *qorain* on the conjugate leaf f103r | as above |
| `fig_es_locality` | executive summary: one bar per named text, pooled k=2–5 r100 (known prose / verse windows cut to manuscript length; manuscript as bound, light blue, and "bifolia hypothesis" = best unconstrained stacked sheet order, dark blue; all bars sorted by size), log x, ×1 = chance; redrawn 2026-09-06 (was k=2 with Currier A/B and RF1b bars) | corpus_sweep.json (§9), order_optimize_none.json (§10) |
| `fig_es_metric` | executive summary (S3, added 2026-09-09), the burst-scaled seriation score on two real words of quire T (IT2a, word unit): three rows on one word scale, each read as a candidate order (home sheet band with the uses as orange ticks, then the sheet holding the word's one stray use, blue tick); *qokaly* bursts (uses 51 and 115 apart, burst scale 173, stray on sheet 5 at 1,651 words = 9.6 bursts), *chckhedy* spreads (1,480 and 437 apart, scale 756, stray on sheet 4 at 1,857 words = 2.5 bursts: same distance, a quarter of the cost), third row *qokaly* with sheet 2 read between (3,390 words = 19.6 bursts; the order moves the distance, never the scale); footer: score = bursts summed over every stray use, lowest wins. Replaced the same-day first draft (ratio-to-chance curves), which explained the easy metric | derived_burst_example.json (§14; from `derive_burst_example.py`, burst scale per `scripts/quire_order_burst.py`) |
| `fig_es_leaftest` | (generated, not placed — removed from the executive summary 2026-09-06, restates `fig_leaf_test`(a)) formerly executive summary: D z for words k2–10, n7 all, n7 cross; both transcriptions; hollow = hand-controlled; stacked/nested control washes; plain-language axis | rare_types.json, rare_types_controls.json, glyph_ngrams.json, hand_control.json (§6.2, §7, §12) |
| `fig_es_nesting` | executive summary, quire T only, one chart: the metric itself (mean rare recurrence distance, closer fit upward) for the two extreme classes only — 720 fully nested vs 720 fully stacked orders — grouped by unit — all words boxes on the left half (left axis, words), all 7-glyph-string boxes on the right half (right axis, glyphs; axes tied by glyphs per word 56,889/10,673) — nested pair then stacked pair inside each half; colour = class (grey nested, blue stacked), lighter/darker shade = IT2a/RF1b; mean ± sd boxes, mean line, min–max whiskers; dots: the binding in each nested box, the best stacked order in each stacked box (labelled "best stacked hypothesis"; best of all on glyph strings, 5th/2nd on words); no x-axis title (earlier 2026-09-06 forms: log rank chart + staircase; linear rank ladder; all-class box plot, now `fig_nesting_classes`; red/aqua two-class chart) | derived_quire_T_nesting.json (§19; from `derive_quire_T_nesting.py`, ranks checked against quire_order_nesting_T.json) |
| `fig_nesting_classes` | main text §7: every nesting pattern of quire T by number-of-nested-blocks class (1 = 720 fully nested … 6 = 720 stacked), standardised L1 (n7) closer fit upward, mean ± sd box / mean line / min–max whiskers, binding and best stacked marked with a dashed line at the latter; one panel per transcription | derived_quire_T_nesting.json (§19) |
| `fig_es_quireT` | (generated, not placed — removed 2026-09-06, restates `fig_quire_T`(a) and the chains figure) formerly executive summary: (a) histogram of the 720 orders' z (n7 L1 IT2a) with winner and shuffled best-of-720; (b) sheets as bound vs the chain 1-6-5-4-2-3 with folios, no arrow | derived_quire_T_costs.json (§15, §17) |
| `fig_direction` | (a) direction gap Δ (sd) real vs null distribution per quire (T, M, C) per unit — box/strip of null Δ with real marked, p annotated; (b) burst front-loading: fraction first<last for prose texts (pooled + by text) and manuscript (all sheets, Currier A, Currier B, pages) per unit, 0.5 reference | quire_order_direction_*.json, quire_order_nullshape_*.json, burst_frontloading.json (§16–18) |

Made in TikZ inside the .tex by the writer:

- `fig:es-schematic` — executive summary: three-sheet quire seen from its top edge (spine left), as bound vs stacked side by side; each leaf a band with its front (r) face numbered above and back (v) face below in reading order; the conjugate pair marked on the fold (aqua), one binding-neighbour pair by a red bracket. Drawn with the `\q…` pile macros in the preamble (2026-09-04; replaced the row-of-boxes form with red ticks and aqua arcs).

- `fig:codicology` — the same pile diagram for a quire of 4 sheets, nested (a) vs stacked (b),
  faces numbered in reading order; conjugate 𝒞 (fold, aqua), nested-adjacent 𝒩 (red bracket)
  and the excluded innermost sheet ℬ (grey bracket) marked. Same macros as `fig:es-schematic`.
- `fig:burstmetric` — the burst-scaled cost: a type's home-sheet occurrences (gaps →
  λ_t), an outlier on another sheet at distance d under a candidate order.
- `fig:chains` — sheet-adjacency chains found for T (1-6-5-4-2-3), C (1-4-3-2), M
  (2-3-5-1-4 family): per quire, the sheets as bound (`\qnestedmini`) beside the chain as a
  stacked pile in chain order (`\qstackedorder`), colour by physical sheet (redrawn 2026-09-04
  in the Figure 1 vocabulary); direction arrows explicitly absent / marked "direction not established".

## Style rules for figures

- `from style import *`; widths `W_FULL` (6.3 in) or `W_HALF`; heights 2.2–4.5 in.
- Colours by meaning only (see style.py docstring). Never a rainbow; never colour by rank.
- Bands: red wash = nested-written control range; aqua wash = stacked-written range.
- Legend present whenever ≥ 2 series; direct labels on ≤ 4 key points; no number on
  every point. Tabular figures use `ax.text` in INK2. Grid y-only, hairline.
- z-axes: draw a zero line (AXIS colour). p-values on log10 scale with 1/200 = 0.005
  floor marked (200 shuffles) where applicable.
- Every figure saved as PDF + PNG via `save(fig, "<name>")`. Look at the PNG (Read tool)
  before finishing: no label collisions, no clipped text, no overflow.
- Panel labels (a), (b) via `panel_label`.

## Executive summary (added 2026-09-04)

Unnumbered `\section*{Executive summary}` after the TOC, ~4½ pages, for an academic but
non-technical reader: the question, the instrument, four findings (each one figure +
"what this means"), the establishes / does-not-establish boxes in plain language, and a
order table S1 (added 2026-09-04: per quire T/C/M/A/B, sheets as bound and the order the
text prefers as coloured sheet chips `\shf{n}{fa}{fb}` with the folio pair beneath, an
evidence column, and a dagger note that direction is not claimed) and the reading-guide
table S2 (claim, strength dots, basis, section). Its floats are numbered S1–S4 / S1–S2
(counters reset afterwards so the main text keeps Figure 1 …). Wording must
match the Discussion: composition-unit vs work-unit not separable, direction not
established, small mass.

## LaTeX conventions (writer)

- `\documentclass[11pt,a4paper]{article}`, `geometry` margins 2.4 cm (textwidth ≈ 6.3 in),
  `mathpazo` + `helvet`, `microtype`, `booktabs`, `siunitx`, `amsmath,amssymb`, `graphicx`,
  `tikz`, `caption`/`subcaption`, `xcolor`, `tcolorbox` (for the "what this establishes /
  does not establish" boxes), `hyperref` + `cleveref` last.
- `\graphicspath{{figures/}}`; include every figure with `\includegraphics[width=\textwidth]`
  guarded by `\IfFileExists{figures/<name>.pdf}{…}{\fbox{missing <name>}}` so the document
  compiles before figures exist.
- Named colours in the preamble matching style.py (it2a, rf1b, stacked, nested, accent).
- Equations: define the D statistic, the permutation null and z, the empirical p
  (#null ≥ obs + 1)/(n+1), the L1 / burst / blog / modal costs, λ_t shrinkage, the
  direction cost asymmetry (L_A − x) + y vs (L_B − y) + x, the best-of-N minimum heuristic.
- No bibliography; mention Davis's stacked-bifolia proposal and Montemurro–Zanette /
  Reddy–Knight only in passing by name (no reference list required by the user).
- Tone: academic, precise, every quantitative claim traceable to a §; state the caveats
  (sequential testing, correlated cells, composition-vs-work-unit, direction not
  established) in the body, not only in a footnote.
