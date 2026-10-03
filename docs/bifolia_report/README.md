Record status: standalone report of `docs/doubleton_gaps.md` (bifolia ordering, §2–19), 2026-09-03. Side study; not on the decipherment critical path; `docs/project_status.md` remains the arbiter of current status.

# Bifolia-ordering report

A self-contained LaTeX write-up of the bifolium-order findings of the doubleton-gaps side study: the physical sheet (bifolium) as the unit that shares rare material in the Voynich Manuscript, the one-quire seriation that follows from it, and the negative result on reading direction.

## Contents

| path | what |
|---|---|
| `bifolia_order.tex` | the report (pdflatex; one-column article, Palatino); front matter is title + abstract, then contents + record-status note, then a ~6-page executive summary (added 2026-09-04; 'sheet' is the dominant term, related to 'bifolium' once per section; the generic phrase is 'rare text') written for a non-technical academic reader, with its own figures S1–S5 (S3, the burst-scaled score, added 2026-09-09; S2, added 2026-09-04, redrawn crisper 2026-09-05: the two units, words and 7-letter glyph windows, on real text in Voynich glyphs, one rare word (qorain, f103r → f116r) and one rare window (shorshy, f19r → f22v: whole words on one leaf, inside a longer word on the other, a maximal match) each joined by a connector across an ellipsis to its occurrence on the conjugate leaf, in manuscript order; the detailed version with every window classified and the recurrences on the conjugate leaf moved to the main text as Figure 2 next to the definition of rare types), the quire-by-quire order table S1 (sheet chips + folios, added 2026-09-04) and reading-guide table S2 |
| `bifolia_order.pdf` | built output |
| `Makefile` | `make` builds the PDF with latexmk; `make figures` regenerates the figures; `make clean` |
| `SPEC.md` | the working spec shared by the writing and figure agents (terminology, figure list, style rules) |
| `figures/style.py` | shared matplotlib style and palette (colour by meaning: manuscript blue family, stacked-written aqua, nested-written red, nulls grey, winner orange) |
| `figures/fig_*.py` | one script per figure; each writes `fig_<name>.pdf` (for LaTeX) and `.png` (for review). `fig_es_*.py` are the executive-summary figures (simplified views of the same data) |
| `figures/make_all.py` | runs every figure script |
| `figures/fonts/` | `VoynichUnicode.ttf` + `EVA.TXT` (Bettencourt's Voynich Unicode package, UCSUR block U+FF400, after Landini's EVA Hand 1) used by `fig_es_raretext.py` / `fig_raretext.py` (shared data + glyph layout in `raretext_data.py`) to set EVA text as manuscript glyphs; provenance and licence note in `figures/fonts/README.md` |
| `figures/derive_quire_T.py` | re-derives quire T's full candidate-cost vectors and between-sheet affinities (the recorded JSON holds summaries only) into `data/derived_quire_T_costs.json`; ~10 min CPU, only needed if that file is deleted |
| `figures/derive_quire_T_nesting.py` | re-derives the L1 cost of every one of quire T's 23 040 nesting patterns, five units, both transcriptions (the recorded JSON holds ranks and best-per-class values only) into `data/derived_quire_T_nesting.json`, asserting the recorded stacked / nested / bound ranks; feeds `fig_es_nesting` |
| `figures/derive_burst_example.py` | re-derives the two quire-T words behind the executive summary's metric figure `fig_es_metric` (S3, added 2026-09-09: the burst-scaled seriation score on the bursting word *qokaly* and the spread word *chckhedy*, home-sheet burst scale per `scripts/quire_order_burst.py`) into `data/derived_burst_example.json`; seconds, needs DATA_ROOT |
| `data/` | the JSON / log / CSV artifacts the figures and tables are built from, copied 2026-09-03 from `DATA_ROOT/analysis/doubleton_gaps/` |

Table S1 (the sheet orders the text prefers, per quire, with folios) is built from the `\shchip` / `\shf` preamble macros (sheet number in a box coloured by physical position, folio pair beneath). Four figures are drawn in TikZ inside the .tex: the executive summary's three-sheet schematic S1 and the four-sheet nested-vs-stacked Figure 1 (both from the `\q…` quire-pile macros in the preamble: a quire seen from its top edge, leaves as bands with front/back faces numbered in reading order, conjugate pair on the fold, binding-neighbour pair bracketed; redrawn 2026-09-04), the burst-scaled cost, and the sheet-adjacency chains (Figure 10, same pile macros: as-bound pile beside the chain as a stacked pile).

## Rebuild

```bash
cd docs/bifolia_report
make figures      # cd figures && uv run python make_all.py   (seconds; CPU only)
make              # latexmk -pdf bifolia_order.tex
```

The document compiles without the matplotlib figures (each `\includegraphics` is guarded and shows a placeholder), so the text can be edited and built independently of the figure code.

## Provenance

Every number in the report is taken from `docs/doubleton_gaps.md` and marked with the record section it comes from (`[record: §N]`). The analysis scripts live in `scripts/` and were not copied; the artifacts they wrote are in `data/`:

| script (`scripts/`) | artifacts (`data/`) | report section |
|---|---|---|
| `doubleton_gaps.py` | `summary.json`, `vms_*_doubletons.csv`, `page_affinity_*.csv` | §3 baseline |
| `doubleton_leaf_affinity.py` | `leaf_affinity.json`, `leaf_affinity_controls.json` | §4 leaf test (doubletons) |
| `rare_type_clustering.py` | `rare_types.json`, `rare_types_controls.json`, `rare_types.log`, `rare_controls.log` | §3, §4 (pooled words) |
| `glyph_ngram_leaf_test.py` | `glyph_ngrams.json`, `hand_control.json`, logs | §4 (n-grams, hand control) |
| — | `window_tokens_per_type.log`, `containment.log`, `corpus_sweep.json` / `.log` | §3, §5 |
| `order_optimize.py` | `order_optimize_none.json`, `order_optimize_inverted.json`, logs (`order_optimize.json`/`.log` = superseded first run) | §5 |
| `leaf_test_pvalue.py` | `leaf_test_pvalue.json`, log | §4 empirical tail |
| `quire_order_poc.py` | `quire_order_poc_M.json`, log (`_maskbug.log` superseded) | §6 |
| `quire_order_burst.py` | `quire_order_burst_{M,T,A,B,C}.json`, logs; `burst_v1/` archived first runs | §6 |
| `quire_order_nesting.py` (+ `_summary.py`, `_table.py`) | `quire_order_nesting_{A,B,C,M,T}.json`, logs | §7 |
| `quire_order_direction.py` | `quire_order_direction_{T,M,C,A,B}.json`, logs | §7 |
| `quire_order_nullshape.py` | `quire_order_nullshape_{T,M,C}.json`, logs | §8 |
| `burst_frontloading.py`, `burst_frontloading_control.py` | `burst_frontloading.json`, logs | §7 |

Number style (2026-09-06, user decision): thousands separators are commas throughout the report — text, tables and figure labels (`23,040`, `17,000`); do not reintroduce `\,` or thin spaces between digit groups.

Figures: executive summary `fig_es_locality` (rare-word locality, named texts; since 2026-09-06 the manuscript is two bars on the pooled k 2–5 statistic, as bound vs best re-ordered stacked sheet order, from `corpus_sweep.json` + `order_optimize_none.json`), `fig_es_raretext` (the two units in Voynich glyphs: words on f103r/f116r with counts and the rare word *qorain*; glyph strings on f19r/f22v with the rare window *shorshy*, washed and connected across each sheet; counts hard-coded in `raretext_data.py`, `--recount` re-derives them from the IT2a stream), `fig_es_nesting` (quire T, one chart: the 720 fully nested vs the 720 fully stacked orders as mean ± sd / min–max boxes of the raw L1 distance, closer fit upward, grouped by unit: all words boxes on the left half with the left axis, all glyph-string boxes on the right half with the right axis (tied by glyphs per word), grey = nested, blue = stacked, shade = transcription, binding and best stacked order ("best stacked hypothesis") marked by dots; from `derived_quire_T_nesting.json`; the four earlier 2026-09-06 forms were replaced the same day — log rank chart + staircase, a three-arrangement rank ladder, an all-class box plot (moved to the main text as `fig_nesting_classes`), and a red/aqua two-class chart); `fig_es_leaftest` and `fig_es_quireT` are still generated but no longer placed (removed from the executive summary 2026-09-06 as restatements of `fig_leaf_test`(a) and of `fig_quire_T`(a) + the chains figure); main text `fig_raretext` (§2: every window of the line classified + recurrences on the conjugate leaf f103r), `fig_locality_baseline` (§3), `fig_leaf_test`, `fig_leaf_test_controls`, `fig_leaf_test_tail` (§4), `fig_containment`, `fig_order_optimize` (§5), `fig_seriation_power`, `fig_quire_T`, `fig_quires_all` (§6), `fig_nesting_classes` (§7, every nesting pattern by class as box plots), `fig_direction` (§8).

## Findings in one paragraph

Provenance: the stacked-bifolia proposal is Lisa Fagin Davis's, made in a public lecture (c. 2023–24; link to be added) where she also reported a University of Malta colleague's attempt to test it with latent semantic analysis; no published account of that test has been found, and this report is an independent second attempt with a different instrument, not the first.

The manuscript's rare text (word types and glyph n-grams used 2–10 times) has about one tenth of the passage-scale locality of prose and clusters like rhymed verse at every frequency class; this sets the power of every order test. Even so, the two leaves of one physical sheet share rare text more than leaves that are neighbours only in the nested binding (words z +2.4/+2.5; glyph n-grams z +3 to +5 at n 5–8, both transcriptions, unchanged with quire, Currier language and hand held fixed; 0/5 000 even-spread draws at n ≥ 7), and six known texts reproduce that sign only when written sheet by sheet (36/36 vs 36/36). The effect is small in mass: rare types are not contained by sheet, solver tokens/type move by under 1 %, and no re-ordering of the sheets lifts the locality above what an optimizer extracts from noise. No other way of gathering a quire's sheets (reordered nestings, sub-gatherings, mixed patterns; up to 23 040 per quire) does as well as the best stacked order, and every fully nested order ranks far down (T: 5 007–10 535 of 23 040; the binding lower still). Within a quire, a burst-scaled seriation that recovers a prose writing order about a third of the time picks one sheet chain in quire T (1-6-5-4-2-3, 31/32 cells, p ≤ 0.02 against shuffled contents and against geometry), replicates the shape in quire C (1-4-3-2) and is consistent in quire M; in each the outermost sheet neighbours the innermost. Reading direction along a chain is not established (the gap to the sheet-reversal is what selection produces on noise; prose bursts carry no time arrow), and no rare-text statistic can distinguish the sheet as the unit of composition from the sheet as the unit of work.
