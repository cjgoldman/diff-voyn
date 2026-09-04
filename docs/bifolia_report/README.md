Record status: standalone report of `docs/doubleton_gaps.md` (bifolia ordering, §2–19), 2026-09-03. Side study; not on the decipherment critical path; `docs/project_status.md` remains the arbiter of current status.

# Bifolia-ordering report

A self-contained LaTeX write-up of the bifolium-order findings of the doubleton-gaps side study: the physical sheet (bifolium) as the unit that shares rare material in the Voynich Manuscript, the one-quire seriation that follows from it, and the negative result on reading direction.

## Contents

| path | what |
|---|---|
| `bifolia_order.tex` | the report (pdflatex; one-column article, Palatino); opens with a ~4½-page executive summary (added 2026-09-04) written for a non-technical academic reader, with its own figures S1–S5 and reading-guide table S1 |
| `bifolia_order.pdf` | built output |
| `Makefile` | `make` builds the PDF with latexmk; `make figures` regenerates the figures; `make clean` |
| `SPEC.md` | the working spec shared by the writing and figure agents (terminology, figure list, style rules) |
| `figures/style.py` | shared matplotlib style and palette (colour by meaning: manuscript blue family, stacked-written aqua, nested-written red, nulls grey, winner orange) |
| `figures/fig_*.py` | one script per figure; each writes `fig_<name>.pdf` (for LaTeX) and `.png` (for review). `fig_es_*.py` are the executive-summary figures (simplified views of the same data) |
| `figures/make_all.py` | runs every figure script |
| `figures/derive_quire_T.py` | re-derives quire T's full candidate-cost vectors and between-sheet affinities (the recorded JSON holds summaries only) into `data/derived_quire_T_costs.json`; ~10 min CPU, only needed if that file is deleted |
| `data/` | the JSON / log / CSV artifacts the figures and tables are built from, copied 2026-09-03 from `DATA_ROOT/analysis/doubleton_gaps/` |

Four figures are drawn in TikZ inside the .tex: the executive summary's three-sheet schematic S1 and the four-sheet nested-vs-stacked Figure 1 (both from the `\q…` quire-pile macros in the preamble: a quire seen from its top edge, leaves as bands with front/back faces numbered in reading order, conjugate pair on the fold, binding-neighbour pair bracketed; redrawn 2026-09-04), the burst-scaled cost, and the sheet-adjacency chains (Figure 10, same pile macros: as-bound pile beside the chain as a stacked pile).

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

Figures: executive summary `fig_es_locality` (rare-word locality, named texts), `fig_es_leaftest` (leaf-pair test, three units), `fig_es_nesting` (quire T, all 23 040 gatherings), `fig_es_quireT` (quire T seriation histogram + chain); main text `fig_locality_baseline` (§3), `fig_leaf_test`, `fig_leaf_test_controls`, `fig_leaf_test_tail` (§4), `fig_containment`, `fig_order_optimize` (§5), `fig_seriation_power`, `fig_quire_T`, `fig_quires_all` (§6), `fig_direction` (§7).

## Findings in one paragraph

Provenance: the stacked-bifolia proposal is Lisa Fagin Davis's, made in a public lecture (c. 2023–24; link to be added) where she also reported a University of Malta colleague's attempt to test it with latent semantic analysis; no published account of that test has been found, and this report is an independent second attempt with a different instrument, not the first.

The manuscript's rare vocabulary (types used 2–10 times) has about one tenth of the passage-scale locality of prose and clusters like rhymed verse at every frequency class; this sets the power of every order test. Even so, the two leaves of one physical sheet share rare material more than leaves that are neighbours only in the nested binding (words z +2.4/+2.5; glyph n-grams z +3 to +5 at n 5–8, both transcriptions, unchanged with quire, Currier language and hand held fixed; 0/5 000 even-spread draws at n ≥ 7), and six known texts reproduce that sign only when written sheet by sheet (36/36 vs 36/36). The effect is small in mass: rare types are not contained by sheet, solver tokens/type move by under 1 %, and no re-ordering of the sheets lifts the locality above what an optimizer extracts from noise. No other way of gathering a quire's sheets (reordered nestings, sub-gatherings, mixed patterns; up to 23 040 per quire) does as well as the best stacked order, and every fully nested order ranks far down (T: 5 007–10 535 of 23 040; the binding lower still). Within a quire, a burst-scaled seriation that recovers a prose writing order about a third of the time picks one sheet chain in quire T (1-6-5-4-2-3, 31/32 cells, p ≤ 0.02 against shuffled contents and against geometry), replicates the shape in quire C (1-4-3-2) and is consistent in quire M; in each the outermost sheet neighbours the innermost. Reading direction along a chain is not established (the gap to the sheet-reversal is what selection produces on noise; prose bursts carry no time arrow), and no rare-material statistic can distinguish the sheet as the unit of composition from the sheet as the unit of work.
