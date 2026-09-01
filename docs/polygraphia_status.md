# Polygraphia digitization — status

> **Record status (banner added 2026-09-01):** record of the 2026-08-31 digitization session; still the most recent statement on Polygraphia. **Read §5 first: Trithemius/Polygraphia is QUARANTINED from every control-set workflow (user decision 2026-08-31, also in `CLAUDE.md`)** — the §4 "provisional anchor" is NOT a usable anchor or control; its cyclic cells are the pipeline's first false CALLS (margins 1.76–1.96 at SER 0.74–0.77). Transcription is paused at 92/692 columns (§3); every §6 item is gated on the user's explicit say-so. **Current project position: `docs/project_status.md`** (§2 row 2026-08-31, §4, §6 "Anchors / data").

*2026-08-31. Scope: `docs/polygraphia_digitization_scope.md`. This records
what is built, what the sources actually contain (three scope assumptions
died on contact with the scan), and the provisional-anchor results.*

## 1. Sources acquired (deliverable §3.1 — DONE)

`DATA_ROOT/external/polygraphia/` — see `SOURCES.md` there for the full
record. Primary scan: Wellcome **b33552137**, the 1518 Basel printing, all
392 relevant images at ≥2400 px width via IIIF. Structure verified page by
page (contact sheets in `sheets/`):

- Book I tables images 0033–0224 (192 pages, 384 column-slots),
- Book II tables images 0229–0382 (154 pages, **308 columns** — and the
  Book II preface, image 0228, itself says "paraphrasmata … octo &
  trecenta simplici progressione locantes": the printed count matches),
- 24 rows per column keyed `a b c d e f g h i k l m n o p q r s t v x y z`
  + `w` printed as an ꝛv ligature ("vicesima quarta litera w in fine
  alphabeti assumitur", image 0029 — w for German).

Cipher rules from the printed instructions (images 0029–0030): one word
per column ("una duntaxat capiatur"), columns in order, ciphertext reads
as Latin prayer prose; wrap-around ("reversion") when columns run out.
Small-type auxiliary words beside some rows are the "minutae dictiones"
of the design (image 0030) — recorded as annotations, not entry text.

## 2. Scope corrections (all verified on the scans)

1. **The tables are gothic rotunda, not "clean roman type"** (§4.2's OCR
   plan assumed roman). Any conventional OCR pass needs a medieval-print
   model (Kraken/CATMuS class), not Tesseract `lat`.
2. **Collange 1561 is a French TRANSLATION of the tables** ("le créateur…
   immortel, omnipotent, clement"), not a reprint — §2's cross-check role
   is void at word level (structure/order check only). Replacement pinned:
   MDZ **bsb00026190**, the Frankfurt 1550 Latin edition (VD16 T 1995),
   independent typesetting, IIIF.
3. **Within-column duplicate words are real**: Book I col 22 prints
   "diuitias" under both *o* and *y* (verified at full resolution).
   §4.3's "24 distinct entries" validation must allow printed duplicates;
   deciphering such a word was ambiguous in 1518 too. The wrapper models
   this (`decipher_candidates`).

## 3. Layout + transcription state (§4 — PARTIAL, paused by user)

- `scripts/polygraphia_layout.py`: all 692 column crops cut
  (`columns/bK_colNNN.jpg`, geometry in `layout.json`), zero failures.
- Transcription protocol: `transcripts/PROTOCOL.md` (transliteration,
  tilde marking, annotation and bleed-through rules).
- Double-keyed **vision** transcription (pass1 fable / pass2 opus agents)
  was piloted and scaled one wave before the user stopped it over token
  cost: pilot 6 columns = 144 rows with **0 disagreements** and an exact
  match against dcode's opening tables; on stop, 92 columns covered by ≥1
  pass (contiguous prefix: Book I cols 1–41), 7 columns by both.
- `scripts/polygraphia_validate.py` (shape + double-key diff),
  `scripts/polygraphia_build_csv.py` (CSV build; refuses unresolved
  disagreements; nasal-tilde expansion with `raw` kept; within-column
  duplicate report). The frozen v1 CSV is NOT yet built — the plan for the
  remaining ~600 columns is OCR-first (Kraken + medieval model as key 1,
  one vision pass as key 2, LLM adjudication only on diffs).

## 4. Provisional anchor (§6 — first pass, provisional tables)

`provisional_tables.csv` (92 columns, mostly single-keyed) → cols 1–41
feed `diff_voyn/ciphers/polygraphia.py` (`PolygraphiaTables`,
`PolygraphiaCipher`, `polygraphia_pre_map` u→v; tests
`tests/test_polygraphia.py`). `scripts/polygraphia_instances.py` builds
wordhom-format instances (3 languages × cyclic-8000 / hapax-41,
`DATA_ROOT/analysis/polygraphia_anchor/instances/`);
`scripts/polygraphia_run.py` solves/scores/reports them through the
Phase-6 harness (frozen evaluator, MDL scale, ABSTAIN_RULE).

Structural fact the instances encode: the column cipher is **not
type-deterministic** — cycling 8000 letters over 41 columns yields 50–58
word types that stand under different letters in different columns, so
the best possible wordhom key decodes only ~96 % of tokens
(`truth.oracle_type_acc` ≈ 0.956–0.960). The §6.2 contrast (wordhom head
vs a future table-aware positional head) is therefore measurable even
before the full tables exist.

### Results (solve + score + report run 2026-08-31, provisional 41-col tables)

`DATA_ROOT/analysis/polygraphia_anchor/{solves,scores,report}.json`:

| instance | tok/type | oracle ceiling | abstain | margin | SER vs plain |
|---|---|---|---|---|---|
| latin/cyclic | 11.0 | 0.960 | **no (CALLED)** | 1.96 | 0.773 |
| italian/cyclic | 11.0 | 0.957 | **no (CALLED)** | 1.76 | 0.766 |
| german/cyclic | 10.2 | 0.958 | **no (CALLED)** | 1.90 | 0.737 |
| latin/hapax | 1.0 | 1.0 | no† | 3.99 | 0.829 |
| italian/hapax | 1.05 | 0.951 | yes† | 3.69 | 0.854 |
| german/hapax | 1.0 | 1.0 | yes† | 4.26 | 0.805 |

† 41-char streams are below the instrument's floor (ELBO unreliable
< 100 chars); hapax rows are not meaningful — a real hapax test needs the
full 384-column Book I.

**Finding — the first controls that falsely cross the frozen 1.5-margin
threshold.** On the cyclic cells the SA lands, on every arm tried, in a
*periodic pseudo-language* local optimum: the 41-column cycle makes the
type sequence periodic, and the optimizer composes one fluent clause per
cycle that repeats ~195× ("…lebn wolt haben sonder schent seinem ander
sein leben…"; MDL-top = german on all three, the known n-gram German
attractor). The judge then endorses it: margins 1.76–1.96, all three
CALLED at SER 0.74–0.77. Nothing in the Phase-6/wordhom batteries had
falsely exceeded 1.51.

It is an **objective trap** (higher-is-better objective): found key
−13,727 vs oracle −28,201 on latin/cyclic — and it survives correction of
the u→v truth handicap. Decomposition (latin/cyclic, 8000 chars, nats):
u→v pre-map decode off-distribution ≈ 7,600 (fixed by keeping truth in
model space: oracle raw −25,433 → −17,834, ceiling SER 0.053); intrinsic
collision noise ≈ 4,600; repeat-rule mismatch 246 violations ≈ 1,100.
After the u-fix the periodic key still wins by ≈ 6,000 (−12,902 vs
−18,983). Heavy SA (32×1.5M), wildcard→anneal (129/725 wild types), and
SA seeded AT the oracle all end in the periodic basin — seeded-from-truth
SA *walks away* (obj −28,201 → −18,349, SER 0.04 → 0.17), so truth is not
even locally stable under this objective at this table length.

Scope of the finding: short-table cyclic verbose ciphertext (period ≪
window). The full books (384/308 columns) have ~9× weaker periodicity per
window — untested until transcription completes. Blind mitigations that
would catch this false call: the decode's near-perfect periodicity, and
the lexical-coverage criterion (the mantra reuses a handful of word
forms). Neither is in the frozen rule.

## 5. Decision (2026-08-31): not a control anchor — QUARANTINED

User decision after the §4 results: Trithemius/Polygraphia is **not a
usable historical anchor for now** and must stay out of every control-set
workflow — no Polygraphia instance in any control battery, wordhom/Phase-6
control manifest, abstention-threshold calibration, or pooled "controls"
statistic (also recorded in `CLAUDE.md`). Grounds: (a) the cyclic cells
falsely CALL (the §4 objective trap), so as "controls" they would poison
any threshold calibration; (b) the cipher is outside the wordhom key
space, with collisions growing in table size (73/889 words under >1
letter at 41 columns, 275/1746 at 92 — the hypothesis fits *worse* as
coverage grows), so oracle ceilings keep dropping; (c) most transcribed
columns are single-keyed and unvalidated. The work and findings are kept:
scans, layout, transcripts, wrapper, instances, and the false-call result
are all recorded here and under `DATA_ROOT/analysis/polygraphia_anchor/`.
Verified 2026-08-31 that no control workflow references it ("polygraphia"
appears only in the seven polygraphia-named files; control manifests come
only from `build_wordhom_controls`). Any future use is its own
explicitly-named study, gated on the user.

## 6. Open (all gated on the §5 decision)

- Resume transcription with the OCR-first double-keying (~600 columns),
  then adjudicate, build and freeze `references/polygraphia_tables_v1.csv`.
- 5–10 % cross-check against MDZ bsb00026190 (1550).
- Table-aware positional head (key = start column) for the §6.2 contrast.
- Book II instances once its columns are transcribed; longer cyclic
  instances over the full 384/308-column books.
