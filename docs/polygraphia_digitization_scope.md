# Polygraphia digitization — scope

> **Record status (banner added 2026-09-01):** the plan as written on the morning of 2026-08-31, before any scan was opened. What was actually built the same day, the three scope corrections (gothic rotunda not roman type; Collange 1561 is a translation, not a reprint; within-column duplicates are real — `docs/polygraphia_status.md` §2), the provisional 41-column anchor's **false CALLS** (§4 there) and the **QUARANTINE decision** (§5 there; `CLAUDE.md`) supersede §2, §4 and §6 below: **Polygraphia must not be used as a control or anchor** in any control-set workflow. The findability figure in §6.2 is corrected in place. **Current project position: `docs/project_status.md`** (§2 row 2026-08-31, §4, §6 "Anchors / data").

*2026-08-31. Context: under the "no 1:1 / 1:n glyph substitution" premise the
pipeline's only historical anchors (Zodiac-408, Borg — both 1:n class) drop
out, leaving the verbose tier with no real historical cipher. Trithemius's
**Polygraphia I–II (1518), the "Ave Maria" cipher**, is the direct historical
instantiation of the word-homophonic hypothesis — each plaintext letter is
replaced by a Latin *word* drawn from the current column's table, columns
consumed in order so the ciphertext reads as a fluent prayer — i.e. word
tokens as homophones for letters under a compact positional key rule, the
"compact key rule (what the arithmetic was)" that `docs/wordhom_study.md`
concluded would make Boxer's hypothesis testable. This document scopes making
the tables machine-readable and turning them into an anchor. It is a plan;
nothing here is built.*

## 1. Object to digitize

- **Books I and II of *Polygraphiae libri sex*** (1518): the cipher tables
  proper. Reported structure (to be verified against the scan, it is the
  first acceptance check below): ~**383 columns in Book I and ~308 in
  Book II**, each column a table of **24 Latin words**, one per letter of
  the 24-letter alphabet (no j/u distinction: i/j and u/v merged). A message
  of L letters consumes L consecutive columns; each column's words are
  grammatically compatible with its neighbours so any letter choice extends
  the same prayer.
- Books III–VI are out of scope (III is the "artificial language" variant —
  see the HistoCrypt paper below — and V–VI are the tabula recta material,
  glyph-class and excluded under the working premise).

## 2. Sources

| source | role |
|---|---|
| 1518 Basel printing (Adam Petri / Haselberg); digitized by Herzog August Bibliothek, Wolfenbüttel; further copies via Wellcome Collection ([ukbqdd75](https://wellcomecollection.org/works/ukbqdd75)) and Google Books | **primary scan** to transcribe |
| Gabriel de Collange's French edition, *Polygraphie et vniuerselle escriture cabalistique* (Paris 1561), scanned on the [Internet Archive](https://archive.org/details/polygraphieetvni00trit) and [Wellcome](https://wellcomecollection.org/works/ndg23cgk) | **independent reprint of the tables** — cross-check copy (the tables are reprinted, not translated) *[corrected 2026-08-31 — the 1561 tables are a French TRANSLATION, void as a word-level cross-check; replacement pinned MDZ bsb00026190 (Frankfurt 1550): `docs/polygraphia_status.md` §2.2]* |
| [dcode.fr Ave Maria implementation](https://www.dcode.fr/trithemius-ave-maria) (first table page only) | spot-check for the opening columns |
| [trithemius.com bibliography](https://trithemius.com/bibliography/) | edition census; the community has **no machine-readable table set** (checked 2026-08-31; DECODE/HistoCrypt likewise — the [Polygraphia III paper](https://ceur-ws.org/Vol-3313/paper7.pdf) digitized Book III alphabets, not the Book I–II word tables) |

## 3. Deliverables

1. `DATA_ROOT/external/polygraphia/` — page images of the table pages of the
   chosen scan, with a `SOURCES.md` recording edition, library, URL, and
   page↔column concordance.
2. `references/polygraphia_tables_v1.csv` — mirror of the Naibbe convention
   (`references/naibbe_tables.csv`: one row per (code, value)):
   `book,column,letter,word` with letters in the 24-letter A..Z order the
   book uses. **Versioned like the vocab: v1 frozen once validated; errors
   fixed by bumping, never by mutating.**
3. `diff_voyn/ciphers/polygraphia.py` — `PolygraphiaCipher` wrapper:
   - `encipher(text, start_column, rng)` → word tokens (the only RNG use is
     the historically-documented freedom, if any survives validation; the
     base cipher is deterministic given the start column);
   - cipher-side pre-map onto the 24-letter alphabet (analogue of
     `naibbe_pre_map`: k/w handling per what the tables actually contain —
     to be discovered in validation; j→i, v→u style folds recorded
     explicitly);
   - `decipher(tokens)` for truth alignment.
4. Anchor instances through the existing machinery
   (`wordtypes_presentation` + the wordhom head, and a new table-aware
   positional head — see §6).

## 4. Transcription route

Estimated volume: ~691 columns × 24 words ≈ **16.6k short Latin words**, on
~90–120 printed pages of clean 1518 roman type in tabular layout.
*[Corrected 2026-08-31 on the scans: 692 columns (384 slots Book I + 308
Book II) on 346 table pages, set in **gothic rotunda**, not roman type — a
medieval-print OCR model is required: `docs/polygraphia_status.md` §1, §2.1.]*

1. **Layout pass**: crop each table page into column images (the tables are
   ruled; a simple projection cut suffices; manual fallback for ornamented
   pages).
2. **OCR pass**: early-print Latin OCR (Kraken/Tesseract with a 15th–16th c.
   Latin model). The type is clean roman but expect long-s, ct/st ligatures,
   tildes for nasal abbreviation (õ→on/om), and `ę` for ae. Normalization
   rules recorded in the CSV build script, not applied silently.
3. **Structural validation (the strong lever — do not rely on OCR
   accuracy)**:
   - every column has exactly 24 entries;
   - every entry is a single token from a closed morphological family per
     column (Trithemius's columns are near-paradigms: same stem or same
     part-of-speech/inflection class down a column) — flag outliers;
     *[corrected 2026-08-31: 24 entries are NOT necessarily distinct —
     within-column duplicate words are printed (Book I col 22 "diuitias"
     under both o and y), so the check must allow duplicates:
     `docs/polygraphia_status.md` §2.3]*
   - **grammatical-continuation property**: for random letter sequences,
     consecutive drawn words must parse as the running prayer; sample and
     eyeball per book section;
   - cross-check a 5–10 % random sample of columns against the 1561
     Collange reprint (independent typesetting → independent OCR errors);
   - opening columns against the dcode implementation.
4. **Acceptance**: double-keyed (OCR + correction) error rate < 0.5 % of
   words on the sampled cross-check; any column failing structure checks
   re-transcribed by hand.

Effort estimate: layout + OCR tooling 1–2 days; correction/validation is the
bulk — at ~2 min/column of human checking, ~25 h; total **roughly one
focused week**, parallelizable by book.

## 5. Known risks

- **Edition variance**: 1518 vs later printings (1550, 1571) and the 1561
  French reprint may differ in individual words; the CSV pins ONE edition
  (1518) and the cross-check only validates OCR, not edition identity.
- **Alphabet ambiguity**: which 24 letters, and whether w/k appear at all —
  same issue as Naibbe (23 letters, `w→uu, k→c` pre-map). German plaintext
  through the anchor inherits whatever lossy pre-map the tables force;
  record it in the wrapper, and treat German anchor cells with the same
  caveat as Naibbe/German.
- **Column-order rules**: Book II may restart or interleave differently
  from Book I; the *Clavis*/instructions pages must be transcribed too, and
  the wrapper's column-advance rule tested by enciphering Trithemius's own
  worked examples if any are printed.
- **Abbreviation expansion**: expanding printed abbreviations changes token
  identity; the CSV stores the expanded form plus a `raw` column when they
  differ.

## 6. What the anchor buys once built

*[Note 2026-08-31/09-01: retained as the original motivation only. The
provisional 41-column anchor was built and run the same day; its cyclic
cells produced the pipeline's first false CALLS (margins 1.76–1.96 at SER
0.74–0.77, a periodic pseudo-language objective trap), the cipher is not
representable in the wordhom key space (collisions grow with table size),
and Polygraphia is QUARANTINED from every control workflow —
`docs/polygraphia_status.md` §4–§5, `CLAUDE.md`. None of the items below
is a usable asset as of 2026-09-01.]*

1. **Known-truth verbose decodes at any length** (generated pairs; no
   Borg-style edition-alignment problem): judge margins of true verbose
   decipherments of all three inventory languages at manuscript lengths.
2. **A findability contrast that mirrors the manuscript**: a message
   shorter than ~691 letters uses each column once → tokens/type ≈ 1
   (hapax-dominated, beyond the findability wall of the unstructured wordhom
   head — ≈ 4 tokens/type under the wildcard→anneal solver of record, ≥ 8
   under plain SA; `docs/alt_loop_plan.md` §8, `docs/wordhom_study.md`,
   `docs/project_status.md` §3; *corrected 2026-09-01, originally read
   "beyond the ≥8 tok/type findability wall from `docs/wordhom_study.md`"*);
   cyclic reuse (message length ≫ 691) raises tokens/type continuously. The unstructured wordhom head should fail on
   the former and a **table-aware positional head** (new, small: key =
   start column + advance rule; decoding is table lookup) should succeed on
   both — demonstrating the pipeline separates "unsolvable key shape" from
   "solvable given the right key grammar".
3. **Statistics caveat, stated up front**: Ave Maria ciphertext has natural
   Latin prose statistics — it anchors the verbose *judge and heads*, not
   the manuscript's statistical shape (Naibbe remains the shape-matched
   generator).

## 7. Out of scope here

- Steganographia Books I–II (selection/null cipher): different program — a
  new selection head plus the documented statement that the MDL frame is
  structurally blind to steganography (a hidden message adds no
  compression; the cover text is paid for either way).
- Any claim that the VMS *is* a Polygraphia-type cipher; this is
  instrumentation, not a hypothesis.
