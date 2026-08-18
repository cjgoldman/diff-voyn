# Phase 0 — Frozen decisions and data spec

Status record for Phase 0 of the [task breakdown](../reference_docs/Diffusion%20Model%20Training%20-%20Task%20Breakdown.md).
Numbers here are produced by `scripts/build_corpora.py`, `scripts/tune_ciphers.py`,
`scripts/ingest_vms.py`, and `scripts/g0_check.py`; re-run those to regenerate.

## 0.1 — Alphabet (FROZEN, vocab v1)

**Resolution of the "~22-letter vs 23-symbol" discrepancy (X.4):** Greshko's
actual tables (`references/naibbe_tables.csv` in greshko/naibbe-cipher @
`df3d074`, the repository behind Zenodo 10.5281/zenodo.16415087) contain
exactly **23 letters** — `a b c d e f g h i l m n o p q r s t u v x y z`
(Latin alphabet minus j, k, w), each covered by all 18 tables
(unigram/prefix/suffix × alpha/beta1–3/gamma1–2). The "~22-letter" wording in
the inverse note was an approximation. **The drift-log entry is closed.**

Frozen vocabulary (`diff_voyn/vocab.py`, 32 symbols):

| ids | symbols |
|---|---|
| 0–5 | `<pad> <mask> <null> <bos> <eos> <lang>` (`<lang>` = reserved conditioning slot, unused in v1) |
| 6–30 | `abcdefghiklmnopqrstuvwxyz` (25 letters = Naibbe 23 + **k, w**) |
| 31 | `<res0>` (reserved) |

- **K/W extension:** k and w are first-class letters (Germanic requirement,
  design §2). Greshko's own `clean_line` k→c / w→uu mapping is a property of
  the *Naibbe cipher*, applied only inside `diff_voyn/ciphers/naibbe.py`.
- **i/j:** j→i in normalization (medieval convention; Naibbe has no j table).
- **u/v:** kept **distinct** — both are separate Naibbe cipher letters; a
  merge would orphan the v tables.
- **No SPACE token** — whitespace removed in preprocessing, unrecoverable by
  design.

## 0.2 — Language inventory + corpora (FROZEN)

**Latin, Italian, German.** Sources:

- **Latin, German** — reference corpora bundled in alexanderboxer/voynich-attack
  (pinned `e324bee`): sentence-level CSVs with verbatim `textstring_orig`
  (which we normalize ourselves — the repo's `textstring_simple` applies a
  v→u fold we must not inherit). Latin: Corpus Corporum (mlat.uzh.ch,
  medieval/scientific Latin) + classical legacy texts + apothecary. German:
  Deutsches Textarchiv (1570s–1650s early-modern prints, CC BY-SA 4.0),
  zeno.org (Luther Bible 1545), legacy FNHD texts.
- **Italian** — assembled under `DATA_ROOT/raw/italian/` (see its
  `manifest.csv`): Dante (Commedia, c. 1320), Boccaccio (Decameron, 1353),
  Boiardo (Orlando innamorato, c. 1495), Ariosto (Orlando furioso, 1532),
  Machiavelli (Principe, Istorie fiorentine, plays, 1513–1532), Tasso
  (Gerusalemme liberata, 1581). Public-domain texts via Project Gutenberg and
  it.wikisource.org.

Period-appropriateness caveat (recorded, not hidden): German skews early-modern
(FNHD, 1520s–1650s) rather than strictly 15th-century; Latin mixes classical
and medieval; Italian is Trecento–Cinquecento literary Tuscan. Domain matching
is imperfect for all three languages *in the same direction* (printed
editions, literary/learned register).

Corpus v1 summary (authoritative per-document table with licenses and sha256:
`DATA_ROOT/corpora/v1/manifest.json`):

| language | docs | normalized chars | letter-drop rate | foreign-script rate | low-resource flag |
|---|---|---|---|---|---|
| latin | 72 | 27,270,074 | 0.00008% | ~1.5% (embedded Greek quotations) | no |
| italian | 7 | 4,176,750 | 0.00000% | ~0% | no (>2M) |
| german | 568 | 89,533,982 | 0.00017% | ~0.002% | no |

Italian works: Divina Commedia, Decameron, Orlando innamorato, Orlando
furioso, Il Principe, Mandragola/Clizia/Belfagor, Gerusalemme liberata.
*Istorie fiorentine* was attempted but dropped — Wikisource rate-limiting
made its 297 subpages impractical; add in a corpus v2 if Italian volume
becomes a constraint (v1 is comfortably above the 2M low-resource floor).

Held-out splits v1 (seed 20260818, document-level, domain-proportional):
latin 494,953 chars (6 docs) / italian 528,216 (2 docs: Commedia + the
Machiavelli plays) / german 821,630 (8 docs). Oversized single-doc domains
(e.g. zeno's Luther Bible 1545) are never force-held-out.

## 0.3 — Normalization (v1)

One pipeline (`diff_voyn/normalize.py`) for every stream. Uniform stages:
NFC → casefold (ß→ss) → ligature/abbreviature expansion (æ→ae, œ→oe, and the
medieval print letters attested in the sources: ꝛ→r r-rotunda, ꝙ/ꝗ→q, ʒ→z,
ꝟ→v, ø→o, ı→i, …) → NFKD diacritic strip (ü→u — the German-specific ü→ue
convention is deliberately not used) → j→i, ſ→s → whitespace removed (counted
separately) → punctuation/digits removed (counted separately).

Two loss categories, accounted separately:

- **`letters_dropped`** — Latin-script letters the pipeline failed to map:
  true lossiness, under the <0.1%-of-alphabetic-input acceptance budget.
- **`foreign_script_removed`** — Greek/Hebrew/Cyrillic content (mostly Greek
  quotations embedded in Corpus Corporum Latin works, ~1.5% of Latin input).
  Removal is *by design* and uniform across languages: the frozen 25-letter
  alphabet cannot represent these scripts, and transliterating them would be
  a per-language lossy mapping (R1 violation; the design treats "Greek
  (transliterated)" as its own tier-2 candidate language, not part of Latin).
  Per-language rates are published in the corpus manifest, not hidden.

Idempotent; verified in `tests/test_normalize.py` and on real corpus text by
`scripts/g0_check.py`.

## 0.4 — Held-out splits (v1)

`DATA_ROOT/corpora/v1/splits_v1.json`: document-level, domain-proportional,
~500k chars per language (floor 200k), seed 20260818, content-addressed
(sha256 per doc). Carved before any training run exists.

## 0.7 — Cipher generators (pinned)

- **Naibbe**: greshko/naibbe-cipher @ `df3d074` (`naibbe_v2.py`), MIT +
  citation clause (Greshko 2025, Cryptologia, doi:10.1080/01611194.2025.2566408).
- **Arithmetic**: `voynpy.pseudo_vms` @ voynich-attack `e324bee` (MIT),
  upstream default parameters, our 25-letter alphabet with upstream default
  values (a=3…z=28, j unused), seed 42; `doubling_strength` tuned per language
  to the VMS 0.92% token-doubling rate; tuned tables persisted under
  `DATA_ROOT/ciphers/` (persisted-determinism level).
- **Negative control**: `voynichesque.py` (same Naibbe repo/pin).

## 0.8 — VMS ingest

Transcriptions: **Takahashi (IT2a-n)** and **Reference (RF1b-e)**, IVTFF
files from voynich.nu (CC BY-NC-SA per site terms; used for research).
Currier dialect from `$L=` page headers; A and B never pooled; uncertain
glyphs (`?`) and extended-EVA codes dropped and counted; word/char counts
recorded pre-strip (they reconcile with published figures: 37,026 words;
191,545 chars + 37,026 word separators ≈ 228.6k ≈ the commonly cited ~230k),
then all separators stripped. Outputs under `DATA_ROOT/vms/`.

## Infrastructure (0.6)

ClearML project `diff-voyn` at `clearml.acet.network`. Canary metric layout:
scalar title `heldout_nelbo_bits_per_char`, one series per language; sampling
weights logged under `language_sampling_weights`. Checkpoints capture
optimizer + EMA (0.9999) + all RNG streams; kill-and-resume reproduces the
loss curve exactly (`tests/test_infra.py`).
