# Language identification of the Voynich Manuscript by trial decipherment under a frozen diffusion likelihood — Phase-6 write-up (draft)

> **Record status (banner added 2026-09-01):** written 2026-08-24; every number is the frozen Phase-6 record and is unchanged. Not updated for post-Phase-6 work, except by the in-place notes below: the Borg `elbo_polish` account (cause is the choice-bits term in the polish objective, 2026-08-25, `docs/race_polish_plan.md` §7), the uncovered-symbol charge (best held-out n-gram cross-entropy, not order-0 — corrected in §5), the "two cells" sentence in §2, and the voynichesque band (content-inflated; a wrong-hypothesis control, strict-gibberish ceiling 1.40, 2026-08-31). The abstention was re-confirmed on the manuscript by three later campaigns (2026-08-26, 08-29, 08-31), all NOISE. **Current project position: `docs/project_status.md`.**

*Status: exploratory, assumption-dependent. This document states what the
instrument measured on the manuscript, under which assumptions, at which
resolution, and what it cannot say.*

## 1. What was measured

A multilingual masked-diffusion character model (85M, Latin / Italian /
German, frozen at Gate G4) serves as a per-language likelihood. Four
differentiable-in-principle cipher heads, validated on synthetic ciphertexts
in Phase 5 (1:1 substitution, unigram homophonic, Naibbe mixed
unigram–bigram, Boxer's arithmetic sum-to-target), are applied to the
manuscript **per Currier dialect** (A and B never pooled), on two EVA
transcriptions (Takahashi IT2a primary, Reference RF1b replicate) and on
Boxer's independent glyph transcription. Every (cipher × language) cell is
ranked on the **minimum-description-length total per ciphertext symbol**
(calibrated plaintext bits under the language condition + key bits + the
cipher's choice bits), the rule Phase 5 showed to be the only one that
survives verbose heads; within a cipher hypothesis the language order is by
the same total. Each cell carries its per-window spread, replicate
flip-rate, calibration margin uncertainty, coverage, and a per-instance
shuffled-text structure margin. A fixed abstention rule (frozen before any
manuscript number was read) decides whether any cell is a language-like
decipherment at all; a negative-control battery (structured gibberish,
shuffled text, out-of-inventory languages under in-inventory-fit ciphers,
in-inventory positives) runs through the identical pipeline.

## 2. Results

**The instrument abstains on the Voynich Manuscript.** Across 87
(cipher × language × window × presentation) cells — four cipher hypotheses,
three languages, two dialects, two EVA transcriptions plus Boxer's glyph
transcription — **no cell is a language-like decipherment**. The decision
does not rest on a threshold: the whole VMS table's structure margins
(decode vs. a shuffle of itself, 0.04–1.25 bits/char) fall **below every one
of the nine true decipherments** put through the identical pipeline
(1.49–2.48) and inside the band of `voynichesque.py` gibberish (0.92–1.51)
and of out-of-inventory languages under in-inventory ciphers (0.60–1.43).
Any cut between 1.26 and 1.48 abstains on the entire manuscript and on no
positive control. *[Note added 2026-09-01: the `voynichesque` band is
generated from real held-out text and its homophonic cells are
content-inflated by Δ −0.27 bits/char relative to strict-negative twins on a
letter-shuffled source (27/27 pairs; 2026-08-31,
`docs/voynichesque_nocontent_restart.md`) — it is a wrong-hypothesis
control, not a strict negative. Strict gibberish peaks at 1.40, so the
corridor also safe against a strict negative is ≈ 1.41–1.48; the manuscript
(≤ 1.25) is below both. `docs/project_status.md` §5 items 6–7.]*

Two rungs reach clean-text-level *bits* — the 1:1 head at 2.81–3.05 and the
homophonic head at 2.47–2.76 bits/char (`docs/phase6_status.md` §6.2) — but
do so the way Phase 5 showed a verbose head can on any stream, by finding a
low-entropy mapping *(corrected 2026-09-01; originally read "Two cells reach
clean-text-level bits — the homophonic head at 2.47–2.62 bits/char")*; §2.1 shows what those decodes
actually look like. The two heads with real cipher grammars (Naibbe,
arithmetic) return decodes at 3.5–4.3 bits/char, i.e. no key was found.

**The language ranking among these non-decipherments is not evidence.**
Within-head margins are 0.001–0.083 bits per ciphertext symbol against a
calibration margin uncertainty of 0.067–0.193; the four heads disagree
with each other (head agreement 0.25–0.67), the two transcriptions disagree
on Currier B, and A and B disagree under the homophonic head — while the
replicate flip-rate is 0 for 80 of 87 cells. The disagreement is systematic,
not Monte-Carlo noise: different hypotheses genuinely prefer different
languages, which is what ranking non-decipherments produces.

**One structural result survives, and it is about description length, not
language.** The Naibbe hypothesis — a mixed unigram/bigram token grammar —
ranks first on the MDL total in every table, and on **Currier A** (both
transcriptions independently) its total is below the manuscript's own
held-out bigram cross-entropy (2.091 vs 2.179 and 2.123 vs 2.231 bits per
ciphertext symbol). On **Currier B** no cipher hypothesis beats that
surface baseline. So a Naibbe-shaped token grammar compresses Currier A
slightly better than a character n-gram of Currier A does — while the
plaintext it yields is not language (3.5 bits/char, structure margin
0.07–0.09). The dialects differing here is consistent with their known
token-structure difference; it is a statement about the manuscript's word
structure, not a decipherment.

**Controls** (39 instances, identical pipeline): shuffled text 9/9
abstained, out-of-inventory contamination 12/12 abstained, positives 8/9
called correctly at both language and family level with all nine 1:1 solves
at SER 0.000, `voynichesque` 8/9 abstained — 0.89, below the > 0.95 the
task requires, and reported as a failure rather than re-tuned. The one
false accept (margin 1.51) and the one false abstention (1.49) straddle the
frozen threshold: the margin is very nearly a separating statistic on this
battery, with one error on each side out of 39. With the abstention rule
switched off, untrained languages are assigned in-inventory labels with
family accuracy 7/12 — barely above chance — which is precisely why the
rule exists.

**Anchors**: Zodiac-408 at **0.98% SER** (target ≤ 1.9%) confirms the inner
search at literature difficulty, but only as a pre-diffusion n-gram
baseline — English is outside the frozen inventory. Borg (Latin, 55 symbol
types) is **ranked Latin with a 0.250 bits/symbol margin, the largest in
Phase 6**, and its n-gram decode is readable Latin (median page SER 0.110,
best 0.035) — but it **fails** the ≤ 4.10% target as we measure it (0.129
n-gram, 0.226 final), for two reasons that are ours and not the head's: the
published plaintext is a corrected and expanded edition rather than a
symbol-aligned transcription, and we drop glyph types occurring fewer than
20 times. BnF fr2988 could not be obtained. A separate negative finding:
the Phase-5 discrete `elbo_polish` **degrades** Borg (median page SER 0.110
→ 0.217, 25 of 55 symbols reassigned) when the homophonic choice-bits term
is part of the polish objective — the judge dislikes the moves (`e` → `z`,
+0.087 bits/char) but the choice term rewards them (−0.243); with the term
removed the polish holds the key (SER 0.1195 → 0.1194, race; 0.1198,
greedy) and polishes now run on the ELBO alone by default
(`docs/race_polish_plan.md` §7, 2026-08-25) *(corrected 2026-09-01;
originally read "through selection bias over a large move neighbourhood — it
does not transfer beyond the rung-2 synthetics"; selection noise adds damage
on the wrong objective but does not set the direction)*.
Re-scoring the best VMS cells with pre-polish keys leaves the abstention
unchanged (margins 0.83–0.84 before, 1.05–1.11 after), so the manuscript
verdict is not an artifact of that step.

**What this does and does not say.** It says: under this instrument, with
this language inventory, these four cipher hypotheses, and these
transcriptions and tokenizations, the Voynich Manuscript does not yield a
decipherment that behaves like Latin, Italian, or German text — and the
instrument demonstrably does yield one for true ciphers of those languages
put through the same pipeline. It does not say the manuscript is
meaningless, nor that it is not one of these languages under some cipher
outside this inventory or some tokenization other than the ones tried.

### 2.1 What the abstention was judging — representative decodes

The verdict of §6.2 is a scalar (0 of 87 cells language-like); these are
the decodes behind it, so the reader can see what the instrument judged
rather than trust a threshold. For each head, the cell shown is the one
with the **lowest** full-stream plaintext bits anywhere in the 87-cell
grid — the most language-like letter stream the manuscript produces under
any (cipher × language) hypothesis. Excerpts are at fixed, deterministic
positions (stream start; one mid-stream replicate for the headline cell);
nothing is re-scored and nothing was selected by eye
(`scripts/phase6_samples.py` → `analysis/phase6/samples.{json,md}`).
`.` marks the manuscript's word boundaries; the evaluator sees the
unspaced stream.

**Homophonic head, German hypothesis, Currier B (IT2a) — the best cell in
the table** (plaintext 2.468 ± 0.030 bits/char, structure margin 1.25 of
the required 1.5, MDL total 2.750 vs no-cipher baseline 2.085
bits/symbol):

```text
cipher: psheoky.odaiir.qoyofseod.chypchey.ypchedy.ainchofochcphdy.dchey.aiin.adeeodyykecthhy.chedy.ytedy.dychecthedy.lr.oaiin.shcthy.eteeda.oloyykeedy.olchedy.galy.sheey.saiin.s.qokedy.cheos.ytedy.qokedyytedy.chekedydaiin.odam
decode: kersuni.utaccm.wuiuvesut.erikersi.ikersti.acperuvuerekrti.tersi.accp.atssutiinsenrri.ersti.insti.tiersenrsti.mm.uaccp.erenri.snssta.umuiinssti.umersti.oami.erssi.eaccp.e.wunsti.ersue.insti.wunstiinsti.ersnstitaccp.utal
```

mid-stream (chars 59,738–60,046), same key:

```text
cipher: cheody.ytchey.otaiin.kshd.qotar.chear.or.am.yteeo.dy.chedy.qokal.yteey.qotar.am.dcheey.teeody.oty.otchedy.daiir.aiim.ykeeedy.qoteey.qodaiin.okeeey.tchedy.opaiis.chedaiin.dsheedy.qopchedal.keo.daiin.otalaiin.oar.dor
decode: ersuti.inersi.unaccp.nert.wunam.ersam.um.al.inssu.ti.ersti.wunam.inssi.wunam.al.terssi.nssuti.uni.unersti.taccm.accl.insssti.wunssi.wutaccp.unsssi.nersti.ukacce.erstaccp.terssti.wukerstam.nsu.taccp.unamaccp.uam.tum
```

This is what "clean-text-level bits without a language-like structure
margin" looks like: the head reaches 2.47 bits/char by mapping the EVA
suffix system onto a small set of recurring pseudo-word endings — `ersti`,
`insti`, `wunam`, `accp`, `uni` repeat constantly, far beyond the
repetition rate of any natural text — so the low entropy is the
manuscript's own word-level repetitiveness re-expressed in Latin letters,
not German morphology. No content words, no inflectional variety, no
function-word skeleton.

**1:1 substitution, German hypothesis, Currier B (RF1b)** (2.805 ± 0.042
bits/char, structure margin 1.16): the same phenomenon under the strictest
head — `senit.salld.siiosttrievnnt.enit.tvist.stenievnit...` — pronounceable,
repetitive, wordless.

**Naibbe head, Latin hypothesis, Currier A (RF1b) — the MDL-table top**
(total 2.123 vs baseline 2.231 bits/symbol, but plaintext 3.538 bits/char,
structure margin 0.07; one letter per unigram token, two per bigram
token):

```text
cipher: ykal.ar.taiin.shol.shory.y.kor.sory.ckhar.ory.kair.shar.cthar.cthar.dan.syaiir.or.ykaiin.shod.cthes.daraiin.sy.soiin.oteey.oteor.daiin.okaiin.or.okan.sairy.chear.cthaiin.cphar.odar.shol.cphoy.yshey.shody.okchoy.otchol
decode: nc.e.na.i.ta.e.ae.li.is.ui.ar.p.is.is.ii.oc.i.na.ta.il.la.ii.is.o.st.i.s.i.ei.bi.eu.ia.ms.or.i.me.ni.tu.te.si
```

The Naibbe hypothesis wins the description-length comparison by
compressing the glyph stream ~2:1 — and this is the plaintext that
compression buys: an unstructured one-and-two-letter stream at 3.5
bits/char, deep in the wrong-key plateau. The arithmetic head's best cell
(Italian, Currier B, one letter per Boxer token) is the same story at 4.0
bits/char: `i.o.n.v.o.n.n.n.t.h.b.s.a.c.o.e.t...`.

**Contrast — a true decipherment through the identical pipeline.** The
positive control (synthetic 1:1 Latin, 2,000 chars) solved by the same
inner search, letter accuracy 1.000 against the ground truth:

```text
bresnonsuntfleuothomianecesseestadhibereautcucurbitamquodsimaioreritingluuiesettraicerenonpossuntaquamulsaeritadhibendausqueaddeclinationemetomniacataplasmasnoninfrigdentsedsubindecaliderefouentur...
```

**Contrast — the other side of the structure margin.** Every scored window
is paired with a letter-shuffled copy of itself; the abstention rule asks
the decode to beat its shuffle by ≥ 1.5 bits/char. The best VMS decode
beats it by 1.25 — visibly more ordered than its own shuffle
(`ieeiiktwnnactpsiiise...`), but by the amount Phase 3 measured for
wrong-hypothesis decipherments (0.5–1.9), not for true ones (1.6–2.9).

## 3. Assumptions the result depends on

1. **The language inventory.** Latin, Italian, German — a three-language
   frozen inventory (task 0.2). Any ranking is a ranking *within* this
   inventory. The contamination set (Dutch, English, French, Spanish)
   measures what an untrained language looks like to the instrument; it
   does not extend the inventory.
2. **The cipher inventory.** Four cipher classes, each as pinned in Phase 0
   (Naibbe @ `df3d074`, `pseudo_vms` @ `e324bee`). A manuscript enciphered
   by none of them — or by a verbose cipher whose word-identity statistics
   the framework excludes by construction (design §10) — is expected to
   produce plateau-level cells under every hypothesis, which is what the
   abstention rule is for.
3. **Presentations.** How a transcription becomes each head's input is a
   Phase-6 decision (`docs/phase6_status.md` §6.1): EVA characters as
   symbols; Naibbe-parseable words only (65–78% of characters); Boxer's
   16 most frequent glyph units for the arithmetic head, with the observed
   word boundaries as the segmentation because the cipher's sorted-token
   signature is absent from the glyph stream. Each choice is reported with
   its coverage; a different tokenization of EVA (glyph units, benched
   gallows as single symbols) is a different ciphertext.
4. **Bound-tightness comparability (R1).** The ranking compares per-language
   ELBOs, i.e. *bounds*. The bound gap relative to an autoregressive
   reference was measured (latin +0.138, italian +0.013, german +0.205
   bits/char; `docs/phase6_fairness_audit.md`) and is **not** subtracted —
   subtracting it broke every same-text comparison in Phase 3 — but carried
   as the systematic uncertainty of every cross-language margin. Margins
   below 0.01–0.19 bits/char (pair-dependent) are unresolved at the
   calibration's precision. The audit's escalated findings (the offsets
   depend on the reference tier more than on their own s.e.m.; they differ
   by language beyond document dispersion) mean the gap estimate is an
   estimate, not a proof, and this is the residual bound-comparability risk
   the design §10 names.
5. **Search fairness.** The inner n-gram search is harder for Latin at every
   rung (Phase 5: 93–97% vs 100% solve success at ≥ 200 chars; one of six
   Zodiac-class Latin instances unsolved at 480 restarts). "Could not find
   the key under Latin" and "not Latin" are distinguishable only through the
   per-language solve-success numbers reported next to every recovery
   figure, and the same budget was given to every hypothesis here.
6. **Length and regime.** Language-level ranking is supported at ≥ 200
   plaintext characters of a *correct* decipherment (0.989 language /
   0.991 family on the 1:1 suite); the ELBO is the worse judge below ~100
   characters; on partial decipherments (rung 4 on synthetics: SER 0.4–0.7)
   the language signal survives only at family resolution and at margins of
   the order of the calibration uncertainty.

## 4. Resolution of the claims

- **Family level is the honest resolution.** Within the inventory the
  within-Romance pair (Latin / Italian) is resolved at ≥ 100 characters of a
  correct decipherment; the residual error at ≥ 200 characters is
  cross-family (high-entropy Latin documents tying with the German
  condition). Within-Germanic resolution does not exist inside the inventory
  (German is its only Germanic member); the Dutch / English contamination
  set measures whether a Germanic language outside the inventory is called
  German, which is a family-level statement only.
- **A language call requires a language-like cell.** If every cell of a
  dialect fails the abstention rule, the language ranking among the
  non-decipherments is reported but carries no evidential weight — the
  Phase-4 plateau finding: every wrong-key cell sits 0.8–1 bit below
  shuffled text and looks like partial structure.
- **Cross-head MDL ranking is a statement about description length, not
  decipherment.** On synthetics it picks the true cipher class 24/24 when
  the true head yields real plaintext. On the manuscript the comparator
  says which cipher grammar compresses the glyph stream best relative to
  the no-cipher baseline (the stream's own n-gram cross-entropy); a head
  that compresses the manuscript without producing a language-like decode
  is a model of the manuscript's word structure, not a key.

## 5. Residual risks, stated

- Bound comparability across languages (R1) — measured, not eliminated.
- Transcription dependence — two EVA transcriptions and one independent
  glyph transcription were used; agreement between them is reported, but
  all three share the EVA-family decomposition of glyphs.
- Coverage — the Naibbe and arithmetic heads explain subsets of the
  manuscript (65–78% and 90–92% of symbols); totals are reported per
  covered symbol and per all symbols with uncovered symbols charged at
  the stream's own best held-out n-gram cross-entropy (order-0 entropy
  only as a fallback; `vms/apply.py:810–818`, `docs/phase6_status.md`
  §6.2) *(corrected 2026-09-01; originally read "charged at order-0
  entropy")*.
- Anchors — Borg (Latin) is the only literature anchor the frozen
  evaluator can score end-to-end; Zodiac-408 (English) is a pre-diffusion
  n-gram baseline only; BnF fr2988 was not available.
- Single evaluator seed — the 25M seed replication (task 4.7) is paused;
  every Phase-6 number is one 85M evaluator. *[Still paused as of
  2026-09-01.]*

## 6. Reproducibility

`scripts/fetch_external.py` (pinned repos + anchor texts) →
`scripts/vms_apply.py --stage prepare|solve|score|report` →
`scripts/vms_controls.py --stage generate|solve|score|report` →
`scripts/anchors.py --stage prepare|solve|score|report` →
`scripts/phase6_analysis.py` → `scripts/fairness_audit.py --phase-tag phase6`
→ `scripts/phase6_check.py`. All artifacts under `DATA_ROOT/analysis/phase6/`;
evaluator fingerprint in `analysis/phase5/evaluator_freeze.json`.
