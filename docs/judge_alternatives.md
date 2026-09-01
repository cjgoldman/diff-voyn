# Alternative language judges for partial decodes — team exploration (2026-08-29)

## The problem

The frozen judge (`vms/apply.py::ABSTAIN_RULE`: plaintext ≤ 3.0 bits/char AND
structure margin ≥ 1.5, margin = bits(letter-shuffled decode) − bits(decode)
under the Phase-C diffusion evaluator) cannot call a *partial* decode. The
margin falls ~0.085 bits/char per 0.01 SER and crosses 1.5 at SER ≈ 0.10
(German) / 0.045 (Latin) / < 0.03 (Italian, whose truth is only 1.56)
(`docs/alt_loop_plan.md` §8.7). The wildcard→anneal loop leaves residuals at
SER 0.05–0.24, so a decode that a fluent reader would read at a glance is
scored "not language". Previously refuted: blanking low-confidence letters
(`docs/confidence_mask_probe.md` §9); exact-word lexical coverage works for
German/Italian but leaks on Latin (`docs/lexical_coverage_note.md`).

Five explorers took one family each and ran cheap probes on the existing
`judge_at_ser` keys (A-like wordhom truth, uniform/rare-first corruptions at
recorded SER, loop finals, stuck keys), the wordhom/Phase-6 control decodes
and the 87 VMS cells. Scratch code and JSON artefacts:
`/tmp/claude-1000/-workspace/*/scratchpad/{lexical,slope,learned,noisychannel}/`
(session-local; nothing tracked was modified, no threshold has been frozen).
The full per-family reports are in the same directory (`*_report.md`).

## Why the current judge fails — the shared diagnosis

Three independent measurements agree that the margin is **context-limited**,
not averaging-limited:

* the cost of one more wrong letter *halves* by SER 0.15 (German 9.9 bits at
  truth → 6.4 at SER 0.10 → 4.5 at 0.20; slope explorer) — a wrong letter
  costs what its context could have predicted, and the context is itself
  wrong at the same rate;
* the per-position surprise is never bimodal (bimodality 0.41–0.59 for every
  key, positives and negatives alike): the t-averaged diffusion surprise
  charges ~4 bits at every position as t → 1, so "wrong letters" never form a
  separable mode and no quantile/mixture statistic can find them;
* repairing letters before judging (noisy channel) helps the shuffled
  baseline *more* than the decode (a 5-gram Viterbi repair with λ = 7–9 bits
  per edit changes 0.5–13 % of the decode's letters but 18–28 % of the
  shuffle's), so every repair budget shrinks the margin — the
  difference-in-differences (how much more a fixed budget helps the decode
  than its own shuffle) is negative on every key of both languages (−0.9 at
  truth, −0.5 at SER 0.10, −0.4 on voynichesque), and charging the edits
  merely adds λ × (edits_shuffle − edits_decode) ≈ 1.2–1.9 bits/char to
  positives and controls alike. Deeper reason (noisychannel explorer §3):
  the Phase-C ELBO is *already* a noisy-channel marginal — the Phase-2
  curriculum trained on the noised stream as target under a self-consistent
  wrong-key channel — and its price was the compressed clean→wrong dynamic
  range recorded in `docs/phase2_status.md` (2.2–2.8 → 0.9–1.5 bits/char);
  the ~8.5 bits per wrong letter is what is left *after* that softening, so
  an evaluation-time channel just repeats the trade at another λ.

Consequently **no per-character likelihood statistic — level, slope, shape,
repaired, or masked — is SER-invariant**. What survives are statistics that
(a) exploit *where* the search's errors are (rare cipher types, > 95 % of
residual errors), (b) read *words*, or (c) are trained to be error-tolerant.
The literature survey confirms this from the outside: no published
decipherment system calls partial decodes automatically; every historical
call (Copiale `CEREMONIE…DER`, Zodiac-340 `HOPE YOU ARE HAVING LOTS OF FUN`)
was a human reading contiguous word runs, and the only key-error-tolerant
automated statistics in the field are dictionary/coverage statistics and
cipher-invariant pattern statistics (Hauer & Kondrak 2016).

## Results by family

| family | statistic | callable SER (la / it / de) | negatives | verdict |
|---|---|---|---|---|
| frozen rule (reference) | structure margin ≥ 1.5 | 0.045 / < 0.03 / 0.10 | voynichesque ≤ 1.26 (Phase-6 heads), ≤ 0.66 (wordhom head) | baseline |
| **rare-aware margin** (slope explorer §3) | paired margin charged only at positions whose own type and ±2 neighbour types have ≥ 10 occurrences; nothing masked | loop finals SER 0.17 / 0.21 / 0.13 → 1.69 / 1.35 / 2.04 (full margin 1.01 / 0.83 / 1.42); rare@0.65 (SER 0.23) → 1.67 / 1.46 / 1.93 | fall: max 0.80 (contamination), voynichesque ≤ 0.68, VMS 0.25–0.50 | **adopt as criterion 2**; needs 4-seed replicate (SEM ≈ 0.13 at 400–600 charged positions) and own threshold |
| **lexical density** (lexical explorer) | distinct exact top-5000 words of length ≥ 7 per 1000 chars, minus shuffle; per-language threshold 0.35 × held-out median | ≈ 0.12 / 0.10 / 0.12 on the worst of 8 corruption seeds at 3000 chars (mean on threshold at 0.20); anneal finals score at or above truth | all ≤ 0.34 normalised (max: Spanish contamination under Italian, wordhom voynichesque); VMS ≤ 0.19 | **adopt as criterion 3**; Latin now separates 4–8×; fuzzy matching REJECTED (helps n-gram-fitted negatives more: max-neg/truth 0.91 → 1.01 Latin); stem lexicon inferior |
| **learned judge** (learned explorer) | logistic regression on language-symmetric features: Phase-4 LID head (max-prob, entropy, abstain, 2nd) + denoiser posterior profile (`heads/posterior.py`, mask 0.3, 8 draws); threshold 5 % FPR | 1.00 call rate at SER ≤ 0.15 in every language under leave-one-language-out; 0.70–1.00 at 0.20 | shuffle / voynichesque / random key 0.00–0.03; **soft band** wrong keys SER 0.45–0.60 leak 0.05–0.97 (corrupted German ≈ clean Italian) | **strongest tolerance, weakest validity**; pursue as the acceptance instrument only with the soft band and real search residuals in the negative set; fairness = symmetric data + pre-registration, not a bound |
| LID head alone | Phase-4 head argmax | language correct ≥ 0.97 through SER 0.30, 0.92 at 0.45 | abstain class fires 0 % on wrong keys to SER 0.75 | error-tolerant *language namer*, useless *rejector* (Phase-4 caveat, now quantified) |
| noisy-channel repair (noisychannel explorer) | 5-gram Viterbi repair, λ = 7–9 bits/edit, then frozen judge; also "charged" margin adding λ × edits | SER falls only 0.10 → 0.08, 0.20 → 0.16; margin *drops* (German anneal 2.17 → 1.25–1.56, Latin 1.77 → 1.10–1.23; uni@0.10 1.55 → 1.05–1.25) | negatives also drop (voyn 0.66 → 0.12–0.24) but the positive/negative gap does not widen (charged: German uni@0.15 2.03 vs voyn 1.65; original 1.26 vs 0.66) | **reject**; DiD < 0 everywhere, per-language false-edit bias (clean German 0.5 % vs Latin 1.2 % rewritten at λ = 7 — an R1 violation), and the repairer *worsens* the anneal finals (0.050 → 0.054, 0.069 → 0.076: their residual errors sit at the n-gram optimum); the denoiser-posterior version was also negative — iterative argmax re-fill *raises* SER (0.10 → 0.14, clean text gains 5–9 % errors) |
| corruption slope | d(bits)/dδ under injected errors | ≈ 4 × margin + 2: the level's own derivative; separates worse than the margin at SER 0.10 (6.4 vs 4.4 negatives) and not at all at 0.20 | — | reject |
| surprise-distribution shape | bimodality, quantiles, fraction < 1 bit | no bimodality; "frac < 1 bit" is language-unfair (German truth 0.13, Italian 0.013) | — | reject |
| type-level consistency | Spearman(type-mean surprise, log occurrence) | −0.34 … −0.67 for search-profile partial decodes; truth, negatives, VMS ≈ 0 (−0.16 … +0.12) | — | **diagnostic only**: the signature of "frequent types right, rare types wrong"; the manuscript shows none of it |
| clean-run statistic (literature #2) | longest / 95th-pct DP-segmentable in-lexicon run vs shuffle | not run | — | cheap follow-up inside the lexical script; the human criterion in every historical call |

## Synthesis

1. **Three complementary instruments beat one.** The rare-aware margin keeps
   the bits/char units, CRN scoring and the R1 argument (the charged set is
   defined by ciphertext type counts only, identical for every hypothesis and
   the shuffled control) and is *specific to the search's error profile*
   (uniform errors: uni@0.10 1.51 vs 1.50, no gain). The lexical density is
   human-shaped, cheap (CPU, seconds), and tolerant of *any* error profile
   to SER ≈ 0.12, but is genre-dependent (Latin pharmacological recipe text
   scores as a negative) and exact-orthography-dependent. The learned judge
   tolerates the most (SER 0.15–0.20) but is a classifier, not a bound: its
   fairness rests on symmetric training data and pre-registration, and its
   soft band (SER 0.4–0.6, where the manuscript's own stuck keys live at
   0.64–0.76 on the A-like positives) is exactly where an over-optimistic
   call would come from.
2. **The two hard cases remain hard.** Italian: its truth margin is 1.56–1.70
   under both margin variants, its ≥ 7-letter lexicon is thin (0.86 M-token
   corpus, f_needed 0.34, the binding language), and it is the language whose
   clean baseline is highest for the learned judge — every instrument has
   the least headroom on Italian. Latin: fixed for lexical density (factor
   4–8 vs 0.91), but its true-text coverage has a genre tail.
3. **The manuscript is negative on every instrument** that produced VMS
   numbers: rare-aware margin 0.25–0.50 (threshold candidates ≥ 1.3 from the
   positive/negative gap), lexical density ≤ 0.19 normalised (threshold
   0.35), type-level Spearman ≈ 0. This is *not* a pre-registered result
   (thresholds were read off the same probes) — it is consistency with the
   abstention, nothing more.
4. **Negatives are thin exactly where they bind.** The binding negatives are
   the wordhom-head fits on voynichesque ciphertext (n = 3, one per language)
   and Spanish/Dutch contamination; the 1.5 rule itself was set by the
   sub1to1/homophonic voynichesque ceiling, and wordhom-head negatives never
   exceed 0.66 under the frozen pipeline (the threshold is inherited, not
   measured, on that head). The manuscript-shaped wordhom battery
   (`analysis/altloop/battery`, chains running) is the right enlargement and
   must be scored on every candidate before any threshold is fixed.

## Recommended programme (pre-register before any manuscript number is re-read)

1. **Implement all three as reportable statistics**, no call rule yet:
   * `rare_aware_margin` in `heads/two_tier.py` / `masked_bits.py`
     (position filter from ciphertext type counts, occ ≥ 10, ±2 — both
     chosen a priori, keep them), run through the exact Phase-6 loop
     (13 windows × 4 seeds × 3 conditions); report the type-level Spearman
     alongside;
   * `lexdens7` from the lexical probe (exact, K = 5000 and 20000, L ≥ 7,
     whole available window, ≥ 6000 chars where possible, per-language
     held-out medians from prose only); add the clean-run statistic to the
     same script;
   * learned judge as a module: fixed-seed feature extractor (LID features +
     posterior profile, ~6 s per 1024-char window per key), LR trained on
     symmetric synthetic positives (SER 0–0.25, three error profiles)
     **plus** the real search residuals in `judge_at_ser.json` /
     wordhom artefacts, negatives covering wrong keys at SER 0.4–1.0,
     shuffles, voynichesque, cross-language and dirty decodes.
2. **Score the full control battery** (Phase-6 controls on all four heads +
   wordhom battery when its chains finish) on all three; set thresholds per
   instrument — rare-aware margin one threshold, lexical density one
   normalised factor f, learned judge one FPR quoted *on the soft band*, not
   on shuffles — and write them into a `docs/judge2_rule.md` with the
   decision rule (proposed: a call requires the frozen rule OR
   [rare-aware margin AND lexical density], with the learned judge reported
   and used for ranking/acceptance in the loop, not for the call) before the
   VMS cells are scored.
3. **Only then** re-score the 12 VMS wordhom anneal finals and the Phase-6
   cells. Expected outcome from the probes: German and Latin A-like loop
   finals become calls, Italian stays borderline, the manuscript stays
   negative.
4. Use the learned judge (or the rare-aware margin) inside the alternating
   loop's acceptance rule — the surviving ranking signal at SER 0.3 was the
   argument for a judge-in-acceptance (`docs/alt_loop_plan.md`), and the
   frozen judge's noise there was the obstacle.

Not pursued further: noisy-channel repair, slope/shape statistics, fuzzy
lexical matching, argmax denoising, position-level confidence masking.
