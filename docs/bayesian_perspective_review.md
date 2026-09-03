# A Bayesian search-and-detection review of diff-voyn (2026-09-02)

> **Record status (banner added 2026-09-02):** an outside-perspective review written 2026-09-02, after the four manuscript campaigns and the control batteries of 2026-08-24 → 08-31. It changes no verdict. It adds one reportable statistic (the evidence-odds table, `scripts/evidence_odds.py`, §5) and a ranked list of proposals (§4) that are **not yet pre-registered, implemented or run** except where §5 says so. **Current project position: `docs/project_status.md`.**

Written for a reader who knows the project's goal but not its jargon. Where a term from the project's own records is unavoidable it is explained on first use. Sources read: `docs/project_status.md`, `docs/diffusion_guided_search_method.md`, `docs/project_goals_and_progress.md` §2.7–2.9 and §8–9, `docs/wordhom_study.md` §1–2, `docs/alt_loop_plan.md` §8.4–8.6 and §10, `docs/judge_alternatives.md`, and the solver code in `diff_voyn/heads/{wordhom,wordhom_state,altloop,posterior}.py` and `scripts/altloop_vms.py`.

---

## 1. What the project is doing, in plain words

The project asks: "if the Voynich text were a coded version of Latin, Italian or German, could we find the code, and would the result read like language?" It tries several shapes of code, searches for the best code under each shape and each language, and then a fixed *referee* (the frozen judge, `vms/apply.py::ABSTAIN_RULE`) decides whether the result is language-like. The answer so far is "no" on every cell, and the team has been careful to show on planted test cases that the search and the referee do work when there is something to find.

Two facts drive everything:

- **Most words in the manuscript occur only once or twice.** Currier A has about 3 uses per distinct word, Currier B about 4.6, and 69–75 % of the distinct words occur exactly once. Under a "each word stands for a letter" code (the word-homophonic hypothesis), most entries of the codebook are supported by a single observation. No search can pin those entries down one by one.
- **The referee is strict.** It says "language" only when the decode is nearly perfect: roughly 5 % wrong letters for German, 5–7 % for Latin, and never for Italian at the manuscript's shape, because even a perfect Italian decode scores just above the line.

The team already found the single most Bayesian idea in the project by trial and error: the *hapax wildcard* objective. Charging a once-seen word zero and letting the surrounding context reset (`wordhom_state.py`, the wildcard branch of the scorer) is, at that word's own position, exactly what "I do not know this letter, so average over every possibility" costs, because the language model's probabilities over all letters sum to one and the log of one is zero. The context reset for the *following* letters is an approximation of the same averaging. That change moved the solvable frontier from about 8 uses per word to about 4. The recommendations below mostly say: do that idea on purpose, everywhere, and stop turning uncertainty back into hard guesses.

A second thing to say up front: nowhere in the project is there a prior probability, a *power* statement ("if the manuscript were a cipher of this kind, how often would we call it?"), or a likelihood ratio. Every decision is a fixed two-part threshold on a point estimate, and every multi-hypothesis comparison is a bit-count ranking. That is not wrong, but it is why the verdict cannot say *how much* it excludes. §5 fills that gap from numbers the project already has.

---

## 2. Where the current approach leaves value on the table

### 2.1 Every stage commits to a single best answer

The search returns one key. The anneal step forces every once-seen word back to one letter. The referee scores that one committed decode. A hard threshold turns the referee's number into yes or no. At no point does the pipeline carry "how sure are we about each entry" forward.

This matters because the project's own diagnostics say the errors in a near-solved decode come almost entirely from rare words (more than 95 %) and the referee charges roughly 8.5 bits for each wrong letter. The pipeline therefore commits its least-evidenced guesses and then pays the referee's full penalty for them. A soft-assignment routine (`WordHomophonicHead.em_phase`) already exists in the head, but `solve` never calls it.

### 2.2 "Abstain" is one word for two different situations

For Currier B the solver finds, and the referee calls, a planted clean German or Latin cipher of B's shape, and the manuscript still fails. That is genuine evidence against the hypothesis. For Currier A nothing of A's shape has ever been solved, so there is no evidence either way. The status page says this in prose, but the reported verdict is the same word in both cases, and the number that separates them (how much more likely the manuscript's score is under "real cipher" than under "not a cipher") was never computed. It now is (§5).

### 2.3 The test has little power against the *plausible* version of the hypothesis

A real fifteenth-century cipher would reach us with transcription errors. The battery shows that at 10 % noise the *true* key's referee score drops to 1.54 / 1.06 / 0.91 bits (German / Latin / Italian), below the 1.5 line for two of the three languages, and the solver fails on every dirty-10 % positive, leaving margins of 0.37–0.46 that are indistinguishable from gibberish. The manuscript's word-homophonic cells sit at 0.30–0.51. Against the hypothesis "a somewhat noisy real cipher", the current test essentially cannot say no. This is the first thing a detection specialist would want stated.

### 2.4 The threshold is a point estimate wrapped around noisy numbers

The referee's margin comes from 64 masking draws times 4 replicate seeds, and the same-text uncertainty is 0.07–0.19 bits. Decisions compare a point estimate against 1.5 (`apply.py`, the verdict lines; the standard error is computed but unused). The 1.5 line was set as a round number just under the Phase-3 positive band, not tuned against negatives, and the margin is the only load-bearing half of the rule (gibberish passes the ≤ 3.0 bits/char half). Two cheap sources of noise remain: the "shuffled copy" reference is a **single** random permutation per window, and the battery negatives were run with **one seed**. A Bayesian would average the shuffle reference over several draws, carry the replicate uncertainty into the decision as "probability the true margin exceeds the line", and replicate the negatives that bound the corridor.

### 2.5 The key space for Currier A is too free, and the fix is being sought as a guess

The status page says A needs "a compact key rule" and that no design exists; the one historical rule tried (Polygraphia) is not representable and is quarantined. From a Bayesian view the missing ingredient is a *prior that ties codebook entries together*, learned from the data rather than guessed as a formula. Word tokens that share a prefix, a suffix or a length could share a letter distribution, and how strongly they share is itself a quantity to estimate. That lowers the effective number of free entries from about 5 000 to a few hundred, which is exactly the lever the findability wall depends on.

### 2.6 The manuscript sits *above* every same-pipeline negative

On the word-homophonic pipeline the manuscript scores 0.30–0.51 against negatives at 0.17–0.48; across the 72 alternating-loop cells of 2026-08-26 it spans 0.36–1.05 against negatives ≤ 0.48. The status page correctly says this depends on how the key was obtained, but it is a consistent signal that the searcher finds *something* exploitable in the manuscript that it does not find in shuffled or gibberish text. The Phase-6 voynichesque controls (glyph grammar with no content) reach the same band, so the leading explanation is word-internal glyph structure, not language. That is a hypothesis one can test directly (R5) rather than leave as a caveat.

---

## 3. Recommendations, ranked by expected information per unit of effort

### R1. Report odds, not a pass/fail flag — **done 2026-09-02, §5**

Build two *reference piles* of referee scores per (dialect shape × language): one from planted true ciphers run through the identical solver (clean, dirty-5 %, dirty-10 %, mixed), one from same-shape non-ciphers (shuffled, voynichesque, the true cipher under the wrong language). For each manuscript cell, report how many times more likely its score is under the "real cipher" pile than under the "not a cipher" pile. This is a likelihood ratio; multiplied by prior odds it is a Bayes factor. Add a **power line** to every verdict: "if the manuscript were a cipher of this kind with this much noise, the pipeline would call it X % of the time."

### R2. Stop hard-committing the rare entries; let the referee average over them (moderate; pre-register first)

After the wildcard stage, instead of annealing the once-seen words to a single letter, draw their letters from the denoiser's own posterior (`heads/posterior.py` already computes it), score the referee on several such fills, and average. This is a *posterior-predictive* score. It differs from the confidence-mask probe that was rejected (`docs/confidence_mask_probe.md` §9): masking removes context, which the probe showed hurts; sampling supplies plausible context and charges the model's actual uncertainty. A planted A-like Italian decode at 12 % error scores 1.39, its truth 1.56, and most of that gap is in rare words. This is a pre-registrable test on the existing `judge_at_ser` keys: if the averaged score closes the gap on positives without lifting negatives, adopt it as a reported statistic, not yet a call rule.

### R3. Carry a population of keys with weights instead of one chain (moderate; mostly reuses code)

The solver is simulated annealing from 16 restarts, keeping the best. A sequential-Monte-Carlo or parallel-tempering version keeps many keys alive, weighted by the objective, and resamples. Three things fall out:

- **Per-entry confidence.** Across the population, does every key agree on what word W means? Frequent words that agree across seeds and arms are settled; entries that vary are not. The 62/72 identical-key result of 2026-08-26 already hints that the manuscript's optimum is stable. A stable optimum the referee rejects is evidence the hypothesis is wrong for that space; an unstable one is a search failure. That distinction is what the B verdict needs.
- **The anneal becomes tempering**, a principled schedule instead of the hand-set 0–40-round window the battery found binding on perturbed positives.
- **A marginal-likelihood estimate** (the normalising constant the population tracks) is what a real Bayesian model comparison uses instead of "best key found" (R5).

The unused `em_phase` is a soft-assignment starting point; a tempered population of `WordHomObjectiveState` chains is the smallest version worth building.

### R4. A learned, tied key prior for Currier A (higher effort; the only lever proposed for A)

Give the codebook structure: each word token's letter distribution depends on a few glyph features (first one or two EVA glyphs, last glyph, length) through a small shared table with a concentration parameter that says how strongly tokens sharing a feature share a letter. Fit the concentration from the data. Then compare the evidence for "tied" versus "untied" codebooks.

- On synthetic ciphers whose key *has* such structure, the tied model should recover keys well below the current wall, because a once-seen token borrows evidence from its neighbours in feature space.
- On synthetic ciphers with a random key, the tied model should learn to untie itself (concentration → 0) and lose nothing.
- On the manuscript, the evidence comparison is itself a result: "is there any feature-shared rule under which A reads as language?" The Naibbe compression finding (a token grammar compresses A, not B) suggests A's tokens are feature-structured, which is the precondition.

This is the Bayesian form of the "compact key rule" the status page asks for: learn the rule's strength rather than guess its formula.

### R5. Test for content without solving for the key (research bet; medium cost)

The Naibbe twin test (real versus letter-shuffled source through the same pipeline, indistinguishable at 3.4 uses per word) compared *best keys*. Below the wall the best key is meaningless, but the *average over all keys* (the marginal likelihood from R3) can still differ between text with content and text without. Compute a population-based marginal-likelihood estimate for real-source and shuffled-source twins at manuscript shape, then for Currier A and B, and ask which pile the manuscript resembles. This attacks §2.6 directly: shuffled-source twins keep the glyph grammar and lose the content.

### R6. Decide with uncertainty and choose experiments by information value (cheap)

- Report P(margin > 1.5) from the 4 replicate seeds; average the shuffle reference over several permutations (one line in `apply.py`); replicate the single-seed battery negatives.
- Before spending more GPU on the manuscript, rank candidate runs by how much they would move the R1 odds. Cost anchors: ≈ 40 min per A-like cell, ≈ 100 min per B-like cell *[measured 2026-09-03: ≈ 2 h per B-shape seed-run]* on the two GPUs. Highest value: (a) dirty-5 % / 10 % positives at **B shape** in all three languages, since B is the dialect where a null result is informative and the only dirty positives so far are A-shape (about 5 h for six cells); (b) second seeds on the borderline positives and on the negatives that set the 0.48 ceiling; (c) Italian, where the referee has no headroom even at truth, so any Italian claim needs R2 or a second criterion first. Currier A runs have near-zero information value until R4 exists.

### R7. Things a Bayesian would *not* recommend redoing

- Noisy-channel repair before judging: the difference-in-differences result (repair helps the shuffle more than the decode, `docs/judge_alternatives.md`) is the right diagnostic and the negative conclusion stands.
- Putting the referee inside the acceptance rule: tested 2026-08-28, no gain, and the reason (the referee's per-symbol preference is not truth-directed either) is sound.
- Shortening the text to parallelise: lowers uses per word; correctly rejected 2026-09-01.

---

## 4. What this changes about the headline

Nothing here contradicts the abstention. It sharpens it into three statements a reader can weigh, each now backed by §5:

1. Currier B is not a *clean* word-homophonic cipher of German or Latin (strong evidence, odds worse than 1 : 400 on every B cell).
2. Currier B as a *noisy* cipher, or as an Italian one, is not excluded: the odds against a 10 %-noisy cipher are only about 1 : 3–4 in any language, and the pipeline has never called a dirty-10 % positive.
3. Currier A is untested by current methods on the noisy variants; R4 is the first proposal that could change that, and R5 the first that could say whether there is content at all.

---

## 5. R1 as run: the evidence-odds table (2026-09-02)

`scripts/evidence_odds.py` (CPU, seconds) reads the recorded judge artifacts and writes `DATA_ROOT/analysis/evidence_odds/odds.{json,md}`. Nothing was re-scored. The statistic is the structure margin of the final key; each pile is summarised by a Student-t predictive with a floor of 0.10 bits on its spread (the judge's replicate noise is ≈ 0.07; results below were also checked at a 0.20 floor and the conclusions do not move). Piles are small (2–6 cells), so read the ratios as orders of magnitude, and use the nonparametric column (fraction of the pile at or below the manuscript's margin) as the sanity check.

### 5.1 Word-homophonic head, solver of record (wild → anneal, post-all arm, both seeds)

Reference piles per hypothesis language (anneal-final structure margin, bits/char; `called` = fraction the frozen judge called):

| hypothesis | not a cipher (shuffled, voynichesque, wrong language) | clean real | mixed 80/20 | dirty 5 % | dirty 10 % |
|---|---|---|---|---|---|
| German | n 6, 0.17–0.44, called 0 | n 5, 1.88–2.40, called **1.00** | n 1, 2.13, called 1 | n 2, 1.13–1.51, called 0.50 | n 1, 0.44, called 0 |
| Latin | n 6, 0.18–0.48, called 0 | n 5, 1.70–1.88, called **1.00** | n 3, 0.52–1.24, called 0 | n 2, 0.41–0.43, called 0 | n 1, 0.37, called 0 |
| Italian | n 6, 0.21–0.43, called 0 | n 5, 1.39–1.46, called **0** | n 1, 0.59, called 0 | n 1, 0.62, called 0 | n 1, 0.46, called 0 |

Manuscript cells. "LR" is the likelihood of the cell's margin under the named real pile divided by its likelihood under the not-a-cipher pile; values below 1 favour "not a cipher of this kind". The two seeds agree to within 0.05 bits on every cell, so one row per cell is shown (seed 0; both are in `odds.md`).

| cell | margin | position in not-a-cipher pile | LR clean | LR mixed | LR dirty 5 % | LR dirty 10 % |
|---|---|---|---|---|---|---|
| A :German (IT2a / RF1b) | 0.40 / 0.44 | 0.67 / 1.00 | < 1/1000 | 1/64 / 1/46 | 1/21 / 1/15 | 1/2.6 / 1/1.9 |
| A :Latin (IT2a / RF1b) | 0.50 / 0.51 | 1.00 / 1.00 | < 1/1000 | 1/2.5 / 1/2.2 | 1.6 / 1.6 | 1/1.5 / 1/1.4 |
| A :Italian (IT2a / RF1b) | 0.48 / 0.51 | 1.00 / 1.00 | 1/400 / 1/230 | 1.0 / 1.5 | 1/1.1 / 1.4 | 1.1 / 1.5 |
| B :German (IT2a / RF1b) | 0.30 / 0.33 | 0.50 / 0.50 | < 1/1000 | 1/96 / 1/92 | 1/34 / 1/32 | 1/4.0 / 1/3.7 |
| B :Latin (IT2a / RF1b) | 0.35 / 0.37 | 0.50 / 0.67 | < 1/1000 | 1/8.6 / 1/7.9 | 1/1.5 / 1/1.2 | 1/3.2 / 1/3.0 |
| B :Italian (IT2a / RF1b) | 0.36 / 0.37 | 0.67 / 0.67 | < 1/1000 | 1/4.4 / 1/4.1 | 1/4.8 / 1/4.4 | 1/3.3 / 1/3.1 |

*[Superseded 2026-09-03 for the dirty columns — after the B-shape dirty run the regenerated table gives German dirty-5 % LR 1/100 / 1/96 on B and 1/58 / 1/39 on A (seed 0), German dirty-10 % 1/1.8 / 1/1.4 on B, Italian dirty-5 % 1/12 / 1/10 on B; see the addendum below and `docs/project_status.md` §5.18. The clean and mixed columns are unchanged.]*

Power (fraction of the real pile the frozen judge called): clean German 1.00, Latin 1.00, Italian 0; dirty-5 % German 0.50 *[0.75 after the B-shape run, 2026-09-03 addendum]*, Latin 0, Italian 0; dirty-10 % 0 in every language.

**Reading.**
- Against a *clean* word-homophonic cipher of German or Latin the odds are worse than 1 : 400 on every cell, A and B alike, and the pipeline would have called such a cipher every time it was planted. That is the strong half of the abstention, and it now has a number.
- Against Italian the clean pile is never called even at truth, so "not called" carries no information; the odds above for Italian come only from the margin gap (1.39–1.46 versus 0.36–0.51) and should be read as "not a clean Italian cipher *as this referee sees one*".
- Against a *dirty-10 %* cipher the odds are 1 : 2–4 in every language and dialect: essentially no evidence. Against dirty-5 % Latin the manuscript's A cells sit exactly where the unsolved dirty positives sit (LR ≈ 1.6, i.e. slightly *favouring* the real pile). The "plausible" version of the hypothesis is not excluded.
- Every A cell sits at or above the top of its not-a-cipher pile (position 0.67–1.00); B cells sit in the middle (0.50–0.67). The pipeline extracts a little more from the manuscript than from same-shape gibberish, most so on Currier A, which is the §2.6 observation in numbers.
- Missing piles that would sharpen this: dirty positives at B shape (none exist) *[done 2026-09-03 — addendum below]*, second seeds on the dirty-5 % cells *[German done at B shape, 2/2 called]*, and anything at all for Italian that the referee can call.

**Addendum 2026-09-03 — dirty positives at B shape (record: `docs/alt_loop_plan.md` §10.6).** Six `dirty/<lang>/Blike_{s05,s10}` cells were run through the solver of record (two seeds on s05). The dirty-5 % German cell is **called on both seeds** (SER 0.085–0.086, margins 1.69–1.70, ceiling 1.98) — the first replicated dirty call in the project. Dirty 10 % is uncalled everywhere (finals 0.28–0.35). Latin s05 drew a hard text (3.00 bits/char) and is uncallable at truth (1.24); Italian is uncallable at truth in both cells (1.26 / 0.93), though its s05 search reaches SER 0.22. With the regenerated table (the script pools A and B shapes per language): German dirty-5 % pile n 4, power 0.75; the manuscript's B cells sit at **1 : 83 – 1 : 100** against a dirty-5 % German cipher (was 1 : 29 – 1 : 34), the A cells at 1 : 27 – 1 : 58; German dirty-10 % stays at no evidence (1 : 1.2 – 1 : 1.8); Latin dirty-5 % LR ≈ 1 on B; Italian dirty-5 % 1 : 10 – 1 : 14 on B by margin gap only. The §4 wording becomes: B excludes the clean German/Latin hypothesis and a 5 %-noisy German one; 10 %-noisy, noisy-Latin and Italian variants are not excluded.

### 5.2 Phase-6 heads (ELBO-polished keys; sub1to1, homophonic, naibbe)

Piles per (head, hypothesis) from the Phase-6 controls plus the strict-negative twins; "not a cipher" pools strict twins, shuffled text and the true cipher under the wrong language, so it is broad (sd 0.4–0.7 for the two letter heads) and the ratios are correspondingly weaker. Voynichesque (real-text source) is kept as its own comparison column.

| head / hypothesis | not-a-cipher max (strict-twin max) | clean real: n, mean, min, called | manuscript LR clean, range over cells | manuscript position in voynichesque-real-text pile |
|---|---|---|---|---|
| homophonic / German | 1.14 (1.14) | 3, 2.40, 2.30, 1.00 | 1/26 – 1/280 | 0.00–0.78 |
| homophonic / Latin | 1.29 (1.24) | 3, 1.73, 1.60, 1.00 | 1/1.8 – 1/33 | 0.00–0.78 |
| homophonic / Italian | 1.24 (1.15) | 3, 1.69, 1.49, 0.67 | 1/1.2 – 1/17 | 0.00–0.89 |
| sub1to1 / German | 1.79 (1.40) | 3, 2.37, 2.28, 1.00 | 1/68 – 1/200 | 0.00–0.67 |
| sub1to1 / Latin | 1.65 (1.33) | 3, 1.72, 1.60, 1.00 | 1/10 – 1/47 | 0.00–0.44 |
| sub1to1 / Italian | 1.20 (1.20) | 3, 1.67, 1.45, 0.67 | 1/2.8 – 1/30 | 0.00–0.89 |
| naibbe / any | 0.63–0.72 | 3, 0.34–0.55, **called 0** | 1/2 – 2.4 | 0.00 |

**Reading.**
- The Naibbe head's own positives are never called on the margin statistic (their true keys score 0.33–0.55), so on this head the manuscript's naibbe cells carry no evidence either way. The Naibbe result that stands is the description-length one (`docs/project_status.md` §1), not a margin.
- For the letter heads the evidence against a clean German cipher is again the strongest (1 : 26 to 1 : 280); against Latin and Italian it is one order of magnitude or less, because the negative pile is broad and the Italian/Latin positives sit close to the strict-negative ceiling.
- Currier B's homophonic and sub1to1 cells sit in the upper part of the voynichesque real-text pile (position up to 0.89) while A's sit near its bottom: on these heads B looks more like a *wrong-hypothesis decode of real text* than A does. That is consistent with, not evidence for, hidden content in B, since the strict twins show the voynichesque pile is itself inflated by content.

### 5.3 What is and is not claimed

These ratios are descriptive summaries of recorded controls, computed after every manuscript number had been read. They were not pre-registered and are not a call rule. They change the *wording* of the abstention (§4), not its status. Adopting any threshold on them would require the pre-registration route the status page describes for a second judge criterion.

---

## 6. Housekeeping the survey turned up

For the next coherence pass, not part of the review:

- `docs/judge_alternatives.md` results table quotes the frozen rule's voynichesque negatives as "≤ 1.26 (Phase-6 heads)"; every other record has 1.51. The lower figure makes the rule look cleanly separating.
- Per-round wall clock is given as ~13 s (`docs/altloop_vms_plan.md` §13) and ≈ 24 s (`docs/wordhom_throughput.md`, `docs/project_status.md` §6); reconcilable (manuscript cells skip the SER computation) but never stated side by side.
- `analysis/phase6/controls_nocontent/report.json` has a null acceptance block; the strict-twin headline numbers were read per cell.
