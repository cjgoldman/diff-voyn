# Interim Status Report — diff-voyn

**Date:** 2026-08-19 · **Audience:** general / non-specialist

## What this project is trying to do

The Voynich Manuscript is a 15th-century book written in an alphabet nobody can read. A live scientific question is whether it is an enciphered natural language — and if so, *which* language. This project builds a measuring instrument to attack that question: a neural network trained on medieval-era Latin, Italian, and German text, paired with "decoder" modules that attempt trial decipherments of Voynich-like ciphertext. The idea, well established in code-breaking research, is that even a *partial* decryption of a text still carries fingerprints of its underlying language. If our decoders can turn Voynich-like ciphertext into something the Latin model finds far more plausible than the German model (or vice versa), that is evidence about the plaintext language — even without a full solution.

The end goal, as stated in the project abstract, is a ranked table of (cipher system × language) candidates for the actual Voynich text, using two modern cipher systems that are known to produce convincingly Voynich-like text: the Naibbe cipher (Greshko 2025) and Boxer's arithmetic "sum-to-target" cipher. Handling ciphers this complex is the project's main claimed advance over prior work, which topped out at simpler substitution ciphers.

## Where we are against the plan

The project plan runs through seven phases (0–6) with hard quality gates between them. We are currently mid-Phase 1 of 6, with a substantial piece of Phase 5 prototyped ahead of schedule.

**Phase 0 — data and groundwork: complete.** The character alphabet, text-cleaning pipeline, and language corpora are built and frozen: ~121 million characters of period-appropriate Latin, Italian, and German, with held-out test sets carved off before any training. Both cipher generators are pinned to exact published versions, and the Voynich transcription itself has been ingested and cleaned. Freezing these early matters because the science depends on treating all three languages identically — any asymmetry would quietly bias the final ranking.

**Phase 1 — training the core model: in progress, on track.** The diffusion language model (the "instrument") is fully coded and verified against exact mathematical references, and full pretraining launched on 2026-08-18 on two GPUs. The smaller 25M-parameter model has **finished** its 20,000-step run; the larger 85M-parameter model is at step ~9,250 of 20,000 (roughly halfway, expected to finish within a day or so). Training is healthy: the model's "surprise" on unseen text (measured in bits per character) is falling smoothly for all three languages, with no language stalling — the key health check the plan requires.

**Phase 5 preview — the cipher decoders: prototyped early, and the core premise holds.** While the big model trains, the four decoder modules were built and tested against a cheaper stand-in scorer (classical n-gram language models). All four difficulty rungs now have working prototypes, tested on synthetic ciphertexts where we know the right answer:

- **Rung 1 (simple substitution):** essentially solved — near-zero decryption error on texts of 200+ characters. Crucially, the first language-identification probe ranked the **true language first in 6 of 6 trials** — direct evidence for the project's core premise.
- **Rung 2 (homophonic substitution):** 4 of 6 test cells at ≤2% error; remaining failures are traced to search budget, not the method.
- **Rung 3 (Naibbe cipher):** a structured decoder recovers 85–95% of the secret key — at or near the 95% acceptance bar, not yet consistently over it.
- **Rung 4 (arithmetic cipher, the hardest):** acceptance criterion **met** on 2026-08-19 — the decoder identified the true language 4/6 times and the right language *family* 5/6 (chance would be 2/6 and ~3.3/6), and in one case recovered a complete cipher key from scratch.

## Progress relative to the abstract's aspirations

The abstract makes three claims. Here is honestly where each stands:

1. *"Decoding heads capable of handling mixed unigram-bigram and arithmetic ciphers."* — **Prototypes exist and work on synthetic data**, but so far only at small scale (short texts, a handful of trials per condition) and against the stand-in n-gram scorer, not yet the diffusion model.
2. *"A multilingual diffusion backbone with a jointly trained language-detection head."* — The backbone is **half-trained**; the language-detection head (Phase 4) does not exist yet.
3. *"Germanic candidates receive the highest likelihood [for the Voynich text]."* — **Not yet tested.** No Voynich scoring has been run; the abstract describes the intended end state, not a current result.

In short: the foundation and the riskiest technical machinery are validated; the headline experiment still lies ahead.

## Technical risk: what has been retired, and what remains

**Risk taken off the table.** The early decoder work was deliberately sequenced to kill the project's biggest risks first, and the largest ones are now retired:

- **"Does the whole approach work at all?"** This was the existential risk: that partial decipherments simply wouldn't carry enough language signal to rank languages. The rung-1 probe answered it — the true language ranked first in 6 of 6 trials. The method's core premise is now demonstrated, not assumed.
- **"Can the hard ciphers be cracked in principle?"** The Naibbe and arithmetic ciphers were designed to defeat statistical analysis, and it was genuinely unknown whether any decoder could recover their keys. Both now have working prototypes: a first Naibbe parameterization failed as theory predicted (~20% key recovery), but a structured redesign reaches 85–95%; the arithmetic decoder met its acceptance bar and once recovered a complete key from scratch. These were the two novel-architecture claims of the paper, and they no longer rest on hope.
- **Instrument correctness.** The diffusion model's loss function is verified against exact brute-force mathematical references, and two subtle numerical bugs that would have silently corrupted results (NaN traps in the gradient path) were caught by tests and are permanently guarded.
- **Training stability.** Training all three languages jointly could have caused them to interfere or one to stall. Halfway through the big run, all three are improving smoothly with zero numerical failures.
- **The eventual "swap".** The plan calls for replacing the stand-in scorer with the trained diffusion model inside the decoders. The plumbing for this (gradients flowing correctly through a frozen model) is already built and tested, so the swap is designed to be a one-line change rather than a rewrite.

**Risk that remains, roughly in descending order of size:**

1. **The Voynich Manuscript itself may not yield a decisive answer** (largest, and irreducible). Everything so far is validated on *synthetic* ciphertext where we control the ground truth. The real manuscript may involve a cipher outside our candidate set, scribal quirks, or an unrepresented language — in which case the honest outcome is an assumption-dependent ranking, not a solution. This is a scientific risk no engineering can remove; the abstract already frames results as exploratory for this reason.
2. **Calibration fairness (Phase 3).** Cross-language score comparisons must be corrected so no language wins by statistical accident — the n-gram experiments already showed German wins every uncorrected comparison purely because German text is inherently more predictable. The correction method is known and gated, but if done poorly it silently biases the headline result. This is the most consequential *avoidable* risk.
3. **Does the diffusion model actually beat the stand-in scorer?** The decoders currently run against classical n-gram models. The project's bet is that the diffusion model closes the remaining gaps (e.g., the rung-2 failures). The plumbing is de-risked, but the payoff itself is untested.
4. **Scale and search budget** (engineering, moderate). Current results are smoke-test scale — a few trials per condition, short texts. The full acceptance grids, and running decoders at full manuscript length (~160,000 characters), need performance work and compute. The remaining decoder failures trace to search budget rather than method, which makes this a resourcing risk, not a conceptual one.
5. **Noise curriculum drift (Phase 2, gated, small).** Teaching the model to tolerate cipher-like noise must not disturb its clean-text baseline; the gate explicitly checks for this.
6. **External validation pending (small, low effort).** The published benchmark ciphertexts (Zodiac-408 and others) that would validate the decoders against literature results still need to be fetched and run.

The overall shape: the risks retired were mostly *"is this possible?"* risks; the risks remaining are mostly *"can we measure it fairly and at scale?"* risks — smaller and more controllable than what has been cleared, with the one exception of item 1, which is the nature of the problem rather than of the method.

## What comes next

1. Finish the 85M training run and pass Gate G1 (all languages plateaued, no interference).
2. Phase 2: teach the model to tolerate cipher-like noise without disturbing its clean-text baseline.
3. Phase 3: calibrate the scoring metric so cross-language comparisons are fair — the gate everything downstream depends on.
4. Phase 4: attach the language-ID head; Phase 5: swap the trained diffusion model in behind the already-built decoders (designed as a one-line change).
5. Phase 6: score the actual Voynich Manuscript, Currier dialects A and B separately.
