# Diffusion Model Training — Task Breakdown

Execution plan for the backbone specified in [[Design and Training of the Multilingual Diffusion Backbone]].

This document has two complementary views of the same work:
- **Phases 0–6** (below) — the time-ordered spine: what happens in what order, with hard **gates G0–G5** you must clear before advancing.
- **Workstreams WS0–WS7** (§ *Workstream map*) — the parallel-ownership view: which track each task belongs to, for assigning owners and running things concurrently.

Every task carries a **priority**, its **depends-on** predecessors, and an **acceptance criterion** (how you know it's done). The ordering enforces the design's three load-bearing decisions: all in-scope languages trained from step one (§7.1), the LID head attaches only after the backbone stabilizes (§7.2), and the evaluator is frozen before any cipher head is optimized (§7.4).

> **Language scope.** The implementation trains three languages — **Latin, Italian, German** (task 0.2). Wherever this document says "all languages" it means *this frozen inventory*.

**Priority legend**
- **P0** — on the critical path; nothing downstream starts until done.
- **P1** — required for the paper's claims, but parallelizable or deferrable past the first training run.
- **P2** — strengthens the result or de-risks review; cut first under time pressure.
- **P3** — nice-to-have.

---

## Critical path at a glance

```
Phase 0 Data/Infra  →  G0  →  Phase 1 Backbone (clean)  →  G1  →  Phase 2 Noise curriculum  →  G2
      →  Phase 3 ELBO metric + calibration  →  G3  →  Phase 4 LID head (joint)  →  G4
      →  Phase 5 Cipher-head integration (ladder)  →  G5  →  Phase 6 VMS application
```
Phase 3 (metrology) and Phase 4 (LID head) overlap — the calibration harness can be built while the head trains — but **G3 must pass before G4**: a mis-calibrated metric makes head evaluation meaningless. Metrology (WS3) runs alongside training throughout and must never slip behind it.

---

## Phase 0 — Decisions and data/infrastructure

Prerequisite for everything. These are cheap now and expensive later; nothing downstream is trustworthy if the corpus pipeline or the frozen decisions are wrong.

| ID  | Task                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Priority | Depends on | Acceptance criterion                                                                                                                                                                                                                          |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0.1 | **Resolve the alphabet.** Pull Greshko's actual Tables 5–9 (repo + Zenodo 10.5281/zenodo.16415087); reconcile the "~22-letter" vs 23-symbol listing; decide the K/W extension for Germanic; fix u/v & i/j normalization; freeze the ~32-symbol vocab + specials (MASK, NULL, BOS/EOS, language slot — **no SPACE token**: all whitespace is removed in preprocessing, design §2).                                                                                                                                                                                   | **P0**   | —          | A single-source-of-truth vocab spec with per-language normalization table; round-trip normalize→denormalize loses nothing the ciphers need (whitespace excepted — it is removed by design and not recoverable).                               |
| 0.2 | **Freeze language inventory + corpus list — Latin, Italian, German.** Scope the implementation to these three languages. Identify a concrete period-appropriate corpus per language with license and size; candidate starting point: the reference corpora bundled in the voynich-attack repo (~20M tokens across German 14M+, Latin, plus other languages) — vet period-appropriateness and upstream licenses per language.                                                                                                                                        | **P0**   | —          | Table of corpus → language → token count → license for Latin, Italian, German; any language below ~2M chars flagged for explicit low-resource treatment.                                                                                      |
| 0.3 | **Shared normalization pipeline** applied identically across all languages (no per-language lossy mapping). **Removes all whitespace as an early step** — every downstream stream (corpora, synthetic ciphertexts, partial decipherments, VMS text) is an unsegmented character sequence.                                                                                                                                                                                                                                                                           | **P0**   | 0.1        | Per-language cleaned text passes the normalizer with <0.1% dropped characters (whitespace removal excluded from the drop count); zero whitespace characters survive in any output; no duplicate documents across train/held-out.              |
| 0.4 | **Held-out calibration splits**, size- and domain-matched per language, carved out *before* any training touches the data (design §5b).                                                                                                                                                                                                                                                                                                                                                                                                                             | **P0**   | 0.2, 0.3   | Fixed, versioned splits; per-language held-out ≥200k chars; never fed to any training phase.                                                                                                                                                  |
| 0.5 | **Data loader**: char→id, masking sampler, additive language-embedding injection, temperature-balanced sampling (τ≈0.7) with per-language weights logged every run.                                                                                                                                                                                                                                                                                                                                                                                                 | **P0**   | 0.1        | Empirical draw frequencies within 2% of target over a 1M-example simulation.                                                                                                                                                                  |
| 0.6 | **Training infra**: config/versioning, checkpoint+resume, EMA (0.9999), bf16, run manifests (data/vocab version, seeds), and per-language held-out NELBO logged as a first-class metric every eval step (the canary used at every gate). **All runs tracked in ClearML** (server at `clearml.acet.network` on LAN): one ClearML Task per run, auto-capturing hyperparameters/config, scalar metrics (per-language NELBO), console logs, and checkpoints as artifacts.                                                                                                                                                                                                                                                                                                                            | **P0**   | —          | Kill-and-resume reproduces the loss curve; every run auto-registers in ClearML with config, scalar metrics, and checkpoint artifacts; per-language NELBO dashboard (ClearML scalars) exists before Phase 1 launches.                                                                                                                                       |
| 0.7 | **Acquire cipher generators + controls, and pin the arithmetic cipher.** Greshko `naibbe_v2.py` (design §9) paired plaintext/ciphertext corpora; Boxer `voynpy.pseudo_vms` arithmetic cipher (design §10) — use the **latest** version with its default parameters (`zipf_exponent`, `tokens_per_char`, token-length distribution, ~500 homophones/letter), set the custom alphabet to our frozen vocab (0.1), tune `doubling_strength` per language via `tune_to_vms` to the VMS ~0.92% token-doubling rate, and fix a seed; `voynichesque.py` (negative control). | **P1**   | 0.1        | Generators run; arithmetic encode→decode round-trip exact on sample text per language; per-language tuned doubling rate within tolerance of the 0.92% target; sample outputs' statistics (entropy, token-length dist., doubling rate) logged. |
| 0.8 | **VMS ingest**: choose transcription(s) (Takahashi, RF1b-e, and/or the independent character-level transcription in the voynich-attack repo, CC-BY 4.0), EVA parsing, Currier A/B page split, uncertain-glyph policy; all whitespace/word-break markers stripped after parsing (0.3), same as every other stream.                                                                                                                                                                                                                                                   | **P2**   | 0.1        | Both dialects load; counts reconcile with published figures (~37k words / ~230k chars, counted before stripping); post-strip character counts recorded per dialect.                                                                           |

**Gate G0:** vocab spec frozen; normalization byte-stable on a round-trip across all languages; corpus table complete with size flags; per-language NELBO metric logs correctly to ClearML on a random-init model. → Phase 1.

---

## Phase 1 — Backbone pretraining (clean text)

Establishes the calibrated conditional distributions that *are* the measurement instrument. Everything else is downstream of this being right.

| ID | Task | Priority | Depends on | Acceptance criterion |
|---|---|---|---|---|
| 1.1 | **MDLM-style masked discrete diffusion**: absorbing/mask forward process, continuous-time SUBS parameterization, Rao-Blackwellized NELBO, log-linear schedule. | **P0** | 0.5, 0.6 | Loss matches a reference MDLM implementation on a toy corpus to numerical tolerance; text8-scale bits/char in the expected range (≈1.4–1.5, the D3PM-era anchor to beat). |
| 1.2 | **Encoder-only transformer** (RMSNorm, SwiGLU, RoPE) at 85M (12L/d768/12H) + 25M sibling (6L/d512). Additive per-position language embedding + learned NULL-language embedding, 10% conditioning dropout. | **P0** | 1.1 | Conditional vs unconditional NELBO differ in the expected direction on pilot data; dropout rate verified in logs. |
| 1.3 | **Pilot run** (25M, short budget, subset of the language inventory) to shake out data pipeline, dashboards, and scoring harness. *A plumbing test — short-budget and possibly single-language; its weights are discarded.* | **P0** | 1.1, 1.2, 0.5, 0.6, 3.1 | End-to-end run completes; per-language NELBO curves plausible; harness scores pilot checkpoints without manual intervention. No NaNs (watch the log-space blending trap). |
| 1.4 | **Phase A full pretraining**: 85M + 25M, all in-scope languages from step one — the frozen inventory in 0.2 (Latin, Italian, German) — trained jointly (design §7.1 — no single-language-first curriculum), clean data, backbone only. AdamW β₂=0.98, peak LR 3e-4, 2k warmup, cosine, batch ~0.5M chars, dropout 0.1, EMA. | **P0** | 1.3 | Per-language held-out NELBO plateaued (<0.5% improvement over trailing window) for *every* in-scope language; no language stagnating while others improve (else adjust τ and continue). |
| 1.5 | **Interference watch**: monitor per-language NELBO; if any language stalls while others improve, adjust sampling temperature, not schedule. | **P1** | 1.4 | Documented τ adjustments; no language pathologically behind at plateau. |
| 1.6 | **25M/85M ranking-agreement probe** on a clean-text sample (cheap bound-stability check, design §3). | **P2** | 1.4 | Agreement rate reported; disagreement cells investigated. |

**Gate G1:** all languages' held-out NELBO plateaued; interference check passed; 25M and 85M rank a clean-text sample consistently; calibration table v1 (from 3.4) produced. → Phase 2.

---

## Phase 2 — Noise curriculum

Turns the clean model into a noise-robust evaluator (design R2), matching training corruption to deployment corruption. A permanent clean fraction is retained so §5b calibration stays valid.

| ID | Task | Priority | Depends on | Acceptance criterion |
|---|---|---|---|---|
| 2.1 | **Structured substitution noise**: sample self-consistent many-to-one wrong-key maps, apply to a fraction of positions (*not* i.i.d. flip noise). | **P0** | G1 | Unit test; severity sweep yields monotone NELBO degradation on the pilot model ("hard but lawful"). |
| 2.2 | **Segmentation noise**: letter-stream parse errors — spurious insertions/deletions/duplications mimicking wrong unigram-vs-bigram Naibbe parses. (No space noise: whitespace is already stripped in preprocessing, 0.3; Naibbe's 50-50 re-spacing layer is absorbed by the whitespace removal.) | **P1** | G1 | Unit test; generator parameterized by severity; generator output verified whitespace-free. |
| 2.3 | **Transcription noise** at ~5% (level Bruton 2026 tolerates at ≥0.99 F1). | **P1** | G1 | Unit test; severity-parameterized. |
| 2.4 | **Phase B fine-tune**: 30–50% of examples carry the noise mixture; clean fraction retained forever. | **P0** | 2.1, 2.2, 2.3 | Clean-text NELBO within 1% of Phase A (clean anchor held); noised-input NELBO degrades smoothly. |
| 2.5 | **NULL-token exposure** at rates matching the 2N-slot scheme, so the backbone treats empty slots as in-distribution (prerequisite for Phase 5 interface). | **P1** | 2.4 | NULL slots score as in-distribution after Phase B. |
| 2.6 | **Robustness curve**: score degradation vs substitution-noise level. | **P2** | 2.4 | Curve published; no catastrophic cliff. |

**Gate G2:** noised-input NELBO degrades smoothly, not catastrophically; clean-text NELBO has *not* drifted from its G1 value (calibration anchor intact). → Phase 3.

---

## Phase 3 — ELBO metrology (the instrument itself)

Makes the per-language likelihood a *fair* comparison — this is what makes the result publishable rather than anecdotal. The framework's load-bearing assumption (comparable bound tightness across languages) is *measured and corrected here*, not assumed. Do not let this slip behind training.

| ID | Task | Priority | Depends on | Acceptance criterion |
|---|---|---|---|---|
| 3.1 | **Scoring harness**: stratified timestep sampling (64 strata × k), **common random numbers across language conditions**, per-window bits/char with mean and spread over documents. | **P0** | 1.1, 1.2 | Variance of *between-language score differences* under CRN ≥5× smaller than under independent sampling on pilot data (verify the whole point of CRN). |
| 3.2 | **Sample-budget study**: samples needed until language *ranking* is stable per length {50…700}. | **P1** | 3.1 | Curve published; chosen budget marked; ranking flip-rate <1% at chosen budget. |
| 3.3 | **Per-window document scoring** → mean + spread, so rankings carry uncertainty. | **P1** | 3.1 | Long docs scored with reported spread. |
| 3.4 | **Per-language bound calibration**: held-out NELBO per language; small char-AR reference model per language on identical data; per-language `NELBO − NLL_AR` offsets computed, versioned, applied in exactly one place in the code. Re-run after every phase (A/B/C). | **P0** | 0.4, 1.1 | Calibration table produced per phase; offsets stored and single-sourced. |
| 3.5 | **Bound-fairness audit**: does bound looseness correlate with language family, corpus size, or morphology? | **P1** | 3.4 | One-page audit per phase; any correlation above noise escalated as a design issue, not footnoted. |
| 3.6 | **Synthetic language-recovery validation**: on 1:1-enciphered known-plaintext text, calibrated ranking recovers true language across length cells (bar: Hauer & Kondrak 97.1%/380-lang on the easy end). | **P0** | 3.4 | Full recovery report; near-ceiling on 1:1 at ≥200 chars. |
| 3.7 | **Family-vs-language reporting** at both granularities (close-pair confusion is the expected dominant error mode). | **P1** | 3.6 | Rankings reported at both granularities. |

**Gate G3:** calibrated ranking recovers true language on the synthetic 1:1 suite within target; per-language offsets estimated and stored; fairness audit shows no un-escalated language-dependent bias. **Do not proceed to G4 without this.** → Phase 4.

---

## Phase 4 — Language-ID head (delayed, then joint)

Implements the paper's "jointly trained" head, but joint from the *end* (design §7.2) to protect the instrument. Harness work can begin during Phase 3.

| ID | Task | Priority | Depends on | Acceptance criterion |
|---|---|---|---|---|
| 4.1 | **Head architecture**: mean-pool final hidden states (averaged over several masking levels) → 2-layer MLP → softmax over languages + one "no-language/synthetic" abstain class; stop-gradient switch; λ ramp for Phase C. | **P0** | G2 | Trains to >99% on clean long text (anything less = wiring bug); stop-gradient verified (backbone grads exactly zero in Phase B mode). |
| 4.2 | **Phase B attach behind stop-gradient**: train head on clean + noised + simulated-partial-decipherment inputs, backbone unchanged. | **P0** | 4.1, G3 | Head converged; LID accuracy vs noise-severity curves produced. |
| 4.3 | **Abstain-class training data**: `voynichesque.py` output + shuffled text. | **P1** | 4.1, 0.7 | Abstain class triggers on negative controls at >95%. |
| 4.4 | **Phase C joint fine-tune**: release stop-gradient, `L = L_NELBO + λ·L_LID`, ramp λ 0→~0.05 (LID grad norm <10% of diffusion grad norm). | **P0** | 4.2 | Calibration table v3 produced; λ schedule logged. |
| 4.5 | **Abort/canary monitoring during C**: per-language held-out NELBO; if any degrades >1% relative, halve λ. If synthetic ranking (3.6) changes between end-B and end-C, flag it — report, don't tune away. | **P0** | 4.4 | Joint model's synthetic-grid ranking identical to end-of-Phase-B ranking; any change documented as a red flag. |
| 4.6 | **Head calibration**: temperature-scale on held-out decipherments; record ELBO-ranking vs head-ranking agreement as a diagnostic. | **P1** | 4.4 | Calibrated head; agreement matrix reported. |
| 4.7 | **Seed replication**: repeat Phase A–C at 25M with 2 extra seeds. | **P2** | 4.4 | Ranking stability across seeds reported. |

**Gate G4:** joint model passes the G3 synthetic ranking test *unchanged or improved*; per-language NELBO not degraded beyond threshold; head and ELBO rankings agree on clean synthetics. → Phase 5.

---

## Phase 5 — Cipher-head integration (frozen evaluator, difficulty ladder)

Backbone is **frozen (EMA weights)** throughout (design §7.4). Heads validated in strict difficulty order, each rung gated. Inner search uses the cheap frozen n-gram DP scorer; the diffusion ELBO scores shortlisted maps and supplies dense gradients via expected embeddings.

| ID | Task | Priority | Depends on | Acceptance criterion |
|---|---|---|---|---|
| 5.1 | **Head interface**: expected-embedding (mixture) inputs on the 2N-slot frame; NULL-blended slot-2 by unigram/bigram weight; straight-through fallback. Uniform bits-per-plaintext-char scoring + description-length complexity penalty. Includes the `logaddexp(−∞,−∞)` smoke test from the inverse note. | **P0** | G4, 2.5 | Gradients reach a toy head's parameters; NaN smoke test passes. |
| 5.2 | **Rung 1 — 1:1 substitution head** (Sinkhorn/bijective). Sanity that the frozen evaluator drives correct map recovery. | **P0** | 5.1 | Near-perfect recovery on synthetic 1:1 at ≥200 chars. |
| 5.3 | **Rung 2 — unigram homophonic head.** | **P0** | 5.2 | ≤1.9% SER on Zodiac-408-class problems; Borg ≤4.10%; BnF fr2988 ≤1.13% (literature anchors). |
| 5.4 | **Rung 3 — Naibbe mixed unigram-bigram head** (differentiable inverse): U/Pre/Suf soft maps, exact semi-Markov DP inner scorer, ELBO shortlist scoring/refinement, entropy/row-sparsity annealing, discrete-move interleaving, many restarts. | **P0** | 5.3, 0.7 | Recovers ground-truth maps on synthetic Naibbe Latin/Italian pairs (Greshko repo alignments) at ≥95% letter-map accuracy; restart budget documented. |
| 5.5 | **Rung 4 — arithmetic-encoded head** (pseudo-VMS sum-to-target inverse, design §10): jointly infer the 16 cipher-character values and the letter-value map; marginalize the latent 2–6-char token segmentation (generalizes the Naibbe semi-Markov DP — whitespace stripping removes token boundaries); model the doubling mechanism. No interval tracking or dummy-symbol marginalization needed. Split to its own design note. | **P1** | 5.4, 0.7 | Recovers plaintext language on synthetic pseudo-VMS ciphers (0.7 pinned config; ground truth via the generator's exact sum-lookup decode) at better-than-family-random rates. |
| 5.6 | **Uniform-scale cross-head comparison**: verify (cipher × language) cells are rankable on one calibrated scale with the complexity penalty applied. | **P1** | 5.3, 5.4 | Cross-head scores demonstrably comparable on a shared scale. |

**Gate G5:** rungs 1–3 meet their SER targets on synthetics; cross-head scores are on a comparable scale. (Rung 4 is no longer spec-blocked — pinned to the pseudo-VMS generator per 0.7 — but remains the hardest rung and may trail; proceed to Phase 6 without it if needed, noting the gap.) → Phase 6.

---

## Phase 6 — VMS application and reporting

| ID | Task | Priority | Depends on | Acceptance criterion |
|---|---|---|---|---|
| 6.1 | **Per-dialect scoring** of the VMS (Currier A and B separately — never pooled; Naibbe's B replication is known-incomplete and Parisel's signatures are dialect-calibrated). | **P0** | G5, 0.8 | Both dialects scored independently. |
| 6.2 | **Rank (cipher × language) cells** with calibrated ELBO + uncertainty (per-window spread) + head agreement. | **P0** | 6.1 | Ranked table with uncertainty and head-agreement columns. |
| 6.3 | **Negative-control battery**: `voynichesque.py` (must abstain), shuffled text, cross-contamination set (out-of-inventory languages — ones not in the trained inventory — enciphered under in-inventory-fit ciphers, to confirm untrained languages don't masquerade as an in-inventory fit). | **P0** | 6.1, 0.7 | Abstention >95%; contamination confusions documented. |
| 6.4 | **Bound-fairness audit re-run** (§5b) reported alongside every result. | **P1** | 6.2 | Audit attached to the results. |
| 6.5 | **Length-sensitivity + family-confusion analysis**: accuracy-vs-length curves and within-Germanic / within-Romance confusion matrices (predicted dominant error mode; LID anchor ~50 chars clean). | **P1** | 6.2 | Curves + confusion matrices published; claims restated at the granularity the data supports. |
| 6.6 | **Known-benchmark anchors** (Zodiac-408 ≤1.9%, Borg ≤4.10%, BnF fr2988 ≤1.13%) reported to validate the head machinery independent of the VMS. | **P1** | G5 | Anchor results reported. |
| 6.7 | **Write-up with honest framing**: exploratory, assumption-dependent; family-level resolution; explicit statement of residual bound-comparability risk. | **P0** | 6.2, 6.3, 6.4 | Draft states assumptions and residual risks explicitly. |

---

## Workstream map (parallel-ownership view)

The same tasks, grouped by track for assigning owners and running concurrently. A workstream spans multiple phases; a task's phase tells you *when*, its workstream tells you *who*.

| Workstream | Covers | Tasks |
|---|---|---|
| **WS0 — Decisions to close before code freezes** | Alphabet, language inventory | 0.1, 0.2 |
| **WS1 — Data pipeline** | Corpora, splits, loaders, noise + synthetic cipher generators, controls, VMS ingest | 0.3, 0.4, 0.5, 0.7, 0.8, 2.1, 2.2, 2.3 |
| **WS2 — Model implementation** | Diffusion core, transformer, conditioning, mixture-input path, LID head, infra | 1.1, 1.2, 4.1, 5.1, 0.6 |
| **WS3 — ELBO metrology** | Scoring harness, calibration, fairness audit, sample-budget study | 3.1–3.7 |
| **WS4 — Training runs** | Pilot, Phase A/B/C, seed replication | 1.3, 1.4, 1.5, 2.4, 2.5, 4.2, 4.4, 4.5, 4.7 |
| **WS5 — Validation gauntlet** | Synthetic grid, negative controls, length + family analysis | 3.6, 4.3, 6.5, 6.6 |
| **WS6 — Cipher heads (Phase D)** | Rungs 1–4 on the frozen backbone | 5.2, 5.3, 5.4, 5.5, 5.6 |
| **WS7 — VMS application + reporting** | Per-dialect scoring, ranking, controls, write-up | 6.1, 6.2, 6.3, 6.4, 6.7 |

---

## Cross-cutting / continuous tasks

| ID | Task | Priority |
|---|---|---|
| X.1 | **Per-language held-out NELBO dashboard** (ClearML scalars, server `clearml.acet.network`) — the canary referenced at G1, G2, G4. Maintain from Phase 1 on. | **P0** |
| X.2 | **Reproducibility**: seed control, config versioning, dataset hashing — all captured in ClearML (Task config/hyperparameters + ClearML Data dataset versioning); MIT release planned "upon publication". | **P1** |
| X.3 | **Compute-cost tracking** — the 25M sibling exists to keep restart-heavy search affordable (design R6); monitor eval cost per cryptanalysis call via ClearML resource/scalar logging. | **P2** |
| X.4 | **Alphabet/spec drift log** — the 23-vs-22 discrepancy (0.1) stays visible until resolved. The arithmetic cipher is pinned to the latest `voynpy.pseudo_vms` (design §10, implemented in 0.7); log any parameter drift against the pinned generator. | **P1** |

---

## Priority summary (what to do if resources are cut)

- **Minimum viable result** (P0 only): Phases 0–4 + Phase 5 rungs 1–3 + Phase 6 core. Yields calibrated per-language ranking with Naibbe as the most complex validated cipher — enough for the paper's headline claim with honest caveats.
- **Defer safely** (P2/P3): robustness curve (2.6), 25M/85M probe (1.6), seed replication (4.7), compute dashboard (X.3). Improve confidence but don't gate a result.
- **Former hard blocker, now cleared**: Phase 5 rung 4 (arithmetic head) is pinned to the latest pseudo-Voynich generator in the voynich-attack repo (design §10, task 0.7) — no longer spec-blocked. It is still the hardest rung; do not let it hold up Phases 1–4 or the Naibbe result.

---

*Ordering rationale traces to the design doc's curriculum decisions: all-languages-from-start (§7.1) sets Phase 1; delayed-then-joint LID head (§7.2) sets the G3→G4 ordering; frozen-evaluator + difficulty ladder (§7.4) sets Phase 5. The gates encode the design's non-negotiables — stabilize the instrument before measuring, calibrate before ranking, freeze before attacking.*
