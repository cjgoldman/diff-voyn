# Design and Training of the Multilingual Diffusion Backbone

Companion design document to [[A Diffusion-Based Framework for Language Identification of Voynich-Like Ciphertext]]. It specifies the diffusion model that serves simultaneously as (a) the per-language likelihood instrument (ELBO), (b) the feature extractor for the jointly trained language-ID head, and (c) the differentiable plaintext evaluator behind the cipher decoding heads (see [[A Differentiable Inverse for a Naibbe-Style Homophonic Substitution Cipher]]).

Each section states the **decision**, the **alternatives considered**, and the **reasoning**, with evidence drawn from the reference summaries (files 1–8). Choices that are informed defaults rather than evidence-forced are flagged as such.

---

## 0. Requirements the design must satisfy

Derived from the framework paper and the differentiable-inverse note:

1. **R1 — Comparable per-language likelihood.** The ELBO must be tractable, tight, and *comparably tight across languages*, because language ranking is argmax over per-language bounds, not exact likelihoods. A bound gap that varies by language is a silent bias (flagged explicitly in the Diffusion LM summary).
2. **R2 — Noise robustness.** The operational input is *partially decrypted text*: mostly-right letters with systematic substitution errors and segmentation errors. The model's score must degrade gracefully, not cliff.
3. **R3 — Dense per-position gradients.** The Naibbe-inverse note is explicit: "a classifier's single scalar will starve the optimization." The backbone must expose per-letter gradients to the decoding heads through soft (expected-embedding) inputs.
4. **R4 — Latent-length tolerance.** Homophonic expansion means a ciphertext of T tokens corresponds to a plaintext of latent length in [T, 2T] (Naibbe: each token = 1 or 2 letters; the pseudo-VMS arithmetic cipher: exactly 1 letter per token, but with token boundaries themselves latent after whitespace removal, since its 2–6-character tokens are space-delimited in raw generator output). The evaluator must score sequences whose true length is itself a latent variable.
5. **R5 — Uniform head interface.** Simple homophonic, mixed unigram–bigram (Naibbe), and arithmetic-encoded heads must all be scoreable on the same scale, since the paper ranks (cipher × language) pairs jointly.
6. **R6 — Scale realism.** The target corpus is small (VMS ≈ 37k words / ~230k EVA characters; 43 Currier B pages = 17.6k words) and cryptanalysis needs *many restarts*, so evaluation must be cheap per call.

---

## 1. Diffusion formulation

**Decision: masked (absorbing-state) discrete diffusion in continuous time, MDLM-style — Rao-Blackwellized NELBO with the SUBS parameterization, trained directly over character tokens.**

Alternatives considered:

| Option | Verdict | Why |
|---|---|---|
| Continuous embedding diffusion (Diffusion-LM) | Rejected | ELBO lives in embedding space and connects to token likelihood only through the rounding model — the metric *is* the product here, so an indirect bound is disqualifying (R1). The reference notes call this "the road not taken." |
| DDPM-style weighted bound | Rejected | Explicitly trades likelihood for sample quality; we need likelihood, not samples. |
| D3PM (discrete-time) | Rejected as primary | Foundational but looser objective; its hybrid ELBO + auxiliary cross-entropy loss is retained as the *precedent* for our joint LID objective (§7). |
| SEDD (score entropy) | Strong runner-up | Principled NLL upper bound (Thm 3.6), but MDLM's Rao-Blackwellized bound is ~17% tighter on LM1B (≤27.04 vs ≤32.79) and the loss reduces to a weighted MLM — simpler to train and to reason about. |
| Uniform-transition noise | Rejected | Absorbing beats uniform "by a wide margin" in both D3PM and SEDD. No reason to revisit. |

Reasoning:

- **Tightness is the ranking's error bar.** Among formulations with token-level bounds, MDLM's continuous-time NELBO is the tightest documented in our references. Tighter bounds shrink the room for language-dependent bound-gap variation (R1).
- **The NELBO is a weighted MLM loss** (`ℒ_NELBO = 𝔼_q ∫ α′_t/(1−α_t) · log⟨x_θ(z_t,t), x⟩ dt`), i.e., the model is literally trained to reconstruct characters from partially masked context. This is structurally the trial-decipherment regime: judge plausibility of text from partial, corrupted evidence (R2). The training task and the deployment task coincide, which is the strongest argument available.
- **Encoder-only, bidirectional** — supports a classification head "in the same way BERT-style encoders support classification heads" (R5, §6), and bidirectionality matters because Voynich-relevant signal is bidirectional (Parisel: right-to-left optimization within words, left-to-right dependency across boundaries — an autoregressive model sees only one direction natively).
- **Masking as noise is also the right noise model for R4/R3:** the MASK token gives us a principled "this position is unknown" symbol, which the 2N-slot padding scheme for latent segmentation (§8) exploits directly.

Noise schedule: continuous-time with the standard log-linear α_t. Under the SUBS parameterization the objective is invariant to schedule reparameterization and explicit time conditioning can be dropped from the network (informed default; the reference summaries do not record schedule details — flagged as a gap there).

---

## 2. Tokenization and vocabulary

**Decision: character-level, over a normalized ~26-symbol Latin alphabet plus specials. All whitespace is removed from every text stream — training corpora, synthetic ciphertexts, partial decipherments, and the VMS transcription — as an early preprocessing step, so the model never sees spaces or word boundaries. No subwords, ever.**

- Base letters: the Naibbe plaintext alphabet `A B C D E F G H I L M N O P Q R S T U V X Y Z` (23 symbols as listed in the inverse note, which calls it "~22-letter" — a discrepancy to resolve against Greshko's Tables; track it, don't paper over it).
- **Alphabet extension decision:** Germanic candidates (German, Danish, Dutch) need `K` and `W`, and the framework's headline result is about Germanic languages. Extend the alphabet rather than lossily mapping (K→C, W→VV) — a lossy mapping would systematically *deflate* Germanic likelihoods and contaminate exactly the comparison we care most about. Normalize u/v and i/j merges per medieval orthographic convention, identically across languages.
- Specials: `MASK`, `NULL` (§8), `BOS`/`EOS`, and one conditioning slot (§4). There is deliberately **no `SPACE` token** — whitespace is stripped in preprocessing before tokenization, so no space symbol ever enters the vocabulary. Total vocab ≈ 32 — pad to 32 for hardware alignment.

Reasoning: every method in the decipherment literature that works is character- or symbol-level (the neural-methods summary is blunt: "no reference uses subword tokenization; the field is uniformly character/symbol-level"). Subwords would (a) make per-language bound tightness depend on tokenizer fit — a direct R1 violation and a known cross-language perplexity confound; (b) break the letter-granular interface the cipher heads need (R3: the Naibbe inverse emits *distributions over letters*); (c) buy nothing at a 23-letter alphabet scale.

Whitespace removal is a modeling decision, not a convenience: VMS "word" boundaries have uncertain semantics (Naibbe's re-spacing layer rewrites them arbitrarily, and the arithmetic cipher need not preserve them at all), so a space token would carry cipher-layer artifacts rather than plaintext signal — and any language whose spacing conventions survived preprocessing better would gain an unearned likelihood edge (an R1 violation). Stripping spaces from every stream, identically, keeps all languages and all cipher heads on the same unsegmented footing; word-boundary phenomena are deferred to future work (§10).

The tiny vocabulary also means embedding/output layers are nearly free, so parameter budget goes almost entirely into depth/width.

---

## 3. Architecture and scale

**Decision: encoder-only transformer, ~85M parameters — 12 layers, d_model 768, 12 heads, SwiGLU FFN (inner dim ~2048), RMSNorm, RoPE. Context length 1024 characters. A 6-layer/512-dim (~25M) sibling is trained for ablations and restart-heavy search loops.**

Reasoning:

- **Scale calibration from the nearest working systems.** Kambhatla 2023 solves homophonic ciphers with a 12-layer, ~23M decoder; ALICE reaches 1.09% SER with an ~85M encoder-only LLaMA-style model (RMSNorm, SwiGLU, RoPE) — we adopt ALICE's recipe since it is the closest architectural cousin (encoder-only, symbol-level, cipher domain) with published success. Going much larger is unsupported: the constraint is medieval-corpus data, not capacity, and an overfit backbone has language-dependent bound gaps (R1 again — memorization of the best-resourced language's corpus tightens its bound preferentially).
- **Context 1024.** The homophonic-cipher literature operates at 300–700 characters; LID judgments stabilize by ~50 characters (Apple bi-LSTM result); Naibbe expansion of 1.5–2.5× means a 1024-glyph ciphertext window corresponds to ~400–680 plaintext letters — comfortably inside the regime where decipherment methods work and beyond where LID saturates. O(n²) attention made ~1500-char sequences infeasible for Kambhatla 2023; we stay under that and score long documents by averaging per-window bits-per-character (which also gives a variance estimate across windows for free).
- **No time-conditioning network** (see §1); this simplifies the head interface — cipher heads see one static evaluator, not a t-indexed family.
- **Small sibling model:** the Naibbe note budget is "~38k tokens × many restarts." The 25M model runs the inner search loops; the 85M model does final scoring. Both are trained on identical data so their rankings can be sanity-checked against each other (a cheap bound-stability probe).

---

## 4. Multilingual conditioning

**Decision: one shared backbone conditioned by a learned language embedding added at every position (not a prefix token), trained with 10% language-embedding dropout (replaced by a learned NULL-language embedding).**

Alternatives considered:

- **Per-language expert models.** Rejected. This is the classical design whose scaling failure motivates the whole paper (Hauer & Kondrak: ~1 hour CPU per ciphertext at 380 languages because per-language key search doesn't amortize; every classical method carries per-language n-gram tables). It also makes bound-tightness comparability *worse*: N independently-trained models have N independent optimization outcomes. A shared backbone shares its inductive biases across languages, which is exactly what R1 wants.
- **Prefix language token.** Workable, but an additive embedding at every position conditions the reconstruction of every masked character symmetrically regardless of position, and doesn't consume context length. Minor choice; flagged as a default.
- **No conditioning (pure LID head on an unconditional model).** Rejected — we need conditional ELBOs `-log p(x | L)` per candidate language as the ranking metric; an unconditional model gives only one number.

The 10% conditioning dropout gives us an unconditional ELBO from the same network. That enables:
1. A Bayesian read-out: `p(L|x) ∝ p(x|L)·p(L)` with the unconditional score as a normalization sanity check.
2. An **abstention channel**: text that scores poorly under *all* language conditions but decently unconditionally is "structured but not any trained language" — precisely what the `voynichesque.py` negative control (Voynich-like gibberish with no plaintext) must trigger (§9).

**Language inventory (two tiers, all trained from the start — see §7 for why):**
- *Tier 1 (cipher-demonstrated):* Latin, Italian, German — the languages Naibbe is demonstrated/claimed on and the stub languages in the inverse note.
- *Tier 2 (candidate set from prior statistical work):* Danish, Dutch, English, French, Spanish, Czech, Hungarian, Greek (transliterated), Hebrew (transliterated, abjad-aware). Motivated by Arutyunov's distance results (Danish L1=0.10, Latin 0.11, Latin+Danish 2:1 mixture 0.09; conclusion "~30% Germanic, ~70% Romance") and Hauer & Kondrak's Hebrew ranking.

Corpora: period-appropriate (medieval/early-modern) text per language, orthographically normalized to the shared alphabet by one common pipeline — which strips all whitespace (§2) so every language's training text is an unsegmented character stream. A concrete candidate source: the voynich-attack repo bundles ~20M tokens of reference corpora across German (14M+), Latin, Dutch, French, English, Spanish, and Hebrew, sourced from historical archives — vet period-appropriateness and upstream licenses per language before adoption. Domain matching matters more than volume — Vo & Khoury found retraining a general LID system on mismatched domains *hurt*; the digram-table swap in Dhavare (13% → 84% by matching the LM to the actual source) is the same lesson from the classical side.

**Balancing:** temperature-smoothed sampling (τ ≈ 0.7 over corpus sizes) so no language is starved, but with the *evaluation-side* correction in §5 carrying the real burden of fairness — data balancing alone cannot equalize bound tightness.

---

## 5. The ELBO as a language metric: estimation and calibration

The metric is the model; these choices matter as much as architecture.

**Decision 5a: report length-normalized bits-per-character, estimated with stratified timestep sampling and common random numbers across language conditions.**

- Stratify t over [0,1] (e.g., 64 strata × k samples) rather than i.i.d. sampling — the integrand's variance is concentrated at low masking rates.
- **Common random numbers:** score every candidate language with the *same* masking realizations (same t values, same masked positions). Language ranking is a paired comparison; sharing the noise cancels most Monte Carlo variance out of the *differences*, which is what the argmax consumes. This is cheap and, for a ranking instrument, the single highest-leverage variance-reduction trick available.
- Report per-window scores across the document (§3) → mean and spread, so "Germanic candidates rank highest" can carry an uncertainty statement, as the paper's "exploratory, assumption-dependent" framing requires.

**Decision 5b: per-language bound calibration against held-out clean text.**

The known methodological risk (stated verbatim in the references): "cross-language comparisons implicitly assume comparable bound tightness across languages" — assumed, never validated. We do not assume it; we measure and correct:

1. Hold out a clean-text set per language, matched in size and domain.
2. Measure each language's NELBO on *its own* held-out text; the spread across languages estimates language-specific bound looseness plus intrinsic entropy differences.
3. Where an AR reference is affordable, train a small character AR transformer on the same data and use `NELBO − NLL_AR` per language as a direct bound-gap estimate; apply it as an additive per-language offset at ranking time.
4. Validate end-to-end on synthetic ciphers with known plaintext language: the calibrated ranking must recover the true language across all (cipher × language × length) cells before any VMS number is reported. Hauer & Kondrak's 97.1%-over-380-languages figure is the bar for the easy (1:1) end.

Expected residual failure mode: confusion *within* language families (Serbian/Bosnian in Hauer & Kondrak; the survey's close-pair problem). Report results at both language and family granularity — the paper's own headline ("Germanic candidates") is already family-level, which is the honest resolution.

---

## 6. Language-ID head

**Decision: mean-pool over final-layer hidden states (computed at several masking levels, averaged) → 2-layer MLP → softmax over languages + one "no-language/synthetic" class.**

- **Character-level features, learned in-domain, no lexicon.** Duvenhage's ablation is the cautionary tale: lexicon-anchored LID drops from 96% to 75% when test vocabulary leaves the lexicon — and trial decipherments are *guaranteed* to be out-of-lexicon. Vo & Khoury show attention/pooling learns to ignore garbage spans without preprocessing, which is what we need for residual cipher artifacts.
- **The head sees the same corruption distribution as deployment:** clean text, masked text at all ratios, and simulated partial decipherments (§7, noise curriculum). Training the head only on clean text would repeat the domain-mismatch failure the LID literature documents.
- **The extra class** is trained on `voynichesque.py` output and shuffled text — giving the system a learned abstention channel to complement the ELBO-based one (§4).
- The head is a *cross-check*, not the metric. The ELBO ranking is primary (it is the principled likelihood); the head is fast, calibratable (temperature-scaled on held-out decipherments), and its disagreement with the ELBO ranking is itself a diagnostic worth reporting.

---

## 7. Training curriculum

This is where most silent failure modes live. The schedule is three phases, with the two headline curriculum questions answered explicitly.

### 7.1 All languages from the start — not single-language-first

**Decision: train multilingual from step one, with temperature-balanced sampling. Do not pretrain on one language and add others.**

Reasoning:

- **Symmetry is a correctness property here, not a convenience.** The instrument is a *comparison* of per-language bounds. Sequential language introduction gives earlier languages more optimization steps, better-settled representations, and tighter bounds — a bias with exactly the same signature as the finding we want to report ("language X scores best"). Simultaneous exposure is the only schedule that is symmetric by construction; the calibration in §5b then corrects second-order asymmetries instead of a first-order one.
- **Curriculum-by-language buys nothing identifiable.** The usual argument for staged curricula is task difficulty ordering; languages here are not ordered by difficulty, and the shared character vocabulary means there is no tokenizer-warmup argument either.
- **Positive transfer is real at this scale.** Kambhatla 2023's single 13-language model beats its baseline on Borg (4.10% vs 5.47%) with *no* language ID; ALICE handles 7 languages in one model. With 23–26 shared characters and closely related candidate languages, shared representations help the low-resource tier (medieval Danish is not a large corpus).
- The one *real* risk of joint training — interference degrading every language a little — is monitored, not assumed away: per-language held-out NELBO is tracked throughout; if any language's bound stagnates while others improve, adjust sampling temperature, not schedule.

### 7.2 LID head: delayed attachment, then joint fine-tuning — not concurrent from step zero

**Decision: Phase A trains the backbone alone. The LID head attaches in Phase B behind a stop-gradient; the stop-gradient is released in Phase C with a small loss weight. The final model is "jointly trained" (as the paper states) — but joint from the *end*, not from the beginning.*

Reasoning:

- **Protect the instrument.** The NELBO is the measurement device (R1). Discriminative gradients from a classification loss shape features toward *between-language contrast*, which is not the same objective as *within-language density* and can distort it — and any distortion that differs by language is a ranking bias. Early in training, when the backbone is far from converged, the classification loss is large and its gradients are proportionally most distorting. Letting the generative objective stabilize first means the joint phase makes small, monitorable adjustments to a settled model.
- **The head needs almost nothing.** LID over ~10 languages from rich character features is a nearly linear problem (fastText-class models solve the clean version); the head trains to convergence in a small fraction of backbone compute. There is no data-efficiency argument for concurrent training from step zero.
- **But pure post-hoc (frozen backbone forever) is also wrong.** The paper claims — and the references support — that joint training improves *noise robustness*: D3PM's hybrid loss (ELBO + auxiliary cross-entropy) is the direct precedent that a small auxiliary loss on top of the variational bound stabilizes rather than harms, and a head trained only behind a stop-gradient cannot recruit features the backbone never built (e.g., features that separate German from Danish under 30% letter noise may not be needed for reconstruction alone).
- **Concrete schedule:** Phase C uses `L = L_NELBO + λ·L_LID` with λ ramped 0 → ~0.05 (λ chosen so the LID gradient norm stays under ~10% of the diffusion gradient norm — informed default). **Abort criterion:** per-language held-out NELBO is the canary; if any language's bound degrades by more than ~1% relative during Phase C, halve λ. If ranking on the synthetic validation suite (§9) changes between end-of-B and end-of-C, that is a red flag for the whole framework, and is reported, not tuned away.

### 7.3 Noise curriculum: clean → masked-only → decipherment-noise mixture

**Decision: Phase A = clean text (masking is the only corruption). Phase B adds simulated partial-decipherment noise on 30–50% of examples. Never remove the clean fraction.**

- The masking process itself already trains reconstruction-from-partial-evidence; Phase A therefore isn't "noise-free" in the relevant sense — it establishes calibrated conditional distributions.
- Phase B noise is *structured to match deployment*, not generic: (a) **systematic substitution noise** — sample a random many-to-one letter map and apply it to a fraction of positions, simulating a partially wrong cipher key (wrong keys are self-consistent, not i.i.d.; i.i.d. flip noise is the wrong model); (b) **segmentation noise** — with whitespace already stripped in preprocessing (§2), there are no spaces to corrupt; segmentation errors instead manifest in the letter stream itself: spurious insertions, deletions, and duplications mimicking wrong unigram-vs-bigram parses of a Naibbe-style ciphertext (whose 50-50 re-spacing layer is absorbed by the whitespace removal); (c) **transcription noise** at ~5%, the level at which Bruton 2026 stays ≥0.99 F1, evidencing that training-time noise injection is what buys robustness.
- The clean fraction is retained forever because §5b's calibration is defined on clean text; letting the clean-text NELBO drift invalidates the offsets.

### 7.4 Cipher heads: strictly after the backbone, frozen evaluator, easy-to-hard validation ladder

**Decision: the backbone is frozen (EMA weights) during all cipher-head optimization. Heads are validated in difficulty order: 1:1 substitution → unigram homophonic → Naibbe mixed unigram-bigram → arithmetic-encoded.**

- Freezing is inherited directly from the differentiable-inverse note (the n-gram LM is "a buffer, no grad") and from every working classical attack: the language model is the fixed measuring stick against which cipher maps are optimized. A moving evaluator makes the map-recovery landscape (already "full of self-consistent-but-wrong maps") non-stationary, and would let the cipher head *train the evaluator into agreeing with a wrong map*.
- The difficulty ladder mirrors the literature's own history (1:1 solved → homophonic solved → verbose/arithmetic open) and gives per-rung go/no-go gates with known targets: ≤1.9% SER on Zodiac-408-class problems at the homophonic rung (Kambhatla 2023's greedy result) before attempting Naibbe; Naibbe recovery on synthetic Latin/Italian pairs (ground-truth alignments ship with the Greshko repo) before the arithmetic head, now pinned to the pseudo-Voynich generator (`voynpy.pseudo_vms` in Boxer's voynich-attack repo). It remains the hardest rung — letter identity is hidden behind the unknown cipher-character value assignment (only a token's value-sum identifies the letter), and with whitespace stripped (§2) token boundaries are latent too — but the interface is simpler than the interval-coding design previously assumed: no dummy symbols, no interval tracking; the head is a sum-constrained homophonic inverse with segmentation marginalization over 2–6-character token lengths, a direct generalization of the Naibbe head's semi-Markov DP.
- Search still runs on the cheap frozen n-gram DP scorer (exact semi-Markov marginalization) as the inner loop; the diffusion ELBO scores shortlisted maps and provides dense gradients for refinement via expected embeddings (§8). This two-tier design respects R6: n-gram DP for the many restarts, 85M ELBO for the decisions.

### 7.5 Optimization defaults (flagged: not evidence-derived — the references record no optimizer details at scale)

AdamW (β₂ = 0.98 for small-vocab stability), peak LR 3e-4 with 2k-step warmup and cosine decay, batch ~0.5M characters, bf16, EMA of weights (decay 0.9999) used for all evaluation and as the frozen cryptanalysis evaluator. Token budget: train until per-language held-out NELBO plateaus; with corpora this small, expect multiple epochs and rely on the noise curriculum as the effective regularizer. Dropout 0.1 as a starting point given corpus size.

---

## 8. Interface to the decoding heads

**Decision: the backbone accepts mixture inputs — expected embeddings under the head's soft letter distributions — on a 2N-slot frame with a learned NULL token; per-slot NELBO terms provide the dense signal.**

- **Soft inputs:** cipher heads emit row-stochastic distributions over letters (the Naibbe `U`/`Pre`/`Suf` matrices). Input at each slot is the expectation of character embeddings under that distribution. Gradients flow to the head parameters through the embedding table (R3). Straight-through sampling is the fallback if expectation inputs prove too smooth to discriminate sharp maps.
- **Latent length (R4) via the 2N-slot scheme** from the inverse note: each ciphertext token owns two plaintext slots; slot 2 is blended between its real letter distribution and NULL by the token's unigram-vs-bigram weight `w_t`. NULL is a first-class trained token (it appears in Phase B training data at matching rates) so the backbone treats "this slot is structurally empty" as in-distribution, not as noise. The known `logaddexp(−∞,−∞)` NaN trap from the note applies to any log-space blending here — blend before re-padding, and keep the smoke test.
- **Scoring scale (R5):** all heads report calibrated bits-per-plaintext-character under the same backbone, plus a complexity penalty (parameter count of the head / description length of the map) so verbose heads cannot win by capacity alone. This is the uniform scale on which "(cipher × language)" cells are ranked.

---

## 9. Evaluation protocol and controls

1. **Synthetic grid:** every (cipher class × language × length ∈ {50, 100, 200, 400, 700} chars) cell, ≥50 ciphers per cell (Hauer 2014 convention). Primary metrics: language-ID accuracy at language and family granularity; SER for head decipherments; ranking stability across restarts.
2. **Negative controls:** `voynichesque.py` text (must abstain), shuffled real text, and Tier-2-language text enciphered with ciphers fit on Tier-1 (tests cross-contamination).
3. **Bound-fairness audit (§5b)** re-run after every training phase; published alongside results.
4. **Currier A/B split:** all VMS numbers reported per dialect. Naibbe's replication of B is documented as incomplete, and Parisel's four signatures are dialect-calibrated — pooling A and B would blur the very statistics the ciphers are tuned to.
5. **Known-benchmark anchors** for the head machinery: Zodiac-408 (≤1.9% SER), Borg (≤4.10%), BnF fr2988 (≤1.13%) — not because the VMS is these ciphers, but because failing solved problems invalidates any unsolved-problem claim.

---

## 10. Key risks and open questions

- **Bound-tightness comparability (R1)** remains the load-bearing assumption even after calibration; the AR-reference gap estimate (§5b.3) is an estimate, not a proof. This inherits the paper's own "exploratory, assumption-dependent" framing.
- **The arithmetic-encoded cipher is now pinned** to the pseudo-Voynich generator in Boxer's voynich-attack repository (`voynpy.pseudo_vms.PseudoVmsEncoder`, MIT). Concrete parameters: a 16-character cipher alphabet (hex digits `0`–`9` valued 0–9, `A`–`D` valued 10–13, `E` = −1, `F` = −2); plaintext letters assigned integer values (default a=3 … z=28 — replace with our frozen alphabet via the encoder's custom-alphabet option); each letter enciphered as one 2–6-character token whose values sum to the letter's value, drawn Zipf-weighted (`zipf_exponent`) from ~500 pre-sampled homophones per letter with a fixed length distribution (2: 10%, 3: 22%, 4: 26%, 5: 26%, 6: 16%); a doubling mechanism (`doubling_strength`, default 0.26) reuses the previous token for repeated letters and is tuned via `tune_to_vms` to the VMS's ~0.92% token-doubling rate — a per-source-language calibration, since it depends on the language's letter-doubling rate. There are no dummy symbols and decode is a deterministic sum lookup, so the Ryabko–Fionov/Ziegler interval-decoder assumption from the references is superseded: the head (§7.4 rung 4) is a sum-constrained homophonic inverse, not an interval tracker. The generator emits space-delimited tokens; after whitespace removal (§2) token segmentation is latent, so the head marginalizes over 2–6-character token lengths. Pin to the **latest** `voynpy.pseudo_vms` with its default parameters (`zipf_exponent`, `tokens_per_char`, token-length distribution, ~500 homophones/letter); set the custom alphabet to our frozen vocab (§2), tune `doubling_strength` per language via `tune_to_vms` to the ~0.92% rate, and fix a seed — recorded in the data spec and implemented under the cipher-generator acquisition task. Its head still warrants its own design note.
- **The Naibbe generator is pinned** to `naibbe_v2.py` (greshko/naibbe-cipher, commit `df3d074`). It is distribution-identical to v1 (same six tables, card-deck weights, `RESPACING=17`, 3% space removal) but hardens decipherability — a full bigram-collision catalog guarantees uniquely reversible ciphertext (v1 checks only unigram collisions) and bounded retries (`MAX_BIGRAM_RETRIES=10000`) make corpus-scale generation terminate safely. Its built-in `clean_line` normalization is superseded by our shared pipeline (§2), which sets the encoder's custom alphabet to our frozen vocab.
- **Alphabet discrepancy:** the "~22-letter" alphabet in the inverse note lists 23 symbols and omits K/W needed by Germanic candidates; resolve against Greshko's actual tables (repo / Zenodo 10.5281/zenodo.16415087) before freezing the vocabulary.
- **Word-boundary and line-position phenomena** (line-as-functional-unit, word-length autocorrelation — the effects Naibbe fails to reproduce) are deferred, consistent with the paper's future-work section — and, since whitespace is removed in preprocessing (§2), they are excluded from the model's input *by construction*, not merely unmodeled. Extension point: they enter as additional conditioning or positional features, not as changes to the diffusion formulation or as a reintroduced space token.
- **Verbose ciphers vs word-identity statistics:** Naibbe-style homophony maps one plaintext word to many ciphertext forms, degrading exactly the word-level structure (Zipf, keyword clustering at ~800-word scales) that Montemurro & Zanette show the VMS retains. This tension is unresolved in the project notes and is a genuine argument the framework may eventually need to address at the cipher-ranking level.

---

*Sources: project reference summaries 1–8, the framework paper draft, and the differentiable-inverse design note. Numbers cited from summaries inherit their caveats (several were compiled from abstracts/secondary sources; see each summary's provenance notes).*
