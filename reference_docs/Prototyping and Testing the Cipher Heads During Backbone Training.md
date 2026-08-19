# Prototyping and Testing the Cipher Heads During Backbone Training

A working plan for advancing the cipher decoding heads (Phase 5 / workstream WS6 of the [[Diffusion Model Training - Task Breakdown]]) **concurrently with Phase 1 backbone pretraining**, rather than waiting behind gate G4. Companion to [[Design and Training of the Multilingual Diffusion Backbone]] and [[A Differentiable Inverse for a Naibbe-Style Homophonic Substitution Cipher]].

---

## 1. The problem and the opening

In the time-ordered plan, cipher heads are the *last* modelling work: Phase 5 depends on G4, which depends on a fully pretrained, noise-hardened, LID-attached, **frozen** diffusion backbone. Read literally, nobody touches a cipher head until Phases 1–4 are done. That serializes the two hardest, most independent research problems in the project — training the backbone and cracking the verbose-homophonic heads — one after the other.

They do not have to be serial. The design already contains the escape hatch, in §7.4 of the backbone doc:

> "Search still runs on the cheap frozen n-gram DP scorer (exact semi-Markov marginalization) as the inner loop; the diffusion ELBO scores shortlisted maps and provides dense gradients for refinement via expected embeddings. This two-tier design respects R6: n-gram DP for the many restarts, 85M ELBO for the decisions."

The inner loop — the differentiable inverses, the semi-Markov DP, the restart/annealing/discrete-move search that actually recovers a cipher map — is scored by a **frozen n-gram language model**, not the diffusion ELBO. The diffusion model is the *outer* scorer: it re-ranks a shortlist and supplies dense gradients for final refinement. The [[A Differentiable Inverse for a Naibbe-Style Homophonic Substitution Cipher|Naibbe-inverse note]] says the same thing from the other side — its §4 evaluator fork lists the character n-gram LM first and flags it "**the sweet spot. Start here.**"

So the plan is:

> **Build every cipher head against a scorer-abstraction interface. Develop and validate them now, against a frozen n-gram evaluator, on synthetic ground truth and the literature's solved-cipher anchors. When the frozen diffusion backbone lands (post-G4), swap the evaluator behind the same interface, re-run the identical harness, and measure the delta.**

This is not building a mock and throwing it away. The n-gram DP scorer is a **permanent, load-bearing component** of the final system (§7.4). Work done here is the real inner loop.

---

## 2. The core enabling abstraction: a frozen `Evaluator` interface

Every head is written against one narrow contract: *given soft (row-stochastic) distributions over the plaintext alphabet, return a differentiable scalar plaintext-quality score, and its gradient w.r.t. those distributions.* Nothing about a head's parameterization or search loop should name "n-gram" or "diffusion."

```python
class Evaluator(Protocol):
    """A frozen plaintext-quality scorer. No parameters are ever updated
    through this object; it is the fixed measuring stick (design §7.4)."""

    A: int  # plaintext alphabet size (frozen vocab, task 0.1)

    def score_fixed(self, soft_letters, *, language) -> Tensor:
        """soft_letters: (L, A) row-stochastic. Fixed 1:1 alignment
        (one emitted letter per position). Returns scalar log P(plaintext).
        Used by rungs 1-2, where there is no latent segmentation."""

    def score_segmental(self, emissions, segmentation, *, language) -> Tensor:
        """emissions + a per-token candidate structure (unigram / bigram /
        variable-length token). Returns scalar log-likelihood marginalized
        over segmentation via the semi-Markov forward DP. Used by rungs 3-4."""

    def as_embedding_frame(self, soft_letters, null_weights) -> Tensor:
        """Optional: project soft letters onto the 2N-slot expected-embedding
        frame (design §8). Only the DiffusionEvaluator uses this; the n-gram
        evaluators ignore it. Present so heads emit the frame from day one."""
```

Three concrete implementations, in increasing fidelity and availability:

| Implementation | Available | Role | Scoring mechanism |
|---|---|---|---|
| `NgramEvaluator` | **Now** (days) | The workhorse; the permanent inner loop | `logT` transition tensor; exact forward / semi-Markov DP (inverse-note §5). Fixed-alignment heads can also score with a direct high-order (penta-gram) n-gram, no DP. |
| `SmallARLMEvaluator` | **~Phase 1** (needed anyway for §3.4 calibration) | De-risks the *neural* path: tests the sampling + straight-through fallback and the "expected embeddings too smooth?" question before the backbone exists | Small char-AR transformer per language, `p(letter \| history)`; no closed-form marginalization → sample parses + straight-through |
| `DiffusionEvaluator` | Plumbing **now** (random-init backbone, a G0 artifact); real **post-G4** | The final outer scorer and dense-gradient source | Frozen EMA backbone, expected-embedding 2N-slot inputs → per-slot NELBO (design §8) |

The random-init `DiffusionEvaluator` is worth calling out: a randomly-initialised backbone **already exists at G0** (the gate requires "per-language NELBO metric logs correctly to ClearML on a random-init model"). Its *scores are meaningless*, but the *interface is fully testable* against it now — gradient flow through the embedding table, the 2N-slot NULL blend, and the `logaddexp(−∞,−∞)` NaN smoke test (task 5.1) all run against random weights. That makes the eventual swap mechanical rather than exploratory.

---

## 3. What "progress" means without the real backbone

Three things can be genuinely established now, with n-gram scoring, and none of them depend on the diffusion model:

1. **Head mechanics.** Each head recovers ground-truth maps on synthetic ciphers whose plaintext, key, and alignment are all known (the generators of task 0.7 emit exactly this). This is a correctness property of the head + search loop, independent of which frozen scorer drives it.
2. **The literature anchors.** Zodiac-408 (≤1.9% SER), Borg (≤4.10%), BnF fr2988 (≤1.13%) are public ciphers with known solutions, and they were *originally solved with n-gram/pentagram scorers* (Ravi & Knight word+3-gram; Kopal/AZdecrypt pentagram SA). They are the right validation target for the n-gram-scored heads — passing them proves the machinery independent of both the VMS and the diffusion backbone (this is task 6.6, brought forward).
3. **The core language-discrimination premise.** Score a synthetic ciphertext by trial decipherment under each candidate language's n-gram LM and check the true language wins. This is the n-gram analogue of the diffusion-ELBO ranking, and it de-risks the framework's *headline claim*, not just the plumbing. Dhavare et al.'s 13%→84% jump on the Zodiac challenge when the digram table was swapped to the correct source is the precedent: decipherment quality is a function of the assumed language, so a decipherment score ranks languages.

What **cannot** be settled until the real backbone exists is deferred honestly to §9.

---

## 4. Head-by-head prototyping plan

The difficulty ladder is unchanged (design §7.4). Two structural families:

- **Fixed-alignment heads (rungs 1–2):** one emitted plaintext letter per cipher symbol. No latent segmentation → no DP needed; score `soft_letters` directly with a high-order n-gram (`score_fixed`). This is where the penta-gram-scale scorers that the anchors demand are cheap.
- **Segmental heads (rungs 3–4):** token → 1..k latent letters, token boundaries latent after whitespace stripping. Score with the semi-Markov forward DP (`score_segmental`), tractable to trigram/4-gram; supplement with discrete moves rescored under a penta-gram.

### Rung 1 — 1:1 substitution head (Sinkhorn / bijective)

- **Parameterization.** ALICE-style: a learnable square matrix Sinkhorn-normalized to doubly-stochastic; Gumbel-Sinkhorn sampling during training to enforce the bijective (permutation) constraint. This structurally encodes the 1:1 key — the design principle ALICE validates and §7.4 rung 1 inherits.
- **Scorer.** `score_fixed` — apply the soft permutation to the cipher stream to get `(L, A)` soft letters, chain through a bigram/trigram `logT`. Trivial DP (no segmentation).
- **Ground truth.** Apply a random permutation to normalized corpus text (tasks 0.1/0.3); recover it.
- **Acceptance (mirrors 5.2).** Near-perfect recovery on synthetic 1:1 at ≥200 chars. This is the sanity check that a *frozen* evaluator drives correct map recovery — here the n-gram LM stands in for the diffusion ELBO. Matches the classical result (Kambhatla 2018 LM-guided search; ALICE 0.06% SER >128 chars).

### Rung 2 — unigram homophonic head

- **Parameterization.** Row-stochastic soft-assignment matrix `(V_sym, A)`, many cipher symbols → one letter — **not** bijective, so no Sinkhorn/permutation prior (inverse-note §6: "you *cannot* use the permutation/Sinkhorn prior... the map is many-to-one"). Regularize instead with row-entropy annealing (push to one-hot) and a **letter-frequency-prior KL** on the expected emitted-letter distribution.
- **Scorer.** `score_fixed`, but with a **penta-gram** `logT` for the anchors — the anchors were achieved at pentagram order (AZdecrypt/Kopal) and word+3-gram (Ravi & Knight). A pure char-trigram may not reach 1.9% on Zodiac-408; use ≥5-gram and/or interleave discrete simulated-annealing moves (the classical workhorse) rescored by the penta-gram. Report the n-gram baseline honestly; closing any residual gap is part of what the diffusion backbone is later asked to do.
- **Search loop.** Hybrid, per inverse-note §7/§9: gradient steps on the soft matrix against the DP score, entropy/temperature annealing, frequency-prior KL, **many random restarts**, interleaved with hill-climb / fixed-temperature SA on the argmax map. Log evals-per-solve and wall-clock (R6 cost realism / task X.3).
- **Ground truth + anchors.** Synthetic unigram-homophonic generator (N symbols/letter by frequency) for map-accuracy; **Zodiac-408, Borg, BnF fr2988** (public transcriptions + solutions) for SER anchors.
- **Acceptance (5.3 / 6.6).** ≤1.9% SER Zodiac-408-class; Borg ≤4.10%; BnF fr2988 ≤1.13% — with n-gram scoring, reported as the pre-diffusion baseline.

### Rung 3 — Naibbe mixed unigram-bigram head

This head is already specified end-to-end in the [[A Differentiable Inverse for a Naibbe-Style Homophonic Substitution Cipher|inverse note]], with a **validated PyTorch skeleton** (its §8 smoke test passes: finite likelihood, gradient flow to `U`/`Pre`/`Suf` and the segmentation weights, frozen LM, loss decreases). What remains is production work, all doable now:

- **Parameterization.** Three soft inverse maps `U` (unigram type→letter), `Pre`, `Suf` (~9k logits total), each a softmax over the frozen alphabet.
- **Structural prior.** The Zattera slot grammar as a *hard* prior (inverse-note §3): parse each token deterministically into a unigram candidate and/or a (prefix, suffix) pair *before* training, collapsing latent structure to one binary variable per token. Implement `parse_token` from the type-1/type-2 marker rule; build the three vocabularies. (These are the two stubs the skeleton leaves: `parse_token` and `load_ngram_lm`.)
- **Scorer.** `score_segmental` — the exact semi-Markov forward DP that marginalizes over both segmentation and soft-letter identity in one pass (inverse-note §5). **Keep the `logaddexp(−∞,−∞)` guard**: mix the finite real-letter entries before re-padding the BOS slot.
- **Scale-up work (inverse-note §8 notes).** The `for t in range(T)` HMM loop is fine as a skeleton but slow at VMS scale (~38k tokens × many restarts). Vectorize / `torch.jit` / chunk it, or express as a WFST in GTN. Move to a **trigram** LM (α becomes `(A, A)`, `advance` an einsum).
- **Ground truth.** `naibbe_v2.py` (pinned commit `df3d074`, task 0.7) on Latin/Italian corpus → paired ciphertext with the **ground-truth alignments that ship with the Greshko repo**.
- **Acceptance (5.4).** Recovers ground-truth maps at ≥95% letter-map accuracy on synthetic Naibbe Latin/Italian pairs; restart budget documented. Expect to *need* the restarts + discrete interleaving — the cipher is engineered (conditional char entropy ~2.0) to destroy exactly this signal (inverse-note §7).

### Rung 4 — arithmetic sum-to-target head

- **Parameterization (design §10).** Jointly infer the 16 cipher-character integer values (hex `0`–`9`→0–9, `A`–`D`→10–13, `E`→−1, `F`→−2) and the letter-value map; the head is a **sum-constrained homophonic inverse** — a token decodes to the letter whose assigned integer equals the token's value-sum. No dummy symbols, no interval tracking (the Ryabko–Fionov interval-decoder assumption is *superseded* — §10).
- **Scorer.** `score_segmental`, generalizing the Naibbe semi-Markov DP to marginalize the latent **2–6-character token segmentation** (whitespace stripping removed token boundaries) plus the doubling mechanism.
- **Ground truth.** `voynpy.pseudo_vms` at the **pinned config** (task 0.7: latest version, default `zipf_exponent`/`tokens_per_char`/length-dist/~500 homophones-per-letter, custom alphabet = frozen vocab, `doubling_strength` tuned per language to the ~0.92% VMS rate, fixed seed). Ground truth via the generator's exact sum-lookup decode.
- **Acceptance (5.5).** Recovers plaintext *language* on synthetic pseudo-VMS at better-than-family-random rates. This is the hardest rung and may trail — proceed without it if needed, noting the gap (per G5). It "warrants its own design note" (§10); this early-phase work is where that note gets written, against n-gram scoring, well before the backbone is ready.

---

## 5. The shared test harness (evaluator-agnostic)

One harness, parameterized by `Evaluator`, so a swap is a single argument. Built now against `NgramEvaluator`; re-run unchanged against `DiffusionEvaluator` later.

- **Synthetic grid (design §9.1 / task 3.6).** Every (cipher class × language × length ∈ {50, 100, 200, 400, 700}) cell, ≥50 ciphers/cell (Hauer 2014 convention). Primary metrics below.
- **Negative controls (§9.2).** `voynichesque.py` output (must fail to resolve to any language / trigger abstain), shuffled real text, and Tier-2-language text under Tier-1-fit ciphers (cross-contamination).
- **Known-benchmark anchors (§9.5 / 6.6).** Zodiac-408, Borg, BnF fr2988 — runnable now.
- **Metrics.**
  - **Letter-map accuracy** (maps with ground truth).
  - **SER** (field convention; Kambhatla / ALICE lineage).
  - **Language-recovery accuracy** at both language and family granularity (the framework's real target; family-level is the honest resolution per §5b).
  - **Ranking / recovery stability** across restarts and seeds (flip-rate).
  - **Cost:** evals-per-solve, wall-clock per restart (R6 / X.3) — motivates the 25M sibling later.
- **Scoring scale (design §8, R5), built once and evaluator-agnostic.** Calibrated bits-per-plaintext-char **plus a description-length / parameter-count complexity penalty**, so verbose heads cannot win by capacity alone. Implement this now so cross-head comparison (5.6) is ready the moment the diffusion evaluator lands.

---

## 6. Building forward-compatible: the diffusion interface, stubbed now

To make the eventual swap mechanical rather than a rewrite, each head emits the diffusion interface's inputs **from day one**, even though only the n-gram scorer consumes gradients today:

- **Soft inputs / expected embeddings (§8, R3).** Heads already emit row-stochastic letter distributions; the `as_embedding_frame` path takes their expectation under the embedding table. Test gradient flow through a **random-init** embedding table now.
- **2N-slot latent-length frame (§8, R4).** Each ciphertext token owns two plaintext slots; slot 2 is blended between its real-letter distribution and the learned `NULL` by the token's unigram-vs-bigram weight `w_t`. Wire this now; the **same `logaddexp(−∞,−∞)` NaN trap** applies to any log-space blend — blend before re-padding, keep the smoke test (task 5.1).
- **Calibration hooks (§3.4 / 5b).** Leave a single, single-sourced place where a per-language additive offset is applied at ranking time. The `NgramEvaluator` uses a `NELBO − NLL_AR`-style offset of its own; the `DiffusionEvaluator` later slots its offset into the identical hook.

This is exactly task **5.1** (head interface: expected-embedding inputs, NULL-blended slot-2, straight-through fallback, NaN smoke test). It is listed under Phase 5 but has no true dependency on a *trained* backbone — only on a backbone *object* with an embedding table, which exists at G0. **Pull 5.1 forward.**

---

## 7. Build order mapped onto Phase 1

Phase 1 (backbone pretraining) is the long pole — pilot + full Phase-A training to per-language NELBO plateau across three languages. That is the window. Suggested ordering of the concurrent cipher-head track:

1. **Week 0 (prereqs, §11).** Freeze vocab (0.1); run the normalization pipeline (0.3); stand up the generators + controls (0.7); train per-language char n-gram LMs (new task, §11). Implement the `Evaluator` interface + `NgramEvaluator`.
2. **Rung 1** (1:1 Sinkhorn) — fastest to a green synthetic result; exercises the harness end-to-end and the scoring-scale machinery.
3. **Task 5.1 interface + random-init `DiffusionEvaluator`** — get the NaN/gradient smoke tests green against random weights while it's cheap.
4. **Rung 2** (unigram homophonic) + the **literature anchors** (Zodiac-408 / Borg / BnF). First externally-checkable milestone.
5. **Rung 3** (Naibbe) — the flagship head; productionize the validated skeleton, scale the DP, hit ≥95% map accuracy on Greshko pairs. Largest single chunk of work; it can run the whole length of Phase-A training.
6. **`SmallARLMEvaluator`** (also needed for §3.4 calibration) — de-risk the neural/straight-through path and the "expected embeddings too smooth?" question before the diffusion model exists.
7. **Rung 4** (arithmetic) + its own design note — start whenever rung 3 is stable; expected to trail.

By G1/G2, rungs 1–3 should be passing their acceptance criteria *under n-gram scoring*, with rung 4 in progress. Phase 5 then collapses from "build and validate four heads" to "swap evaluator, re-run harness, measure delta, refine."

---

## 8. Task table (house style)

IDs prefixed `CH` (cipher-head early track); priorities and acceptance mirror the corresponding Phase-5 tasks so they merge cleanly.

| ID | Task | Priority | Depends on | Acceptance criterion |
|---|---|---|---|---|
| CH.0 | **Per-language char n-gram LMs** (trigram for the DP; penta-gram for fixed-alignment anchors), trained on the normalized corpora. Exposed as `logT` / n-gram tables. | **P0** | 0.1, 0.3 | LMs load; held-out bits/char sane per language; both DP-order and penta-gram-order variants available. |
| CH.1 | **`Evaluator` interface + `NgramEvaluator`** (`score_fixed`, `score_segmental`, `as_embedding_frame` stub). | **P0** | CH.0 | Interface stable; n-gram DP matches inverse-note skeleton to tolerance; one-line evaluator swap demonstrated. |
| CH.2 | **Shared test harness** (synthetic grid, anchors, negative controls, metrics, scoring scale + complexity penalty). | **P0** | CH.1, 0.7 | Harness runs any head × any evaluator; metrics + uncertainty reported; scoring scale single-sourced. |
| CH.3 | **Rung 1 — 1:1 Sinkhorn head.** | **P0** | CH.1 | Near-perfect recovery on synthetic 1:1 at ≥200 chars (= task 5.2, n-gram-scored). |
| CH.4 | **Task 5.1 head interface + random-init `DiffusionEvaluator`** (2N-slot, NULL blend, straight-through, NaN smoke test). | **P0** | CH.1, G0 | Gradients reach a toy head; `logaddexp(−∞,−∞)` smoke test passes against random-init weights. |
| CH.5 | **Rung 2 — unigram homophonic head** + literature anchors. | **P0** | CH.2, CH.3 | ≤1.9% SER Zodiac-408-class; Borg ≤4.10%; BnF fr2988 ≤1.13% under n-gram scoring, reported as pre-diffusion baseline (= 5.3 / 6.6). |
| CH.6 | **Rung 3 — Naibbe head** (productionize inverse-note skeleton: `parse_token`, vocabularies, vectorized semi-Markov DP, restarts + discrete moves). | **P0** | CH.2, 0.7 | ≥95% letter-map accuracy on synthetic Naibbe Latin/Italian pairs (Greshko alignments); restart budget documented (= 5.4). |
| CH.7 | **`SmallARLMEvaluator`** — neural scorer; tests sampling + straight-through fallback and embedding-smoothness. (Reused for §3.4 calibration.) | **P1** | CH.1 | Straight-through path recovers a synthetic map; smoothness probe reported. |
| CH.8 | **Rung 4 — arithmetic sum-to-target head** + own design note. | **P1** | CH.6, 0.7 | Better-than-family-random language recovery on pinned pseudo-VMS synthetics (= 5.5). |
| CH.9 | **Language-discrimination probe** — trial-decipherment ranking under per-language n-gram LMs recovers true language on the synthetic grid. | **P1** | CH.5 | True language wins at target rate per length cell; family-level confusion reported (n-gram precursor to 3.6). |

---

## 9. The evaluator swap and delta-measurement protocol

When the frozen EMA backbone passes G4:

1. **Swap.** Replace the random-init `DiffusionEvaluator` weights with the frozen EMA weights. No head or harness code changes (that is the whole point of §2).
2. **Re-run the identical harness.** Same synthetic grid, anchors, controls, seeds.
3. **Measure the delta**, i.e. the diffusion ELBO's marginal value over the n-gram baseline, along the axes the design predicts it should help:
   - **Noise robustness (R2):** re-score partially-decrypted / noised inputs; the ELBO should degrade more gracefully than the n-gram score (this is the argument for the whole backbone).
   - **Dense gradients (R3):** does expected-embedding refinement improve map accuracy / SER over the n-gram-only result on the hard rungs (3–4)?
   - **Cross-head comparability on one calibrated scale (R5 / 5.6):** the diffusion bits-per-plaintext-char + complexity penalty is the scale the paper actually reports.
4. **Two-tier operation as designed (§7.4):** keep the n-gram DP as the inner-loop restart engine; use the diffusion ELBO to re-rank shortlists and refine. The early-phase work built the inner tier; the swap adds the outer tier.

Because the harness, metrics, scoring scale, and calibration hook are all in place, this is measurement, not construction.

---

## 10. What this establishes — and what it does not

**Establishes now (n-gram scoring):**
- The heads are correct: they recover known maps on synthetic ground truth.
- The machinery clears the field's solved-cipher anchors (task 6.6), independent of the VMS.
- The core premise holds: trial-decipherment scores rank languages (Dhavare precedent).
- The diffusion integration is de-risked to a one-line swap: interface, 2N-slot frame, NaN guard, scoring scale, and calibration hook are all built and smoke-tested.

**Does *not* establish (needs the real frozen backbone):**
- The diffusion ELBO's specific noise-robustness advantage (R2) — measurable only once Phase 2 hardening exists.
- Whether expected-embedding inputs are "too smooth to discriminate sharp maps" (§8) — probed on the small AR LM, but the definitive answer needs the trained backbone; the straight-through fallback is the hedge.
- The final reported SER / language-recovery numbers — those are on the diffusion evaluator's calibrated scale; n-gram results are a lower-bound baseline and a de-risking instrument, not the paper's headline figures.
- Cross-head comparability on the *diffusion* calibrated scale (5.6) — the machinery is testable now, the numbers are not final until the swap.

Stated plainly, in the spirit of the inverse note's §7: this track reparameterizes and de-risks the hard cryptanalysis problem and removes it from the critical path — it does not pre-solve it.

---

## 11. Dependencies to pull forward from Phase 0

For the concurrent track to start at Phase 1, these Phase-0 items must be done early (none require the diffusion backbone; all are already parallelizable WS0/WS1 work):

- **0.1 — vocab freeze** (defines `A`, resolves the 23-vs-22 / K-W discrepancy). Hard prerequisite for every head.
- **0.3 — normalization pipeline** (whitespace strip) — produces the corpora the n-gram LMs and synthetic ground truth are built from.
- **0.7 — cipher generators + controls, pinned** (`naibbe_v2.py` @ `df3d074`; `voynpy.pseudo_vms` latest, pinned config; `voynichesque.py`). Currently **P1**; for this track it is effectively **P0** — no synthetic ground truth without it.
- **CH.0 — per-language char n-gram LMs** (new task, §8). Small, fast, backbone-independent.
- **G0 artifact — random-init backbone object** (already required by G0) — enables the interface smoke tests (CH.4 / task 5.1).

Nothing here touches the Phase-1 training run. The two hardest problems in the project — training the backbone and cracking the verbose-homophonic heads — proceed in parallel, and meet at a scripted evaluator swap.

---

*Ordering rationale: the design's §7.4 two-tier scorer (n-gram DP inner loop, diffusion ELBO outer decision) means the cipher-head inner loop is separable from the backbone. Writing every head against a frozen `Evaluator` interface makes the separation explicit and the eventual integration mechanical. The gates are unchanged; this document only moves WS6 off the critical path and alongside WS4's Phase 1.*
