# Cipher-head early track (CH) — status

Status record for the concurrent cipher-head track defined in
[Prototyping and Testing the Cipher Heads During Backbone Training](../reference_docs/Prototyping%20and%20Testing%20the%20Cipher%20Heads%20During%20Backbone%20Training.md),
running alongside Phase-A pretraining (task 1.4). Started 2026-08-18.

Code lives in `diff_voyn/heads/`; scripts `scripts/train_ngram_lms.py` (CH.0)
and `scripts/cipher_head_harness.py` (CH.2/3/5/9 grid). Tests:
`tests/test_heads.py` (CI-safe, synthetic-only). All CPU work in this track
runs **single-threaded** (`torch.set_num_threads(1)`) — the Phase-A trainers
own the cores, and thread-pool contention makes small-tensor torch ~24×
slower otherwise.

## CH.0 — per-language char n-gram LMs ✅ (2026-08-18)

`diff_voyn/heads/ngram.py`. Interpolated Witten–Bell, orders 1–5, trained on
the **train** side of splits v1 only (held-out is the calibration set for
every evaluator; it also supplies the synthetic-grid plaintexts, so map
recovery is never "the LM memorized this text"). Dense float32 tables,
persisted `DATA_ROOT/ngram_lms/v1/<lang>.npz` (+ `summary.json`), versions
verified at load.

Held-out bits/char (acceptance: sane, monotone in order — met):

| order | latin | italian | german |
|---|---|---|---|
| 1 | 4.03 | 4.01 | 4.09 |
| 3 | 3.34 | 3.15 | 3.00 |
| 5 (pentagram) | **2.955** | **2.842** | **2.405** |

(Whitespace-stripped streams, hence higher than word-segmented literature
numbers. German lowest — its calibration offset matters, see CH.2.)

## CH.1 — `Evaluator` interface + `NgramEvaluator` ✅

`diff_voyn/heads/evaluator.py`. The frozen contract every head is written
against: `score_fixed` (chained-expectation soft score, any order; exact
`score_hard` for discrete moves), `score_segmental` (trigram-state
semi-Markov forward DP; start state = stationary bigram, correct for
mid-stream excerpts; generalized per-token branch lists for rung 3),
`as_embedding_frame` (diffusion-only). Verified against brute-force
enumeration on tiny alphabets (`tests/test_heads.py`). The
`logaddexp(−∞,−∞)` guard: infeasible branches are *skipped before* any
log-space blend; tested at the degenerate weights.

**Calibration hook** (single-sourced, §6 of the plan):
`EvaluatorBase.calibrated_bits_per_char` — per-language additive offsets in
exactly one place. The n-gram evaluator's offsets are −(held-out bits/char),
so rankings compare *excess* bits; without this German wins every
cross-language ranking by LM entropy alone (R1 in n-gram form).

## CH.4 — task 5.1 pulled forward: frame + random-init DiffusionEvaluator ✅

`diff_voyn/heads/frame.py`, `diff_voyn/heads/diffusion_eval.py`;
`Backbone.forward_soft` added (mixture-input path, expected embeddings — the
id path is untouched; one-hot equivalence tested). 2N-slot frame with
probability-space NULL blend (w=0/1 corners tested finite), safe log, exact
straight-through fallback. `DiffusionEvaluator` scores frames with a
soft-target Rao-Blackwellized NELBO under CRN masking (same seed across
language conditions), differentiable w.r.t. the frame; backbone frozen.

Acceptance met against **random-init** weights: gradients reach a toy head's
parameters through the embedding table; NaN smoke test passes. Two real traps
were caught and fixed by these tests: (a) SUBS makes the MASK logit −∞ and a
naive soft-target CE produces `0·−∞ = NaN` — and a `torch.where` guard still
NaNs in *backward*; the fix masks `logq` before the product; (b) the
straight-through estimator must group `hard + (frame − frame.detach())` —
the other association is not exactly one-hot in float32.

## CH.9 — language-discrimination probe: first result ✅ (rung 1)

Trial-decipherment ranking with calibrated offsets on rung-1 synthetics
(L=200, 2 trials × 3 languages, common solver seed across language
conditions): **true language ranked first 6/6**. The framework's core
premise (decipherment score ranks languages — the Dhavare precedent) holds
under n-gram scoring at rung 1. Full grid + homophonic probe pending.

## CH.2 — shared harness ✅ / CH.9 probe wired

`diff_voyn/heads/synth.py` (generators + metrics), `harness.py` (grid,
aggregates, JSON out), `scripts/cipher_head_harness.py`. Plaintexts sampled
from held-out docs; generators emit full ground truth. Metrics: SER,
occurrence-weighted letter-map accuracy, evals-per-solve + wall-clock (R6),
and the CH.9 trial-decipherment language ranking through the calibration
hook (same solver seed across language conditions).

## CH.3 — rung 1, 1:1 Sinkhorn head ✅ code, acceptance run pending

`diff_voyn/heads/rung1_sinkhorn.py`. Gumbel–Sinkhorn gradient phase
(trigram soft score) → Hungarian projection → exhaustive 2-swap hill-climb +
iterated local search (3-transposition kicks) under exact pentagram, plus a
frequency-rank-init restart. Preliminary (5 trials/cell, restarts=3):

| | L=100 | L=200 | L=400 |
|---|---|---|---|
| italian SER | 0.010 | 0.000 | 0.000 |
| latin SER | 0.000 | 0.049 (1 stuck trial) | 0.000 |
| german SER | 0.032 | 0.000 | 0.000 |

≥200 chars is near-perfect (= task 5.2 criterion under n-gram scoring);
~10–16 s/solve single-threaded.

## CH.5 — rung 2, unigram homophonic head — in progress

`diff_voyn/heads/rung2_homophonic.py`. Gradient phase (soft assignment +
row-entropy annealing + letter-frequency KL) feeding a pentagram SA over
single-symbol reassignments (+ pair swaps) with greedy polish — the
classical workhorse.

**Measured finding (important): the pure-LM discrete objective has
degenerate optima.** On Zodiac-408-class synthetics (408 chars, 54 symbols)
sufficiently long SA finds maps that *outscore the true map by >150 nats*
while decoding to hyper-likely repetitive junk (SER ≈ 0.9). The fix is the
plan's own prescription applied to the *discrete* objective too, not just
the gradient phase: `_objective = pentagram LL − λ·L·KL(decoded letter freq
‖ LM unigram prior)`, λ=1 nat/char of KL. With the penalty, no found map
beats the truth, and the true basin polishes to SER 0–1.5 % (≤ the 1.9 %
anchor threshold). Remaining constraint is the basin-hit rate (~1/12–1/36
per 100k-step restart on latin) — i.e., restart budget, as the plan
predicted ("expect to need the restarts"). `solve_parallel` fans restarts
over forked nice(10) single-thread workers (~linear wall-clock scaling; the
Phase-A trainers keep priority).

**Pre-diffusion baseline, Zodiac-408-class synthetics** (408 chars, 54
symbols, 180 restarts x 100k SA steps ~ 17.4M evals ~ 290 s/solve, 6
workers; 2 instances/language):

| | SER (t=0) | SER (t=1) |
|---|---|---|
| german | **0.000** | **0.000** |
| latin | **0.000** | 0.659 (basin never hit) |
| italian | 0.061 (in-basin, polish-limited) | **0.020** |

4/6 cells at or below ~2 % SER. Known remaining levers, in expected-yield
order: pair-swap moves in the final polish (the 6.1 % instance is inside the
true basin), larger restart budgets for the hit-rate tail, >=6-gram scoring.
This is the honest n-gram baseline the plan asks for; closing the residual
is part of what the diffusion evaluator is later asked to do (and the real
anchors still need their transcriptions fetched).

Literature anchors (Zodiac-408 / Borg / BnF fr2988 transcriptions) are NOT
yet fetched — external data acquisition is an open task (extend
`scripts/fetch_external.py`; anchor results reported as the pre-diffusion
n-gram baseline per the plan).

## CH.6 — rung 3, Naibbe head — block-Sinkhorn head at threshold ✅(prelim)

`diff_voyn/heads/naibbe_parse.py`: the two inverse-note stubs closed
*exactly* — vocabularies (137 unigram / 138 prefix / 138 suffix glyph types)
and `parse_token` built from the pinned tables (structural prior = the
published apparatus; the glyph→letter key stays unknown). On generated
streams every token parses and truth maps decode every aligned segment
(property tested in CI). ~42 % of tokens are parse-ambiguous
(uni + spurious bigram splits), marginalized exactly by the generalized
semi-Markov DP.

Two parameterizations were built and compared (`rung3_naibbe.py`), on
10k-char held-out Italian enciphered with the pinned generator:

1. **v1 — free categorical maps** (the inverse-note parameterization:
   independent softmax rows per glyph type). Plateaus at **~20 % map
   accuracy** under both SGD and exact EM (E-step via the `q·∂LL/∂q`
   autograd identity, deterministic annealing, prior-informed init). The
   engineered homophony destroys type-level identifiability, as the inverse
   note warned.
2. **v2 — block-Sinkhorn (`NaibbeBlockHead`)**: the published apparatus
   fixes which codes live in which (state × table) block and the deck
   weights; each block's key is a 23×23 **bijection**, parameterized
   ALICE-style with per-block Gumbel–Sinkhorn matrices (18 blocks), emission
   likelihood `b(c) = Σ_cells w_table · P_block[row, c]` consumed directly
   by the semi-Markov DP. Result: **95.2 % overall code-map accuracy**
   (unigram 97.1 / prefix 95.7 / suffix 92.8) with 2 restarts × 250 steps,
   ~6.5 min single-threaded — at the ≥95 % acceptance threshold of 5.4.
   The bijection prior is the identifiability the free maps lack.

Acceptance pairs (10k chars, 3 restarts, 350 steps, ~12-14 min/solve):
italian **95.2 %** (first instance) / **92.0 %** (second), latin **84.8 %** —
at/near the >=95 % bar but not consistently over it yet. Next levers, in
order: within-block discrete swap polish (found scores still trail truth
slightly, e.g. -40010 vs -39889 — the residual is search, not objective),
larger restart/step budgets, VMS-scale DP vectorization. Restart budget so
far: 2-3 restarts of 250-350 chunked-SGD steps each.

## CH.8 — rung 4, arithmetic sum-to-target head ✅ acceptance met (2026-08-19)

`diff_voyn/heads/rung4_arithmetic.py` + char-lattice semi-Markov DP
(`NgramEvaluator.score_lattice` / `viterbi_lattice`, verified against
brute-force enumeration and the token-level segmental DP);
`gen_arithmetic` ground truth on the Phase-0-pinned per-language
`pseudo_vms` tables. Full design record with all measured numbers:
**[rung4_arithmetic_design.md](rung4_arithmetic_design.md)** (the §10
design note the plan asks for). Highlights:

- Identifiability chain, each link measured: canonical char order recovered
  from ciphertext by exact LOP + a factorized boundary-model tie-break (raw
  LOP is *exactly* degenerate; cyclic rotations killed by the min-token-
  length constraint) → admissibility prunes ~60% of the segment lattice →
  order-derived `v` (split 2) reproduces the true char values exactly →
  `u` becomes a Gumbel–Sinkhorn assignment between integer segment sums
  and letters over the lattice DP (a scalar Gaussian-kernel `u` cannot
  travel and collapses the value scale — measured, documented).
- True-key Viterbi: SER 0.7%, boundary recall 99.7% (scorer ~lossless).
- **Measured reversal of the rung-2 KL defense**: the emission-posterior
  frequency proxy penalizes the TRUE key (~320 nats/300 letters) and flips
  the objective ordering; with it off the truth outscores every found key
  and polish-from-truth stays exactly at the truth. The assignment's
  injectivity is the structural defense; `freq_penalty_weight` defaults 0.
- **Acceptance 5.5 (language recovery better than family-random): met.**
  Probe (2 instances/language × 3 conditions, common seed, calibrated
  bits/char): language top-1 **4/6**, family top-1 **5/6** (random: 2/6 and
  ~3.3/6); both misses within 0.02 bits/char; latin t0 recovered the
  complete key from scratch (u map acc 1.000, SER 0.003). Best config
  (head defaults): 400 Sinkhorn steps, 768-char chunks, 2 restarts,
  ~15–25 min/solve single-threaded. `data/cipher_heads/rung4_probe.json`.

## Deliberately deferred

- `SmallARLMEvaluator` (CH.7), anchor-data fetch, DP vectorization for
  VMS-scale (38k tokens ≈ 160k chars) rung-3/4 runs.
- Full-grid acceptance runs (50 ciphers/cell) — current numbers are 2–5
  trials/cell smoke levels.
- Rung-4 next levers (design note §6): more restarts for the 0.02-bit
  near-misses, exact-EM interleave on the assignment, pair-swap polish.

## Side study — robustness of the n-gram judges vs the diffusion judges (2026-08-22)

`scripts/ngram_robustness.py` scores the CH.0 LMs (orders 1/2/3/5) on the
*same* noised windows as the task-2.6 robustness curves and ranks languages
with both judges. Record: **[ngram_judge_robustness.md](ngram_judge_robustness.md)**.
Short version: under a wrong key the n-gram cost never saturates (it climbs
through the shuffled-letters ceiling, 1.3–2.7× the clean→shuffled gap at a
fully wrong key) while the diffusion judge saturates at ~0.55 of its gap —
the n-gram objective is the better far-from-key search signal, the ELBO the
better near-key judge; under parse/transcription errors the n-gram judges
spend about half the share of their range the diffusion judge does; and as
*language* judges the n-gram LMs drift to "German" under any corruption
(Italian lost at 5–10 % wrong key, Latin at 20–30 %, German never), a
severity-dependent bias no static offset removes, whereas the Phase-C judge's
call is flat to a 50 %-wrong key — a property of the Phase-B curriculum, not
the architecture (the clean Phase-A model fails as fast as the n-grams).
