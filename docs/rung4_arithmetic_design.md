# Rung 4 — arithmetic sum-to-target head: design note

The design doc (§10) flags the arithmetic cipher head as warranting its own
design note; the prototyping plan (CH.8) schedules writing it now, against
n-gram scoring, well before the diffusion backbone is ready. This is that
note. Code: `diff_voyn/heads/rung4_arithmetic.py`; lattice DP in
`NgramEvaluator.score_lattice` / `viterbi_lattice`
(`diff_voyn/heads/evaluator.py`); ground truth via
`diff_voyn.heads.synth.gen_arithmetic` on the Phase-0-pinned per-language
`PseudoVmsEncoder` tables (`data/ciphers/pseudo_vms_<lang>.csv`, tuned
doubling strengths in `acceptance_stats.json`).

## 1. The cipher, and what the head is allowed to know

`voynpy.pseudo_vms` (Boxer, pinned @ `e324bee`): each plaintext letter
carries an integer value (our config: frozen 25-letter vocab, a=3 … z=28,
j's slot unused); a cipher token is 2–6 characters from a 16-char alphabet
with values 0–9 → 0..9, A–D → 10..13, E → −1, F → −2, chosen so the char
values **sum to the letter's value** (~500 Zipf-weighted homophones per
letter). Tokens are written in a canonical order — negatives first
high-to-low, then positives high-to-low — i.e. every token is sorted by a
fixed global order on the 16 chars. Consecutive identical plaintext letters
repeat the same token with a tuned probability (~0.92% output doubling rate,
matching the VMS). Our pipeline strips all whitespace (design §2), so the
head receives an **unsegmented char stream**; token boundaries are latent.

Following the Naibbe convention (rung 3), the *published apparatus* is
structural prior, the *key* is learned:

- **Structural prior**: the sum-to-target rule; the 2–6 length range and the
  VMS-calibrated length distribution {2: .10, 3: .22, 4: .26, 5: .26,
  6: .16}; the existence of a canonical within-token order; the doubling
  mechanism's existence.
- **Key (learned)**: the 16 char values `v`, the 25 letter values `u`, and
  the global char order itself (the cipher chars reach the head under a
  random symbol permutation — `gen_arithmetic` shuffles ids so no code path
  can leak the natural hex order). A segment decodes to the letter whose
  value equals the segment's sum. Per design §10 there are no dummy symbols
  and no interval tracking (the Ryabko–Fionov interval-decoder assumption is
  superseded).

## 2. Identifiability chain

The head's leverage comes from a chain of structural deductions, each
measured before being relied on:

**(a) The canonical order is recoverable from ciphertext alone.** Adjacent
char pairs within a token always respect the global order; only
cross-boundary pairs can violate it (~21% of adjacencies at mean token
length ~4.2). The true order is therefore an optimum of the **linear
ordering problem** on the adjacency-count matrix, solved *exactly* by a
2^16-subset DP (`_lop_order`). Two degeneracies had to be broken, both
found empirically:

1. *Boundary-only pairs.* Boundary pairs are not symmetric noise: token-final
   chars are low positives, token-initial chars negatives / high positives.
   A pair that never co-occurs within a token (e.g. E/0 — a token cannot
   contain a negative after positives, and no token both ends in 0 and is
   followed by nothing systematic) is oriented purely by boundary counts,
   and moving such a char through the order is *exactly* LOP-cost-neutral
   (measured: the true and a scrambled order tie at 833.0 on a 1428-char
   instance). Tie-break: `_boundary_score` — descents under a candidate
   order are forced boundaries; fit P_end × P_start on them and score the
   descent mass under the factorization. Wrong orders must explain
   systematic within-token pairs as boundaries, which the factorization
   cannot do cheaply.
2. *Cyclic rotations.* A rotation of the true order reinterprets the
   boundary wrap-around and fits the factorized model well — but it forces
   two adjacent descents somewhere in a long stream, i.e. a length-1 token,
   impossible under min length 2. Candidates with adjacent forced
   boundaries (or more forced boundaries than L/2 tokens allow) score −inf.

`infer_char_orders` samples the LOP optimum class with Gumbel-noised exact
solves, scores candidates, and returns the top-k distinct orders; the solver
carries them as restart seeds and lets the end objective select. Measured on
300-letter instances (2/language): the true order is top-1 in most
instances and in the candidate pool otherwise.

**(b) The order prunes the segment lattice and near-determines `v`.** Any
descent forces a token boundary; a segment is admissible iff
non-descending (`admissible_mask`, ~40% of the (position, length) lattice
survives). Under the scheme's value convention (negatives −1..−s prefix,
positives descending to 0, consecutive integers), the order determines `v`
up to the split parameter s: `v = (−1..−s, 15−s, …, 0)` — for the true
split s=2 this reproduces the upstream values **exactly**
(`order_derived_values`; splits {2, 1, 3} are carried as restart
alternatives, and the discrete polish can leave the convention with ±1
moves, preserving the plan's "jointly infer" requirement).

**(c) Given integer `v`, `u` is an assignment problem, not a regression.**
Every admissible segment's sum lands on a small integer grid (~50 values);
the true `u` is an injective map letters → grid values. A scalar-`u`
Gaussian-kernel gradient phase cannot travel the ~8 value units a frequency
init is typically off by (measured — it plateaued at ~10% map accuracy and,
worse, collapsed the value scale when `v` was free). The working
parameterization is a **Gumbel–Sinkhorn partial permutation** over (grid
values × letters, with dummy columns absorbing unused values), ALICE-style
as in rungs 1/3: emissions index the Sinkhorn matrix rows, the lattice DP
scores them, Hungarian projects at the end (`_gradient_phase`,
`_project_u`). Init biases the logits with a mass-quantile ↔
frequency-quantile affinity.

## 3. Scorer: the char-lattice semi-Markov DP

`NgramEvaluator.score_lattice` generalizes the (verified) token-level
segmental DP to a lattice over char positions: `alpha[i]` = forward mass
after consuming i chars; each admissible segment (i, n) emits one letter
with weight `log b(i, n, ·) = log P(len=n) + log P_assign[sum(i, n), ·]`
(sums via cumsum differences, so the whole emission tensor is a gather).
State is the last 1 or 2 letters (order 2 for the inner loop — ~2.5×
cheaper; order 3 for final scoring), start state stationary as in
`score_segmental`. `viterbi_lattice` is the max-product version with
traceback, used for decode/SER. Verified in CI against (i) brute-force
enumeration over segmentations in linear-space float64 numpy and (ii) the
token-level `score_segmental` on single-admissible-segmentation instances
(`tests/test_heads.py`).

Numerical traps (all hit during development, now guarded + tested):

- Inadmissible slots use a −1e30 sentinel, NOT −inf: an all-−inf
  `logsumexp` NaNs in *backward* (the rung-3 lesson recurs).
- A mid-stream **chunk** need not begin or end on a token boundary. An
  edge-misaligned chunk has a dead lattice (LL ≈ −1e30) whose gradient
  blows the parameters up, after which `(V−u)²` overflows float32 and the
  run is NaN. Fix: `start_window`/`end_window` marginalize the chunk edges
  over a max-token-length window, plus a skip-guard on sentinel-magnitude
  LLs and non-finite grads.
- The doubling mechanism needs no special handling in the DP (identical
  adjacent segments decode to the same letter automatically; the LM scores
  the doubled letter), but it is a small unexploited segmentation hint.

Degenerate-optimum defense — a measured *reversal* of the rung-2 lesson.
The rung-2 frequency-KL penalty was ported first (with decoded frequency
approximated by the emission-posterior average, since a Viterbi decode per
polish move would dominate cost), and it **mis-targets the true key**: the
average over all admissible — mostly spurious, overlapping — segments
diverges from the LM prior even under the truth, charging it ~320 nats on a
300-letter instance, after which found keys "outscore" the truth. With the
penalty off, the truth outscores every found key (search-side gap only) and
gradient-phase map accuracy improved 0.42 → 0.66. The structural
injectivity of the Sinkhorn/Hungarian `u` assignment is itself the defense
rung 2's free many-to-one maps lacked; `freq_penalty_weight` defaults to 0
(knob retained for exact-decoded-frequency experiments).

## 4. Search pipeline

Per restart (combos of order-candidate × split, then jittered repeats):
order-derived integer `v` → Sinkhorn assignment phase for `u` (chunked
bigram-lattice SGD, τ annealed, ~0.4 s/step at 384 chars) → Hungarian → 
integer snap → greedy polish over `u_ℓ ± 1`, `v_c ± 1`, and near-value
`u` swaps under the penalized bigram objective → final trigram rescore +
Viterbi decode. Restarts are selected by the penalized score.

## 5. Results (pre-diffusion n-gram baseline)

Scorer validation (italian, 300 letters / 1428 chars, pinned tables):
order inference exact; true segmentation fully admissible; order-derived
`v` (split 2) equals the true char values exactly; **true-key Viterbi
decode: SER 0.7%, boundary recall 99.7%** — the scorer side of the head is
essentially lossless, so remaining error is search, not model.

Objective validation: with the frequency penalty off (§3), the greedy
polish started FROM the true key stays exactly at the truth (447 move
evaluations, no improving move found) — the true key is a local optimum of
the search objective, so residual map error is attributable to search
budget, not scoring.

Search (single instance above, single restart, config sweep):

| config (steps / chunk chars) | u map acc (occ-weighted) | SER | score (truth −839.0) |
|---|---|---|---|
| 400 / 384 | 0.660 | 0.35 | −844.4 |
| 600 / 384 | 0.563 | 0.44 | −847.9 |
| **400 / 768** | **0.773** | **0.27** | **−840.7** |
| 600 / 768, lr 0.2 | 0.540 | 0.40 | −844.7 |

The best config lands 1.7 nats below the truth with the frequent letters
essentially all correct; head defaults are set to it. Seed variance across
configs is large (single-seed sweep) — restarts are the lever, as with
rungs 2/3.

**Language-recovery probe (= acceptance 5.5)** — 6 instances (2 per
language, 300 letters, pinned per-language tables), each solved under all 3
language conditions with a common solver seed (CRN convention), ranked by
calibrated bits/char of the final trigram lattice LL over decoded length
(offsets = −held-out bits/char, the CH.2 hook):

| instance | ranking (calibrated bits/char) | true-cond SER | u map acc |
|---|---|---|---|
| latin t0 | **latin (1.03)** > italian (2.37) > german (2.81) | 0.003 | **1.000** |
| latin t1 | **latin (1.46)** > italian (2.17) > german (2.75) | 0.277 | 0.740 |
| italian t0 | **italian (1.73)** > latin (2.06) > german (2.85) | 0.537 | 0.573 |
| italian t1 | latin (1.743) > italian (1.765) > german (2.59) | 0.407 | 0.620 |
| german t0 | **german (1.75)** > latin (1.84) > italian (2.19) | 0.197 | 0.857 |
| german t1 | latin (2.00) > german (2.02) > italian (2.22) | 0.333 | 0.757 |

**Language top-1: 4/6; family top-1: 5/6** — better than language-random
(2/6) and family-random (~3.3/6): **acceptance 5.5 met at this budget** (2
restarts × 400 steps + polish, ~15–25 min/cell single-threaded). Both
misses are within 0.02 bits/char of the true language (and one is
latin-over-italian, same family); latin t0 recovered the complete key from
scratch. Full rows: `data/cipher_heads/rung4_probe.json`.

Cost: gradient step ~0.6 s at 768 chars (bigram lattice, single-threaded,
`torch.set_num_threads(1)` — the Phase-A trainers own the cores); a full
solve (2 restarts + polish) ~10–15 min on a 300-letter instance.

## 6. Known limitations / next levers

- The `u` assignment search is the binding constraint (see results); levers
  in expected-yield order: more restarts (the probe's two near-miss
  rankings are 0.02 bits/char gaps — better in-condition solves separate
  them), longer streams, an exact-EM interleave on the assignment (the
  rung-3 `q·∂LL/∂q` identity applies verbatim), and pair-swap enrichment
  of the polish move set.
- Order inference assumes enough stream (LOP + boundary score were
  validated at ≥300 letters); very short ciphers (< ~100 tokens) will need
  the candidate pool widened.
- The acceptance criterion (5.5) is language recovery better than
  family-random, not map recovery — the probe result below is the
  deliverable; map accuracy is diagnostic.
- VMS-scale (38k tokens ≈ 160k chars) needs the position loop vectorized or
  chunk-parallelized before Phase 5 runs on real Currier streams.
