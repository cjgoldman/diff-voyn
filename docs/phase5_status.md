# Phase 5 — Cipher-head integration (frozen evaluator, difficulty ladder): status

Status record for Phase 5 of the [task breakdown](../reference_docs/Diffusion%20Model%20Training%20-%20Task%20Breakdown.md)
(design §7.4, §8; prototyping doc §9 "the evaluator swap and delta-measurement
protocol"). Started 2026-08-22 after Gate G4 (`docs/phase4_status.md`). The
heads themselves were built and validated against the frozen n-gram evaluator
in the CH early track (`docs/cipher_heads_status.md`); Phase 5 is the swap to
the frozen diffusion evaluator, the two-tier search it enables, the delta it
buys, and the uniform cross-head scale. Code: `diff_voyn/heads/diffusion_eval.py`
(real-weights evaluator), `two_tier.py` (shortlist, paired scoring, selection),
`scale.py` (uniform scale), `ladder.py` (pools, soft refinement); scripts
`phase5_freeze.py`, `rung{1,2,3,4}_diffusion.py`, `crosshead_scale.py`,
`g5_check.py`; artifacts under `DATA_ROOT/analysis/phase5/`; ClearML tag `g5`.

## 5.1 — Evaluator freeze and head interface on the real weights (DONE)

**The frozen evaluator** is the Gate-G4 joint Phase-C 85M EMA checkpoint,
`DATA_ROOT/runs/phase_c-85m-seed0/ckpt_final.pt` (step 2000, EMA 0.9975,
85,006,080 parameters, sha256 `e2cfb3c6…e52df3`), loaded through
`DiffusionEvaluator.from_checkpoint` (`load_backbone` ignores the LID-head
keys), every parameter `requires_grad=False`, eval mode; calibration
`v3-phase_c-ro` (report-only; offsets measured +0.138 / +0.013 / +0.205
bits/char are the margin uncertainty, not subtracted). The 25M sibling
(`phase_c-25m-seed0`, sha256 `305b47a1…2864a0`) is recorded for restart-heavy
loops but every Phase-5 number below is the 85M. `scripts/phase5_freeze.py`
→ `analysis/phase5/evaluator_freeze.json` (all checks PASS):

| check | result |
|---|---|
| `forward_soft` on one-hot input == id path | max \|Δlogit\| 0 |
| gradients reach a toy head's parameters through the frozen backbone; backbone grads absent | \|∂/∂logits\| 1.6, \|∂/∂w\| 3.1; all backbone `.grad is None` |
| `logaddexp(−∞,−∞)` smoke test: NULL-blend at w = 0 / 1 / 0.5 and a hard 0/1 frame | all finite (score and gradient) |
| masking is a pure function of the seed (CRN) | same seed identical; other seed / language differ |
| soft frame path == Phase-3 metrology estimator on one-hot input, same seed | latin 2.2223 vs 2.2219, italian 2.6707 vs 2.6705, german 1.8599 vs 1.8603 bits/char (bf16 noise); `paired_bits` / `score_ids` identical to 4 decimals |

Two measurements that set how the rest of the phase scores:

- **The 2N NULL frame costs ~0.5 bits per plaintext character more than the
  plain stream** on the same held-out text (plain / frame letter-slots-only /
  frame all-slots, bits per plaintext char: latin 2.125 / 2.633 / 2.867,
  italian 2.683 / 3.138 / 3.427, german 1.849 / 2.367 / 2.616). The gap is
  language-symmetric (0.51 / 0.45 / 0.52), so it does not bias a
  within-frame language ranking, but it means frame scores and plain-stream
  scores are not the same number. Decision: the **uniform scale scores the
  collapsed hard decode on the plain path** (the G3/G4 scale, NULL slots
  dropped); the frame is the *soft-gradient* instrument (R3) only.
- Cost on one 3090 (bf16 autocast): forward 16 × 1024 chars 82 ms; one soft
  gradient step on a 700-slot frame at 8 strata 0.37 s; paired scoring of 8
  candidates × 3 conditions × 64 draws on 700 chars 5.8 s. Outer-tier
  decisions are cheap; the n-gram inner search stays the restart engine
  (design §7.4, R6).

### Two-tier protocol (all rungs)

`diff_voyn/heads/two_tier.py`. The inner n-gram search now hands up a
**shortlist** of distinct local optima (restart argmaxes, ILS optima, SA
restart optima, polished block maps, per-restart arithmetic keys) instead of
one winner. The outer tier scores every shortlist decode of an instance
under every language condition with **paired masking realizations**
(`paired_bits`: one t-draw and mask sequence shared by all candidates and
conditions of the instance — CRN inside the shortlist, where the decision is
a difference between near-identical texts; for a single row it reproduces
`per_window_nelbo_bits` exactly), picks the lowest own-condition NELBO,
**refines** the pick through the backbone (expected-embedding gradients on
the head's soft parameterization, `ladder.refine_assignment` for rungs 1–2,
`NaibbeBlockHead.refine_frame` on the 2N frame for rung 3; rung 4's
segmentation is n-gram-lattice-only, so it gets shortlist re-ranking only),
projects back to a hard key, and lets the refined key re-enter the
shortlist. Three decisions are recorded per (instance × hypothesis): the
n-gram winner, the diffusion winner, the oracle (best SER in the shortlist)
— the delta of prototyping doc §9 is measured per cell, not assumed.

### Uniform scale (5.1 / 5.6)

`diff_voyn/heads/scale.py`: per cell, **calibrated plaintext bits/char**
(frozen evaluator, single calibration hook) + **key bits** (description
length of the key class: rung 1 log2(25!) = 83.7; rung 2 n_symbols·log2 25;
rung 3 18·log2(23!) = 1274; rung 4 log2(16!) + 25·log2 26 = 162) +
**choice bits** (the cipher's encoding freedom given plaintext and key: 0
for a bijection; Σ log2(#homophones of the emitted letter) for homophonic;
deck-draw + unigram/bigram-state entropy per token for Naibbe, from the
published card weights; the Zipf-weighted homophone-draw entropy per letter
for the arithmetic cipher). Heads emit different plaintext lengths for the
same ciphertext, so the cross-head comparator is the **total per ciphertext
symbol**; the within-head language ranking uses the calibrated plaintext
bits/char (the penalty terms are identical across a head's language cells
and cancel). Both normalizations are reported side by side (§5.6).

## 5.2 — Rung 1, 1:1 substitution (DONE, acceptance PASSED)

`scripts/rung1_diffusion.py`: 3 languages × L ∈ {50, 100, 200, 400, 700}
× 20 held-out instances (300), each deciphered under every language
hypothesis by the CH.3 Sinkhorn head on the frozen n-gram evaluator
(restarts 3 + ILS; 12 forked workers, 29 min) returning an 8-deep shortlist
of distinct local optima; every shortlist decode of an instance (24) scored
by the frozen diffusion evaluator under all three conditions with paired
masks at budget 64, the ELBO pick refined (20 soft steps) and re-entered
(GPU 0, 68 min). `analysis/phase5/rung1_{solves,scores,report}.*`.

| cell | SER n-gram winner | SER diffusion pick | SER final | SER oracle | diffusion pick better / worse than n-gram | language recovery final (n-gram winners) |
|---|---|---|---|---|---|---|
| german L50 / L100 | 0.404 / 0.042 | 0.414 / 0.043 | 0.414 / 0.043 | 0.340 / 0.037 | 5% / 10% ; 0% / 5% | 60 / 95% (60 / 100%) |
| italian L50 / L100 | 0.314 / 0.041 | 0.349 / 0.043 | 0.349 / 0.043 | 0.286 / 0.038 | 10% / 30% ; 0% / 5% | 80 / 95% (90 / 95%) |
| latin L50 / L100 | 0.524 / 0.141 | 0.496 / 0.149 | 0.496 / 0.149 | 0.430 / 0.134 | 30% / 15% ; 5% / 10% | 70 / 80% (70 / 90%) |
| german L200 / 400 / 700 | 0.000 / 0.000 / 0.000 | same | same | same | 0% / 0% | 100 / 100 / 100% |
| italian L200 / 400 / 700 | 0.000 / 0.000 / 0.000 | same | same | same | 0% / 0% | 100 / 100 / 100% |
| latin L200 / 400 / 700 | 0.006 / 0.005 / 0.008 | 0.006 / 0.006 / 0.003 | 0.006 / 0.006 / 0.003 | 0.006 / 0.005 / 0.003 | 0%/0% ; 0%/5% ; 10%/0% | 100 / 100 / 95% (100 / 100 / 90%) |

**Acceptance: ≥ 200 chars mean SER 0.0016 (n-gram winner 0.0021), 98.9% of
instances solved (SER < 5%), calibrated language recovery 99.4% (ranking the
n-gram winners: 98.9%) — PASS.** Per-language solve success at ≥ 200 chars
(the search-fairness number the Phase-4 assessment asked for): latin 93% →
97% with the outer tier, italian 100%, german 100%; the residual Latin
misses are the known high-entropy Latin documents, and the outer tier
recovers one of three of them (latin L700: SER 0.008 → 0.003, two oracle
candidates the pentagram ranked second).

What the delta says at this rung: the n-gram inner search already solves
1:1 at ≥ 200 chars, so the outer tier can only act on the tail — and it
does, mildly and in the right direction (picks differ from the n-gram
winner in 0–10% of long instances, never worse at L ≥ 200 except one latin
L400 instance by one symbol). **Below 100 chars the ELBO is the worse
judge**: at L = 50 the diffusion pick is worse than the pentagram's in
10–30% of instances and better in 5–30%, and language recovery at L50/L100
is 3 pp below ranking the n-gram winners. This is the Phase-3 finding from
the other side: on 50 characters the ELBO margin between near-equal
candidate texts is a few bits of Monte-Carlo noise, while the pentagram is
exact. The shortlist is also simply not deep enough there (oracle SER at
L50 is 0.29–0.43): short ciphers need a deeper inner search, not a better
judge.

### R3 probe — do the frozen evaluator's dense gradients move a key? (NO at rung 1; finding)

Prototyping doc §9 asks whether expected-embedding refinement improves map
accuracy over the n-gram-found keys (R3), and design §8 names the risk
("expected embeddings too smooth to discriminate sharp maps", with
straight-through as the fallback). Measured on the short rung-1 cells where
the n-gram search fails (latin / italian, L = 50 / 100, 8 instances each,
starting from the diffusion shortlist pick; `scripts/r3_refinement_probe.py`,
`r3_probe2.py`, `r3_probe3.py`, artifacts `analysis/phase5/r3_*`):

| variant | what happens |
|---|---|
| soft refinement as run in the rung-1 pipeline (init 4, lr 0.1, 20 steps, fresh masks) | soft loss falls 4.0 → 2.95 nats/char at L50 (3.6 → 1.9 at L200) **by sharpening the Sinkhorn rows toward the same permutation** — the hard key never changes (0 of 300 instances) |
| aggressive soft / straight-through refinement (init 2, lr 0.3, 50 steps) and ELBO-driven Gumbel–Sinkhorn search from scratch (150 steps), soft or straight-through | destructive: every key changes, SER 0.77–1.00 from picks at 0.01–0.59, hard bits 6–7.7 (above shuffled text) — Adam on mask-sampled gradients at this step size random-walks the logits |
| conditioned (fixed mask realization, 32 strata, init 6, lr 0.05, 40 steps, best hard projection along the trajectory) | never changes a key, soft or straight-through |
| larger steps (init 6, lr 0.2, 80 steps, fixed or fresh masks, best-of-trajectory) | straight-through never changes a key; the soft variant changes keys in 12–62% of short instances for ELBO gains of 0.006–0.05 bits/char, SER better in 0–12% and worse in 0–12% — a wash (`r3_probe3.md`) |

The diagnostic that explains it (12 latin L700 instances, 32 strata,
`scratchpad/soft_diag.py`): the soft objective is *not* degenerate — a
50/50 blend of the true and a wrong key's frames scores 4.8 bits/char and
the uniform frame 7.0, against 2.1 for the true hard key and 3.8 for the
wrong hard key — but **softening the true key alone costs +0.5 / +2.7 /
+5.2 bits/char at row scale 6 / 4 / 2**, because the soft-target CE pays
the frame's own entropy on every masked slot. The gradient of any soft
parameterization is therefore dominated by "make the rows one-hot"; the
"which letter" component is an order of magnitude smaller and, once the
rows are sharp, the Hungarian projection is stable against it. This is the
§8 risk in its concrete form, and straight-through does not rescue it at
this rung: with a hard forward pass the per-slot gradient is the model's
log-probability ratio between letters on a 50–100-char text — Monte-Carlo
noise of the masking dominates it at any step size that can flip an
assignment.

**Decision:** the outer tier is **shortlist re-ranking** (paired ELBO
scoring of the inner search's distinct optima), which is safe (never
worse than the n-gram winner at ≥ 200 chars) and helps on the tail; soft
refinement stays in the pipeline as an instrumented no-op (it is recorded
per instance as changed / better / worse) and is not relied on. The
differentiable-heads property of the architecture — gradients reach head
parameters through the frozen backbone — is verified (5.1), but at rung 1
it buys nothing over the discrete search; rung 3 below tests the same
question on the frame. Shortlists deeper than 8 and a cheaper inner search
(the 25M sibling is not needed — the n-gram DP is the cheap tier) are the
levers for the short cells.

## 5.3 — Rung 2, unigram homophonic (Zodiac-408-class) — the degenerate-optimum finding

`scripts/rung2_diffusion.py`: 3 languages × 6 held-out instances of 408
chars under a 54-symbol frequency-allocated homophonic key; each instance
solved under every language hypothesis by the CH.5 head (`solve_parallel`,
120 SA restarts × 100k steps on the penalized pentagram objective, 12
workers, ~3 min per (instance × hypothesis), 2.8 h for the 54), returning
a 12-deep shortlist of distinct restart optima **plus** 6 optima of 24
restarts on the *unpenalized* LM objective — the hyper-likely degenerate
maps the CH.5 record found (maps that send most symbols to a few letters
and decode to repetitive junk that out-scores the true map under the raw
n-gram LL). These were put into the shortlist on purpose, to test the
outer tier against them.

**Finding — the frozen diffusion ELBO prefers the degenerate maps, and by
a wide margin.** On every one of the first scored instances the lowest
own-condition NELBO in the shortlist was a penalty-free map: e.g. latin t0,
a map using **2 letters** (44 of 54 symbols → one letter, 86% of the decode
one character) scores **1.395 bits/char** against **2.293** for the true
map (SER 0.002); latin t1 1.413 vs 2.386, latin t2 1.493 vs 2.325. A
repetitive string is extremely predictable text, and a density model —
any density model, n-gram or diffusion — rates it as such. The pure ELBO is
therefore not a usable selection rule for a homophonic head: the ELBO-only
outer tier returned SER 0.86–0.93 decipherments where the n-gram inner
search had SER 0.000–0.005 (recorded per instance as `elbo_pure`).

**Resolution — the uniform scale's choice-bits term is the principled form
of the CH.5 frequency penalty.** The description length of a ciphertext
under a homophonic hypothesis is plaintext bits + key bits + *choice bits*
(which of the letter's homophones the encipherer emitted, Σ log2 #homophones
of each decoded letter, `scale.choice_bits`). A map that pours 44 symbols
onto one letter pays log2 44 = 5.5 bits on 86% of the positions: ~4.8
bits/char of choice cost against 1.79 for the true map, so on the MDL total
the true map wins 4.09 to ~6.2 bits/char. The rung-2 outer tier selects by
**plaintext bits + choice bits per plaintext char** (`_mdl_annotate`; key
bits are identical across a head's candidates), which rejects every
degenerate map while leaving the within-language comparison of legitimate
maps to the ELBO. This is the same decision the CH.5 discrete search made
with its KL-to-unigram penalty, now derived from the scale that task 5.6
needs anyway rather than tuned (λ = 1 nat/char there). The pure-ELBO pick
is kept in every record so the rate at which it would have chosen a
degenerate map is reported (`elbo_pure_picked_degenerate_rate`) — it is
the number that says the choice term is load-bearing, not decorative. The
same term is constant across candidates for bijective keys (rung 1),
Naibbe block bijections (rung 3) and the arithmetic key (rung 4), so those
rungs' ELBO selections are unchanged.

Results table: see below once the score stage completes.

## 5.4 — Rung 3, Naibbe mixed unigram-bigram (DONE, acceptance PASSED)

`scripts/rung3_diffusion.py`: 3 languages × 4 held-out instances of 10,000
letters (k→c, w→uu pre-map; ~6.5k tokens, ~20k glyph characters) enciphered
by the pinned `naibbe_v2` generator; each instance solved under every
language hypothesis by the CH.6 block-Sinkhorn head (`NaibbeBlockHead`, 3
restarts × 350 chunked-SGD steps on the n-gram semi-Markov DP) **followed by
the new fixed-parse within-block swap polish** (`NaibbeBlockHead.polish`:
Viterbi parse → exhaustive 18 × C(23,2) two-row swaps scored by the exact
pentagram of the decoded letters, milliseconds a move → re-parse, ≤ 3
rounds). One instance (3 hypotheses) per forked worker: 2.06–2.12 h each,
12 in parallel. Shortlist = every restart's maps before and after polish
(6 candidates). Outer tier (GPU 0, 5.6 min per hypothesis, 3.4 h): every
shortlist decode scored under every condition with paired masks (budget
16 over ten 1024-char windows), ELBO pick, **frame refinement**
(`refine_frame`: block Sinkhorn logits from the pick, emissions collapsed
onto the 2N-slot frame, 40 steps on random 500-token windows) → polish →
re-entered; final by own-condition bits.

Two accuracy metrics are reported, and the acceptance is read on the
first: **occurrence-weighted glyph-type accuracy** (each type's letter
weighted by how often the type occurs in the ciphertext — what the decode
depends on) and the strict **unweighted code accuracy** over all 414 codes
of the 18 block bijections (the CH.6 number). A γ-table code (deck weight
4/52) for a rare letter occurs 0–2 times in 10k characters and cannot be
identified from the data; the unweighted metric counts those.

| language | n | weighted type acc n-gram → **final** | unweighted code acc n-gram → final (oracle) | instances ≥ 95% unweighted | decode SER final | refine changed / better / worse (after polish) | language recovery (margin, bits/char) |
|---|---|---|---|---|---|---|---|
| italian | 4 | 1.000 → **0.999** | 0.961 → 0.960 (0.961) | 75% | 0.002 | 100% / 0% / 75% | 100% (0.70–0.86) |
| latin | 4 | 0.994 → **0.998** | 0.931 → 0.931 (0.931) | 0% | 0.003 | 100% / 25% / 25% | 100% (0.06–1.25) |
| german | 4 | 0.995 → **0.998** | 0.945 → 0.947 (0.948) | 50% | 0.010 | 100% / 0% / 100% | 100% (1.42–2.09) |

**Acceptance: occurrence-weighted letter-map accuracy 0.998 (every instance
≥ 0.997; n-gram inner tier alone 0.996, lowest instance 0.980) on synthetic
Naibbe pairs of all three languages — PASS.** Restart budget: 3 restarts ×
350 steps + polish per hypothesis (~40 min single-threaded per hypothesis at
10k chars); the polish is the lever — it lifts a restart's maps by 0–1.9
code-accuracy points at 10k chars (and by 36 points on a deliberately short
30-step solve, `scratchpad/r3_smoke.py`), at ~3 min per restart. Unweighted
code accuracy is 0.946 (italian 0.952–0.969, german 0.908–0.990, latin
0.906–0.942): the residual is concentrated in rare codes and, for Latin, in
the γ tables of the least frequent letters.

What the outer tier did here: the ELBO pick among the six shortlist
candidates equals or beats the n-gram pick on weighted accuracy in 11 of 12
instances (latin t2 0.980 → 0.982, latin t1 0.9986 → 0.9999, german t1
0.988 → 0.997, german t3 0.992 → 0.994) and loses 0.07 pp once (italian
t2). Frame refinement + polish changed the maps in every instance and
produced the final pick in 3 of 12 (latin t2: 0.982 → **0.997**, the
largest single gain of the phase), was worse than the pick in 7 of 12 and
never selected there — the ELBO-ranked shortlist protects against it. The
soft frame path behaves as in the rung-1 probe: it moves keys (the block
logits are softer than a 25 × 25 permutation and the frame objective's
sharpening gradient scatters rare rows), and the subsequent discrete polish
is what turns the movement into an improvement when there is one.

Language recovery on Naibbe ciphertexts: 12/12 by calibrated own-condition
bits of the final decodes (the choice term is constant across hypotheses at
this rung, so the MDL ranking coincides). The Latin margins are the Phase-3
same-text near-ties (0.06–0.09 bits/char on three of four instances — the
Italian-hypothesis solve of a Latin cipher decodes mostly to the Latin text;
an Italian ciphertext solved under Latin still reaches 73% code accuracy);
the German margins are 1.4–2.1 bits/char.

## 5.5 — Rung 4, arithmetic sum-to-target (P1; DONE, acceptance PASSED)

`scripts/rung4_diffusion.py`: 3 languages × 3 held-out instances of 300
letters enciphered by the Phase-0-pinned per-language `pseudo_vms` tables
(~1.4k cipher characters), each solved under every language hypothesis by
the CH.8 head (`ArithmeticHead`, 2 restarts × 400 Sinkhorn steps + polish
over the char-lattice DP; 27 solves on 12 workers, 40–45 min each under
full CPU contention, 1.7 h). The rung-4 segmentation is n-gram-lattice-only,
so the outer tier is shortlist re-ranking of the hard Viterbi decodes (one
per restart) under every condition (budget 64, paired masks for equal-length
decodes); languages ranked by calibrated own-condition bits of the chosen
decode per hypothesis (the choice term is constant per decode length).

| language | n | SER n-gram pick / diffusion pick / oracle | u-map accuracy n-gram / diffusion / oracle | language top-1 (95% CI) | family top-1 | n-gram excess-bits ranking |
|---|---|---|---|---|---|---|
| latin | 3 | 0.446 / 0.446 / 0.418 | 0.522 / 0.522 / 0.591 | 67% (0.21–0.94) | 100% | 100% |
| italian | 3 | 0.661 / 0.660 / 0.660 | 0.346 / 0.374 / 0.374 | 100% (0.44–1.00) | 100% | 33% |
| german | 3 | 0.513 / 0.523 / 0.513 | 0.493 / 0.506 / 0.493 | 67% (0.21–0.94) | 67% | 0% |

**Acceptance: language top-1 7/9 = 77.8% (random 33%), family top-1 8/9 =
88.9% (random 56%) — better than family-random, PASS** (the CH.8 n-gram
probe: 4/6 and 5/6). The diffusion judge is the better judge of these
*partial* decipherments: ranking the same solves by the n-gram excess bits
recovers the language 4/9. Both misses are Romance/Germanic confusions at
margins of 0.08–0.35 bits/char (latin t1 → italian by 0.075; german t2 →
italian by 0.35) on decodes with SER 0.5–0.7.

The honest reading is that these are decipherments of 300-letter streams
by a search budget of two restarts run under contention — the decodes sit
at 3.4–4.4 bits/char under their own condition, 1–2 bits above clean text
and within a bit of the shuffled-text plateau the Phase-4 assessment
warned about. The language signal survives at that level (7/9), which is
what 5.5 asks, but the margins (median 0.08–0.24 bits/char) are of the
order of the calibration's systematic uncertainty (0.07–0.19) and the
rung-4 row of any VMS table must carry the full error bars. The levers are
the design note's (§6): restarts, longer streams, the exact-EM interleave.

## 5.6 — Uniform-scale cross-head comparison (DONE, acceptance PASSED)

`scripts/crosshead_scale.py`: the (cipher × language) table a VMS run will
contain — every head applied to the same ciphertext — built on 24 synthetic
ciphertexts (2 per language per cipher class: rung-1 L700, rung-2 408/54,
rung-3 10k-char streams cut to their first 2000 glyph characters for the
cross-application, rung-4 300-letter streams). The true head's cells come
from the rung artifacts above; the **cross-application** cells (126 solves,
12 workers, 20 min) apply the rung-1 head (wherever the stream has ≤ 25
symbols) and the rung-2 head (always) to the other classes' ciphertexts
under every language hypothesis; rung 3 / rung 4 are structurally inapplicable
off their own class (a Naibbe stream parses only as Naibbe; the arithmetic
head needs a 16-value stream). Every cell: calibrated plaintext bits/char
of the decode under the hypothesis + key bits + choice bits, totalled **per
ciphertext symbol** (`analysis/phase5/crosshead_{solves,scores,report}.*`).

Mean total description length per ciphertext symbol (rows: ciphertext
class; columns: head / language hypothesis; each entry averages the six
instances of that row):

| ciphertext | rung-1 head la / it / de | rung-2 head la / it / de | rung-3 head la / it / de | rung-4 head la / it / de |
|---|---|---|---|---|
| 1:1 | **3.16 / 3.39 / 2.56** | 3.83 / 3.64 / 3.48 | — | — |
| homophonic | (inapplicable: 54 > 25 symbols) | **5.74 / 5.72 / 5.32** | — | — |
| Naibbe | 3.18 / 3.30 / 3.11 | 3.41 / 3.50 / 3.47 | **1.64 / 1.69 / 1.58** | — |
| arithmetic | 3.59 / 3.59 / 3.57 | 5.10 / 3.74 / 5.51 | — | **2.50 / 2.53 / 2.48** |

and the calibrated plaintext bits/char alone (the within-head language
scale):

| ciphertext | rung-1 head | rung-2 head | rung-3 head | rung-4 head |
|---|---|---|---|---|
| 1:1 | 3.04 / 3.27 / 2.44 | 2.82 / 2.88 / 2.48 | — | — |
| homophonic | — | 3.17 / 3.18 / 2.69 | — | — |
| Naibbe | 3.14 / 3.26 / 3.07 | 2.61 / 2.76 / 2.38 | 3.32 / 3.45 / 3.16 | — |
| arithmetic | 3.53 / 3.53 / 3.51 | **1.73** / 2.78 / 1.94 | — | 4.01 / 3.87 / 3.98 |

Checks (`acceptance` in the report):

- **Same instrument.** On the 18 cells whose decode *is* the plaintext
  (SER < 5%), the decode's bits and the true plaintext's bits under the same
  condition and seed differ by at most 0.098 bits/char (the SER itself) —
  every head's plaintext term is the one frozen estimator, the G3/G4 scale.
- **The MDL total picks the true cipher class on every ciphertext**: 18/18
  contested instances (plus the 6 homophonic ones where only the rung-2
  head applies), minimum margin 0.16 bits per ciphertext symbol (1:1 german
  t1), typically 0.7–2.3. The table shows what the total buys: the rung-2
  head applied to an arithmetic stream emits a hyper-Latin decode at
  **1.73 plaintext bits/char** — lower than any real text in the inventory —
  and the rung-2 head on a Naibbe glyph stream reaches 2.4–2.8; by plaintext
  bits alone those junk cells would beat the true heads (4.0 and 3.3). The
  choice term (log2 of the homophone fan-in, paid per emitted symbol) turns
  them into 5.1 and 3.5 bits per ciphertext symbol against 2.5 and 1.6 for
  the true hypotheses. This is the homophonic-degeneracy finding of §5.3 in
  its cross-head form: a verbose head can make *any* stream look like
  language, and only the description length of the freedom it used stops
  it from winning.
- **Simplicity ordering.** On the six 1:1 ciphertexts the rung-2 head
  (25-symbol homophonic hypothesis) recovers the same text as rung 1
  (plaintext bits within 0.2 of each other, 2 instances where it found a
  slightly better-scoring decode), and the key-bits term ranks rung 1 first
  in 6/6 (e.g. latin t1: 2.19 vs 2.53 bits/symbol). Verbose heads cannot
  win by capacity alone (design §8 R5).
- **Language within the true head**: 20/24 on this subset by calibrated
  plaintext bits; the four misses are three rung-2 cells ranked by plaintext
  bits (which the §5.3 finding says is the wrong rule for a verbose head —
  the rung-2 report's MDL ranking has them 18/18) and the rung-4 latin t1
  miss of §5.5.

**Decision for Phase 6.** The (cipher × language) table is ranked on the
MDL total per ciphertext symbol (plaintext + key + choice bits); within a
cipher hypothesis the language ranking is by the same total (which reduces
to calibrated plaintext bits for bijective and block-bijective keys, and
is the only correct rule for homophonic keys). Plaintext bits/char, the
key and choice terms, the per-instance spread, the calibration-precision
flag and the replicate flip-rate are reported alongside every cell.

### R3, discrete form — the ELBO polish (finding, rung 2)

With 480 restarts the Latin basin is found on latin t4 but the inner
search stops at SER 0.037, and the reason is not search: **the found map
out-scores the true map under the penalized pentagram objective** (−792.8
vs −826.8 nats). At 408 characters the n-gram objective's optimum is not
the truth. The frozen ELBO, scored with paired masks, ranks them the other
way: true map 1.902 bits/char vs found 1.997 at budget 64 (MDL 3.81 vs
3.98). So the outer tier can improve on the inner objective's own optimum
— but, per the probe above, not through gradients. `ladder.elbo_polish`
does it discretely: every single-symbol reassignment of the current map
(optionally every symbol-pair swap) is decoded and scored in one paired
batch (same masks for all ~1.3k–2.6k candidates, budget 8, ≈15–30 s a
sweep on the 85M), plus the choice-bits term; the best move is taken and
the sweep repeated; the result is confirmed against the start map at
budget 64 and discarded if not better there. On latin t4 it reaches **SER
0.000** — the complete true key on every occurring symbol — in three
accepted moves (8 sweeps, 330 s with pair swaps; the accepted moves were
all single reassignments, so the pipeline runs single moves on fixed
masks, ≈2 min a hypothesis). This is the dense-judge result the two-tier
design was built for, in the form the instrument supports: the ELBO is a
better *judge* than the pentagram near the key, and a judge is used by
comparing discrete candidates, not by differentiating through the soft
input. It runs for every (instance × hypothesis) of the rung-2 suite below
(`elbopolish` variant; language ranking by MDL total of the finals).

### 5.3 — results (final, with the ELBO polish; acceptance: per-instance PASS, mean-SER WARN)

`analysis/phase5/rung2_{solves,scores,report}.*` (18 instances; latin t4 and
t5 re-solved at 480 restarts). Outer tier per (instance × hypothesis): MDL
selection among the shortlist → soft refinement (no-op, as at rung 1) →
n-gram pair-swap polish (`polish_pairs`, never changed a map: the SA's
greedy phase already exhausts single moves and the pair moves found nothing)
→ **ELBO polish** (`elbo_polish`, single moves, fixed masks, budget 8, ≤ 6
sweeps, confirmed at 64; 20–120 s) → final by MDL total.

| language | n | SER n-gram (penalized objective) | SER pure-ELBO pick | SER MDL pick | SER ELBO polish | **SER final** | oracle (shortlist) | instances ≤ 1.9% | pure-ELBO picked a degenerate map | language recovery: MDL total (plaintext bits / n-gram excess) |
|---|---|---|---|---|---|---|---|---|---|---|
| latin | 6 | 0.141 | 0.741 | 0.124 | 0.119 | **0.119** | 0.122 | 83% (5/6) | 83% | **100%** (67% / 83%) |
| italian | 6 | 0.005 | 0.005 | 0.005 | 0.003 | **0.003** | 0.005 | 100% | 0% | **100%** (33% / 100%) |
| german | 6 | 0.000 | 0.157 | 0.000 | 0.000 | **0.000** | 0.000 | 100% | 17% | **100%** (100% / 100%) |

All 18: **median SER 0.000, 17/18 instances ≤ 1.9% (94%), mean 0.0407**
(n-gram inner tier alone 0.0486 and 16/18; shortlist oracle 0.0423). The
ELBO polish was accepted on 8 of 18 true-hypothesis cells, improved the
decode on 6 (latin t1 0.005 → 0.002, t3 0.002 → 0, t4 0.037 → 0.010;
italian t3 0.017 → 0.007; german t3 0.002 → 0; latin t5 0.797 → 0.699) and
never worsened one; on italian and german it ends *below the shortlist
oracle* — the judge improves on the inner objective's optimum, which no
amount of re-ranking can do. The pure-ELBO pick would have returned a
degenerate map on 6 of 18 instances (SER 0.30 mean); the MDL selection on
none.

The one failure is latin t5: the inner search never reaches its basin —
120 restarts, then 480 (SER 0.797, oracle 0.684; the true map is 0.35–0.44
bits/char below every found map under the ELBO, so the judge is not the
problem). This is the Latin basin-hit tail the CH.5 record measured
(1/12–1/36 per restart on typical Latin; effectively zero on this window),
i.e. the search-fairness risk of the Phase-4 assessment made concrete on
one instance: under a larger budget or a better inner search it would
resolve, and its language is still ranked correctly by the MDL total (the
unsolved Latin decode beats the wrong-hypothesis decodes by 0.07–0.11
bits/symbol). Reading of the acceptance: the literature target (≤ 1.9% SER
on Zodiac-408) is a per-cipher number; 17 of 18 Zodiac-class instances meet
it and the median is 0; the mean (0.041) does not, because of that one
instance. G5 records the per-instance reading as PASS and the mean as WARN.

Language recovery on homophonic ciphertexts — **18/18 by the MDL total**
against 12/18 by calibrated plaintext bits alone (latin 4/6, italian 2/6).
With 54 symbols for 25 letters the wrong-hypothesis solve of, say, an
Italian cipher under German produces a decode at 2.1 bits/char *under
German* (SER 0.76): the key has enough freedom to fake the other
language's statistics, and a same-text cancellation no longer exists. The
MDL total charges it for that freedom (2.2–2.3 vs 1.6–1.8 bits/char of
choice cost) and restores the ranking; the n-gram excess-bits ranking (17/18)
does so by subtracting each language's intrinsic entropy. **For verbose
ciphers the cross-language ranking must be an MDL ranking** — carried into
5.6 and Phase 6.

## Gate G5 — verdict: **PASS** (2026-08-23 ~06:00 UTC)

`scripts/g5_check.py` → `DATA_ROOT/runs/g5_report.json`, ClearML tag `g5`.
Gate wording: rungs 1–3 meet their SER targets on synthetics; cross-head
scores are on a comparable scale; rung 4 may trail.

| check | status | value |
|---|---|---|
| 5.1 evaluator frozen (EMA G4 checkpoint, sha256 fingerprint), interface verified on real weights | PASS | `phase_c-85m-seed0/ckpt_final.pt` step 2000, `e2cfb3c6…`; gradients reach a toy head, NULL-blend corners finite, frame path == metrology estimator |
| 5.1 NULL-frame vs plain-stream gap measured; uniform scale scores collapsed decodes | PASS | +0.45–0.52 bits/plaintext char, language-symmetric |
| 5.2 rung 1 near-perfect at ≥ 200 chars | PASS | SER 0.0016 (n-gram 0.0021), 98.9% solved, language 99.4% |
| 5.2 per-language solve success ≥ 90% at ≥ 200 (search fairness) | PASS | latin 93 → 97%, italian 100%, german 100% |
| 5.3 rung 2 ≤ 1.9% SER, Zodiac-408-class — mean over instances | **WARN** | mean 0.0407 (one Latin basin miss at 480 restarts) |
| 5.3 rung 2 — per-instance reading (≥ 80% of instances ≤ 1.9%, median ≤ 1.9%) | PASS | 94% (17/18), median 0.000; n-gram alone 89% |
| 5.3 degenerate maps rejected by the MDL selection (pure-ELBO preference recorded) | PASS | pure ELBO 33% degenerate picks (SER 0.30); MDL 0% |
| 5.3 / 6.6 literature anchors | **WARN** | Zodiac-408 is English (outside the inventory); Borg / BnF fr2988 not fetched — Phase 6 |
| 5.4 rung 3 ≥ 95% letter-map accuracy on Naibbe pairs | PASS | occurrence-weighted type accuracy 0.998 (n-gram 0.996); unweighted code accuracy 0.946; SER ≤ 1.3%; language 12/12 |
| 5.4 restart budget documented | PASS | 3 × 350 steps + fixed-parse swap polish per hypothesis, 10k chars (~40 min/hypothesis single-thread) |
| 5.5 rung 4 (P1) better than family-random | PASS | language 7/9, family 8/9 (random 3/9, 5/9) |
| 5.6 / G5 cross-head scores on one scale | PASS | same instrument (max gap 0.098 bits on 18 solved cells); MDL picks the true cipher class 24/24; rung 1 first on 1:1 6/6 |
| freeze discipline: every artifact scored by the same frozen evaluator under `v3-phase_c-ro` | PASS | rung 1–4 and cross-head |
| 5.1 NaN smoke test + two-tier unit tests in CI | PASS | `tests/test_heads.py` (29 tests) |

**Verdict: PASS**, with the two WARNs above stated as such: rung 2 clears
its SER target on 17 of 18 Zodiac-class instances (median 0), not on the
mean; the literature anchors remain a Phase-6 item.

## Assessment — what Phase 5 established and changed

1. **The two-tier design works, but the outer tier's working form is
   discrete, not differentiable.** Gradients reach head parameters through
   the frozen backbone (5.1), and at every rung the expected-embedding
   refinement either left the key unchanged or made it worse (rung 1 probe;
   rung 3: helped 3/12, hurt 7/12, never selected). The mechanism is
   measured, not guessed: the soft-target NELBO pays the frame's own
   entropy, so the gradient is "sharpen", and the "which letter" signal
   under it is smaller than the masking noise at any step that can flip an
   assignment. What the frozen ELBO *is* good at is judging hard candidates
   near the key — and used that way (`elbo_polish`: batch-scored discrete
   moves, paired masks, confirmed at budget 64) it improves on the n-gram
   objective's own optimum, to the exact true key on latin t4 and below the
   shortlist oracle on two languages at rung 2. The design's R3 hedge
   ("straight-through if expectation inputs are too smooth") does not
   rescue the gradient path either; the hedge that works is discrete. This
   should go into the design doc as a revision of §8.
2. **Any density judge prefers degenerate verbose decodes; the MDL scale is
   what makes the ELBO usable on verbose ciphers.** The frozen evaluator
   scored a 2-letter homophonic decode at 1.40 bits/char against 2.29 for
   Latin, and wrong-hypothesis 54-symbol solves at the other language's
   clean-text level. The description length of the cipher's freedom
   (choice bits) — built for task 5.6 — is the principled form of the CH.5
   KL penalty, rejects every degenerate map, picks the true cipher class on
   24/24 cross-head instances, and restores the rung-2 language ranking from
   12/18 to 18/18. Every (cipher × language) cell of the Phase-6 table is
   ranked on the MDL total per ciphertext symbol; plaintext bits/char are
   the secondary scale (and the only term for bijective keys, where the two
   coincide).
3. **Search fairness is now measured, and Latin is the hard language at
   every rung**: rung 1 solved 93% (n-gram) → 97% at ≥ 200 chars vs 100%
   for Italian and German; rung 2 5/6 vs 6/6; rung 3 the lowest unweighted
   code accuracy (0.906–0.942, γ-table codes of rare Latin letters). The
   judge is symmetric (the per-language NELBO canary held at every gate);
   the inner n-gram search is not, because Latin held-out windows are the
   highest-entropy texts in the inventory. Phase 6 must report per-language
   solve success next to every recovery number, and a Latin-specific search
   budget is a legitimate, documented choice.
4. **The ELBO is the worse judge below ~100 chars** (rung 1: the pentagram
   picks better maps in 10–30% of L50 instances), and the 2N NULL frame
   costs 0.5 bits/char over the plain stream — both properties of the
   instrument that Phase 6 inherits: score collapsed decodes on the plain
   path, do not rank on windows shorter than ~200 characters.

### Carry-overs

- **Phase 6 ranking rule**: MDL total per ciphertext symbol (calibrated
  plaintext bits × n_plain + key bits + choice bits) for every (cipher ×
  language) cell; within a cipher hypothesis the same total; report the
  three terms, plaintext bits/char, the per-window spread, the
  calibration-precision flag (`margin_uncertainty_bits`) and the replicate
  flip-rate (budget 64 × 4 seeds). Never rank on the pure ELBO for a
  verbose head.
- **Outer tier per rung**: rung 1 — shortlist re-ranking (8 deep; deeper
  for short texts); rung 2 — MDL selection + `elbo_polish`; rung 3 — ELBO
  shortlist over restarts × polish (an ELBO-scored within-block swap polish
  is the next lever: 4554 moves × ten windows is ~10 min on the 85M per
  hypothesis, untested); rung 4 — shortlist only (`elbo_polish` over the
  letter-value assignment `u` with a re-decode per move is the analogue,
  untested). Soft refinement stays instrumented but is not relied on.
- **VMS scale**: the rung-3/4 DPs still run the position loop in Python
  (40 min per hypothesis at 10k chars); Currier A/B streams are 10–16× that.
  Chunked parallel solves over forked workers (the rung-3 solve stage
  already shards by instance) are the first step; the DP vectorization of
  `rung4_arithmetic_design.md` §6 the second.
- **Literature anchors (6.6)**: Zodiac-408 cannot be scored by this
  instrument (English); fetch Borg (Latin) and BnF fr2988 transcriptions
  and run the rung-2 pipeline with `elbo_polish` on them.
- **Design-doc revisions to record**: §8 (expected-embedding refinement →
  discrete ELBO polish; straight-through does not rescue the gradient
  path), §8 R5 (complexity penalty = key bits + choice bits, ranked per
  ciphertext symbol), §7.4 (the n-gram tier keeps the KL penalty; the
  outer tier's selection is MDL).
- **4.7 is still paused** (`docs/phase4_status.md` §4.7); the GPUs are free
  again after this phase.
