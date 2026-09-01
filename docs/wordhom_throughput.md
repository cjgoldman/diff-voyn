# Throughput of the diffusion-guided n-gram loop on the wordhom head

> **Record status (banner added 2026-09-01):** engineering note, 2026-08-26 (§5 same-day follow-up).
> Still current: the numba objective/SA/polish and the compiled `levenshtein_ser`; a round is ≈ 24 s and judge-bound (§5 supersedes §3's ≈ 14 s prediction). §4–§5 "what is left" (batched `score_stream` behind a per-row CRN-seed extension; optional `metrics_proposed`) is still OPEN as of 2026-09-01 (`docs/project_status.md` §6). **Current project position: `docs/project_status.md`.**

Status: profiled and optimized 2026-08-26. Code: `diff_voyn/heads/wordhom_state.py`
(new), `diff_voyn/heads/wordhom.py` (`sa_phase` / `polish` rewired),
`tests/test_wordhom.py` (exactness tests). Nothing in the scoring,
acceptance rule, judge, or objective *definition* changed; only how the
n-gram objective is evaluated inside the SA and the greedy polish.

## 1. Where a round went (before)

`scripts/altloop_vms.py --head wordhom`, cell IT2a/A latin (3,647 types,
10,846 tokens, 10,904 letters, 30 units), measured on the idle GPU with the
production run still going on GPU 0 (14.5 h in, GPU utilization **0 %**, one
CPU core at 100 %):

| stage of one round | time | note |
|---|---|---|
| `objective()` full-stream recompute | 0.33 ms/eval | `expand_units` 40 %, `score_ids` 33 % |
| SA, 200k steps | ~101 s | one full recompute per step |
| greedy polish, **one** sweep | 36 s | 3647 × 29 full evals; several sweeps on the rand/psamp arms |
| posterior `scores()` (16 draws) | 1.3 s | GPU |
| `metrics()` × 2 (proposed + after SA) | 10 s | GPU, 64 strata × 8 windows × 2 rows, already batched |
| final metrics (3 languages × 4 seeds), once per arm-run | 62 s | GPU |

Log: `none` arm 141–175 s/round, `rand`/`psamp` 400–520 s/round. So ≥ 90 %
of a round is the n-gram inner search, and the inner search re-scores
~11k letters for a move that touches one word type — median occurrence 1,
74 % of types singletons, but the occurrence-weighted proposal has
E[occ] = 38 (Currier A) / 76 (B) and 18–35 % of proposals hit types with
≥ 50 occurrences.

## 2. What changed

`WordHomObjectiveState` (`diff_voyn/heads/wordhom_state.py`) — an exact
incremental form of the objective, with the hot loops compiled by **numba**
(added to the base dependencies with `uv add numba`, 2026-08-26):

* a pentagram conditional depends on the 4 letters before it, so a move on
  token *i* changes only the conditionals of tokens *i..i+4*; per-token
  conditional sums `tokll` are cached, so the OLD side of any delta is a
  lookup and only the NEW side is computed;
* the frequency-KL term is kept as `Σ c ln c` / `Σ c ln prior`
  (O(letters changed)); the repeat-rule count is re-examined on the pairs
  adjacent to the occurrences;
* `_scan` (numba): for every occurrence *p* of a changed symbol it rebuilds
  the letters of tokens *p−4..hi*, `hi = min(p+4, next occurrence − 1)`, so
  overlapping occurrences never charge a token twice, and scores the
  letters of tokens *p..hi* with the same warm-up rule as
  `NgramLM.score_ids` (exact for tokens 0..3 too);
* `_sa_run` / `_polish_run` (numba): the whole SA loop (occurrence-weighted
  inverse-CDF proposal, swap move, Metropolis accept, in-place commit) and
  the greedy best-improvement sweeps run in one compiled pass; the numba
  RNG is seeded from the caller's `Generator`. `WordHomophonicHead.
  sa_phase` / `polish` are thin wrappers; `polish` returns the *full*
  objective recomputed once, so `alternate`'s acceptance comparison sees no
  drift;
* the Python `delta` / `commit` / `deltas_all` methods stay as the
  reference API (the tests pin them, and the compiled loops, to the full
  objective within the float32 summation noise of the full scorer, ≤ 5e-3
  nats on a −26,105 objective).

Edge case found by the tests: types with **zero** occurrences in the window
(the key maps every type of the instance; a window holds a subset). A
first pure-numpy version (per-occurrence vectorized kernel + versioned row
cache) reached 25 s per SA; its floor was ~45 small numpy calls per step,
which is what the compiled loop removes.

## 3. Result

Same cells, same seeds, `sa_phase(steps=200_000, 2.0→0.3)` including its
polish, wall clock on one core (numba compile cached on disk after the
first call, ~2 s):

| | before | numpy version | numba (final) |
|---|---|---|---|
| singleton delta | 330 µs (full) | 17 µs | ~1 µs inside the loop |
| occ 463 delta | 330 µs | 143 µs | ~60 µs |
| polish sweep | 36 s | 0.7 s | ~0.1 s |
| SA 200k + polish, from the optimum (`none` arm) | 137 s | 25 s | **1.9 s** (A) / **1.7 s** (B) |
| SA 200k + polish from a 512-kick (`rand` arm shape) | ~137 s + (k−1)×36 s | 30 s | **2.0 s** (A) / **3.7 s** (B, 9 sweeps) |

Per round on the wordhom head: `none` arm 141 s → ≈ 14 s (2 s SA + 10 s
metrics + 1.3 s scores); `rand`/`psamp` 400–520 s → ≈ 15 s *[revised in §5 below: measured ≈ 24 s per round once the `levenshtein_ser` term was found and compiled; the diffusion metric is the remaining cost]*. The n-gram
inner search went from ≥ 90 % of a round to ~15 %; the GPU metrics are now
the dominant term. Both versions reach the same local optima on the
benchmark cells (identical final objectives from the optimum; from the
kick the compiled SA's different RNG stream lands on a comparable optimum).

Caveats: SA trajectories are **not** bit-identical to the recorded runs
(numba RNG, float64 accumulation); the greedy polish is best-improvement
rather than first-improvement per symbol.

## 4. What is left, in order of expected gain

1. **GPU metrics** — `metrics(prop)` every round (5 s) feeds one logged
   scalar (`structure_margin_proposed`); making it optional halves the GPU
   term. The end-of-run metrics (3 conditions × 4 seeds, 62 s per arm-run)
   are batched and GPU-bound; only a lower budget would cut them.
2. **Parallel cells.** The script runs 12 cells × 3 arms × 2 seeds serially
   on one core with 12 cores and two 24 GB GPUs available; `--only` already
   shards. Four shards ≈ 4× wall clock.
3. The same compiled-scan pattern applies to `HomophonicHead._sa_phase`
   (rung 2, `_objective` over the whole stream per move) and to the
   `posterior_sample` proposer's Python loops — smaller gains (their
   streams are 1–4k symbols).

## 5. Follow-up (2026-08-26, later): the round was still ~97 s

`scripts/altloop_pol.py --only wh/` after the change above logged a flat
**95–105 s per round** (`runs_hm.json`, all arms, all proposal sizes) and
~300 s on the `Alike` stretch cells, against the ≈ 14 s predicted in §3.
Component timing of one round (german/t0, 1,171 types, 7,733 letters, idle
GPU): SA + polish 1.2 s, posterior `scores` 0.9 s, **`metrics` 47.5 s**,
called twice per round by `alternate`. Inside `metrics`: `score_stream`
9.5 s and **`unit_ser` 38 s** — `levenshtein_ser`
(`heads/rung4_arithmetic.py`) ran its O(n·m) edit-distance DP with a
pure-Python inner loop (60 M iterations for a 7.7k-letter decode;
quadratic, so the ~20k-letter `Alike` cells paid ~4× more). §1 never
measured it because that profile was taken on `altloop_vms.py`, whose
manuscript cells have no ground truth and no SER. The wordhom throughput
doc's "metrics 10 s" was the diffusion term only.

Fix: `_levenshtein` compiled with numba (same DP, two rolling rows,
bit-identical to the Python version on random and near-identical pairs of
lengths 0–7.7k): 38 s → **0.13 s**. One round is now ≈ 24 s
(2 × `score_stream` 9.5 s + SA 1.2 s + posterior 0.9 s), i.e. the diffusion
metric is the whole remaining cost. A run launched before this fix keeps
the old function in memory — restart to benefit.

Remaining term, not changed here: `DiffusionEvaluator.score_ids` tiles a
long stream into windows and scores each with `ScoreSettings(batch=1)` —
64 strata × 8 windows = 512 single-sequence forward passes at ~45 ms each
(kernel-launch bound: 100 % of one CPU core, GPU ~40 %). Batching the
windows of one stream as rows (`score_ids(x.reshape(8, 1024))` measured
2.8 s vs 9.4 s) would be ~3.4× faster but changes the per-window CRN seeds
(`seed + 7919·k`), so the recorded Phase-6 numbers would not reproduce —
do it behind a per-row-seed extension of `per_window_nelbo_bits`, not by
re-chunking. Alternatively drop `metrics_proposed` (§4 item 1) to halve it.
