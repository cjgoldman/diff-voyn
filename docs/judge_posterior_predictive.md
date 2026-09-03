# Posterior-predictive judging of the rare key entries (R2) — pre-registration

> **Record status (banner added 2026-09-02):** the pre-registration of proposal R2 of `docs/bayesian_perspective_review.md` §3, written 2026-09-02 **before any run**; §6 stays empty until the study is run and is the only section that will change. Numbers in §4 are the readings fixed in advance and must not be edited after the run. Not a call rule: `vms/apply.py::ABSTAIN_RULE` remains the judge of record whatever §6 says. **Current project position: `docs/project_status.md`.**

## 1. Question

The solver of record (hapax-wildcard → anneal, `docs/alt_loop_plan.md` §8.4–8.6) hands the frozen judge a key in which every hapax type is *committed* to one unit. Those rare types are where the residual errors live: on the A-like anneal finals > 95 % of wrong letters sit on types with ≤ 2 occurrences (error-spread note, 2026-08-28), and the judge pays ≈ 8.5 bits per wrong letter rather than averaging them away. The Italian A-like final (SER 0.12, margin 1.39–1.40) is not called although its truth key is (1.56); German and Latin finals are called (2.13–2.14 / 1.70–1.72).

R2 asks: if the judge sees the hapax entries *integrated out* — sampled from the denoiser's own posterior instead of committed by the search — does the structure margin of a near-perfect decode move toward its truth ceiling, **without** lifting the negatives?

## 2. Protocol (implemented in `scripts/judge_posterior_predictive.py`)

For each key (§3):

1. Decode under the key (`expand_units`). Rare types = types with occurrence ≤ 1 (`--hapax-max 1`, the same `rare_type` mask as `altloop_vms.py`). Their letter positions (both letters of a bigram unit) form the forced mask; they are 9.8–11.1 % of the letters on A-like cells and 4.4–6.9 % on B-like cells.
2. One denoiser pass per fill: `heads/posterior.py::position_posterior(n_draws=1, mask_rate=0.3, force_mask=rare positions, seed=10000+k)` — the rare letters are withheld from the context, plus the standard 30 % random mask.
3. `unit_scores` sums the per-position log-posterior per type within the type's length class (letter types over the 25 letters, bigram-unit types over the hypothesis' doubled units). One unit per rare type is sampled from the softmax of that row at temperature 1 (`numpy` generator seeded 10000+k). Rule violations are counted, not repaired.
4. The filled key is scored by the exact Phase-6 judge as wrapped by `judge_at_ser.score_map`: full stream, ≤ 16 windows, paired decode / letter-shuffled rows, 4 replicate seeds, budget 64, 3 language conditions, calibration `v3-phase_c-ro` report-only. CRN masks stay on seed 0 (the seed of every recorded row); **the shuffled reference is re-drawn from the fill's own letters** (shuffle seed 100+k), never reused.
5. K = 4 fills per key (`--fills 4`). Reported per key: mean and sd over fills of structure margin and plain bits, mean SER (positives), fraction of fills called, the majority top language — next to the key's *recorded* committed numbers (copied from the source row, not re-scored).
6. One extra scoring of the committed key with a fresh shuffle draw (`--committed-rescore 1`, shuffle seed 1) is stored as the shuffle-draw-noise reference. It is diagnostic; the readings in §4 compare against the recorded committed number.

Fixed parameters: `--fills 4 --committed-rescore 1 --hapax-max 1 --mask-rate 0.3 --temp 1.0 --budget 64 --seeds 4 --score-windows 16`. The d5b20 (`big*`) rows are excluded (different hypothesis space, as in `evidence_odds.py`).

## 3. Key set (from the recorded `judge_at_ser*.json` rows, `--dry-run` 2026-09-02)

47 anneal-final keys, in the piles the evidence-odds table uses:

| pile | keys |
|---|---|
| A-like clean positives (the target of the study) | `wild:anneal_de/s0,s1` (German, committed 2.14 / 2.13, called), `wild:anneal/s0–s2` (Italian 1.39 / 1.40 / 1.39, not called), `wild:anneal/s0–s2` (Latin 1.70 / 1.72 / 1.70, called) |
| B-like clean positives | `positive/{german,latin,italian}/Blike` anneal finals (2.22 called / 1.83 called / 1.46 not called) |
| **negatives** (the guard) | `shuffled/<lang>/{Alike,Blike}` ×6 (0.17–0.27), `voynichesque/<lang>/{Alike,Blike}` ×6 (0.21–0.37), cross-language `positive/<lang>/Alike:<other>` ×6 (0.39–0.48) — 18 keys |
| dirty positives (A-like) | `dirty/<lang>/Alike_{s05,s10}` finals incl. the `x1` second runs (0.37–1.51; only `dirty/german/Alike_s05` `g0` called) |
| other positives | `mixed` ×5 (incl. `x1`, `anneal2_x1`), `nodouble` ×3, `revdouble` ×1 |

With `--include-truth`, 29 further truth keys (the fill's cost on a perfect key: how much margin integrating out ~10 % of the letters gives away even when the committed entries are right). Truth rows are informational.

## 4. Pre-registered readings (fixed before the run)

Δ = mean posterior-predictive margin over the 4 fills − the recorded committed margin of the same key.

**Success requires both halves:**

- **Positives.** On the Italian A-like anneal finals Δ ≥ +0.10 on at least 2 of the 3 seeds (1.39–1.40 → ≥ 1.49; ceiling 1.56), **and** on the German and Latin A-like anneal finals Δ ≥ −0.05 on every seed (no loss on the already-called keys).
- **Negatives.** On every one of the 18 negative-pile keys Δ ≤ +0.05.

**Failure on either half = not adopted**, recorded here (§6) the way `docs/confidence_mask_probe.md` §9 records a negative probe, with no further tuning of K, temperature, mask rate or the rare-type threshold on these keys.

**What success means.** Adoption makes the posterior-predictive margin a *reportable statistic* beside the committed margin (a row in the odds tables, a column in the battery reports). It does **not** make it a call rule: turning it into one — a threshold, its own negative corridor, its own power table — needs a second pre-registration and its own control battery.

**Secondary readings (informational, no gate).** Whether the B-like Italian positive (1.46) crosses 1.5; whether any dirty / mixed positive that is not called becomes called on a majority of fills; the truth-key drop (truth margin − pp margin at truth) as the price of the fill; the rescore-vs-committed difference as the shuffle-draw noise the Δ tolerance has to be read against. If the rescore noise on the negatives exceeds 0.05 by itself, that is reported as a limitation of the 0.05 criterion, not used to relax it.

**Expected outcome (stated so the result can be read as a surprise or not).** The Italian final's 1464 rare types carry most of its 0.12 SER; a posterior fill that gets those right more often than the committed key would lift the margin toward 1.56. The negatives are expected not to move: on a wrong key the denoiser's posterior at the withheld positions is close to the language's unigram distribution, which the shuffled reference already matches. The likelier failure mode is that the fill *costs* margin even on German/Latin (sampled letters are noisier than committed ones that are mostly right), which would fail the first half.

## 5. Cost and invocation

≈ 100 s (A-like) / 125 s (B-like) per scoring at budget 64 × 4 seeds; 5 scorings per key (1 rescore + 4 fills) → ≈ 8–10 min per key; 47 keys ≈ 7–8 h on one GPU, truth keys +≈ 4.5 h. Split across the two GPUs by `--only` / `--sources` with distinct `--tag`s (each tag writes its own `judge_posterior_predictive<tag>.json`; `--stage report` renders one file). Standing rule: `nvidia-smi` + the torch check before any launch (memory `check-gpus-before-launch`).

```bash
uv run python scripts/judge_posterior_predictive.py --dry-run --include-truth        # key list, CPU
CUDA_VISIBLE_DEVICES=0 uv run python scripts/judge_posterior_predictive.py --sources judge_at_ser.json judge_at_ser_battery_g0.json judge_at_ser_battery_g1.json judge_at_ser_battery_g0b.json judge_at_ser_battery_rd0.json judge_at_ser_battery_x1.json judge_at_ser_battery_x1c.json --tag _r2_g0
CUDA_VISIBLE_DEVICES=1 uv run python scripts/judge_posterior_predictive.py --sources judge_at_ser_battery_i0.json judge_at_ser_battery_i1.json judge_at_ser_battery_l0.json judge_at_ser_battery_l1.json judge_at_ser_battery_l1b.json --tag _r2_g1
```

A smoke run (2 fills, budget 4, 1 seed, 2 windows, output in the session scratchpad, discarded) verified the script end to end on 2026-09-02.

## 6. Results

*Not run as of 2026-09-02.* This section is filled in after the run; §4 is not edited.
