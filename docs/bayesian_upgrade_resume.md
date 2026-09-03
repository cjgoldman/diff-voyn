# Resume note — Bayesian upgrade track (written 2026-09-02, before a container restart)

> **Record status (banner added 2026-09-02):** a restart/pickup note, in the pattern of `docs/wordhom_battery_restart.md`. It records what the 2026-09-02 Bayesian review left on disk, what is pending, and the exact next actions with their pre-conditions, so a fresh session can continue without re-deriving anything. Superseded the moment the pending items are run — update the state table below rather than this banner. **Current project position: `docs/project_status.md`.**

## 1. Why the restart

`nvidia-smi` returned `Failed to initialize NVML: Unknown Error` and `torch.cuda.is_available()` was `False` inside the container on 2026-09-02 15:00 UTC, with `/dev/nvidia{0,1}` present and host driver 590.48.01 loaded — the container lost its device access and must be restarted from the host. Standing rule from the user (memory `check-gpus-before-launch`): **before any GPU launch, run `nvidia-smi` and the torch check and report; if CUDA is absent, stop.**

## 2. State on disk (all under the persistent mounts; nothing lives only in the container)

| item | path | state |
|---|---|---|
| the review | `docs/bayesian_perspective_review.md` | complete; §5 holds the odds tables with numbers |
| odds script | `scripts/evidence_odds.py` | complete, lint-clean; `uv run python scripts/evidence_odds.py` (CPU, seconds) |
| odds artifacts | `DATA_ROOT/analysis/evidence_odds/odds.{json,md}` | generated 2026-09-02 |
| status / instructions | `docs/project_status.md` (§1 evidence-odds paragraph, §2 row, §6 R2–R6 items, §7 row), `CLAUDE.md` (Current status bullet) | edited |
| memory | `~/.claude/projects/-workspace/memory/{bayesian-review,check-gpus-before-launch}.md`, indexed in `MEMORY.md` | written; `~/.claude` is on the host root filesystem (persistent) |
| plan file of the review session | `~/.claude/plans/please-take-a-look-reactive-petal.md` | the approved plan; §3 there = §3 of the review |
| git | `main`, **uncommitted**: `CLAUDE.md`, `docs/project_status.md`, `docs/bayesian_perspective_review.md`, `docs/bayesian_upgrade_resume.md`, `scripts/evidence_odds.py` | the pre-commit hook runs `scripts/doc_coherence_check.py` (clean as of writing) |
| **§3.1 prep (2026-09-02, after the restart)** | `scripts/wordhom_battery.py` (`--shapes`, dirty loop over shapes; A-like RNG unchanged), instances `dirty/<lang>/Blike_{s05,s10}` in the battery manifest (5.5–5.7 tokens/type), `battery_solves.json` (+6 own-language solves), `DATA_ROOT/analysis/altloop/battery/chain_bd{0,1}.sh` | **run 2026-09-02 15:42 → 03 03:38 UTC**; results `docs/alt_loop_plan.md` §10.6, judge `judge_at_ser_battery_bd{0,1}.json`, report `analysis/wordhom/battery/report_bshape_dirty.md`, odds regenerated |
| **§3.2 prep (2026-09-02)** | `docs/judge_posterior_predictive.md` (pre-registration, readings fixed in §4), `scripts/judge_posterior_predictive.py` (lint-clean, `--dry-run` lists 47 + 29 keys, smoke-tested on GPU at budget 4), `scripts/judge_at_ser.py::score_map(shuffle_seed=…)` | written; **not run** |

Nothing was launched on a GPU in the review session; no run records or queues are in flight.

## 3. Pending — gated on the user's explicit say-so (memory `present-before-running`)

Both were proposed in the review and presented to the user; neither was approved for launch yet.

### 3.1 Dirty positives at Currier-B shape (the biggest gap in the odds piles)

*Why:* `evidence_odds` has no "dirty" pile at B shape, and B is the dialect where a null result is informative. Currently the dirty-10 % odds (1 : 2–4) rest on single A-shape cells.

*Status 2026-09-03:* **run and recorded** (`docs/alt_loop_plan.md` §10.6): dirty-5 % German called 2/2 (1.69–1.70); dirty 10 % uncalled; Latin s05 drew a 3.00-bpc text and is uncallable at truth — the reading below ("a called dirty-5 % Latin raises the power line") could not be tested; a re-drawn Latin instance is the open item. The readings paragraph stands as written before the run. Earlier status 2026-09-02: the code change, `prepare` and `solve` below were done; `chain_bd0.sh` (GPU 0: german s05 ×2 seeds, german s10, italian s05 ×2) and `chain_bd1.sh` (GPU 1: latin s05 ×2, latin s10, italian s10) are written and only need `nohup zsh data/analysis/altloop/battery/chain_bd0.sh &` (and `bd1`) after the GPU check. ≈ 8.5 h / 7 h.

*Code change needed first (small — done):* `scripts/wordhom_battery.py` builds `dirty/<lang>/Alike_{s05,s10}` only (`# dirty positives (A-like)`, the loop at the `battery-dirty` RNG). Extend it to emit `dirty/<lang>/Blike_{s05,s10}` (loop over `SHAPES`, keep the seed derivation `_rng("battery-dirty", seed, lang, shape, tag)` so existing A-like instances are not re-drawn — check `--fresh` semantics before running `prepare`). Then `--stage prepare --controls dirty --only dirty/<lang>/Blike_s05 …` and `--stage solve` for the n-gram MDL start keys.

*Run:* copy `DATA_ROOT/analysis/altloop/battery/chain_g1.sh` as the template (wild 96 / patience 10 → anneal `0,40` 80 / patience 10 → `judge_at_ser.py --battery`), one chain per GPU, cells `dirty/<lang>/Blike_s05:<lang>` and `_s10` for the three languages. Tags **must** be `_bat_wild_<G>` / `_bat_anneal_<G>` / `_battery_<G>` so `evidence_odds.py` picks the judge rows up automatically (it reads `judge_at_ser_battery_*.json`, keys `final:_bat_anneal*`, and skips files starting with `big`). Two seeds on the s05 cells (the A-shape dirty-5 % German call did not replicate). Run chains serially per tag — parallel `altloop` runs on the same tag clobber `runs.json` (memory `naibbe-wordhom-twins`).

*Cost:* ≈ 100 min per B-like cell per seed on one GPU *[observed 2026-09-03: ≈ 2 h per B-shape seed-run — wild 20–60 min, anneal 70–106 min; chains took 15:42 → 00:10 / 03:38 UTC]*; 6 cells × 1 seed ≈ 10 h on one GPU, ≈ 5 h across two; +6 h for second seeds on s05.

*Readings to fix before running:* a dirty-5 % B-shape positive that is **called** in German/Latin raises the dirty-5 % power line from 0.5 / 0 and sharpens the odds against a noisy cipher in B; dirty-10 % B-shape positives are expected uncalled (truth ceilings 1.54 / 1.06 / 0.91 at A shape; B shape not measured) — if they *are* called, the "no power at 10 %" statement in the review §5 is withdrawn for B.

### 3.2 R2 — posterior-predictive judging of the rare entries (pre-register, then run)

*Idea:* instead of judging the anneal final with every hapax type committed, sample the hapax letters from the denoiser's own posterior and average the judge over several fills.

*Pieces that exist:* `heads/posterior.py::position_posterior(force_mask=…)` (mask chosen positions, one forward pass per draw, per-position log-posterior), `unit_scores` (per-type sums, same length class), the Phase-6 scoring loop as wrapped by `scripts/judge_at_ser.py` (paired decode/shuffle windows × 4 seeds × 3 conditions, budget 64, ≈ 100 s per key), the keys in `DATA_ROOT/analysis/altloop/judge_at_ser.json` (A-like truth / stuck / `wild:anneal*` finals) and `judge_at_ser_battery_*.json` (battery finals, negatives, cross-language).

*Status 2026-09-02:* written and pre-registered — `docs/judge_posterior_predictive.md` holds the protocol and the fixed readings; invocation in its §5. The description below is the original sketch.

*To write (done):* `scripts/judge_posterior_predictive.py` — for each key: decode; hapax positions = occurrences of types with `occ ≤ 1` (the same `rare_type` mask as `altloop_vms.py`); draw K fills by (a) `position_posterior` with `force_mask` on the hapax positions plus the standard 0.3 random mask, (b) one letter per **type** sampled from the type's summed posterior (type-consistent, temperature 1), (c) rebuild the decode; score each fill with the judge; report mean margin and plain bits over the K fills next to the committed key's numbers. The shuffled reference must be re-drawn per fill (own letters), not reused.

*Pre-registered readings (fix the numbers before running):* success = on the A-like anneal finals the averaged margin moves toward the truth ceiling by ≥ 0.10 bits for Italian (1.39 → ≥ 1.49; ceiling 1.56) and does not fall for German/Latin, **and** no negative pile cell (shuffled / voynichesque / cross-language anneal finals) rises by more than 0.05. Failure on either half = not adopted, recorded like `docs/confidence_mask_probe.md` §9. Adoption means "reportable statistic", never a call rule without a second pre-registration.

*Cost:* K = 4 fills × ≈ 100 s per key ≈ 7 min per key; ~40 keys ≈ 5 h on one GPU.

### 3.3 Later, not scoped: R3 key population, R4 tied key prior for A, R5 marginal-likelihood content test, R6 hygiene (multi-permutation shuffle reference in `vms/apply.py`, P(margin > 1.5) from replicate seeds, second seeds on the single-seed battery negatives).

## 4. First actions on resume

1. `nvidia-smi` and `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` — report; stop if no CUDA.
2. `uv run python scripts/doc_coherence_check.py` (also runs at SessionStart).
3. `git status` — the five files above should still be modified/untracked; commit only if the user asks.
4. §3.1 is done (2026-09-03). Ask whether to launch §3.2 (the two commands in `docs/judge_posterior_predictive.md` §5); do not start it on the strength of this note.
