# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`diff-voyn` — a research codebase for language identification of Voynich-like ciphertext via trial decipherment. A multilingual masked (absorbing-state) discrete diffusion model over characters serves as the likelihood instrument; differentiable cipher decoding heads are optimized against it. Target output is a ranked (cipher system × plaintext language) table for the Voynich Manuscript.

**Current state: Phase 5 complete — Gate G5 passed 2026-08-23 (cipher heads on the frozen diffusion evaluator: rung 1 SER 0.0016 / language recovery 99.4% at ≥200 chars; rung 2 Zodiac-408-class 17/18 instances ≤1.9% SER, median 0, language recovery 18/18 by MDL total; rung 3 Naibbe occurrence-weighted letter-map accuracy 0.998, language 12/12; rung 4 language 7/9, family 8/9; cross-head MDL scale picks the true cipher class 24/24). Two WARNs: rung-2 mean SER 0.041 (one Latin basin miss at 480 restarts) and the literature anchors (Zodiac-408 is English — outside the inventory; Borg/BnF not fetched). **Phase 6 (VMS application, tasks 6.1–6.7) is next.** Read `docs/phase5_status.md` first — its findings change how Phase 6 ranks: (a) the outer tier's working form is *discrete* (`ladder.elbo_polish`: batch-scored single moves under paired NELBO + choice bits, confirmed at budget 64), expected-embedding gradient refinement never moves a key usefully (soft-target CE pays the frame's entropy); (b) any density judge prefers degenerate verbose decodes (a 2-letter homophonic decode scores 1.40 bits/char vs 2.29 true) — **rank every (cipher × language) cell on the MDL total per ciphertext symbol** (`heads/scale.py`: calibrated plaintext bits + key bits + choice bits; `two_tier.py` shortlist machinery), never on the pure ELBO for a verbose head; (c) the 2N NULL frame costs ~0.5 bits/char over the plain stream — score collapsed hard decodes on the plain path; (d) Latin is the hard language for the n-gram inner search at every rung — report per-language solve success; (e) the ELBO is the worse judge below ~100 chars. Frozen evaluator: `DATA_ROOT/runs/phase_c-85m-seed0/ckpt_final.pt` (sha256 in `analysis/phase5/evaluator_freeze.json`), calibration `v3-phase_c-ro`. Phase-5 code: `diff_voyn/heads/{diffusion_eval,two_tier,scale,ladder}.py`, `NgramEvaluator.viterbi_segmental`, `NaibbeBlockHead.decode/polish/refine_frame`, `HomophonicHead.polish_pairs`, shortlists on every head; scripts `phase5_freeze.py`, `rung{1,2,3,4}_diffusion.py` (solve → score → report, resumable), `crosshead_scale.py`, `g5_check.py`, R3 probes `r3_*.py`; artifacts `DATA_ROOT/analysis/phase5/`. Phase 4 (G4 passed 2026-08-22): joint Phase-C 85M, synthetic 1:1 recovery 98.9% language / 99.1% family at ≥200 chars; task 4.7 (25M seed replication, P2) is **paused at resumable checkpoints** — resume per `docs/phase4_status.md` §4.7. Phase-4 code: `diff_voyn/model/lid_head.py`, `diff_voyn/data/abstain.py`, `Backbone.hidden()`, `load_lid_head`; scripts `train_lid_head.py`, `lid_eval.py`, `train.py --phase phase_c`, `head_calibration.py`, `seed_replication.py`, `g4_check.py`; the head is a short-text cross-check, not a wrong-key abstention instrument (`docs/phase4_status.md`). Phase 3 (G3 passed 2026-08-21): calibration offsets vs the multilingual AR reference v3 are *measured and stored but not subtracted* (`policy: report-only`) — applying any AR-gap table breaks same-text comparisons and drops recovery to ~70% (the Phase-3 finding, an escalated §5b.3 design change); measured offsets are the systematic uncertainty of a ranking margin (`CalibrationTable.margin_uncertainty_bits`). Scoring budget: 64 stratified draws (× 4 replicate seeds for a flip-rate). `docs/phase3_status.md` and `docs/phase3_fairness_audit.md` hold the Phase-3 record.** Phase-3 code: `diff_voyn/metrology/` (`scoring.py` — `score_conditions`, CRN harness; `calibration.py` — `CalibrationTable`, `calibrate_bits` the single application point); scripts `crn_check.py`, `sample_budget.py` (`--restat`), `score_documents.py`, `calibrate.py` (`--nelbo-from`, `--derive-report-only`), `fairness_audit.py`, `language_recovery.py` (`--stage solve|score|report`), `g3_check.py`; AR reference v3 via `train_ar_reference.py --multilingual`. Phase-2 code: `diff_voyn/data/noise.py` (wrong-key / Naibbe-parse / transcription generators + 2N-slot NULL frame, `NoiseConfig` mix), `scripts/robustness_curve.py` (severity sweeps, `--restat`), `scripts/g2_check.py`; side study `scripts/ngram_robustness.py` + `docs/ngram_judge_robustness.md` (CH.0 n-gram judges on the same noised windows: no saturation under wrong keys, severity-dependent drift to "German" as a language judge; the Phase-C judge's robustness is the curriculum, not the architecture); `scripts/train.py --phase phase_b --init-from`. `diff_voyn/model/` holds the MDLM diffusion core (`diffusion.py`, SUBS + Rao-Blackwellized NELBO, verified against exact enumerated references in `tests/test_diffusion.py`) and the RMSNorm/SwiGLU/RoPE backbone (`backbone.py`, presets via `model_preset()` in `infra/config.py`). Training runner: `scripts/train.py` (`--phase pilot|phase_a --model 25m|85m`, `--resume`); checkpoint scoring harness: `scripts/score_checkpoint.py`. The `diff_voyn/` package also holds the Phase-0 deliverables: frozen vocab (`vocab.py`, 32 symbols — the 23 Naibbe letters resolved from Greshko's actual tables, plus k/w), shared normalizer (`normalize.py`), corpus assembly + held-out splits (`corpus/`), data loader with masking sampler (`data/`), training infra with ClearML/EMA/checkpointing (`infra/`), pinned cipher wrappers (`ciphers/`), and VMS ingest (`vms/`). Phase-0 decisions and acceptance numbers are recorded in `docs/phase0_decisions.md`; **read that plus the docs below before changing anything Phase 0 froze** (vocab, normalizer, splits are versioned — bump the version rather than mutating v1).

Key scripts (all `uv run python scripts/<name>.py`): `fetch_external.py` (idempotent clone/pin of greshko/naibbe-cipher @ df3d074 and alexanderboxer/voynich-attack @ e324bee, plus voynich.nu downloads), `build_corpora.py` (assemble + splits), `tune_ciphers.py` (0.7 acceptance), `ingest_vms.py`, `g0_check.py` (gate verification; logs random-init NELBO to ClearML).

Data layout under `/workspace/data` (gitignored bind mount): `external/` (pinned repos — SHA verified at import), `raw/` (italian texts + voynich.nu files), `corpora/v1/` (normalized docs, `manifest.json` corpus table, `splits_v1.json`), `ciphers/` (tuned pseudo-VMS tables + acceptance stats), `vms/` (per-dialect Currier A/B streams).

The design is fully specified in `reference_docs/` — read those before implementing anything:

- `A Diffusion-Based Framework for Language Identification of Voynich-Like Ciphertext.md` — the paper abstract / framing.
- `Design and Training of the Multilingual Diffusion Backbone.md` — the authoritative design doc: every architecture/training decision with alternatives and reasoning (referenced below as "design §N").
- `Diffusion Model Training - Task Breakdown.md` — the execution plan: Phases 0–6 with hard gates G0–G5, task IDs (e.g. 0.1, 5.4), priorities, and acceptance criteria. New work should map to a task ID from this plan.

## Environment and commands

Development happens inside a devcontainer (CPU or GPU variant, selected via `.devcontainer/cpu/` or `.devcontainer/gpu/`; the GPU one uses a PyTorch CUDA base image and `runtime: nvidia`). Dependencies are managed with **uv** (Python 3.12, venv at `/workspace/.venv`):

```bash
uv sync                 # base deps + dev group (black, isort, ruff, pytest)
uv sync --group gpu     # additionally install the full GPU/training stack (torch, lightning, clearml, ...)

uv run pytest                                   # run tests
uv run pytest path/to/test_file.py::test_name   # run a single test
uv run black . && uv run isort .                # format (black is pinned to 24.8.0)
uv run ruff check .                             # lint
```

The `DEVICE` env var (`cpu`/`cuda`) selects which dependency set `post-create.sh` installs.

**You (Claude Code) are running inside the GPU devcontainer.** Two persistence rules follow:

- **All data, results, model checkpoints, logs, and downloaded models go under `/workspace/data`** (a gitignored bind mount from the host, `DATA_DIR` in `.devcontainer/.env`; HF models via `HF_HOME=/workspace/data/hf-models`). Anything written elsewhere in the container filesystem outside `/workspace` is lost on rebuild, and large artifacts must never land in git.
- **Never install packages ad hoc** (`pip install`, bare `uv pip install`) — the environment is recreated from spec on every container rebuild. Record every addition where it persists: Python deps via `uv add <pkg>` (or `uv add --group gpu` / `--group dev` for training-only or tooling deps) so `pyproject.toml` and `uv.lock` are updated and committed; system-level packages in `.devcontainer/Dockerfile`; setup steps in `.devcontainer/post-create.sh`.

Experiment tracking uses **ClearML** (self-hosted at `clearml.acet.network`, credentials passed through from the host environment). Every training run gets a ClearML Task capturing config, per-language NELBO scalars, and checkpoint artifacts.

## Architecture (from the design doc — the "big picture")

One shared backbone, three consumers: (a) per-language ELBO as the likelihood metric, (b) a language-ID head, (c) a differentiable plaintext evaluator for cipher decoding heads.

- **Model**: encoder-only transformer (RMSNorm, SwiGLU, RoPE), ~85M (12L/d768) plus a ~25M sibling for cheap restart-heavy search loops. MDLM-style continuous-time masked diffusion (SUBS parameterization, Rao-Blackwellized NELBO), no time-conditioning network. Context 1024 chars.
- **Tokenization**: character-level only, ~32-symbol vocab (normalized Latin alphabet + MASK/NULL/BOS/EOS). **No subwords, and no SPACE token — all whitespace is stripped from every text stream (corpora, ciphertexts, VMS) in preprocessing.** This is a modeling decision (design §2), not a convenience.
- **Languages**: Latin, Italian, German (frozen inventory, task 0.2), all trained jointly from step one via an additive per-position language embedding with 10% conditioning dropout. Never introduce languages sequentially — symmetry across languages is a correctness property of the instrument.
- **Cipher heads** (Phase 5): optimized against a **frozen (EMA) backbone**, validated in strict difficulty order: 1:1 substitution → unigram homophonic → Naibbe mixed unigram-bigram (`naibbe_v2.py`, greshko/naibbe-cipher @ `df3d074`) → arithmetic sum-to-target (`voynpy.pseudo_vms` from Boxer's voynich-attack repo). Heads feed the backbone expected embeddings on a 2N-slot frame with NULL blending; inner search uses a cheap n-gram DP, the diffusion ELBO scores shortlists.

## Non-negotiables the plan is built around

These come from the design doc's load-bearing decisions; violating them silently biases the scientific result:

1. **Bound-tightness fairness (R1)**: language ranking compares per-language ELBOs, so anything that tightens one language's bound preferentially (sequential training, per-language tokenizer fit, lossy alphabet mapping) is a bias. Calibration offsets (task 3.4) are measured, versioned, and applied in exactly one place.
2. **Gates are ordered**: LID head attaches only after the backbone stabilizes (stop-gradient first, then joint with small λ); the evaluator is frozen before any cipher-head optimization; G3 (calibrated metric) must pass before G4.
3. **Per-language held-out NELBO is the canary** at every gate — if it degrades or a language stalls, adjust sampling temperature or λ, don't change the schedule.
4. **Scoring uses common random numbers** across language conditions (same masking realizations) — the ranking consumes score *differences*, and CRN is the main variance-reduction lever.
5. Clean-text fraction is retained in every training phase (calibration is defined on clean text); Currier A and B dialects of the VMS are never pooled.
6. Known numerical trap: `logaddexp(−∞,−∞)` NaN when log-space blending NULL slots — keep the smoke test (design §8).
