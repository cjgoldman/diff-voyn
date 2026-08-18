# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`diff-voyn` — a research codebase for language identification of Voynich-like ciphertext via trial decipherment. A multilingual masked (absorbing-state) discrete diffusion model over characters serves as the likelihood instrument; differentiable cipher decoding heads are optimized against it. Target output is a ranked (cipher system × plaintext language) table for the Voynich Manuscript.

**Current state: Phase 0 complete, Gate G0 passed (2026-08-18).** The `diff_voyn/` package holds the Phase-0 deliverables: frozen vocab (`vocab.py`, 32 symbols — the 23 Naibbe letters resolved from Greshko's actual tables, plus k/w), shared normalizer (`normalize.py`), corpus assembly + held-out splits (`corpus/`), data loader with masking sampler (`data/`), training infra with ClearML/EMA/checkpointing (`infra/`), pinned cipher wrappers (`ciphers/`), and VMS ingest (`vms/`). Phase-0 decisions and acceptance numbers are recorded in `docs/phase0_decisions.md`; **read that plus the docs below before changing anything Phase 0 froze** (vocab, normalizer, splits are versioned — bump the version rather than mutating v1). Phase 1 (MDLM diffusion core + backbone, tasks 1.1/1.2) is next and does not exist yet.

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
