# Content-free voynichesque negatives — restart context

> **Record status (banner added 2026-09-01):** restart note written 2026-08-31 15:35 UTC; **the task COMPLETED the same day, 17:55 UTC, without the restart — results below** (§ "Results (2026-08-31)"). The state table, restart recipe and expectations are the mid-task record; the motivation (§ "What this task is") and the reading rules (§ "How to read the result") are still current. The Polygraphia remark at the end is superseded by `docs/polygraphia_status.md` §5 (quarantine). **Current project position: `docs/project_status.md`** (§1 "Phase-6 acceptance", §4, §5.6–5.7).

*Written 2026-08-31 ~15:35 UTC, immediately before a container restart (GPU
connections lost mid-chain). This doc holds everything needed to resume and
finish the task. Companion memory: `voynichesque-nocontent-battery.md`.*

## What this task is

A **strict-negative** twin of the Phase-6 `voynichesque` control battery.
Motivation (2026-08-31 session): the Phase-6 voynichesque control enciphers
REAL held-out text, and its token types deterministically encode 1–3-letter
plaintext chunks — so it is a wrong-hypothesis partial decode of real
language, not a negative under the strict definition (*negative = no
meaningful text anywhere in the generation*). The oracle study
(scratchpad `voyn_oracle.py`, session of 2026-08-31) showed a perfect 50-unit
wordhom key on it still only reaches structure margin 0.02–0.37, and the
solved keys' Latin-cell exact token→letter recovery runs ~27× chance — real
content leaks, even though it never produces a call.

**The experiment**: rebuild each of the nine Phase-6 voynichesque instances
with the **same generator seed** (⇒ identical parameter draw, cipher
alphabets, chunk boundaries, y-decoration — the RNG stream is consumed
identically for an equal-length source) but a **letter-shuffled** copy of the
same source window. Unigram statistics and the full glyph grammar survive;
sequential content provably does not.

**Decision it makes**: if the twins reproduce the Phase-6 voynichesque margin
band (glyph heads 0.92–1.51; naibbe 0.19–0.79), that band was **glyph
grammar, not leaked content** — and the twins become the legitimate
"Voynich-shaped gibberish" strict negative for every head. If the twins land
materially lower, part of the old band WAS recoverable content and the
Phase-6 acceptance discussion (voynichesque 0.89 > 0.95 FAIL) needs
reframing either way (the criterion was run on a non-negative).

*[Outcome 2026-08-31 (results section below): mixed — the homophonic band was content-inflated by ≈ 0.27 bits/char (27/27 pairs lower); the sub1to1 and naibbe bands are pure glyph grammar (Δ −0.02 / +0.02); the P0 near-miss `italian/t1` was on a non-negative (1.51 → 1.24). Reading per `docs/project_status.md` §1, §5.6.]*

## State at interruption

| item | state |
|---|---|
| generator script | `scripts/voynichesque_nocontent.py` (committed 2026-09-01 in `bce1dca`; *originally read "new, uncommitted"*) — generate-only; asserts replay of the real twins byte-identically before shuffling; asserts twin chunk counts match |
| instances | **DONE, verified**: 18 files (9 eva + 9 words) + `manifest.json` under `DATA_ROOT/analysis/phase6/controls_nocontent/`; control tag `voynichesque_nocontent`, names `voynichesque_nc/<lang>/t<k>`, truth carries `source: letter-shuffled` and `twin_of` |
| solve stage | **DONE 2026-08-31** — 66/66 jobs in `controls_nocontent/solves.json` *(state at the 15:35 kill: IN PROGRESS, 4/66 jobs — naibbe words jobs, ~7 min each; resumable, `run_solves` skips jobs already in `solves.json`)* |
| score stage | **DONE 2026-08-31** (66 cells scored; *at 15:35: not started*) |
| report stage | **DONE 2026-08-31 17:55 UTC** → `controls_nocontent/report.{json,md}` (*at 15:35: not started*) |
| chain script | `controls_nocontent/chain.sh` (solve → score shard 0/2 GPU0 + 1/2 GPU1 → report), log `chain.log` |

## How to restart (after container is back)

```bash
# 1. verify GPUs
uv run python -c "import torch; print(torch.cuda.device_count())"   # want 2

# 2. re-run the chain — fully idempotent (solve and score both resume via
#    load_done keyed on (instance, presentation, head, window, hypothesis))
cd /workspace
nohup bash data/analysis/phase6/controls_nocontent/chain.sh \
  > data/analysis/phase6/controls_nocontent/chain.log 2>&1 &

# 3. watch
tail -f data/analysis/phase6/controls_nocontent/chain.log
```

Notes:
- The chain re-emits `=== SOLVE/SCORE/REPORT start` lines; solve prints
  `66 jobs, N done, M to solve`.
- If only one GPU comes back, edit `chain.sh` to a single unsharded score:
  `--shard 0/1` (drop the second process and the `wait`).
- Everything runs through the frozen Phase-6 machinery
  (`scripts/vms_controls.py --out-dir .../controls_nocontent`); no code
  changes are needed or wanted — comparability with the Phase-6 controls is
  the point. Estimated remaining: solve ~1.5–3 h (62 jobs, 12 CPU workers;
  homophonic r2=32 jobs dominate), score < 1 h on two GPUs, report seconds.
- Known wrinkle: `stage_report` was written for the Phase-6 control set; the
  new control tag should pass through generically, but if the acceptance
  table code chokes on the unknown control name, fix in the report stage
  only — never in solve/score.

## How to read the result

Compare per-head margin bands in `controls_nocontent/report.json` (cells →
`structure_margin`, grouped by `head`) against the real twins in
`analysis/phase6/controls/report.json` (`control == "voynichesque"`):

- real twins: sub1to1 0.64–1.42 · homophonic 0.85–1.51 · naibbe 0.19–0.79
- per-instance pairing: `voynichesque_nc/<lang>/t<k>` ↔
  `voynichesque/<lang>/t<k>` — the pairing is seed-exact, so per-instance
  margin differences are attributable to content alone (same key, same
  chunking; the character streams differ in length only because glyph
  lengths depend on which letters occur).
- Expectation from the 2026-08-31 Phase-6 parameter analysis: margins track
  the draw's *trigram-heaviness* (the 1.50–1.51 near-miss `italian/t1` is
  93 % trigram letters; the most homophonic-like draw `german/t0`, u=0.78,
  scored lowest, 0.95) — i.e. grammar-driven. If so, twins ≈ real,
  instance by instance.

## Wider session context (for the write-up later)

- The 1.5 margin threshold is anchored on Phase-3 POSITIVES (true 1.6–2.9 vs
  wrong-hypothesis 0.5–1.9); voynichesque never fed it — but the Phase-6
  writeup's phrase "inside the band of voynichesque gibberish" invites the
  invalid inference (band ≈ non-language) and should be re-worded to
  "non-decipherments of real text"; `ciphers/controls.py` "no recoverable
  plaintext" docstrings are false as written.
  *[Status 2026-09-01: the phrase "band of voynichesque gibberish" does not
  appear verbatim in `phase6_writeup.md`/`phase6_status.md` (paraphrase or
  already reworded — the Phase-6 records now carry their own supersession
  notes); the `ciphers/controls.py` and `vms/controls.py` docstrings were
  corrected 2026-09-01 to "wrong-hypothesis control generated from real
  held-out text, not a strict negative". See `docs/project_status.md` §5.6.]*
- Under the user's working premise (no 1:1 / 1:n glyph-substitution heads —
  externally ruled out): VMS ceiling drops 1.25 → 0.50 (wordhom); the entire
  Phase-6 positive band 1.45–2.48 is glyph-head; verbose replacement
  positives are the wordhom truth keys 1.56–2.52; the 0.5–1.55 gap is
  populated by dirty/mismatched truths (0.52–1.56), so the verbose-only
  claim is "excludes a clean SER ≲ 0.1 correctly-specified verbose
  decipherment", not "no language".
- Historical anchor for the verbose tier: `docs/polygraphia_digitization_scope.md`
  (Trithemius Ave Maria tables). *(corrected 2026-09-01; originally read
  "written 2026-08-31, nothing built")* — Polygraphia was built provisionally
  the same day (41 of ~692 columns), produced the pipeline's **first false
  CALLS** (cyclic cells, margins 1.76–1.96 at SER 0.74–0.77, a periodic
  pseudo-language objective trap) and is **QUARANTINED** from every control
  workflow by user decision (`docs/polygraphia_status.md` §4–§5; `CLAUDE.md`).
  It is not an anchor.

## Results (2026-08-31, `analysis/phase6/controls_nocontent/report.{json,md}`)

*Section added 2026-09-01 from the verified report (task completed
2026-08-31 17:55 UTC without the restart). Numbers per
`docs/project_status.md` §1, §4, §5.6–5.7.*

All 66 cells (9 twin instances × heads × languages) ran through the frozen
Phase-6 controls pipeline (`scripts/vms_controls.py`, same evaluator, MDL
scale, `ABSTAIN_RULE`). Per-instance twin − real structure-margin
differences, paired seed-exactly with the Phase-6 `voynichesque` cells:

| head | mean Δ (twin − real) | pairs negative | range | real band → twin band |
|---|---|---|---|---|
| homophonic | **−0.27** | **27/27** | −0.39 … −0.16 | 0.85–1.51 → 0.55–1.24 |
| sub1to1 | −0.02 | 17/27 | −0.13 … +0.07 | (reproduced) |
| naibbe | +0.02 | — | −0.73 … +0.48 | (reproduced, noisier) |

- Highest twin margin anywhere: **1.40** (`italian/t1`, sub1to1 head, German
  hypothesis; plain bits 2.60) — the strict-gibberish ceiling.
- **0/66 cells language-like; 9/9 instances abstain.**
- The Phase-6 P0 near-miss `italian/t1` (homophonic: 1.43 / 1.51 / 1.50 under
  it / la / de) drops to **1.15 / 1.24 / 1.14** on its content-free twin.
- Reading: the homophonic voynichesque band was content-inflated by ≈ 0.27
  bits/char — the real-text `voynichesque` control is a **wrong-hypothesis
  control**, not a strict negative; the sub1to1 and naibbe bands are pure
  glyph grammar. Consequently the Phase-6 P0 FAIL (voynichesque abstention
  0.89 vs > 0.95) was a **mis-specified test run on a non-negative**; the
  frozen verdict is left as recorded (a re-classification, not a pass).
  Against a strict negative the safe corridor for the margin threshold is
  ≈ **1.41–1.48** (Phase 6 quoted 1.26–1.48 against its own battery). The
  margin is the load-bearing half of the rule: gibberish decodes reach plain
  bits 1.98–2.78, i.e. the ≤ 3.0 plaintext criterion alone would pass them.
- Cosmetic: the `acceptance` block of `report.json` is all `null` and
  `summary` is empty (`stage_report` hardcodes the Phase-6 control names, so
  the new control tag `voynichesque_nocontent` is not aggregated there); the
  per-cell `cells` (66) and per-instance `verdicts` (9) entries are complete
  and are what the table above reads.
