# Wordhom battery — restart notes (paused 2026-08-29 ~15:20 UTC)

> **Record status (banner added 2026-09-01):** SUPERSEDED — operational pause note of 2026-08-29; the battery was restarted the same day, completed 2026-08-30 and written up as `docs/alt_loop_plan.md` §10 (commits dcd5600 … b4a62e2).
> Kept for the pre-stated readings ("Readings to make"), which are the pre-registration record for §10. Everything in the state table, "After the German pass" and "Uncommitted changes" is done. **Current project position: `docs/project_status.md`.**

Purpose: manuscript-shaped controls for the wildcard→anneal word-homophonic
pipeline (`docs/alt_loop_plan.md` §8.4–8.7) before an exhaustive treatment.
Requested items: (1) B-like positive, (2) A- and B-like negatives,
(3) "wrong" cross-language hypothesis, (4) dirty plaintext, (5) mixed language
(80 % German with a 20 % Latin block). German first for every set.

## Why paused

The container lost GPU access mid-session: `torch.cuda.is_available()` →
False, `nvidia-smi` → "Failed to initialize NVML: Unknown Error", opening
`/dev/nvidia0` → EPERM although the nodes exist (device-cgroup access dropped).
Fix = restart the container from the host. Nothing GPU-side has run yet.
*[Superseded: the battery ran to completion 2026-08-29/30 — `docs/alt_loop_plan.md` §10.]*

## State on disk (all done, nothing to redo)

| item | where |
|---|---|
| builder / solver / report script | `scripts/wordhom_battery.py` (new, uncommitted) *[committed in dcd5600, 2026-08-29]* |
| loop generalised for battery cells | `scripts/altloop_pol.py --battery --cells NAME:HYP` (WHCell tolerates missing / other-language truth; SER `None` when no plaintext) |
| judge generalised | `scripts/judge_at_ser.py --battery NAME:HYP --run-tags … --tag …` (keys: stuck, truth if hypothesis = generating language, finals of the run tags) |
| 21 instances + manifest | `data/analysis/wordhom/battery/wordtypesall/` |
| German solves (7 instances × 3 hyps, n-gram MDL starts) | `data/analysis/wordhom/battery/battery_solves.json` |
| chain scripts (per GPU) | `data/analysis/altloop/battery/chain_g0.sh`, `chain_g1.sh` |
| memory note | `~/.claude/projects/-workspace/memory/wordhom-battery.md` |

Instances per language: `shuffled/<l>/{Alike,Blike}`, `voynichesque/<l>/{Alike,Blike}`
(negatives), `dirty/<l>/Alike_s05|_s10` (Phase-2 `TranscriptionNoise` 5 %/10 %),
`mixed/<l>+<other>/Alike` (German|Latin|German 5600/2800/5600 letters, one German
key, `truth.sections`). Shapes: Alike 14 000 letters / 5 200 key types (4.2 tok/type),
Blike 30 000 / 7 200 (5.6). Voynichesque draws selected by tokens/type (A 4.4–5.0,
B 6.0–6.4). B-like positive and cross-language cells reuse
`positive/german/{Blike,Alike}` from `analysis/wordhom/controls/wordtypesall` with
their `controls_solves.json` starts. Latin/Italian instances exist but are NOT solved.
*[Done 2026-08-29/30: all three languages solved and run — `docs/alt_loop_plan.md` §10.1.]*

## Restart procedure

1. Restart the container from the host; verify:
   `uv run python -c "import torch;print(torch.cuda.device_count())"` → 2.
2. Optional sanity (fast): `uv run python scripts/wordhom_battery.py --stage report`
   should print an empty table without error.
3. Launch the two chains (each: per cell wild 96 rounds/patience 10 → anneal
   `--wild-anneal 0,40` 80 rounds → judge; 1 seed; set `SEEDS=2` in the env for two):
   ```
   nohup data/analysis/altloop/battery/chain_g0.sh >/dev/null 2>&1 &
   nohup data/analysis/altloop/battery/chain_g1.sh >/dev/null 2>&1 &
   ```
   Verify with `ps -C python3 -o pid,etime,args | grep -v defunct` (not `pgrep -f`,
   which matches the caller). Never put the launch text and a `pkill -f` of it in
   the same shell command (kills the shell, exit 144).
4. Cell order (cheap/informative first) — GPU 0: `positive/german/Alike:latin`,
   `shuffled/german/Alike:german`, `dirty/german/Alike_s05:german`,
   `mixed/german+latin/Alike:german`, `positive/german/Blike:german`,
   `shuffled/german/Blike:german`. GPU 1: `positive/german/Alike:italian`,
   `voynichesque/german/Alike:german`, `dirty/german/Alike_s10:german`,
   `voynichesque/german/Blike:german`. ≈ 40 min per A-like cell, ≈ 100 min per
   B-like; `CELL_DONE …` lines in `data/analysis/altloop/battery/run_g{0,1}.out`,
   judge lines in `judge_g{0,1}.out`.
5. Report as cells finish:
   ```
   uv run python scripts/wordhom_battery.py --stage report \
     --run-tags _bat_wild_g0 _bat_anneal_g0 _bat_wild_g1 _bat_anneal_g1 \
     --judge-tags _battery_g0 _battery_g1
   ```
   → `data/analysis/wordhom/battery/report.md` (SER start→final per stage, obj Δ,
   judge plain/margin/rank/called; per-section SER for the mixed cell; reference
   rows for truth/stuck).

## Readings to make (pre-stated)

- B-like positive: does anneal reach the German A-like residual (SER ≈ 0.05,
  margin ≥ 1.5, CALLED) at 5.5 tok/type?
- Negatives (shuffled / voynichesque, A and B): loop must stay NOISE — judge
  margin in the 0.1–0.5 band, never language-like; any climb toward 1.5 is a
  false-structure alarm for the pipeline.
- Cross-language (German text, latin/italian hypothesis): final margin and
  ranking — does the judge still rank German first on a key fitted to the wrong
  language, and does the loop "invent" the wrong language?
- Dirty 5 %/10 %: SER measured against the noisy plaintext; score the truth key
  too (judge row `truth`) — the noise's own cost is the ceiling.
- Mixed: per-section SER (does the Latin block's key survive under the German
  hypothesis; does it drag the German sections?).

## After the German pass

Solve and run Latin/Italian: `uv run python scripts/wordhom_battery.py --stage solve --only /latin --workers 12`
(then `/italian`), add cells to copies of the chain scripts with new tags, add the
tags to the report. Add seed 2 (`SEEDS=2`) for the cells that matter. Then write up
in `docs/alt_loop_plan.md` (new §10) and update the memory note.

*[Done: Latin/Italian run 2026-08-29/30 (tags `_bat_*_{l0,l1,l1b,i0,i1,x1}`), seed 1 added on the three borderline positives, written up as `docs/alt_loop_plan.md` §10 on 2026-08-30.]*

## Uncommitted changes *[resolved: committed 2026-08-29 in dcd5600]*

`scripts/wordhom_battery.py` (new), `scripts/altloop_pol.py`, `scripts/judge_at_ser.py`
— committed in dcd5600 (2026-08-29) with this note; follow-ups 86efd16 (`nodouble`), 7d4991f (`MAX_OWN_BPC` resample), b4a62e2 (`revdouble`).
