# Phase 3 — bound-fairness audit (task 3.5)

> **Record status (banner added 2026-09-01):** script-generated snapshot of the G3-era audit,
> 2026-08-21 (adopted table `v3-ro`, Phase-B 85M). Superseded by the same script's re-runs
> `docs/phase4_fairness_audit.md` (adopted `v3-phase_c-ro`, 2026-08-22) and
> `docs/phase6_fairness_audit.md` (2026-08-23, **the current audit**); the four ESCALATED
> findings and the report-only policy are unchanged in both. Regenerating this file with
> `scripts/fairness_audit.py` would drop this banner. **Current project position:
> `docs/project_status.md`.**

Generated 2026-08-21T22:57:50Z by `scripts/fairness_audit.py`; adopted table **v3-ro** (`CALIBRATION_VERSION`). Offsets are `NELBO − NLL_AR` in bits/char on the full tiled held-out split (positive = diffusion bound looser than the reference's likelihood).

## Offsets by calibration table (backbone × reference tier)

| table | backbone | reference | policy | latin | italian | german | spread |
|---|---|---|---|---|---|---|---|
| v1-arv1 | phase_a-85m-seed0 | v1 | apply | +0.004 ± 0.009 | -0.083 ± 0.003 | +0.026 ± 0.002 | 0.109 |
| v1 | phase_a-85m-seed0 | v2 | apply | +0.091 ± 0.006 | -0.083 ± 0.003 | +0.149 ± 0.002 | 0.232 |
| v2 | phase_b-85m-seed0 | v2 | apply | +0.096 ± 0.007 | -0.078 ± 0.003 | +0.157 ± 0.002 | 0.235 |
| v3-phase_a | phase_a-85m-seed0 | v3 | apply | +0.132 ± 0.003 | +0.003 ± 0.003 | +0.198 ± 0.002 | 0.195 |
| v3 | phase_b-85m-seed0 | v3 | apply | +0.138 ± 0.003 | +0.008 ± 0.003 | +0.206 ± 0.002 | 0.198 |
| v2-25m | phase_b-25m-seed0 | v2 | apply | +0.298 ± 0.008 | +0.080 ± 0.004 | +0.360 ± 0.003 | 0.280 |
| v3-25m | phase_b-25m-seed0 | v3 | apply | +0.340 ± 0.005 | +0.166 ± 0.003 | +0.409 ± 0.003 | 0.243 |
| v3-phase_a-25m | phase_a-25m-seed0 | v3 | apply | +0.335 ± 0.004 | +0.150 ± 0.003 | +0.394 ± 0.003 | 0.244 |
| v3-ro | phase_b-85m-seed0 | v3 | report-only (from v3) | +0.138 ± 0.003 | +0.008 ± 0.003 | +0.206 ± 0.002 | 0.198 |

## Covariates

| language | family | train chars | AR NLL (adopted) | 5-gram held-out | 1→5-gram gain |
|---|---|---|---|---|---|
| latin | romance | 26,775,121 | 2.218 | 2.955 | 1.076 |
| italian | romance | 3,648,534 | 2.551 | 2.842 | 1.170 |
| german | germanic | 88,712,352 | 1.702 | 2.405 | 1.687 |

## Language dependence beyond document dispersion (per-document offsets)

| table | n docs | doc-mean offset latin / italian / german | between-lang range | within-lang doc sd | ANOVA p | Kruskal p |
|---|---|---|---|---|---|---|
| v2 | {'latin': 6, 'italian': 2, 'german': 8} | +0.058 / -0.070 / +0.170 | 0.240 | 0.103 | 0.0354 | 0.0372 |
| v3-phase_a | {'latin': 6, 'italian': 2, 'german': 8} | +0.120 / +0.020 / +0.220 | 0.199 | 0.041 | 3.56e-05 | 0.00208 |
| v3 | {'latin': 6, 'italian': 2, 'german': 8} | +0.124 / +0.021 / +0.229 | 0.208 | 0.040 | 3.58e-05 | 0.00208 |
| v2-25m | {'latin': 6, 'italian': 2, 'german': 8} | +0.248 / +0.093 / +0.388 | 0.295 | 0.123 | 0.0295 | 0.0536 |
| v3-25m | {'latin': 6, 'italian': 2, 'german': 8} | +0.314 / +0.185 / +0.447 | 0.262 | 0.062 | 0.000305 | 0.00448 |
| v3-phase_a-25m | {'latin': 6, 'italian': 2, 'german': 8} | +0.317 / +0.169 / +0.432 | 0.263 | 0.050 | 4.29e-05 | 0.00376 |
| v3-ro | {'latin': 6, 'italian': 2, 'german': 8} | +0.124 / +0.021 / +0.229 | 0.208 | 0.040 | 3.58e-05 | 0.00208 |

## Correlates (descriptive, n = 3 languages — not testable)

| table | ρ(offset, corpus size) | ρ(offset, AR entropy) | ρ(offset, n-gram gain) | Germanic − Romance |
|---|---|---|---|---|
| v1-arv1 | +1.0 | -1.0 | +0.5 | +0.065 |
| v1 | +1.0 | -1.0 | +0.5 | +0.145 |
| v2 | +1.0 | -1.0 | +0.5 | +0.148 |
| v3-phase_a | +1.0 | -1.0 | +0.5 | +0.130 |
| v3 | +1.0 | -1.0 | +0.5 | +0.133 |
| v2-25m | +1.0 | -1.0 | +0.5 | +0.171 |
| v3-25m | +1.0 | -1.0 | +0.5 | +0.156 |
| v3-phase_a-25m | +1.0 | -1.0 | +0.5 | +0.152 |
| v3-ro | +1.0 | -1.0 | +0.5 | +0.133 |

## Findings

- `reference-dependence/phase_a-85m-seed0` — **ESCALATED**. phase_a-85m-seed0: swapping only the AR reference (v1-arv1, v1, v3-phase_a) moves the offsets by latin 0.128, italian 0.086, german 0.172 bits/char (up to 77× s.e.m.). → the offset is dominated by reference quality, not bound looseness — adopt the most data-fair reference tier (v3-ro: one multilingual model on the backbone's own mix) and treat remaining offsets as bound-gap-plus-architecture terms, never as proof of comparable tightness.
- `reference-dependence/phase_b-85m-seed0` — **ESCALATED**. phase_b-85m-seed0: swapping only the AR reference (v2, v3) moves the offsets by latin 0.041, italian 0.086, german 0.049 bits/char (up to 28× s.e.m.). → the offset is dominated by reference quality, not bound looseness — adopt the most data-fair reference tier (v3-ro: one multilingual model on the backbone's own mix) and treat remaining offsets as bound-gap-plus-architecture terms, never as proof of comparable tightness.
- `reference-dependence/phase_b-25m-seed0` — **ESCALATED**. phase_b-25m-seed0: swapping only the AR reference (v2-25m, v3-25m) moves the offsets by latin 0.041, italian 0.086, german 0.049 bits/char (up to 24× s.e.m.). → the offset is dominated by reference quality, not bound looseness — adopt the most data-fair reference tier (v3-ro: one multilingual model on the backbone's own mix) and treat remaining offsets as bound-gap-plus-architecture terms, never as proof of comparable tightness.
- `language-dependence/v3-ro` — **ESCALATED**. v3-ro: per-document offsets differ by language — range 0.208 bits vs within-language document sd 0.040; ANOVA F=24.9 p=3.58e-05, Kruskal p=0.00208 (n = {'latin': 6, 'italian': 2, 'german': 8}). → a language-level offset exists beyond document noise; under the report-only policy it is NOT subtracted — the 3.6 suite showed that subtracting it breaks every same-text comparison — but carried as the systematic uncertainty of every cross-language margin (CalibrationTable.margin_uncertainty_bits) and re-measured after every phase.
- `correlates/v3-ro` — within noise. v3-ro: Spearman(offset, train chars) = +1.0, (offset, AR entropy) = -1.0, (offset, 1→5-gram gain) = +0.5; Germanic − Romance = +0.133 bits/char. → descriptive only — with three languages no rank correlation is testable; the document-level test above is the inferential statement.

No above-noise finding is left un-escalated.
