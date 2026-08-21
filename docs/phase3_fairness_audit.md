# Phase 3 — bound-fairness audit (task 3.5)

Generated 2026-08-21T19:25:16Z by `scripts/fairness_audit.py`; adopted table **v1** (`CALIBRATION_VERSION`). Offsets are `NELBO − NLL_AR` in bits/char on the full tiled held-out split (positive = diffusion bound looser than the reference's likelihood).

## Offsets by calibration table (backbone × reference tier)

| table | backbone | reference | latin | italian | german | spread |
|---|---|---|---|---|---|---|
| v1-arv1 | phase_a-85m-seed0 | ? | +0.004 ± 0.009 | -0.083 ± 0.003 | +0.026 ± 0.002 | 0.109 |
| v1 | phase_a-85m-seed0 | ? | +0.091 ± 0.006 | -0.083 ± 0.003 | +0.149 ± 0.002 | 0.232 |

## Covariates

| language | family | train chars | AR NLL (adopted) | 5-gram held-out | 1→5-gram gain |
|---|---|---|---|---|---|
| latin | romance | 26,775,121 | 2.259 | 2.955 | 1.076 |
| italian | romance | 3,648,534 | 2.636 | 2.842 | 1.170 |
| german | germanic | 88,712,352 | 1.751 | 2.405 | 1.687 |

## Correlates (descriptive, n = 3 languages — not testable)

| table | ρ(offset, corpus size) | ρ(offset, AR entropy) | ρ(offset, n-gram gain) | Germanic − Romance |
|---|---|---|---|---|
| v1-arv1 | +1.0 | -1.0 | +0.5 | +0.065 |
| v1 | +1.0 | -1.0 | +0.5 | +0.145 |

## Findings

- `reference-dependence/phase_a-85m-seed0` — **ESCALATED**. phase_a-85m-seed0: swapping only the AR reference (v1-arv1, v1) moves the offsets by latin 0.086, italian 0.000, german 0.123 bits/char (up to 55× s.e.m.). → the offset is dominated by reference quality, not bound looseness — adopt the most data-fair reference tier (v1: one multilingual model on the backbone's own mix) and treat remaining offsets as bound-gap-plus-architecture terms, never as proof of comparable tightness.
- `correlates/v1` — within noise. v1: Spearman(offset, train chars) = +1.0, (offset, AR entropy) = -1.0, (offset, 1→5-gram gain) = +0.5; Germanic − Romance = +0.145 bits/char. → descriptive only — with three languages no rank correlation is testable; the document-level test above is the inferential statement.

No above-noise finding is left un-escalated.
