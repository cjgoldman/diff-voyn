# Lexical coverage as a human-shaped call criterion (2026-08-28)

Question: a reader recognises "Italian with 15 % letter errors" easily; the
frozen structure-margin rule (≥ 1.5 bits/char) does not. Is a
word-recognition statistic a better second criterion?

`scripts/lexical_coverage.py` → `analysis/altloop/lexical_coverage_k{K}_l{L}.json`.
Statistic: DP-segment the spaceless decode to maximise the characters covered
by words of length ≥ L from the language's K most frequent training-split
words (equal K across languages); **lexical margin** = coverage(decode) −
coverage(letter-shuffled decode). Primary K=5000, L=5 (K=20000 and L=6
shift every band together; K=236k/count≥2, L=4 is unusable — 73 % of
shuffled text "covered", Latin truth called German).

## Bands (K=5000, L=5), margin under the *hypothesis* lexicon

| group | latin | italian | german |
|---|---|---|---|
| A-like wordhom truth | 0.36 | 0.46 | 0.57 |
| A-like **anneal finals** (SER 0.05 de / 0.07 la / 0.12 it) | 0.36–0.37 | 0.45 | 0.56–0.58 |
| A-like stuck (SER 0.66–0.77) | 0.17 | 0.19 | 0.24 |
| uniform corruption of truth @ SER 0.10 / 0.20 | 0.20 / 0.11 | 0.27 / 0.17 | 0.36 / 0.21 |
| Phase-6 positives, sub1to1/homophonic (SER 0) | **0.29–0.43** | 0.43–0.56 | 0.54–0.63 |
| voynichesque (max over 25 decodes) | **0.31** | **0.32** | 0.31 |
| contamination (max) | 0.11 | 0.17 | 0.23 |
| shuffled | ≤ 0.09 | ≤ 0.10 | ≤ 0.11 |
| **VMS** wordhom / homophonic / sub1to1 / naibbe / arithmetic (max) | 0.18 / 0.13 / 0.07 / 0.03 / 0.01 | 0.20 / 0.12 / 0.08 / 0.03 / 0.03 | 0.21 / **0.28** / 0.13 / 0.04 / 0.03 |

## Findings

1. **The search's residual errors are lexically free.** All nine anneal
   finals sit at the truth's own coverage (German 0.57 vs 0.57, Italian 0.45
   vs 0.46, Latin 0.36 vs 0.36) while *random* errors at the same SER cost
   0.1–0.2 — the n-gram objective places its residual wrong letters where
   they complete words. Ranking by lexicon is right on every A-like key.
2. **German and Italian separate cleanly**: positives ≥ 0.43 (it) / 0.54
   (de) vs every negative ≤ 0.32 — a threshold near 0.38 calls the Italian
   anneal finals (0.45) that the structure margin abstains on (1.40 vs 1.5).
3. **Latin does not separate**: true Latin decipherments (Phase-6 SER 0)
   score 0.29–0.43 and the voynichesque leak reaches 0.31. A 5 000-word
   frequency lexicon covers only 37 % of real Latin (inflection); the
   negatives are real Latin leaking through a wrong cipher, so a weak Latin
   detector cannot tell the two apart. The structure margin is the better
   Latin instrument (truth 1.9 vs voynichesque ≤ 1.26).
4. **The manuscript sits in the negative band** on every head (max 0.28,
   homophonic/German — the same head/language that tops the voynichesque
   negatives), below every positive at truth. Consistent with the abstention.
5. Not a replacement for the frozen rule: the negatives are n=3–25 per
   language, the criterion is language-asymmetric (Latin), and any threshold
   must be pre-registered per language before reading manuscript numbers.
   Reasonable use: a second, *per-language* criterion (Italian/German only)
   for synthetic-positive calls, or a diagnostic alongside the margin.

Note: the Phase-6 `naibbe` candidates' stored `decode` is not the final
decode (SER 0.31–0.78 vs truth) — excluded from the positive band above.
