# Robustness of the prototype n-gram language judges vs the diffusion judges

Side study, 2026-08-22 (after G4). Question: the Phase-2 robustness curve
(task 2.6) characterizes how the diffusion evaluator degrades under partial
decipherments; how do the **CH.0 n-gram judges** — the interpolated
Witten–Bell character LMs (`diff_voyn/heads/ngram.py`, orders 1–5) that the
cipher heads were prototyped against — degrade on the same noise, and what
is their *relative* robustness as language judges? Not a gate; a sense
check for Phase 5, where the n-gram DP stays the inner-loop scorer and the
diffusion ELBO scores the shortlists.

Script: `scripts/ngram_robustness.py`. Outputs
`DATA_ROOT/analysis/phase2/robustness_ngram.{json,_windows.npz,png}`.

## Setup

- **Identical noised texts.** The script imports `robustness_curve.py`'s
  source selection and noise generation unchanged: 48 evenly spaced
  1024-char windows per language from the tiled held-out split, the three
  noise families (self-consistent many-to-one wrong key; Naibbe parse
  errors; transcription errors) on the 2.6 severity grids, noise seed 2024.
  The uint8 arrays are bit-identical to the ones the Phase-A/B checkpoints
  scored, so every comparison below is paired (CRN across judges).
- **n-gram judges:** orders 1, 2, 3 (the DP order) and 5 (the anchor order),
  own-language bits/char per window; shuffled-letter and uniform controls.
- **Diffusion judges:** the saved 2.6 per-window arrays of the Phase-A and
  Phase-B 85M (own-language NELBO, 16 strata) for the cost curves; for the
  language-ranking curves a fresh GPU pass scored a reduced grid
  (24 windows, 8 strata) under **all three language conditions** (CRN
  masking across conditions) with the Phase-C evaluator
  (`phase_c-85m-seed0/ckpt_final.pt`) and, to separate architecture from
  curriculum, the clean Phase-A 85M.
- **Scale-free profile.** A pentagram LM (clean 2.4–3.0, shuffled 7.6–9.2
  bits/char) and a diffusion NELBO (clean 1.9–2.5, shuffled 4.1–4.7) are not
  comparable in absolute bits, so curves are also reported as the fraction
  of the judge's own clean→shuffled gap consumed, `(m(s) − m(0)) /
  (m_shuffled − m(0))`.
- **Ranking conventions.** Each judge is ranked under its adopted
  convention: n-gram = *excess bits* (each language's score minus that LM's
  clean mean, the CH.1 "−held-out bits/char" offsets; without them German
  wins by LM entropy alone); diffusion = *raw* NELBO (Phase-3 report-only
  policy). The other convention is stored for both in the JSON. Applying
  excess offsets to the diffusion judge destroys its ranking (German clean
  top-1 1.00 → 0.00) because its language conditioning is a ~0.01–0.03
  bits/char nudge on one multilingual density, while the per-language clean
  means differ by 0.45 bits — the Phase-3 same-text finding in another form.

## Findings

### 1. Under a wrong key the n-gram cost does not saturate; the diffusion cost does

Own-language cost of a self-consistent wrong key, 85M diffusion vs n-gram:

| judge | lang | clean | shuffled | +5 % key (bits) | +5 % (rel) | +20 % key (bits) | +20 % (rel) | rel @ 50 % | rel @ 100 % |
|---|---|---|---|---|---|---|---|---|---|
| ng3 | latin | 3.35 | 5.79 | 0.51 | 0.21 | 1.91 | 0.78 | 1.75 | 2.63 |
| ng5 | latin | 2.98 | 7.96 | 1.09 | 0.22 | 3.74 | 0.75 | 1.34 | 1.67 |
| phase_a-85m | latin | 2.37 | 4.66 | 0.50 | 0.22 | 1.36 | 0.60 | 0.85 | 0.97 |
| phase_b-85m | latin | 2.38 | 4.11 | 0.24 | 0.14 | 0.64 | 0.37 | 0.55 | 0.55 |
| ng3 | italian | 3.15 | 7.35 | 0.74 | 0.17 | 3.52 | 0.84 | 1.68 | 2.68 |
| ng5 | italian | 2.82 | 9.21 | 1.22 | 0.19 | 5.04 | 0.79 | 1.39 | 1.93 |
| phase_a-85m | italian | 2.54 | 5.07 | 0.52 | 0.20 | 1.54 | 0.61 | 0.88 | 1.06 |
| phase_b-85m | italian | 2.55 | 4.16 | 0.25 | 0.16 | 0.57 | 0.35 | 0.54 | 0.54 |
| ng3 | german | 2.98 | 5.88 | 0.40 | 0.14 | 1.75 | 0.60 | 1.32 | 1.90 |
| ng5 | german | 2.39 | 7.61 | 0.85 | 0.16 | 3.35 | 0.64 | 1.22 | 1.54 |
| phase_a-85m | german | 1.90 | 4.65 | 0.64 | 0.23 | 1.69 | 0.61 | 0.90 | 1.01 |
| phase_b-85m | german | 1.91 | 4.22 | 0.29 | 0.13 | 0.68 | 0.29 | 0.55 | 0.63 |

- The *first* 5 % of wrong-key errors cost every judge a similar share of
  its dynamic range (0.14–0.23); in absolute bits the pentagram pays
  0.85–1.22 bits/char, 4× the Phase-B diffusion (0.24–0.29) and 2× the
  clean Phase-A one.
- Past ~20 % the curves separate qualitatively. The n-gram cost keeps
  climbing *through* the shuffled-letters ceiling (relative 1.3–2.7 at a
  fully wrong key; pentagram Latin 2.98 → 11.3 bits/char vs shuffled 7.96):
  an unseen letter combination backs off to the add-one unigram floor and
  costs more than shuffled text, which at least keeps the letter
  frequencies. The diffusion judge saturates *below* its shuffled ceiling
  (Phase B: 0.55 of the gap at 50 % and at 100 %; Phase A: ≈1.0) — it treats
  a half-wrong key and a fully wrong key alike.
- Consequence for Phase 5: the n-gram objective keeps a usable slope far
  from the key (it is the better *search* signal from random starts, which
  is what the CH track found empirically), while the diffusion score is
  informative mainly in the last 20–30 % of the key — the regime where the
  shortlist is scored. The design's split (n-gram DP inside, ELBO on
  shortlists) is the right way round.
- The 2.6 "monotone / no cliff" verdict: all n-gram curves are monotone;
  the pentagram and several trigram substitution curves fail the
  operational "no cliff" test on the *absolute-bits* criterion (single
  increments > 1 bit), not on acceleration — the curves are concave, just
  steep. (Unigram segmentation curves are flat and their share statistics
  are meaningless; ignore those verdicts.)

### 2. Under local errors (parse, transcription) the n-gram judges are relatively *more* robust

Fraction of the clean→shuffled gap consumed:

| judge | lang | seg 0.1 | seg 0.2 | seg 0.3 | trans 0.05 | trans 0.1 | trans 0.2 |
|---|---|---|---|---|---|---|---|
| ng3 | latin | 0.18 | 0.36 | 0.52 | 0.18 | 0.36 | 0.68 |
| ng5 | latin | 0.24 | 0.45 | 0.62 | 0.19 | 0.38 | 0.67 |
| phase_a-85m | latin | 0.38 | 0.62 | 0.75 | 0.27 | 0.48 | 0.74 |
| phase_b-85m | latin | 0.38 | 0.61 | 0.73 | 0.28 | 0.49 | 0.79 |
| ng3 | german | 0.17 | 0.33 | 0.49 | 0.14 | 0.29 | 0.52 |
| ng5 | german | 0.21 | 0.40 | 0.57 | 0.17 | 0.33 | 0.58 |
| phase_a-85m | german | 0.37 | 0.60 | 0.75 | 0.27 | 0.48 | 0.73 |
| phase_b-85m | german | 0.32 | 0.53 | 0.68 | 0.23 | 0.42 | 0.69 |

(Italian in the JSON; same pattern.) A deletion or insertion corrupts only
a k-gram-wide neighbourhood for an n-gram LM, but disturbs the long-range
context the 1024-char transformer relies on, so per unit severity the
diffusion judge spends roughly twice the share of its range. Phase B barely
moved these families (they were trained at lower severities), consistent
with 2.6. In absolute bits the n-gram judges still pay more (pentagram
+0.2–0.3 bits at the first grid step vs +0.07–0.1 for the diffusion).

### 3. As *language judges* the n-gram LMs collapse asymmetrically under noise

Top-1 accuracy of the true language (n-gram: excess bits, 48 windows;
diffusion: raw NELBO, 24 windows):

| wrong key | judge | lang | 0 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| | ng3 | latin | 1.00 | 1.00 | 1.00 | 0.77 | 0.71 | 0.19 | 0.06 |
| | ng5 | latin | 1.00 | 0.98 | 1.00 | 0.62 | 0.52 | 0.10 | 0.19 |
| | phase_a-85m | latin | 0.75 | 0.75 | 0.75 | 0.62 | 0.29 | 0.38 | 0.79 |
| | **phase_c-85m** | latin | 0.79 | 0.75 | 0.79 | 0.88 | 0.96 | 0.92 | 0.38 |
| | ng3 | italian | 1.00 | 0.67 | 0.35 | 0.06 | 0.00 | 0.00 | 0.00 |
| | ng5 | italian | 1.00 | 0.98 | 0.62 | 0.25 | 0.02 | 0.00 | 0.00 |
| | phase_a-85m | italian | 1.00 | 0.67 | 0.50 | 0.21 | 0.00 | 0.00 | 0.00 |
| | **phase_c-85m** | italian | 0.96 | 0.88 | 0.92 | 0.96 | 0.96 | 0.96 | 0.33 |
| | ng3 | german | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 |
| | ng5 | german | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.69 |
| | phase_a-85m | german | 0.92 | 0.67 | 0.79 | 0.46 | 0.54 | 0.38 | 0.21 |
| | **phase_c-85m** | german | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.79 |

| parse / transcription | judge | lang | seg 0.1 | seg 0.2 | seg 0.3 | trans 0.05 | trans 0.1 | trans 0.2 |
|---|---|---|---|---|---|---|---|---|
| | ng3 | italian | 0.98 | 0.15 | 0.00 | 0.81 | 0.00 | 0.00 |
| | ng5 | italian | 1.00 | 0.98 | 0.19 | 1.00 | 0.96 | 0.00 |
| | phase_a-85m | italian | 0.33 | 0.04 | 0.00 | 0.75 | 0.38 | 0.00 |
| | **phase_c-85m** | italian | 1.00 | 0.92 | 0.88 | 0.92 | 0.96 | 1.00 |
| | ng3 / ng5 | latin, german | 1.00 | 1.00 | ≥0.96 | 1.00 | 1.00 | ≥0.96 |
| | phase_a-85m | german | 0.67 | 0.54 | 0.42 | 0.92 | 0.71 | 0.46 |
| | **phase_c-85m** | latin / german | 0.79 / 1.00 | 0.71 / 1.00 | 0.75 / 1.00 | 0.92 / 1.00 | 0.92 / 1.00 | 0.92 / 1.00 |

- **The n-gram judge's failure mode is a language-dependent bias, not
  noise.** Every noised text drifts toward the call "German": Italian text
  is lost at 5–10 % wrong key (trigram 0.67 → 0.35), at 20 % parse errors,
  at 10 % transcription errors; Latin at 20–30 % wrong key; German is never
  lost. Italian is the most regular orthography → the most peaked LM → the
  fastest-rising cost under any corruption (its shuffled ceiling is the
  highest, 9.2 bits/char at order 5), so a noisy Italian text looks
  "less Italian" faster than it looks "less German". The bias grows with
  severity, so no static calibration offset can remove it — it is the
  bound-tightness asymmetry (R1) in n-gram form, and it is the reason
  n-gram trial-decipherment rankings at *partial* keys cannot be trusted
  across languages.
- **The Phase-C diffusion judge's call is essentially flat to a 50 %-wrong
  key** (Latin 0.79 → 0.92, Italian 0.96 → 0.96, German 1.00 → 1.00) and
  across the whole parse/transcription grids; it only falls at a fully wrong
  key, where the text has no language (≈ chance, 0.33–0.38, German 0.79).
  Its sub-1.0 clean Latin accuracy (0.75–0.79) is the known held-out Latin
  heterogeneity (the three high-entropy Latin documents score as German on
  *clean* text, Phase-3 record), not a noise effect.
- **The robustness is the curriculum, not the architecture.** The clean
  Phase-A 85M loses the language call at least as fast as the n-gram
  judges (Italian 0.67 at 5 % wrong key, 0.33 at 10 % parse errors; German
  0.67 at 5 % wrong key). Phase B/C trained on labelled wrong-key / misparsed
  / mistranscribed text; the n-gram LMs saw clean text only. An n-gram LM
  trained on the same noised mixture would presumably close part of this
  gap — not measured, and not needed: the prototype judges were never meant
  to be the instrument.

## Caveats

- Window length 1024; the n-gram judges are weaker at the 200-char scale of
  the recovery suite. 48 windows/language (24 for the diffusion ranking,
  8 strata) — accuracies are ±0.05–0.1.
- Direct ranking of a noised text is not the Phase-6 use (ranking
  *decipherments*), but it is the quantity a partial-key search consults
  when it compares language hypotheses mid-run.
- Single noise seed / single source set, by design (paired with 2.6).

## Files

- `scripts/ngram_robustness.py` (`--diffusion tag=name` co-plots saved 2.6
  arrays; `--rank-ckpt name=path` runs the GPU cross-language pass;
  `--replot` redraws from the JSON).
- `DATA_ROOT/analysis/phase2/robustness_ngram.json` (curves, controls,
  `judge_accuracy` with both conventions, excess offsets used),
  `robustness_ngram_windows.npz` (per-window own-language arrays
  `ng{k}/{lang}/{fam}`, per-condition arrays `rank/{judge}/{lang}/{fam}/{sev}`),
  `robustness_ngram.png` (3 × 3: absolute, normalized, accuracy).
