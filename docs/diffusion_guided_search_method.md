# Diffusion-guided n-gram search for high-hapax symbol codes

> **Record status (banner added 2026-09-01):** cold-read method note for the posterior re-seeding stage of the loop (2026-08-25), with a §4 correction (2026-09-01) stating the findability wall by solver stage.
> Still current: §2–§3 mechanism, §4 "works / limits" as measured for re-seeding alone, §5 literature pointers. Superseded: the §1 table and the round budget (plain SA / proof-of-life settings); "judge in the acceptance rule … not done" (tested 2026-08-28, no gain; the hapax-wildcard objective and its anneal, `docs/alt_loop_plan.md` §8.4–8.6, are what moved the wall to ≈ 4 tokens per type). **Current project position: `docs/project_status.md`.**

A method note, written to be read cold. It describes what we built, what it
does and does not do, and the vocabulary needed to find the neighbouring
literature. Records: `docs/alt_loop_plan.md` §7–8 (proof of life and the
hapax-masking variant), `docs/altloop_vms_plan.md` §12 (manuscript run),
`docs/wordhom_study.md` §3 (the identifiability curve this attacks); later
stages (added 2026-09-01): `alt_loop_plan.md` §8.4–8.7 (hapax-wildcard
objective, judge in the acceptance rule, wildcard → anneal, judge vs SER),
§9 (small-commitment loop, refuted), §10 (manuscript-shaped control
battery); `altloop_vms_plan.md` §13 (wildcard → anneal on the manuscript,
24/24 NOISE); `docs/project_status.md` §3 (the wall by solver stage). Code:
`diff_voyn/heads/{posterior,altloop}.py`, `scripts/altloop_pol.py`,
`scripts/altloop_vms.py`.

## 1. The problem

A **symbol code** maps ciphertext symbols to plaintext units under an unknown
key: simple substitution (bijection), homophonic substitution (many symbols →
one letter), or *word*-homophonic codes where the ciphertext units are whole
word tokens standing for letters or short letter groups — the historical
"nomenclator"/code-book shape. Recovering the key without a crib is a
combinatorial search whose objective is a plaintext language model, classically
an n-gram log-likelihood optimized by hill-climbing, simulated annealing, or
EM over the key.

That search is reliable as long as each key entry is *evidenced*: every symbol
occurs often enough that the objective can distinguish its correct assignment
from its neighbours. Our measured break point, on synthetic word-homophonic
ciphers with ground truth (8 000 plaintext letters, key size varied; **under plain multi-restart SA**, `wordhom_study.md` §3.1 — see the §4 correction for the current solver's wall):

| tokens per type | hapax types | letter SER after 1.5M SA steps |
|---|---|---|
| ≥ 15 | ≤ 11 | 0.004 – 0.012 |
| ~8 | ~130 | 0.028 – 0.032 |
| ~3.5 | ~690 | 0.73 – 0.77 (found key *below* the truth in objective) |

The failure at the bottom row is not a search failure that more restarts fix —
at 3.5 tokens per type the true key is no longer the optimum of the n-gram
objective (found − truth = −2 600 to −8 400 nats). This is the **high-hapax
regime**: most key entries are supported by a single observation. It is where
short historical ciphers, code-books, and the Voynich manuscript's word
statistics (3.0 tokens/type in Currier A, 4.6 in B, ~70 % hapax types) live,
and where the standard toolchain is silent.

## 2. The instrument: a masked diffusion model as a plaintext judge

Instead of (or alongside) an n-gram model, we use a character-level **masked
(absorbing-state) discrete diffusion** language model — MDLM-style, SUBS
parameterization, Rao-Blackwellized NELBO, trained jointly on Latin, Italian
and German with a per-position language embedding, then frozen. Two things it
gives that an n-gram does not:

1. a per-language likelihood **bound** (NELBO in bits/char) usable to rank
   candidate decodes and their languages;
2. a **per-position posterior over letters**: mask a fraction of the current
   decode, run one forward pass, and read what the model thinks each masked
   position should have been, conditioned on the surrounding (still hypothetical)
   plaintext. This is the denoiser doing exactly the job it was trained for, and
   it is the signal the search lacks.

## 3. The method: the judge proposes, the n-gram disposes

One **alternation round**, given a current key:

1. **Decode** the ciphertext under the current key to a hard letter stream.
2. **Posterior read** — for `n_draws` (16) independent masks at rate 0.3, one
   forward pass each; accumulate mean log-posteriors per position, then sum per
   *cipher symbol* (or per word type) over its occurrences. Cost is a handful of
   forward passes, not a gradient.
3. **Disagreement set** `D = {s : argmax posterior(s) ≠ key(s)}`, ranked by
   occurrence-weighted margin.
4. **Re-seed** the top-`k` entries of `D` (k ∈ {4, 8, all}) *without consulting
   the n-gram objective*. Two proposers: `posterior` (take the argmax letter)
   and `posterior_sample` (sample the letter from the per-occurrence posterior).
5. **Short annealing** of the n-gram search from the re-seeded key (SA, T 2.0→0.3,
   50k–200k steps, then greedy polish).
6. **Accept** the round iff the *n-gram penalized objective* improved.

Repeat ≤ 6 rounds with patience 2 *[round budget superseded: 32 rounds is a free gain (`alt_loop_plan.md` §9.1, 2026-08-27), and the solver of record runs ≤ 96 rounds / patience 10 (wildcard) then ≤ 80 / patience 10 (anneal), §8.4–8.6; the `--rounds 6 --patience 2` CLI defaults still encode this proof-of-life setting]*. The asymmetry in step 6 is deliberate: the
diffusion model is a proposal distribution over key edits, never the acceptance
criterion, so the loop cannot import the judge's own tolerance of wrong symbols
(we have a recorded case where the judge's preferred key is worse cryptanalytically).
Ground truth adjudicates after the fact.

**Controls, run in every cell.** A same-size *random* re-seed (same k, random
symbols, random units), an SA-alone arm (no kick), and a **null** arm started at
the true key (which must not move). Without the size-matched random arm the
result is unreadable: any kick plus annealing sometimes escapes a trap.

**Variant tested and rejected.** Since the decode's hapax positions are the
key's guesses about types with no evidence, we tried withholding them —
additionally masking every letter emitted by a hapax type before the posterior
read (`position_posterior(force_mask=…)`). Effect at 3 % hapax tokens: none. At
11 %: language-dependent and within proposer noise (helps Italian/Latin by
0.3–1.2k nats, hurts German by 0.7–1.3k), no trap escaped. Masking removes bad
context but also removes context the denoiser needs for the *non*-hapax
positions; the effects cancel. Not adopted.

*[Hapax handling was adopted in a different form (2026-08-27/28): hapax types as wildcards in the n-gram *objective* — charged a constant and frozen out of the search until the frequent types converge, then annealed back in — which is what broke the A-like traps (`alt_loop_plan.md` §8.4, §8.6).]*

## 4. What it does and does not do

**Works — the mechanism is real and it is information, not perturbation.**
Under a simulated annealer too cold to escape anything on its own (0/24 escapes),
posterior re-seeding escapes 10/24 recorded traps on 54-symbol homophonic
instances versus 6/24 for a same-size random kick. The decisive comparison is on
word-homophonic cells at 6.6 tokens per type that the search had recorded as
unsolvable: posterior re-seeding of ~500 types per round takes German 0.425 →
0.026 and Italian 0.421 → 0.047 letter SER (the n-gram objective's own optimum
near the truth), while a *512-type random* re-seed produces, in six rounds, zero
keys the annealer can improve on. Same kick size, opposite outcome.

**Limits, all measured.**
- *Blind on badly-wrong keys.* The posterior identifies *which* symbols are wrong
  at 0.62–0.88 precision even from a garbage decode, but its proposed *letter* is
  exactly right ~0 % of the time there — with no readable context the denoiser has
  nothing to condition on. Hence the argmax proposer stalls deterministically
  (it re-proposes the rejected set); sampling fixes the stall.
- *Objective traps are untouched.* Where the true key is *below* the n-gram
  optimum — which is the defining property of the high-hapax regime — no proposer
  helps, because acceptance is on that objective. Putting the diffusion bound
  into the acceptance rule was tested 2026-08-28 with no gain (`alt_loop_plan.md`
  §8.5); what worked instead was changing the n-gram objective — hapax types as
  wildcards, then annealed back in (§8.4, §8.6) *(corrected 2026-09-01;
  originally read "Fixing this requires putting the diffusion bound into the
  acceptance rule, which we have not done")*.
- *It did not transfer to the target regime.* On 72 Voynich manuscript cells
  (3 heads × 3 languages × dialects/transcriptions) at 3.0–4.6 tokens per type,
  the guided arm is indistinguishable from both controls: 62/72 cells end on the
  identical key, the only accepted rounds *lower* the structure margin, nothing
  crosses any pre-registered threshold. Read as "the method did not find it",
  not as evidence of absence.
  *[Also true of the solver of record: the wildcard → anneal pipeline on the 12 wordhom manuscript cells returned NOISE 24/24 (2026-08-29, `altloop_vms_plan.md` §13).]*

The honest summary for re-seeding *alone*: **it buys roughly one octave of key
sparsity** — it converts 6–8 tokens-per-type instances from unsolvable to solved,
and on its own does nothing at 3–5, where the objective rather than the search
is the binding constraint.

*Correction (2026-09-01).* The sentence above was written before the
hapax-wildcard objective and its anneal (`docs/alt_loop_plan.md` §8.4–8.6,
§10.1), which moved the wall again. The findability wall as measured, by
solver stage, on synthetic word-homophonic ciphers with ground truth:

| solver | tokens per type where the key is recovered |
|---|---|
| plain multi-restart SA (`WordHomophonicHead.solve`) | ≥ 8 (SER 0.01–0.03); fails at ≤ 5.7 (`wordhom_study.md` §3.1) |
| + posterior re-seeding (this note) | 6.6 → SER 0.026 / 0.047 |
| + hapax-wildcard objective → anneal (current solver, `altloop_vms.py --wild --wild-anneal`) | B-like 5.5–5.6 → SER 0.026 / 0.036 / 0.070; **A-like and `nodouble` 4.1–4.4 → 0.03–0.12**; A-like German (margin 2.13) and Latin (1.70–1.72, 3 seeds) called by the frozen judge, Italian not (1.39; its truth scores 1.56) — `analysis/altloop/judge_at_ser.md`, tabulated in `alt_loop_plan.md` §10.1 (A-like table) and in the `d5` reference column of `wordhom_bigram_variant.md`; `nodouble` de/la called in §10.1 |
| any stage | ~3.5 → SER 0.73–0.77 with the found key *below* the truth in objective; Naibbe real-vs-shuffled twins at 3.4 indistinguishable |

So the current wall is **≈ 4 tokens per type**, not 6–8: Currier B (4.6) is
inside the regime where clean manuscript-shaped positives solve and still
returned NOISE on 24/24 runs, while Currier A (3.0) sits below it, where the
true key is no longer the n-gram optimum. Anything that shortens the text
(windowing, per-page solves) lowers tokens per type and moves *away* from the
wall — a 1024-token window of the manuscript has 1.75–2.0 tokens per type.

## 5. Where to look in the literature

### 5.1 Hybrid neural-guidance-over-combinatorial-search
The shape here is *learned proposal distribution inside a classical
metaheuristic*, with the classical objective retained for acceptance. Search terms:
"neural-guided combinatorial optimization", "learned local search / neural
large-neighborhood search (NLNS)", "learn-to-improve heuristics", "neural
destroy-and-repair operators", "GNN-guided branching", "policy-guided simulated
annealing", "learned proposals for MCMC / neural adaptive MCMC", "amortized
proposal distributions". The step-4/5 structure is literally *destroy-and-repair*
with a learned destroy set. Related framings: Gibbs-with-a-learned-kernel, and
sequential Monte Carlo with neural twists.

### 5.2 Neural cryptanalysis of classical ciphers
"Neural substitution cipher cracking", "seq2seq / transformer decipherment",
"unsupervised decipherment", "neural machine translation for decipherment",
"Bayesian decipherment", "slice sampling for decipherment", "beam search over
key space with an LM scorer", "homophonic cipher solver", "nomenclator /
code-book decipherment". Landmark systems to trace forward and backward from:
EM-based decipherment of substitution ciphers, Bayesian decipherment with
Gibbs sampling, the Copiale and Zodiac-408/340 solutions, and n-gram
hill-climbing solvers (`AZdecrypt` and its heuristics literature). The specific
contribution to compare against is *which* model scores candidate keys —
character n-gram vs. neural LM vs. masked/diffusion LM — and *where* in the loop
it is used (rescoring a shortlist vs. proposing edits).

### 5.3 Masked / diffusion language models as scorers and infillers
"Masked diffusion language model (MDLM)", "absorbing-state discrete diffusion",
"D3PM", "SEDD", "any-order autoregressive models", "pseudo-log-likelihood scoring
with masked LMs", "MLM as a generative / energy-based model", "text infilling".
Relevant question for a reader: the per-position posterior we exploit is exactly
the conditional a masked LM is trained to emit, so any masked LM would give *a*
signal; what the diffusion training adds is a principled sequence-level bound
(NELBO) to rank whole decodes and languages.

### 5.4 High-hapax benchmarks and the identifiability question
This is the part with the thinnest literature and the most room. Terms:
"unicity distance" (Shannon) and its extensions to homophonic and code-book
ciphers; "identifiability of substitution ciphers"; "sample complexity of
decipherment"; "Zipf's law / hapax legomena / vocabulary growth (Heaps' law)";
"type-token ratio"; "large-vocabulary rare-word estimation", "Good–Turing /
Kneser–Ney smoothing for unseen types"; "one-shot and zero-shot lexicon
induction"; "low-resource / extremely low-resource lexicon induction";
"unsupervised bilingual lexicon induction under sparsity". Concrete corpora and
tasks that instantiate the regime: historical cipher collections (DECODE
database — Copiale, Borg, Zodiac, Ramanacoil-type nomenclators), the Voynich
manuscript, undeciphered scripts with tiny corpora (Linear A, Indus, rongorongo),
and OCR/handwriting transcription of code-books. A grad student looking for a
*benchmark* should note that we had to build ours: sweep a synthetic
word-homophonic generator over tokens-per-type at fixed plaintext length and
report recovery vs. that ratio (§1 table). That curve — recovery as a function of
evidence per key entry — is the missing standard axis; existing cipher benchmarks
report accuracy at a fixed, comfortable ratio.

### 5.5 Evaluation hygiene worth importing
Two habits this project found necessary and that the search terms above rarely
surface: (a) **size-matched random controls per instance**, since any perturbation
plus annealing escapes some traps; (b) a **null arm started at the truth**, which
catches the case where the objective's optimum is not the true key — the
diagnostic that told us the high-hapax failure is an objective failure, not a
search failure. Related reading: "pre-registration", "adversarial controls",
"winner's curse in selection under noisy estimates" (relevant when a polish step
takes an argmin over many noisy score estimates).
