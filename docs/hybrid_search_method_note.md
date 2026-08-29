# Diffusion-guided n-gram key search for high-hapax symbol codes — method note for a literature search

Status: descriptive summary, 2026-08-26. Sources of record: `docs/alt_loop_plan.md`
(§7 proof of life, §8 hapax-masked proposer), `docs/altloop_vms_plan.md` (§12
manuscript run), `docs/wordhom_study.md` (§3.1 feasibility curve). Code:
`diff_voyn/heads/{posterior,altloop,wordhom,homophonic}.py`,
`scripts/{altloop_pol,altloop_vms}.py`.

## 1. Problem setting

We attack **symbol codes with a very long tail of rare symbols**: a ciphertext
alphabet of hundreds to thousands of symbol *types* over a plaintext alphabet of
~23–28 letters (or letters + a few doubled-letter units), where most types
occur once. Concretely:

| instance class | types | tokens per type | hapax types |
|---|---|---|---|
| unigram homophonic (Zodiac-408-class) | 54 | ~7.5 | few |
| word-level homophonic synthetics (Boxer hypothesis, no arithmetic) | 900–1 700 | 8.3 → 3.5 | 14 % → 74 % |
| Voynich MS, word types as symbols (Currier A / B) | 3.6k–5k | 3.0 / 4.6 | 74 % / 69 % |

The key is a map *symbol → unit* (letter, or doubled-letter bigram in the
word-homophonic head). The inner objective is a character **pentagram (5-gram)
n-gram log-likelihood of the decode** plus penalties (rule violations,
homophone-budget terms); the search over keys is simulated annealing (SA) on
random symbol reassignments plus greedy polish, i.e. the standard
homophonic-cipher attack lineage.

The empirical wall (`wordhom_study.md` §3.1): this search recovers the key
essentially perfectly at ≥ 8 tokens/type (SER 0.01–0.03) and fails outright
at ≤ 3.5 (SER 0.73–0.77, hundreds of rule violations, found key thousands of
nats *below* the truth under its own objective — i.e. a **search** failure,
not only an identifiability failure, at 5.6–6.8 tokens/type; at ≤ 4 the n-gram
optimum itself drifts from the truth by 2 000 nats / 10 % of letters, an
**objective** failure).

## 2. The hybrid method

Two models with different inductive biases alternate:

* **Inner / proposer-free tier**: character n-gram objective + SA. Cheap,
  exact, local; gets trapped at rare-type refits.
* **Outer / proposer tier**: a frozen **masked discrete diffusion language
  model** over characters (MDLM-style absorbing-state diffusion, 85M encoder,
  multilingual Latin/Italian/German, 1024-char context). It is the project's
  likelihood instrument (per-language ELBO) and is *not* retrained for search.

One alternation round:

1. **Decode** the ciphertext under the current key into a letter stream.
2. **Posterior read**: mask 30 % of positions at random (16 independent
   draws), run one denoiser forward pass per draw, read the per-position
   letter posterior at masked positions (`heads/posterior.py::position_posterior`).
3. **Aggregate per symbol**: sum log-posteriors over all occurrences of each
   ciphertext symbol → a (symbols × letters) score matrix. Only same-length
   unit reassignments are scored this way; length-changing moves stay with
   the n-gram search.
4. **Disagreement set** `D = {s : argmax P[s] ≠ key[s]}`, ranked by
   occurrence-weighted margin `P[s, argmax] − P[s, key[s]]`.
5. **Re-seed** the top-k symbols of `D` (k = 8 for 54-symbol keys, k = all for
   word-level keys) to either the argmax letter (`posterior`) or a letter
   **sampled** from the per-occurrence posterior (`posterior_sample`) — the
   n-gram objective is *not* consulted.
6. **Short n-gram SA** from the re-seeded key (T 2.0→0.3; 50k steps symbol
   heads, 200k word-level) + greedy polish: the n-gram side may undo what it
   disagrees with.
7. **Accept the round iff the n-gram penalized objective improved.** The
   judge proposes, the n-gram disposes. ≤ 6 rounds, patience 2.

Controls run on every cell from the same start key with the same seeds:
`rand-k` (same number of uniformly random symbols → random units), `none`
(the short SA alone), and a **null arm** starting at the true key (the
proposer must not move a correct key). For 1:1 substitution the proposal is
restricted to pair swaps to preserve the bijection.

A tested variant (§8 of the PoL doc): **hapax masking** — letter positions
emitted by hapax types are additionally forced to MASK in every posterior
draw, so the denoiser does not condition on guesses that carry no evidence.

Design points that matter for comparison with other hybrids:

* the