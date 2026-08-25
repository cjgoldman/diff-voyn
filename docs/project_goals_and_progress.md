# Language identification of the Voynich Manuscript by trial decipherment — goals and progress

*A cross-disciplinary account of the `diff-voyn` project: what it set out to do, how far it has come, and how the current state compares with the aspirational abstract in `reference_docs/A Diffusion-Based Framework for Language Identification of Voynich-Like Ciphertext.md`.*

*Status date: 2026-08-24. Written for readers who are comfortable with careful argument but may not know the vocabulary of machine learning or statistics. Technical terms are introduced where they are first needed; a short glossary closes the document. Specific numbers are drawn from the per-phase status records in `docs/` (listed at the end), which remain the authoritative source.*

---

## 1. Summary in one page

**The question.** The Voynich Manuscript is a 15th-century book written in an unknown script that has resisted every attempt at reading. One family of hypotheses holds that it is an ordinary European language written in a cipher. If that is true, then *which* language? This project builds an instrument to answer that narrower question — not to read the manuscript, but to rank candidate languages (and candidate cipher systems) by how well they explain its text.

**The method.** "Language identification by trial decipherment": for each candidate language, try to decipher the text *as if* it were that language, then measure how language-like the result is. Even a partial, imperfect decipherment of the true language should look more like real text than the best decipherment under a wrong language. The measuring device is a large statistical model of three languages (Latin, Italian, German), and the decipherers are a set of four "cipher heads", each specialised to one class of cipher, from the simplest (one symbol per letter) to two recently proposed systems designed specifically to produce Voynich-like text.

**Where things stand.** The project was planned as seven phases (0–6) separated by six hard checkpoints ("gates"). **All seven phases are complete** (gates G0–G5 passed, the last on 2026-08-23; Phase 6 finished 2026-08-24). That is: the three-language model is trained and frozen; its measurements have been shown to be fair across languages to a stated precision; it recovers the true language of synthetic ciphertexts at 99% for texts of 200 or more characters; all four cipher heads have been validated on synthetic ciphertexts of their own class; and the instrument has been applied to the manuscript.

**The result: the instrument abstains.** Across 87 (cipher × language × window × transcription) cells — four cipher classes, three languages, both Currier dialects, two EVA transcriptions plus Boxer's independent glyph transcription — **not one produces a decipherment that behaves like language.** The decisive measurement is the *structure margin*: how much more predictable a decipherment is than the same letters shuffled. Every manuscript cell scores 0.04–1.25 bits per character on that margin; nine genuine decipherments put through the identical pipeline score 1.49–2.48, and Voynich-mimicking gibberish scores 0.92–1.51. The manuscript's best attempts sit inside the gibberish band and below every true decipherment, so the verdict does not depend on where the pre-registered threshold (1.5) was drawn — any cut between 1.26 and 1.48 gives the same answer. The language ranking *among* those non-decipherments is noise (margins of 0.001–0.083 bits per symbol against a calibration uncertainty of 0.07–0.19; heads, transcriptions and dialects disagree) and is reported as such. One structural finding survives: a Naibbe-shaped word grammar describes **Currier A** — but not Currier B — slightly more compactly than the manuscript's own letter statistics do, while yielding no readable plaintext. That is a statement about word structure, not a decipherment. The Phase-6 acceptance check records a **FAIL on one sub-criterion** (gibberish abstention 8/9 = 0.89 against the required 0.95, one instance at margin 1.51 against the frozen 1.5) and the rule was deliberately *not* re-tuned after the fact.

**How this compares with the abstract.** The abstract was written aspirationally, before the work. Most of its infrastructure claims are now substantiated, often with sharper numbers than it promised. Three things have changed in character, and the headline result must be replaced:

1. The methods are not "end-to-end neural" in the way the abstract implies. The neural model turned out to be an excellent *judge* of candidate decipherments but a poor *search engine* for them; the working system is a hybrid — a cheap classical search proposes keys, the neural model ranks and polishes them. This is not a compromise but a consequence of a contradiction at the heart of the method, explained in §2.7: a score that can *guide a search* toward the key must be sensitive to key errors, and a score that can *fairly judge* the result must not be.
2. Robustness to noise, which the abstract credits to the model architecture, is mostly attributable to *how the model was trained* (on deliberately corrupted text), not to the architecture itself.
3. The raw likelihood score the abstract names as the language metric is, by itself, *unsafe* for the more complex ciphers: it can be fooled by degenerate decipherments. The metric actually used is a "description-length" total that charges a hypothesis for its complexity. This was discovered and fixed in Phase 5.
4. The abstract's headline result — "Germanic candidates receive the highest likelihood … complex heads outscore simple ones" — **is not what was found, and must be replaced**. The instrument abstains: no cipher hypothesis in the inventory yields a language-like decipherment of the manuscript. The second half of the headline is partly borne out in a weaker form — the most complex head (Naibbe) does win the description-length comparison — but it wins by compressing the manuscript's word structure, not by deciphering it (§7). Of the two Voynich-tuned ciphers, the arithmetic one fails to match the manuscript on its own defining signature before any solve is run (§7.3).

The rest of this document explains the concepts needed to read the results, then walks the phases and the comparison in detail.

---

## 2. The idea, without jargon

### 2.1 Trial decipherment

Suppose you are handed a text in an unknown cipher and asked only "is this French or German underneath?" You do not need to finish the decipherment to answer. You can try to solve it as French, try to solve it as German, and compare: the attempt made under the correct language will produce output that looks increasingly like real text, while the attempt under the wrong language will plateau at something that looks like noise. Cryptanalysts have used this "guess and check" idea for decades; its weakness has always been that it needs a reliable and *even-handed* way to measure "looks like real text", and a decipherment engine for each kind of cipher.

The abstract proposes to supply both with modern machine learning: a single statistical model that knows several languages at once, serving as the measurement device, and a family of adjustable decipherment components ("cipher heads") matched to the cipher systems most plausible for the Voynich text.

### 2.2 Measuring "looks like real text": bits per character

A *language model* is a program that, shown part of a text, assigns probabilities to what the hidden parts are. Its quality on a passage is summarised as **bits per character**: roughly, the average number of yes/no questions the model would need to guess each hidden character. A model that knows Latin well needs few questions on a Latin passage (it is rarely surprised); on scrambled letters it needs many. For orientation, in this project real clean text scores about **1.9–2.6 bits per character** depending on the language; the same letters shuffled into random order score about 4; a 25-letter alphabet with no knowledge at all would be about 4.6.

This single number is the instrument's reading. Lower is "more like this language". Ranking candidate languages means scoring the same decipherment attempt under each language and ordering the numbers. Because it is a difference between readings on closely related texts that matters, not the absolute readings, the project goes to considerable trouble to make the readings comparable (§2.5).

### 2.3 The model: a multilingual "fill in the blanks" machine

The model at the centre of the project is a **masked diffusion model** over characters. The unfamiliar name describes a simple training game: take a passage, hide a random fraction of its characters (sometimes 10%, sometimes 90%), and train the model to fill them back in correctly. Do this billions of times across three languages, and the model becomes a dense record of what character sequences are plausible in each language. Scoring a passage means hiding characters in many random patterns and recording how well the model predicts them, averaged.

Three properties explain why this kind of model was chosen rather than the more familiar "predict the next character" model:

- It looks at context on **both sides** of a gap, so it can score any position in a text from its surroundings.
- It is trained on **all three languages together**, with a language label attached to each passage. One model, three settings of a dial. This symmetry is a deliberate fairness property (§2.5): no language gets a better-built model than another.
- It can accept **"soft" inputs** — "this symbol is 70% likely to be *e* and 30% *a*" — which was intended to let the cipher heads be tuned by gradient descent through the model. (As §5 explains, this capability was verified to work mechanically but turned out not to help.)

The model has about 85 million adjustable parameters and reads text as sequences of 25 letters (the Latin alphabet without *j*, plus a handful of bookkeeping symbols). **All spaces are removed from every text** — training corpora, test ciphers, and the manuscript alike — so the model works on unbroken letter streams. This is a design decision: word boundaries in the manuscript are themselves a contested matter, and the methods should not depend on them.

### 2.4 The score is a bound, and why that matters

This is the one piece of statistical machinery the reader must understand, because most of Phase 3 is about it and it shapes how every result is reported. It is also, in our experience, the hardest idea in the project for non-specialists, so this section goes slowly, starting from the problem we are actually trying to solve.

**The problem, restated.** A cipher head has produced a candidate decipherment — a stream of letters that *might* be Latin. We need a number that says how much it looks like Latin, and a comparable number for Italian and for German, so that we can rank the three. The number we use (§2.2) is the cost, in yes/no questions, that a fluent expert in the language would pay to guess the text. A good decipherment of Latin should be cheap for the Latin expert to guess and expensive for the others. Everything therefore depends on *how the expert is asked to guess*, and that is where the subtlety lies.

**Two experts, two ways of guessing.** Imagine two experts, both fluent in English, both asked to guess the word *bread*.

*The Reader* is shown the letters one at a time, left to right, and guesses each next letter from the ones already revealed: the first letter knowing nothing; the second knowing *b*; the third knowing *br*; and so on. Their cost is the total of the five guesses — say, for illustration, 9 bits.

*The Filler* plays hangman instead. A random selection of the letters is hidden, the rest are shown, and the Filler must guess the hidden ones from whatever happens to be visible:

| what the Filler sees | must guess | how hard (illustrative) |
|---|---|---|
| `b _ e a d` | the second letter | easy — *r* is nearly certain; a fraction of a bit |
| `_ r e a _` | first and last letters | moderate — *bread*, *tread*, *dream*, *break*… a couple of bits each |
| `_ _ _ _ _` | everything | hard — the whole word with no help; several bits per letter |
| `b r _ a d` | the third letter | easy again — *e* or *o* (*broad*), about one bit |

The Filler's cost is the *average* over many random hiding patterns and many different fractions hidden. The conventional left-to-right language models of the literature are Readers. The diffusion model at the centre of this project (§2.3) is a Filler — it was trained by exactly this hangman game, billions of times, in three languages — and it is a Filler because of what a Filler can do that a Reader cannot: it can judge any letter from both sides, it can accept "soft" letters from a cipher head, and one Filler can hold all three languages behind a single dial. (On the manuscript it is asked the same hangman questions on windows of 1,024 characters, 64 random hiding patterns each.)

**The fact that links the two experts.** Suppose the Filler's opinions were perfectly *coherent* — their guess for *r* given `b _ e a d`, their guess for *b* given `_ r e a d`, and all their other guesses fitted together into one consistent view of what English words look like. Then a mathematical fact takes over: for a coherent guesser, the cost of guessing a word piece by piece is the same whatever order the pieces come in, hidden at random or revealed left to right. The Filler's average would equal the Reader's 9 bits exactly.

But the Filler's opinions are learned separately for each pattern of hidden letters, and in practice they never fit together perfectly. It can be shown that whenever they fail to cohere, the Filler's average comes out *higher* than the coherent figure — never lower. So the Filler might report 10 bits for *bread* where the Reader reports 9. The Filler's number is a **bound**: guaranteed to be at least as pessimistic as the truth. The jargon name is ELBO, "evidence lower bound" — a lower bound on probability, which is the same thing as an upper bound on surprise in bits. The extra bit in the example is the bound's **slack**.

**Why slack is a problem for *this* project.** A consistently pessimistic ruler is harmless as long as it is equally pessimistic everywhere — differences still come out right. Here, though, we are not measuring one text with one expert; we are comparing the Filler's score on a decipherment candidate with the Latin dial set, against the Italian dial, against the German dial. Each dial is, in effect, a different Filler with its own degree of incoherence and therefore its own slack. Nothing in the training guarantees the three slacks are equal.

Put numbers on it. Suppose a candidate text is, in truth, equally good Italian and German — a Reader in each language would charge it the same 2.30 bits per character. If the Italian Filler's slack is 0.01 bits per character and the German Filler's slack is 0.21, the raw scores we can actually compute read Italian 2.31, German 2.51, and a ranking built on them calls the text Italian by a comfortable 0.2 bits per character — a verdict produced entirely by the ruler, not the text. That is the failure the design calls "bound-tightness fairness" and treats as its first non-negotiable.

**So the slack is measured.** For each language a Reader is trained on the very same text as the Filler, and both are scored on untouched test text in that language. The Reader's figure is exact by construction; the gap between the Filler's bound and it is recorded as the language's **calibration offset**. In the current model the offsets are Latin +0.14, Italian +0.01, German +0.21 bits per character — the numbers of the example above were not invented. The bound is loosest on German, tightest on Italian, and the spread between them is roughly the size of a decisive margin in a close comparison. (One honesty note the audit insists on: the Reader is a *different* model, so the gap mixes genuine slack with any difference in skill between the two experts, and changing the Reader changes the measured offsets. They are estimates of slack, not proofs of it.)

**What is done with the offsets — and why they are not simply subtracted.** The obvious remedy is to subtract each language's offset from its score, and the original design said to do exactly that. Phase 3 tested it and found that it *breaks* the instrument (§5). The offsets are measured on clean, genuine text in each language; but the comparison that actually decides a ranking is between two decipherment attempts of the *same* ciphertext, whose outputs are nearly identical texts differing by margins as small as 0.03 bits per character. An offset of 0.1–0.2 overwhelms a margin of 0.03 and flips correct answers to wrong ones. The adopted policy is to leave the raw scores alone and report the offsets as an **error bar**: any margin between two languages smaller than the uncertainty their offsets imply (0.07–0.19 bits per character, depending on the pair) is labelled "unresolved at calibration precision" rather than counted as a win. A reader of the final table should look at that flag before the point estimate.

### 2.5 Fairness, the central discipline

A recurring theme of the design is that the result is only worth publishing if the instrument is even-handed. Concretely, the project forbids anything that would make one language's score systematically better for reasons unrelated to the text:

- all languages trained together from the first step, never one after another;
- one shared alphabet and one shared text-cleaning procedure (no per-language shortcuts that lose information);
- identical scoring conditions for each language on the same text — the *same* random hiding patterns are reused across language settings, so that the comparison is between the model's answers, not between two different random draws (the statistician's "common random numbers");
- every number the instrument produces is carried with an error bar: the spread across text windows, the flip-rate across repeated random draws, and the calibration uncertainty of §2.4.

### 2.6 The four cipher classes

A *cipher head* is a component that holds a candidate key and turns ciphertext into candidate plaintext. Four were built, in increasing difficulty:

| rung | cipher class | what it does | why it is here |
|---|---|---|---|
| 1 | **simple substitution** | each cipher symbol stands for exactly one letter | the textbook case; a sanity check |
| 2 | **homophonic substitution** | a letter may be written with any of several symbols (common letters get more), so that symbol frequencies are flattened | the classical "hard" case; the Zodiac-408 cipher is of this class |
| 3 | **Naibbe cipher** (Greshko 2025) | a hand-executable system that writes single letters *or pairs* of letters as Voynich-like word-pieces chosen by a card draw; shown to turn Latin and Italian into text with many Voynich statistical signatures | one of two recently proposed ciphers explicitly tuned to produce Voynich-like output |
| 4 | **arithmetic sum-to-target** (Boxer, voynich-attack) | each letter is a number; it is written as a short word whose symbols' values add up to that number, with the choice of word drawn to match Voynich word statistics | the other Voynich-tuned proposal |

**How the heads are used.** No head, and no part of the model, chooses a language. Every ciphertext is deciphered *three times*, once under each language hypothesis, as a fixed protocol: the search for a key must be steered by a model of *some* language (the classical letter-statistics model of that language drives the search; the diffusion judge is then set to the same language's dial to score the result), so the only fair procedure is to run it under every hypothesis and compare the three outputs. For any given ciphertext, then, one of the three attempts is made under the true language and two under a wrong one. When this document speaks of a "wrong-hypothesis decipherment" it means one of those two — an attempt the protocol always makes, not a choice anything made.

Rungs 3 and 4 are *verbose* ciphers: one plaintext letter becomes several ciphertext symbols, and the encipherer makes choices along the way. That freedom is what makes the later description-length argument (§2.8) necessary.

### 2.7 Two jobs that one score cannot do: finding the key and judging the result

The method has two distinct jobs, and it is worth separating them before going further, because the single most consequential design fact of the project — and the one reviewers most often question — is that the two jobs make **contradictory demands** on a score.

**Job 1: finding the key.** A cipher head starts with no idea which symbol stands for which letter and must search through an astronomically large space of possible keys (for a 25-letter simple substitution, about 10²⁵ of them). No search can try them all; every practical search is a form of hill-climbing: change the key a little, ask "is this better or worse?", keep the change if better, repeat — many thousands of times. For this to work the score must have a **slope** everywhere, including far from the answer: a key that is 70% wrong must score noticeably worse than one that is 50% wrong, which must score worse than one that is 30% wrong. Walking uphill in fog only works if the ground actually slopes.

**Job 2: judging the result.** Once each hypothesis's search has stopped, the decipherments are compared across languages. Here the requirement is the opposite. The three searches (§2.6) do not stop at the same place: finding the Latin key is measurably harder than finding the Italian or German one (Phase 5: 93–97% vs 100% solved), and on the manuscript, where no hypothesis may reach a clean key, every decipherment will be partial. A fair ranking therefore needs a judge whose verdict "this is Latin" is the *same* whether the key it is shown is 0% wrong or 40% wrong — a judge that is **insensitive to key errors**. Otherwise the ranking would reward whichever hypothesis happened to search more successfully, which is not a fact about the manuscript.

**One number cannot be both.** Slope toward the key *is* sensitivity to key errors. A score that changes steadily as the key goes from fully wrong to fully right is, by that very fact, a score that depends on how far the search got. The two requirements are not in tension; they are negations of each other. Schematically:

| key wrong by | what a *search target* must report | what a *fair judge* must report |
|---|---|---|
| 100% | very bad | "no language" |
| 70% | bad | — still "Latin", ideally |
| 50% | middling | "Latin" |
| 20% | nearly good | "Latin" |
| 0% | best | "Latin" |

The search target must distinguish every row; the judge must refuse to distinguish the lower four.

**The project has one instrument of each kind, and they were measured against each other.** The classical *n-gram* model — a table of how often each letter follows each sequence of four letters in a language — has slope all the way down: its cost keeps climbing as the key gets worse, even past the point where the text has less structure than shuffled letters. That makes it a good search target. It also makes it a *biased judge*: on a partially wrong key its language verdict drifts toward "German" (the language whose table is most forgiving), and the drift grows with the error, so no fixed correction removes it. The *diffusion* model is the mirror image. Phase 2 trained it on text with 20–50% of positions under a wrong key, labelled with its true language, so it learned to see *through* key errors: by a 50%-wrong key its score has stopped moving, and its language verdict is flat across that whole range. It is a fair judge and a useless search target — **by construction, not by accident**. The clean-trained model from Phase 1, which had not seen wrong keys, was as biased as the n-gram; the robustness was put in deliberately.

**Consequences that run through the rest of this document.**

- The system is *two-tier*: the n-gram drives the search and proposes a shortlist of candidate keys; the frozen diffusion model judges the shortlist. Each instrument does the job it is suited to.
- The diffusion model *does* discriminate in the last 20–30% of the key — that is where a judge needs to be sharp, and it is where the model is used to polish a nearly-right key by scoring discrete single-symbol changes (Phase 5). What it cannot do is guide a search from a random start, and Phase 5's finding that gradients through the model never improve a key is consistent with this, not a surprise.
- "Why not train the diffusion model to be a better search target?" answers itself: that would be training in the sensitivity to key errors that was trained out, and it would make the judge as biased as the n-gram. Making the judge and the search share weights, or updating the judge on the search's errors, would additionally let the search's Latin asymmetry leak into the instrument that is supposed to measure Latin-ness.
- The argument above is mechanistic, and the residual risk it leaves is real: the *judge* is symmetric, but the *search* is not, and the search is where the language hypothesis enters. The two control experiments in §9 (item 6) are there to test the contradiction directly and to measure how much the search asymmetry matters.

### 2.8 Description length: charging a hypothesis for its freedom

A verbose cipher head has a great deal of freedom, and a score that only asks "how language-like is the output?" can be gamed. Phase 5 found the concrete case: a homophonic head can map 44 of 54 symbols to a *single* letter, producing output like "eeeeaeeee…" that any language model rates as extremely predictable (1.4 bits per character — better than real Latin). The output is not language; it is a degenerate explanation.

**Why an expert judge does not simply reject it — predictable is not the same as typical.** A reader may object that no Latin scholar would mistake "eeeeaeeee…" for Latin, and that is true — but the judge was never asked "is this Latin?". It was asked "how predictable is each character from its surroundings?", and on that question the honest answer is *very*: once the visible characters are all *e*, guessing *e* for the hidden one is right almost every time. A perfect Latin density would in fact charge such a string heavily (Latin never has four *e*'s in a row), but a trained model is not a perfect density on text far from anything it has seen, and the extrapolation that comes naturally to a fill-in-the-blanks model is pattern copying. The classical n-gram judge reaches the same verdict by a different route. Nor could training have fixed this: the density model was trained on clean and wrong-key-corrupted language (which keeps a full-width alphabet), and the gibberish and shuffled text used to teach the language head to abstain were deliberately kept *out* of the density's training, because a density model can only learn that what it is shown is normal — showing it repetitive text would make repetition cheaper still. Predictability and language-likeness are different quantities, and they come apart exactly on degenerate inputs. The repair therefore cannot be a better judge; it has to be an accounting rule that charges the *hypothesis* for the freedom it used to manufacture the repetition. (The project does also have an instrument that asks the typicality question directly — the language head of §2.9 — but, as that section explains, it cannot do this job either.)

That rule is an old principle from information theory, **minimum description length** (MDL): the best explanation of a text is the one that lets you *transmit* it most cheaply, counting everything — the plaintext (in bits per character under the language model), plus the key, plus every choice the encipherer made that the recipient would need to know to reproduce the ciphertext exactly ("choice bits"). The degenerate map saves bits on the plaintext but pays far more in choice bits (which of 44 symbols was used at each position?). On the total, the true key wins comfortably. Every cell of the final (cipher × language) table is ranked on this total, expressed per symbol of ciphertext so that heads that produce different plaintext lengths can be compared.

### 2.9 The second instrument: a language head that asks "is this any of my languages?"

§2.8 left a gap. The density judge answers "how predictable is this text?", and we saw that predictability is not the same as being Latin. The project does have a second instrument that asks the typicality question directly: the **language-detection head** of the abstract's second contribution. It is a small classifier bolted onto the same diffusion model (it reads the model's internal representation of a passage, not the raw letters) and trained to answer a four-way question — *Latin, Italian, German, or none of these?* The fourth answer, **abstain**, is what the density judge cannot give. It was taught on exactly the material that was kept out of the density's training (§2.8): Voynich-mimicking gibberish and shuffled real text. On those controls it abstains at least 95% of the time, and 100% on passages of 200 characters or more. So the head is where the scholar's reflex — "that is not Latin" — lives in the system.

It would be natural to hope that the head could therefore police the cipher heads: reject any decipherment that is not real language. Phase 4 tested that and found the hope misplaced, for a reason that is instructive. Because the head was trained (deliberately) on *corrupted* language as well as clean, so that it would still name the language of a half-finished decipherment, it is robust to wrong keys — at a 20%-wrong key it still names the language 99% of the time. That is a virtue for its intended use and a disqualification for the policing job. Recall (§2.6) that every ciphertext is deciphered under all three hypotheses. Take a German ciphertext: the attempt made under the Latin hypothesis is, by construction, the most Latin-looking letter stream the search could make out of those symbols, and the head obligingly labels it Latin well over half the time. Across all such wrong-hypothesis attempts the head abstains only 2–18% of the time. The head is a check on "which language is this text" — used as a cross-check on short passages, where the density judge is weakest (§5, Phase 5) — and on "is this any trained language at all" for gibberish-like input. It is **not** the instrument that decides whether a decipherment is real.

That job falls to the density judge after all, but asked a better question. Instead of "how predictable is this decipherment?" (gameable, §2.8), the question is "how much *more* predictable is this decipherment than the same letters in random order?" — the **shuffled-text margin**. A genuine decipherment is far more predictable than its own shuffle (in Phase 3: 1.6–2.9 bits per character below it); a wrong-hypothesis decipherment is only a little more (0.5–1.9), because the search can impose some local order on any letter stream but not the long-range structure of real text. The margin is computed per text, against that text's own letters, so it does not depend on any cross-language calibration (§2.4). It is the primary "is this language at all" test of the project — and, in the end, the statistic on which the manuscript verdict rests (§5, Phase 6).

Around these two instruments sit the controls that check them:

- **negative controls** — the gibberish generator and shuffled text (on which both instruments must say "no"), text in the three inventory languages (on which both must say "yes", and agree), and text in languages the model was *never trained on* — Dutch, English, French, Spanish — enciphered with the same ciphers, to see whether an unfamiliar language is mistaken for a familiar one, and at what level (language, or only family);
- an **abstention rule** — a fixed pair of thresholds on the density judge, written down *before* any manuscript number was looked at (§6): a decipherment counts as language-like only if its plaintext score is at clean-text level *and* its shuffled-text margin is in the range genuine decipherments occupy. The head's verdict is reported beside it, not folded into it.

An instrument that always names *some* language is useless; the combination above is what lets this one decline.

### 2.10 The manuscript as data

The manuscript's script is recorded in transcriptions using the EVA alphabet (a conventional assignment of Latin letters to Voynich glyph shapes). Two independent EVA transcriptions are used (Takahashi, and the "Reference" transcription), plus a third, independently made glyph-level transcription from Boxer's project. The text falls into two statistically distinct "dialects", **Currier A** and **Currier B**, which are always analysed separately and never pooled. Currier A is about 56,000 characters; B about 119,000.

---

## 3. What the abstract promises

Read as a list of claims, the abstract commits to the following. (Numbering is used in §7 for the comparison.)

- **A1.** Decoding-head architectures that handle mixed unigram–bigram homophonic ciphers (Naibbe) and arithmetic-encoded homophonic ciphers (Boxer).
- **A2.** A multilingual text-diffusion backbone with a jointly trained language-detection head, giving (a) improved robustness to noise and (b) the ELBO as a language metric.
- **A3.** Methods that "extend" the neural approach of Kambhatla et al. (2023) to harder cipher classes — i.e. end-to-end neural decipherment.
- **A4.** Validation on historical corpora enciphered with standard classical systems, and on the two Voynich-tuned ciphers.
- **A5.** Whitespace removed from every stream; word boundaries play no role.
- **A6.** Application to the manuscript with a ranked table of (cipher system × plaintext language).
- **A7.** The headline: Germanic candidates receive the highest likelihood; complex heads outscore simple ones; no single complex head dominates.
- **A8.** Rankings framed as exploratory and assumption-dependent, with limitations discussed.
- **A9.** MIT-licensed code and models released on publication.
- (Implicit) **A10.** The framing criticises earlier methods as hard to scale to many languages.

---

## 4. How the work was organised: phases and gates

The execution plan (`reference_docs/Diffusion Model Training - Task Breakdown.md`) orders the work so that each stage can be trusted before the next depends on it. Three ordering rules are load-bearing: all languages from the start; the language-detection head is attached only after the backbone has settled; and the measuring model is **frozen** before any cipher head is tuned against it (otherwise the instrument and the thing being measured would co-adapt).

| phase | purpose | gate | status |
|---|---|---|---|
| 0 | Freeze the alphabet, the language inventory, the corpora, the cleaning procedure and the held-out test data; pin the cipher generators; ingest the manuscript | G0 | **passed** |
| 1 | Train the backbone on clean text (85M and a cheaper 25M sibling) | G1 | **passed** 2026-08-21 |
| 2 | Continue training on deliberately corrupted text so the model tolerates partial decipherments | G2 | **passed** 2026-08-21 |
| 3 | Turn the score into a fair instrument: variance control, sample budget, calibration, fairness audit, synthetic recovery test | G3 | **passed** 2026-08-21 |
| 4 | Add the language-detection head, first without touching the backbone, then jointly with a small weight | G4 | **passed** 2026-08-22 |
| 5 | Validate the four cipher heads against the frozen model, in difficulty order; put all heads on one comparable scale | G5 | **passed** 2026-08-23 |
| 6 | Apply to the manuscript; controls; audit; length/family analysis; literature anchors; write-up | acceptance check (not a gate) | **complete** 2026-08-24; acceptance **FAIL on one sub-criterion** (gibberish abstention 0.89 < 0.95), all other checks PASS or stated WARN |

A gate is a scripted check with numerical acceptance criteria; each produces a report that is archived. Gate outcomes are recorded as PASS, or PASS with explicit WARN items that are carried forward rather than hidden.

---

## 5. Progress, phase by phase

### Phase 0 — decisions and data (complete)

- **Alphabet.** 25 letters (the 23 letters of Greshko's Naibbe tables plus *k* and *w*, needed for German), with *j* folded into *i* and *u*/*v* kept distinct. A discrepancy in the literature ("~22 letters" vs 23) was resolved by reading Greshko's actual tables.
- **Corpora.** Latin 27.3 million characters (72 documents, medieval and classical), Italian 4.2 million (7 works, Dante to Tasso), German 89.5 million (568 documents, early-modern prints). Period fit is imperfect for all three in the same direction (printed, literary/learned register); this is recorded, not hidden. About 500,000 characters per language were set aside before any training as untouched test material.
- **Cleaning.** One procedure for every language: accents and ligatures resolved, spaces and punctuation removed, with the rate of lost letters measured (below 0.0002%).
- **Cipher generators** pinned to specific versions of the two external repositories, so results are reproducible. The arithmetic cipher's "doubling" parameter was tuned per language to the manuscript's observed rate of repeated words.
- **Manuscript ingest.** Two EVA transcriptions, with counts reconciled against published figures (~37,000 words, ~230,000 characters before space removal).

### Phase 1 — clean-text backbone (complete)

Both models trained to a plateau on all three languages simultaneously. Final held-out readings for the 85M model: Latin 2.35, Italian 2.55, German 1.90 bits per character. (German scores lowest not because the model is better at it but because German text, with its long compounds and regular spelling, is intrinsically more predictable — this is exactly the kind of difference calibration exists to quantify.) A subtle bookkeeping error — the slowly-updated "average" copy of the model's weights lagged the true weights — was diagnosed and fixed by a short additional run; the lesson is recorded in the phase notes.

### Phase 2 — noise curriculum (complete)

The clean model was further trained on a mixture that included text corrupted in the ways a partial decipherment would be: wrong-key substitutions applied self-consistently, spurious insertions/deletions mimicking wrong Naibbe parses, and ~5% transcription noise. Gate G2 confirmed that clean-text readings did not drift (under 0.5%) and that readings degrade *smoothly* as corruption increases — no cliff. A later side study (§5, "n-gram judge robustness") showed this phase was more important than its framing suggested: it is what lets the model rank languages *while* a decipherment is still half-wrong.

### Phase 3 — making the score an instrument (complete)

- **Variance control.** Reusing the same random hiding patterns across language settings reduces the noise in score *differences* by at least 5.5× — confirming the point of the design.
- **Sample budget.** 64 random hiding patterns per window is enough for rankings to flip in under 1% of repeated draws (on windows whose margin is resolvable at all).
- **Calibration — the Phase-3 finding.** The offsets of §2.4 were measured: Latin +0.138, Italian +0.013, German +0.205 bits per character (the diffusion bound is loosest on German). The original design said to subtract them at ranking time. Doing so was tested and **broke the instrument**: recovery of the true language on synthetic ciphers fell from 98% to 70%. The reason is instructive. When the same ciphertext is deciphered under two language hypotheses, the two outputs are *nearly the same text*, and the genuine difference between them can be as small as 0.03 bits per character; an offset of 0.13 swamps it. The offsets are valid on clean own-language text but have no meaning in that comparison. The adopted policy is **report-only**: the offsets are measured, stored and re-measured after every phase, but never subtracted; instead they define the *systematic uncertainty* of every cross-language margin (0.07–0.19 bits per character, depending on the pair), and any margin smaller than that is reported as "unresolved at calibration precision".
- **Fairness audit.** With only three languages, no correlation can be tested statistically, so the audit escalates anything above noise rather than explaining it away. Four findings are escalated: the offsets depend on which reference model is used more than on their own measurement error (so they are estimates of slack, not proofs of comparable tightness), and they differ by language beyond the document-to-document spread. Both are carried into the final write-up as stated residual risk.
- **Synthetic recovery.** On 750 simple-substitution ciphers of held-out text, the instrument recovers the true language in 98.4% of cases at 200 or more characters — above the 97.1% literature bar. The residual errors are not Latin-vs-Italian confusions but a few unusually hard Latin documents that tie with the German setting on the same text.

### Phase 4 — language-detection head (complete)

A small classifier was attached to the backbone: first with the backbone frozen (it reached 100% on clean long text and abstained on ≥95% of negative controls), then trained jointly with a small weight (about 0.003 — far smaller than the 0.05 the design anticipated, set by the rule that its influence stay under 10% of the main training signal). Per-language readings moved by at most 0.2%, and the synthetic ranking test of Phase 3 *improved* slightly (98.9% language, 99.1% family). Two things were learned about the head's role: it is a useful cross-check on short texts, but it is **not** a reliable detector of wrong-key decipherments — that job belongs to the shuffled-text margin. A planned replication across three random seeds (task 4.7) is paused at resumable checkpoints; every later number comes from a single 85M model.

**Side study — n-gram judges.** Classical letter-statistics models (n-grams) were scored on the same corrupted texts. They fail as language judges under noise in a *directional* way: corrupted text of any language drifts toward being called "German" (the most forgiving model), and the drift grows with corruption. The noise-trained diffusion model does not do this. Two consequences: the fairness emphasis was vindicated on a simpler instrument; and the credit for robustness belongs to the training curriculum, not to the architecture (the *clean-trained* diffusion model failed as fast as a trigram model).

### Phase 5 — cipher heads on the frozen model (complete, G5 passed 2026-08-23)

The evaluator was frozen (its exact file fingerprint is recorded), and each head was validated on synthetic ciphertexts of its class made from held-out text. Throughout, the structure is **two-tier**: a cheap classical search driven by letter statistics proposes a *shortlist* of candidate keys; the frozen diffusion model scores every shortlist candidate under every language.

| rung | test | result | acceptance |
|---|---|---|---|
| 1 simple substitution | 300 ciphers, 50–700 chars | at ≥200 chars, 0.16% of symbols wrong; 99.4% language recovery | PASS |
| 2 homophonic (Zodiac-408 class: 408 chars, 54 symbols) | 18 ciphers | 17 of 18 at ≤1.9% symbol error (the Zodiac literature bar), median 0; one Latin cipher never found by the search | PASS per instance; **WARN** on the mean (4.1%) |
| 3 Naibbe | 12 ciphers of 10,000 letters | 99.8% of glyph-type assignments correct (occurrence-weighted); 12 of 12 languages | PASS |
| 4 arithmetic | 9 ciphers of 300 letters | partial decipherments only (45–66% of symbols wrong) but language right in 7 of 9, family in 8 of 9 (chance: 3 and 5) | PASS (P1 task: "better than chance") |
| cross-head scale | 24 ciphers, every head applied to every cipher | the description-length total picks the true cipher class in 24 of 24 | PASS |

Three findings from this phase changed the design:

1. **Gradients do not move keys** (the §2.7 contradiction, seen from the search side). The diffusion model's soft-input capability was verified mechanically, but tuning a key by gradient descent through it never improved a decipherment at any rung; the gradient is dominated by "make the key more certain", not "change which letter". The mechanism was diagnosed, not guessed. What *does* work is using the model as a **judge of discrete moves**: propose every single-symbol change to a key, score them all in one batch, accept the best, repeat. This "ELBO polish" reached the exact true key on a Latin cipher where the classical search had stalled, and on two languages ended *better* than the best candidate in the shortlist — something re-ranking alone cannot do.
2. **Any likelihood judge prefers degenerate verbose decipherments** (§2.8). Ranking rung-2 cells on the raw score would have chosen a degenerate key on 6 of 18 ciphers and recovered the language in only 12 of 18; the description-length total chooses none and recovers 18 of 18.
3. **Latin is the hard language for the search**, at every rung (93–97% solved at ≥200 chars vs 100% for Italian and German; the one rung-2 failure is Latin). The *judge* is symmetric — the per-language readings held at every gate — but the classical search is not, because Latin held-out text is the least predictable in the inventory. "Could not find the key under Latin" and "not Latin" are distinguishable only if per-language search success is reported alongside every result, which is now required.

Two WARNs were recorded at G5: the rung-2 mean (one unsolved instance), and the literature anchors — Zodiac-408 is English, outside the inventory, so it cannot be scored by this instrument; the Borg and BnF fr2988 ciphers had not yet been fetched.

### Phase 6 — application to the manuscript (complete, 2026-08-24)

Every number below comes from the frozen Gate-G4 evaluator under the report-only calibration, 64 random hiding patterns × 4 replicate seeds per window, with the same hiding patterns reused across the three language settings.

**Setup (6.1).**

- **Presentations.** Deciding how a transcription becomes each head's input is itself a set of decisions, frozen and recorded with coverage: EVA characters for rungs 1–2 (100% coverage, 20–23 symbols); manuscript words that Greshko's Naibbe tables could have produced for rung 3 (65–78% of characters — the unparseable remainder is the one Greshko himself reports); Boxer's glyph transcription, its 16 most frequent glyph types, for rung 4 (90–92% of glyphs). Two EVA transcriptions (Takahashi, Reference) and the Boxer transcription are run so that transcription-dependence can be reported. Currier A is ~56,000 characters, B ~119,000; never pooled.
- **A structural finding about the arithmetic cipher, before any solve.** Boxer's cipher produces words whose symbols are in a globally sorted order — its strongest signature, which the rung-4 head exploits to find word boundaries in an unbroken stream. On the manuscript's glyph stream **no sorted order exists that is compatible with the cipher's own word-length rules**. The head therefore cannot run in its validated form; it is run in the weaker form that uses the manuscript's visible word boundaries instead.
- **Ranking rule and abstention rule (6.2).** Every cell carries the description-length total per ciphertext symbol (uncovered symbols charged at the manuscript's own letter-statistics rate, so partial coverage cannot masquerade as compression), the three component terms, coverage, per-window spread, replicate flip-rate, calibration margin uncertainty, the shuffled-text structure margin, and agreement across windows, transcriptions, heads and dialects. A **no-cipher baseline** sits beside every table: how well the manuscript's own surface letter statistics predict it (2.18–2.23 bits per symbol for Currier A, 1.91–1.94 for B). The abstention rule was fixed *before any manuscript number was read*: a cell counts as language-like only if its plaintext score is at clean-text level (≤3.0 bits per character) *and* it sits at least 1.5 bits per character below its own shuffled copy.
- Compute: 87 solve jobs on 12 CPU workers (2.5 h), 87 scoring jobs on two GPUs (3.6 h), then the control battery (39 instances, 309 solves) and the anchors through the same pipeline.

**Result on the manuscript (6.2): every dialect of every transcription abstains — 0 of 87 cells is language-like.**

| table | best cell by description length (bits / symbol) | beats no-cipher baseline? | largest structure margin in table | verdict |
|---|---|---|---|---|
| Takahashi + Boxer, Currier A | Naibbe / Latin, 2.09 | **yes** (2.18) | — | ABSTAIN |
| Takahashi + Boxer, Currier B | Naibbe / Latin, 1.92 | no (1.91) | 1.25 (homophonic / German) | ABSTAIN |
| Reference, Currier A | Naibbe / Latin, 2.12 | **yes** (2.23) | — | ABSTAIN |
| Reference, Currier B | Naibbe / German, 1.97 | no (1.94) | — | ABSTAIN |

Three readings, in order of what the numbers support:

1. **No decipherment.** The two simple heads (1:1 and homophonic) do reach clean-text-level bits (2.47–3.05 bits per character) — but exactly as Phase 5 warned a flexible head can on *any* stream, by finding a low-entropy mapping. Their structure margins (0.87–1.25) say so: shuffling the decode costs about what shuffling a wrong-key decipherment costs. Inspected, the best decode (homophonic, German, Currier B, 2.47 bits per character) is a stream of a few recurring pseudo-words — `ersti`, `insti`, `wunam`, `accp` — that re-express the manuscript's own word-level repetitiveness in Latin letters; no content words, no inflectional variety, no function-word skeleton. The two heads with genuine cipher grammars (Naibbe, arithmetic) return decodes at 3.5–4.3 bits per character, i.e. deep in the wrong-key plateau: no key was found.
2. **The language ranking among these non-decipherments is noise.** Within-head margins are 0.001–0.083 bits per ciphertext symbol against a calibration uncertainty of 0.067–0.193 for the relevant pair — every margin is at or below the systematic uncertainty. The heads disagree with each other (agreement 0.25–0.67), the two transcriptions disagree on Currier B, A and B disagree under the homophonic head. The replicate flip-rate is 0 on 80 of 87 cells, so this is not sampling noise: different hypotheses genuinely prefer different languages, which is what ranking non-decipherments looks like.
3. **One structural result survives, about description length rather than language.** The Naibbe hypothesis ranks first on the description-length total in every table because a Naibbe parse compresses the glyph stream roughly 2:1, and on **Currier A** — both transcriptions independently — that total is *below* the manuscript's own held-out letter-statistics baseline (2.09 vs 2.18; 2.12 vs 2.23). On **Currier B** no cipher hypothesis beats the baseline. So a Naibbe-shaped word grammar (one or two letters per word-piece) describes Currier A slightly better than a character model of Currier A does, while the plaintext it produces is not language (3.5 bits per character, structure margin 0.07–0.09). That A and B differ here is consistent with their known difference in word structure. It is the one place in the whole table where a cipher hypothesis buys anything.

**Negative-control battery (6.3)** — 39 instances through the identical pipeline:

| control | n | abstained | structure-margin band | note |
|---|---|---|---|---|
| positives (real Latin/Italian/German under 1:1 and Naibbe) | 9 | 1 of 9 (false abstention) | **1.49–2.48** | 8/9 language correct; all nine 1:1 keys recovered exactly |
| Voynich-mimicking gibberish | 9 | 8 of 9 = 0.89 | 0.92–1.51 | one miss at margin 1.51 — **fails the > 0.95 criterion** |
| shuffled text | 9 | 9 of 9 | 0.03–0.06 | |
| out-of-inventory languages (Dutch, English, French, Spanish) | 12 | 12 of 12 | 0.60–1.43 | never assigned a language; with abstention switched off, family-correct only 7/12 (chance 5/12) |

The one false accept (1.51) and the one false abstention (1.49) are the same fact from two sides: the frozen threshold of 1.5 sits inside the narrow overlap Phase 3 had measured. The margin is very nearly a separating statistic on this battery — one error on each side out of 39 — but not quite, and the acceptance check therefore records a **FAIL** on the gibberish criterion. The rule was not re-tuned after seeing the data; that is the point of freezing it. Crucially, the manuscript verdict does not depend on the choice: the whole VMS table (0.04–1.25) lies *below every positive control*, so any threshold from 1.26 to 1.48 abstains on the entire manuscript and on no genuine decipherment. The contamination row also quantifies the risk the design named: an untrained language is never *called* a trained one here only because the abstention rule is in force.

**Fairness audit re-run (6.4)** — same offsets as Phase 4 (Latin +0.138, Italian +0.013, German +0.205), same four escalated findings, re-attached; report-only policy unchanged.

**Length sensitivity and family confusion (6.5)** — language accuracy 74% at 50 characters, 94% at 100, 98% at 200, 99% at 400+. At 50 characters the dominant confusion is Latin↔Italian; from 100 on, the residual errors are Latin→German on unusually hard documents. Claims are supported at the language level for ≥200 characters of a *correct* decipherment. Within-Germanic resolution is not possible inside a three-language inventory, and the Dutch/English controls show that an untrained Germanic language is *not* reliably pulled to German — so it is not supported even at family level.

**Literature anchors (6.6):**

- **Zodiac-408** (English, 54 symbols, 408 characters): the classical search alone recovers the key with **0.98%** of symbols wrong (literature target ≤1.9%) — PASS, but as a check on the search only, since English is outside the inventory and the diffusion judge cannot score it.
- **Borg** (a real Latin homophonic cipher, 55 symbol types, 120,000 symbols): through the full pipeline the instrument ranks it **Latin** by 0.25 bits per symbol — the largest margin anywhere in Phase 6, three times the calibration uncertainty — and its Latin decode is readable (median page 11% of symbols wrong, best page 3.5%). It nonetheless **fails** the ≤4.1% literature target as we measure it (12.9% before polish, 22.6% after), for two reasons that are ours: the published plaintext is a corrected, expanded scholarly edition rather than a symbol-aligned transcription, and we drop rare glyph types. Not a like-for-like comparison; obtaining the aligned transcription is a carry-over.
- **BnF fr2988**: no transcription available — not run (standing WARN).

**A negative finding about the Phase-5 outer tier.** The discrete "ELBO polish" that improved keys on the Phase-5 synthetics was applied to Borg, accepted on all six cells, and made the decipherment *worse* every time (median page error 11% → 22%, 25 of 55 symbols reassigned). The mechanism is selection bias: each sweep picks the best of ~1,400 cheaply estimated moves, so the winner is systematically over-rated, and the one careful confirmation at the end compares only the final key against the start and cannot undo a chain of individually biased steps. This is a scale effect the small Phase-5 ciphers could not reveal. It did *not* produce the manuscript abstention — re-scoring the best VMS cells with their pre-polish keys gives margins of 0.83–0.84 against 1.05–1.11 after, both far short of 1.5 — but it must be fixed (per-move confirmation at full budget, or a bias-corrected selection rule) before the polish is reused on anything larger than the synthetics.

**Acceptance roll-up.** `scripts/phase6_check.py`: 6.1, 6.2, 6.4, 6.5 and the freeze discipline all PASS; 6.3 gibberish abstention **FAIL** (0.89); 6.3 positives WARN (one false abstention); 6.6 Borg WARN (SER target missed, language recovered); 6.6 BnF WARN (unavailable). **Overall: FAIL on one P0 sub-criterion, reported as such.**

---

## 6. Comparing the abstract with what exists

| # | abstract claim | status | comment |
|---|---|---|---|
| A1 | Heads for Naibbe and arithmetic ciphers | **delivered; arithmetic head partial** | Naibbe: 99.8% correct assignments on synthetic ciphers. Arithmetic: correct language 7/9 but decipherments only ~half right; on the manuscript the head had to run in a weakened form because the cipher's sorted-symbol signature is absent (§5, Phase 6). |
| A2a | Joint diffusion backbone + language head → robustness to noise | **delivered, credit reassigned** | Robustness is real and measured; it comes from the noise curriculum (Phase 2), not the architecture. The head is a short-text cross-check, not the wrong-key detector the abstract implies. |
| A2b | ELBO available as a language metric | **delivered with a qualification** | Works at ≥200 characters of a good decipherment (99%). For verbose ciphers the raw ELBO is unsafe; the metric is the description-length total (§2.8). Below ~100 characters the classical model is the better judge. |
| A3 | End-to-end neural decipherment | **changed in character** | Gradients through the model never improve a key. The working system is classical search + neural judge + neural discrete polish. The paper's framing should say so. |
| A4 | Validation on classical ciphers and both Voynich-tuned ciphers | **delivered; historical anchors partial** | Rungs 1–4 on held-out text in all three languages; cross-head comparison 24/24. Historical anchors: Zodiac-408 solved to 0.98% error (search only — English is outside the inventory); Borg ranked Latin with the phase's largest margin but misses its symbol-error target under a non-like-for-like alignment; BnF fr2988 unavailable. |
| A5 | Whitespace removed everywhere | **delivered exactly** | Applied from Phase 0 on. One consequence surfaced in Phase 6: with no sorted-order signature, the rung-4 head has to fall back on the manuscript's visible word breaks. |
| A6 | Ranked (cipher × language) table for the manuscript | **delivered — and the table abstains** | 87 cells with error bars, agreement, abstention flag and no-cipher baseline (more columns than the abstract describes). Every dialect of every transcription abstains; the language ordering is printed but flagged as a ranking among non-decipherments. |
| A7 | Germanic highest; complex heads beat simple; no complex head dominates | **not supported; to be replaced** | No language receives a language-like decipherment, so "highest likelihood" has no evidential content. The most complex head (Naibbe) does win the description-length comparison on every table — but by modelling word structure, not by deciphering. See §7. |
| A8 | Exploratory, assumption-dependent framing | **delivered and strengthened** | The write-up lists six assumptions and five residual risks explicitly, shows the actual decodes behind the verdict, and the instrument did in fact decline to name a language. |
| A9 | MIT release | **planned** | Code is in a single repository with pinned external dependencies and fingerprinted model files; not yet released. |
| A10 | Scaling to many languages | **not attempted** | The implementation uses three languages. Everything is built to extend, but "scales better than earlier methods" is not a claim the current work supports. |

---

## 7. What the headline is, now that the numbers exist

### 7.1 "Germanic candidates receive the highest likelihood" — not supported

The sentence has no evidential content on this data, for three reasons that stack.

- **Nothing was deciphered.** A language ranking only means something if at least one cell is a language-like decipherment; none is (§5, Phase 6). What the table shows is which language the *wrong-key plateau* tilts toward under each hypothesis, and §7.1 of the previous draft predicted correctly that this would be unstable: the four heads split between Latin and German at the top, the arithmetic head prefers Italian, and the two transcriptions disagree on Currier B.
- **The margins are inside the error bars.** Every within-head margin (0.001–0.083 bits per symbol) is at or below the calibration uncertainty of the pair involved (0.067–0.193). By the project's own reporting rule, all of them are "unresolved at calibration precision".
- **Even at family level the instrument cannot support "Germanic".** The inventory has one Germanic member, and the Dutch/English controls show that an untrained Germanic language is not reliably called German (family-correct 7/12 across the four contamination languages, chance 5/12). "Germanic" is the finest granularity the instrument could in principle deliver, and it does not deliver it here.

The honest headline is: **under this instrument, with this language inventory, these four cipher hypotheses and these transcriptions, the Voynich Manuscript yields no decipherment that behaves like Latin, Italian or German text — and the same pipeline does yield one for genuine ciphers of those languages** (nine synthetic positives; the historical Borg cipher). It does not say the manuscript is meaningless, nor rule out one of these languages under a cipher or tokenization outside the inventory.

### 7.2 "Complex heads outscore simple heads; no single complex head dominates" — half true, in a weaker sense

On the description-length total the most complex head, Naibbe, ranks first in all four tables, and on Currier A it is the only hypothesis that describes the manuscript more compactly than the manuscript's own letter statistics (2.09 vs 2.18 and 2.12 vs 2.23 bits per symbol). So one complex head *does* dominate the comparison — the abstract's "no single complex head" clause is wrong — and it dominates because a unigram/bigram word-piece grammar captures something real about Currier A's word structure. But the caution written before the numbers arrived applies exactly: a head that compresses the manuscript without producing a language-like decode is a model of the manuscript's *word structure*, not a key. The Naibbe decode sits at 3.5 bits per character with a structure margin of 0.07–0.09 — as far from language as anything in the table. Currier B, by contrast, is described better by its own surface statistics than by any cipher hypothesis.

### 7.3 The arithmetic cipher

The pre-solve finding stands: the arithmetic cipher's global sorted-symbol signature is absent from the manuscript's glyph stream, so of the two Voynich-tuned ciphers, one fails to match the manuscript on the property its own decipherer relies on. Run in the weakened word-boundary form, it produced the worst decodes in the table (4.0–4.3 bits per character) and, alone among the heads, preferred Italian — another sign that its language preference is an artefact of the plateau rather than a signal.

---

## 8. Risks and limitations that will stand regardless of the result

- **Bound comparability (R1)** is measured, not eliminated. The audit's escalated findings mean the calibration offsets are estimates of slack that depend on the reference model used to measure them.
- **Search asymmetry.** Latin is harder to solve at every rung. Per-language search success is reported beside every number, and a larger Latin search budget is a legitimate, documented option. (On the manuscript this cannot have produced the abstention — no hypothesis, including the easier-to-search Italian and German, found a key.)
- **The abstention threshold sits in an overlap band.** The 1.5-bit structure-margin cut is bracketed by a true decipherment at 1.49 and a gibberish instance at 1.51 in a battery of 39. The manuscript verdict is insensitive to it (any cut in 1.26–1.48 gives the same answer), but the rule's operating point is known only to the precision a 39-instance battery allows.
- **The outer-tier polish has a demonstrated failure mode at scale** (selection bias on Borg). It did not produce the manuscript result, but it is not fit for reuse until fixed.
- **Transcription dependence.** Three transcriptions are used and their agreement reported, but all share the EVA decomposition of glyphs; a different decomposition is a different ciphertext.
- **Coverage and tokenization.** The Naibbe and arithmetic heads explain subsets of the manuscript (65–78% and 90–92%); totals are given both per covered symbol and per all symbols. Only one EVA tokenization was tried; the sensitivity of the abstention to tokenization is the obvious study still owed.
- **Period and register of the corpora** — printed, literary/learned, not strictly 15th-century — in the same direction for all three languages.
- **A single trained model.** The seed replication is paused; ranking stability across independently trained models is not yet available.
- **Three languages.** Any ranking is a ranking within Latin, Italian and German. The contamination controls say what an untrained language looks like to the instrument; they do not extend the inventory.
- **Anchors.** Only Borg (Latin) validates the machinery end-to-end on a real historical cipher, and it validates the *language ranking* (Latin, by the phase's largest margin) while missing its symbol-error target under an alignment that is not like-for-like; the other two literature anchors are unavailable or outside the inventory.

---

## 9. What happens next

1. **Revise the abstract** to match what was found: A7 becomes an abstention result (no language-like decipherment; the Naibbe word-grammar finding on Currier A as the one structural positive), A3 (hybrid, not end-to-end), A2a (curriculum, not architecture), A2b (description length, not raw ELBO), A10 (three languages).
2. **Fix the outer tier** before any reuse: per-move full-budget confirmation or a bias-corrected selection rule for the ELBO polish (the Borg finding).
3. **Sensitivity studies the abstention deserves**: a sweep over EVA tokenizations (glyph units, benched gallows as single symbols); a larger control battery — or a length-matched margin normalisation — around the 1.26–1.51 overlap band so the threshold's operating point is better known.
4. **Anchors**: obtain the DECODE-aligned Borg transcription and the BnF fr2988 transcription for a like-for-like symbol-error comparison.
5. Optional strengthening: resume the seed replication (4.7) so ranking stability across independently trained models can be stated.
6. **Two control experiments the current argument owes a reviewer.** Both address the two-tier search of §5 (Phase 5, finding 1) — a cheap classical search proposes keys under each language hypothesis, the diffusion model only judges — whose justification is so far mechanistic rather than experimental. Both run on the existing synthetic suites with known answers, not on the manuscript, and neither touches the frozen evaluator used for every Phase-6 number.
   - **6a. Can one model be both a fair judge and a search target?**
     *The objection.* The paper reports that the diffusion model failed as a target for finding the key — its score is flat far from the key, where the classical n-gram score still has a slope — and then builds a two-tier system around that failure. But the model was never trained to be a search target; it was trained to be a judge. A reviewer is entitled to ask why the authors did not train it for the job — jointly, or in alternation with the key search, in the manner of adversarial training schemes — before concluding it could not do it. Why should anyone be surprised that a model struggles at a task it was never trained for?
     *The authors' answer, which this experiment tests.* The flatness is not an accident of training; it is what the training deliberately did. The noise curriculum (Phase 2) showed the model text with 20–50% of positions under a wrong key, labelled with its true language, so that it would learn to *see through* key errors — because only a judge whose language call does not move as the key goes from 0% to 50% wrong can rank languages fairly when the search is harder for some hypotheses than others (Latin is solved 93–97% of the time, Italian and German 100%). *Slope toward the key* (what a search target needs) and *insensitivity to key errors* (what a fair judge needs) are therefore contradictory requirements for one score. The n-gram judge is the existence proof of the other horn: it has slope all the way down and, as a language judge, drifts toward "German" under any corruption. Training the diffusion model to have slope would be training it to become the n-gram in this respect; joint training would additionally let the search's Latin asymmetry leak into the instrument that is supposed to measure Latin-ness, and would require re-doing the calibration and abstention thresholds (all defined on a frozen model) after every update. This is the §2.7 argument; it is mechanistic, has not been tested, and should be. Test it: fine-tune a *copy* of the evaluator with a key-sensitivity objective on synthetic ciphers (or, more simply, re-run the Phase-2 curriculum *without* the wrong-key family), then measure on held-out synthetics (i) key-search success from random starts under that model, (ii) the stability of its language call as the key is corrupted from 0% to 50% wrong (the side-study grid of `docs/ngram_judge_robustness.md`), and (iii) the per-language held-out canary. Predicted outcome: slope up, robustness and cross-language fairness down. If the prediction fails — if one model can have both — the two-tier design is unnecessary and the paper should say so.
   - **6b. Does the per-hypothesis inner search bias the language ranking?**
     *The objection.* The judge is symmetric across languages (the per-language canary certifies that), but the judge is not where the language hypothesis enters. It enters in the *search*: every ciphertext is solved three times, once under each language's n-gram model, and those three searches are not equally successful — Latin keys are found 93–97% of the time, Italian and German keys 100%. A language that is harder to *search* will, on some fraction of ciphertexts, present the judge with a worse decipherment for reasons unrelated to whether the text is in that language. "Could not find the key under Latin" is then indistinguishable from "not Latin". The current mitigation — reporting per-language solve success beside every number and giving every hypothesis the same budget — is disclosure, not removal. A hypothesis-free inner search would remove the asymmetry at source, and none was ever run; the trade study comparing per-hypothesis search against a search that carries no language hypothesis does not exist.
     *What the experiment settles.* Whether taking the hypothesis out of the search stage changes the language ranking on ciphertexts whose answer is known — and at what cost in search power, since a pooled objective is blunter than a per-language one. Remove the hypothesis from the search stage and see whether the ranking changes: on the rung-1 suite (750 ciphers, 50–700 characters) and then the rung-2 suite (where the one Latin failure occurred), compare under identical budgets (A) the current hybrid; (B) a *language-pooled* n-gram (trained on all three corpora mixed) driving one search per ciphertext, the single decode then scored under the three language dials; (C) arm B followed by the discrete ELBO polish run on the model's language-free (unconditional) dial. Report per arm: solve success per language, language recovery, and the flip-rate on the Latin–Italian and Latin–German pairs. Cost: pooled-LM training minutes; ~1.5 h of CPU for the rung-1 solves; ~1 h of GPU for scoring.

---

## Glossary

- **Bits per character** — the instrument's reading: how many yes/no questions a model needs, on average, to guess each hidden character. Lower = more predictable = more like the language in question.
- **Language model** — a program assigning probabilities to text. *n-gram* models use letter-sequence frequencies; the *diffusion* model here is a large neural network trained to fill in hidden characters.
- **Masked diffusion model** — the fill-in-the-blanks model of §2.3. "Masked" because characters are hidden; "diffusion" from the family of methods it belongs to.
- **Reader / Filler** — the two experts of §2.4. A Reader guesses each letter from the ones before it (a conventional left-to-right model; its score is exact). A Filler guesses randomly hidden letters from whatever is visible (the diffusion model; its score is a bound).
- **ELBO / bound** — the Filler's score: guaranteed to be no more optimistic than the truth. Its *slack* may differ by language; see *calibration*.
- **Calibration offset** — the measured slack of the bound for each language, obtained by comparison with a model whose probabilities are exact. Measured and carried as uncertainty; not subtracted (the Phase-3 finding).
- **Common random numbers** — reusing the same random hiding patterns across the language settings, so that score differences reflect the model rather than the dice.
- **Cipher head** — the component that holds a candidate key for one cipher class and turns ciphertext into candidate plaintext.
- **Two-tier search** — a cheap classical search proposes a shortlist of keys; the frozen diffusion model judges the shortlist and polishes the winner by scoring discrete single-symbol changes.
- **Symbol error rate (SER)** — the fraction of ciphertext symbols that a recovered key assigns to the wrong letter.
- **Description length (MDL)** — the total cost of explaining a ciphertext: plaintext bits + key bits + "choice bits" for every freedom the encipherer exercised. The ranking rule for every cell of the final table.
- **Language head** — the classifier of §2.9 attached to the diffusion model: answers *Latin / Italian / German / none of these*. A cross-check on short texts and a gibberish detector; not a detector of wrong-key decipherments.
- **Shuffled-text margin** — how far below its own letter-shuffled copy a decipherment scores; the primary "is this language at all" test.
- **Abstention rule** — the pre-registered thresholds (plaintext ≤3.0 bits per character and shuffled-text margin ≥1.5) that a cell must meet to be called language-like. No manuscript cell met them.
- **Currier A / B** — the two statistically distinct text groups within the manuscript; never pooled.
- **EVA** — the conventional alphabet used to transcribe Voynich glyphs into Latin letters.
- **Gate** — a scripted checkpoint with numerical acceptance criteria that must pass before the next phase begins.
- **Held-out data** — text set aside before training and never shown to the model, used for every measurement reported here.

## Source records

- Execution plan: `reference_docs/Diffusion Model Training - Task Breakdown.md`; design: `reference_docs/Design and Training of the Multilingual Diffusion Backbone.md`.
- Phase records: `docs/phase0_decisions.md`, `phase1_status.md`, `phase2_status.md`, `phase3_status.md` (+ `phase3_fairness_audit.md`), `phase4_status.md` (+ `phase4_fairness_audit.md`), `phase5_status.md`, `phase6_status.md`, `phase6_fairness_audit.md`, `phase6_writeup.md`.
- Side studies: `docs/ngram_judge_robustness.md`, `docs/cipher_heads_status.md`, `docs/vms_doubling_rate.md`, `docs/vms_token_length_by_hand.md`, `docs/rung4_arithmetic_design.md`.
- Gate reports and all numerical artifacts: `DATA_ROOT/runs/g{0..5}_report.json`, `DATA_ROOT/runs/phase6_report.json`, `DATA_ROOT/analysis/phase{3..6}/` (manuscript table `analysis/phase6/vms_report.md`, decodes `samples.md`, controls `controls/report.md`, anchors `anchors/`).
